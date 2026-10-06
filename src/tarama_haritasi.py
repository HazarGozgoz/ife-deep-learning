# -*- coding: utf-8 -*-
"""
tarama_haritasi.py -- kesimlerin taramadaki yerini numaralandırır
===================================================================
SORUN
------
Defterden hasta adlarını/barkodlarını aldınız ama elinizdeki kırpılmış
görüntünün ana taramanın neresinden geldiğini, yani kaçıncı hastaya ait
olduğunu göremiyorsunuz. Bir taramada 24'e kadar hasta paneli var.

ÇÖZÜM
------
Koordinatlar dosya adında zaten kayıtlı. Bu script her tarama için:

  1) Taramanın küçültülmüş bir kopyasını üretir ve her kırpmanın üstüne
     SIRA NUMARASI basar (kırmızı kutu + büyük numara).
  2) Aynı numaraları içeren bir CSV üretir; siz defterdeki adı/barkodu
     karşısına yazarsınız.

NUMARALANDIRMA
---------------
Okuma sırası: yukarıdan aşağı satır satır, her satırda soldan sağa.
Satırlar SABİT bir ızgarayla değil, y koordinatları kümelenerek bulunur --
tarama düzenleri değişken (kimi taramada 4x4=16 panel, kimisinde 4x6=24).
Böylece hangi düzende olursa olsun numaralar tutarlı çıkar.

Defteriniz farklı bir sırada tutuluyorsa (ör. sütun sütun) haritadaki
numaralara bakıp kendi sıranızı çıkarabilirsiniz -- harita gerçek
görüntüyü gösterdiği için hangi numaranın hangi panele denk geldiği
gözle doğrulanabilir.

KULLANIM
---------
    python tarama_haritasi.py --taramalar A4_Taramalar
    python tarama_haritasi.py --taramalar A4_Taramalar --sadece helena
"""
from __future__ import annotations

import argparse
from pathlib import Path

import itertools
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

MASTER = "annotations_master.csv"
OUT_KLASOR = "tarama_haritalari"
OUT_CSV = "defter_eslestirme.csv"
OUT_SORUN = "harita_sorunlari.csv"


def iou(a, b) -> float:
    """Iki kutunun ortusme orani. 0.5 ustu = ayni paneli iki kez kesmissiniz."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    k = (x2 - x1) * (y2 - y1)
    return k / (aw * ah + bw * bh - k)


def kumelendir(degerler: np.ndarray, tolerans: float) -> np.ndarray:
    """
    Yakın değerleri aynı gruba toplar (satır/sütun bulmak için).
    Sabit ızgara varsaymaz; sıralayıp aradaki boşluk toleransı aşınca
    yeni grup açar.
    """
    sira = np.argsort(degerler)
    grup = np.zeros(len(degerler), dtype=int)
    g = 0
    for i in range(1, len(sira)):
        if degerler[sira[i]] - degerler[sira[i - 1]] > tolerans:
            g += 1
        grup[sira[i]] = g
    return grup


def yazi_tipi(boyut: int):
    for yol in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "C:/Windows/Fonts/arialbd.ttf"):
        try:
            return ImageFont.truetype(yol, boyut)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=MASTER)
    ap.add_argument("--taramalar", default="A4_Taramalar")
    ap.add_argument("--cikti", default=OUT_KLASOR)
    ap.add_argument("--sadece", default=None,
                    help="Yalnızca adında bu geçen taramalar (ör. helena)")
    ap.add_argument("--genislik", type=int, default=2200,
                    help="Harita görüntüsünün piksel genişliği")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="Bu orandan fazla örtüşen kırpmalar TEKRAR sayılır")
    ap.add_argument("--sil-isaretli", action="store_true",
                    help=f"{OUT_SORUN} dosyasında 'sil' sütununa yazdığınız "
                         f"kırpmaları master'dan siler")
    args = ap.parse_args()

    # --sil-isaretli EN BAŞTA: aşağıdaki analiz raporu yeniden yazdığı için
    # sonra çalıştırılsaydı işaretlemeleriniz kaybolurdu.
    if args.sil_isaretli:
        yol = Path(OUT_SORUN)
        if not yol.exists():
            raise SystemExit(f"'{OUT_SORUN}' yok. Önce haritayı üretin.")
        t = pd.read_csv(yol)
        hedef = {str(x).strip() for x in t.get("sil", pd.Series(dtype=str)).fillna("")
                 if str(x).strip() and str(x).strip().lower() != "nan"}
        if not hedef:
            raise SystemExit(
                f"'{OUT_SORUN}' içindeki 'sil' sütunu boş.\n"
                f"Silmek istediğiniz kırpmanın dosya adını o satırın 'sil'\n"
                f"hücresine yapıştırıp tekrar çalıştırın.")
        tam = pd.read_csv(args.input)
        bulunan = tam["filename"].astype(str).isin(hedef)
        print(f"'sil' olarak işaretlenen : {len(hedef)}")
        print(f"Master'da bulunan        : {int(bulunan.sum())}")
        if not bulunan.any():
            raise SystemExit("Silinecek kayıt bulunamadı.")
        yedek = Path(args.input).with_name(
            Path(args.input).stem + "_silme_oncesi_yedek.csv")
        shutil.copy2(args.input, yedek)
        tam[~bulunan].to_csv(args.input, index=False, encoding="utf-8-sig")
        print(f"\n{int(bulunan.sum())} kayıt silindi. Yedek: {yedek}")
        print("Şimdi:  python veri_hazirlama.py")
        return

    df = pd.read_csv(args.input)
    var = df[["x", "y", "w", "h"]].notna().all(axis=1)
    eksik = int((~var).sum())
    df = df[var].copy()
    if args.sadece:
        df = df[df["source_file"].astype(str).str.lower()
                .str.contains(args.sadece.lower())]
    print(f"Koordinatı olan kayıt: {len(df)}"
          + (f"  |  koordinatsız (haritalanamaz): {eksik}" if eksik else ""))

    kok = Path(args.taramalar)
    out = Path(args.cikti)
    out.mkdir(exist_ok=True)
    satirlar: list[dict] = []
    sorunlar: list[dict] = []

    for tarama, g in df.groupby(df["source_file"].astype(str)):
        g = g.copy()
        g["cx"] = g["x"] + g["w"] / 2
        g["cy"] = g["y"] + g["h"] / 2
        # Tolerans: tipik kırpma yüksekliğinin yarısı -> aynı satırdaki
        # panellerin küçük y farkları tek grupta kalır
        tol_y = float(g["h"].median()) * 0.5
        tol_x = float(g["w"].median()) * 0.5
        g["satir"] = kumelendir(g["cy"].to_numpy(), tol_y)
        g["sutun"] = kumelendir(g["cx"].to_numpy(), tol_x)
        g = g.sort_values(["satir", "sutun"]).reset_index(drop=True)
        g["sira_no"] = range(1, len(g) + 1)

        n_satir = g["satir"].nunique()
        n_sutun = g["sutun"].nunique()

        # --- TEKRAR EDEN KIRPMALAR (ayni paneli iki kez kesmissiniz) ---
        arr = g[["x", "y", "w", "h"]].values
        tekrar_idx: set[int] = set()
        for i, j in itertools.combinations(range(len(g)), 2):
            if iou(arr[i], arr[j]) > args.iou:
                tekrar_idx.add(i)
                tekrar_idx.add(j)
                sorunlar.append({
                    "kaynak_tarama": tarama, "sorun": "TEKRAR",
                    "sira_no_1": int(g.at[i, "sira_no"]),
                    "sira_no_2": int(g.at[j, "sira_no"]),
                    "ortusme": round(iou(arr[i], arr[j]), 2),
                    "dosya_1": g.at[i, "filename"], "patern_1": g.at[i, "pattern"],
                    "dosya_2": g.at[j, "filename"], "patern_2": g.at[j, "pattern"],
                    "etiket_ayni": int(g.at[i, "pattern"] == g.at[j, "pattern"]),
                    "sil": "",
                })

        # --- HIC KESILMEMIS OLABILECEK PANELLER (izgara boslugu) ---
        satir_y = g.groupby("satir")["cy"].median().to_dict()
        sutun_x = g.groupby("sutun")["cx"].median().to_dict()
        dolu = set(zip(g["satir"], g["sutun"]))
        gw, gh = float(g["w"].median()), float(g["h"].median())
        bosluklar = []
        for r in sorted(satir_y):
            for c in sorted(sutun_x):
                if (r, c) not in dolu:
                    bx, by = sutun_x[c] - gw / 2, satir_y[r] - gh / 2
                    bosluklar.append((bx, by, gw, gh))
                    sorunlar.append({
                        "kaynak_tarama": tarama, "sorun": "BOS",
                        "sira_no_1": "", "sira_no_2": "", "ortusme": "",
                        "dosya_1": "", "patern_1": "", "dosya_2": "", "patern_2": "",
                        "etiket_ayni": "",
                        "satir": r + 1, "sutun": c + 1,
                        "yaklasik_x": int(bx), "yaklasik_y": int(by),
                        "sil": "",
                    })

        yol = kok / tarama
        if not yol.exists():
            adaylar = list(kok.glob(Path(tarama).stem + ".*"))
            yol = adaylar[0] if adaylar else None

        harita_adi = ""
        if yol is not None and yol.exists():
            try:
                im = Image.open(yol).convert("RGB")
                olcek = args.genislik / im.width
                im = im.resize((args.genislik, max(1, int(im.height * olcek))))
                d = ImageDraw.Draw(im)
                font = yazi_tipi(max(18, int(im.width * 0.022)))
                # Once bos hucreler: SARI kesik cerceve
                for (bx, by, bw, bh) in bosluklar:
                    d.rectangle([bx * olcek, by * olcek,
                                 (bx + bw) * olcek, (by + bh) * olcek],
                                outline=(235, 170, 0), width=6)
                    d.text((bx * olcek + 8, by * olcek + 6), "BOS?",
                           fill=(200, 130, 0), font=font)

                for i, r in g.iterrows():
                    x0, y0 = r["x"] * olcek, r["y"] * olcek
                    x1, y1 = (r["x"] + r["w"]) * olcek, (r["y"] + r["h"]) * olcek
                    # TEKRAR eden kirpma KIRMIZI, normal olan YESIL
                    renk = (220, 30, 30) if i in tekrar_idx else (20, 150, 60)
                    d.rectangle([x0, y0, x1, y1], outline=renk, width=4)
                    et = str(int(r["sira_no"])) + ("  2x" if i in tekrar_idx else "")
                    kutu = d.textbbox((0, 0), et, font=font)
                    kw, kh = kutu[2] - kutu[0], kutu[3] - kutu[1]
                    d.rectangle([x0, y0, x0 + kw + 10, y0 + kh + 10], fill=renk)
                    d.text((x0 + 5, y0 + 2), et, fill="white", font=font)
                harita_adi = Path(tarama).stem + "_harita.png"
                im.save(out / harita_adi)
            except Exception as e:
                print(f"  ATLANDI ({tarama}): {e}")

        print(f"  {tarama:<34} {len(g):3d} panel  ({n_satir} satır x {n_sutun} sütun)"
              + ("" if harita_adi else "   [tarama dosyası yok, sadece CSV]"))

        for _, r in g.iterrows():
            satirlar.append({
                "kaynak_tarama": tarama,
                "sira_no": int(r["sira_no"]),
                "satir": int(r["satir"]) + 1,
                "sutun": int(r["sutun"]) + 1,
                "kirpma_dosyasi": r["filename"],
                "cihaz": r.get("device", ""),
                "mevcut_patern": r.get("pattern", ""),
                "harita": harita_adi,
                # --- defterden dolduracaklarınız ---
                "defter_tarih": "",
                "defter_bas_harf": "",
                "defter_barkod": "",
                "defter_protokol": "",
                # Defterde kayıtlı DİĞER UZMANIN tanısı -- serbest metin
                # yazabilirsiniz ("monoklonal IgG kappa", "normal",
                # "şüpheli bant" gibi); defter_birlestir.py bunu kanonik
                # sınıfa çevirip sizinkiyle karşılaştıracak.
                "defter_uzman_tani": "",
                "not": "",
            })

    son = pd.DataFrame(satirlar)
    son.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    sr = pd.DataFrame(sorunlar)
    if len(sr):
        # Onceki 'sil' isaretlemelerinizi koru
        onceki = Path(OUT_SORUN)
        if onceki.exists():
            try:
                esk = pd.read_csv(onceki)
                if "sil" in esk.columns and "dosya_1" in esk.columns:
                    h = {(str(r["dosya_1"]), str(r["dosya_2"])): str(r["sil"]).strip()
                         for _, r in esk.iterrows()
                         if str(r.get("sil", "")).strip()
                         and str(r.get("sil", "")).strip().lower() != "nan"}
                    if h:
                        sr["sil"] = [h.get((str(a1), str(a2)), "") for a1, a2
                                     in zip(sr.get("dosya_1", ""), sr.get("dosya_2", ""))]
                        print(f"  ({len(h)} önceki 'sil' işaretlemeniz korundu)")
            except Exception:
                pass
        sr.to_csv(OUT_SORUN, index=False, encoding="utf-8-sig")
    print(f"\n{len(son)} panel, {son['kaynak_tarama'].nunique()} tarama")
    if len(sr):
        tek = sr[sr["sorun"] == "TEKRAR"]
        bos = sr[sr["sorun"] == "BOS"]
        print(f"\n=== HARITADA GORECEKLERINIZ ===")
        print(f"  YESIL  : normal kirpma")
        print(f"  KIRMIZI: ayni paneli iki kez kesmissiniz  -> {len(tek)} cift")
        if len(tek):
            fark = int((tek["etiket_ayni"] == 0).sum())
            print(f"           bunlarin {fark} tanesinde FARKLI tani vermissiniz")
        print(f"  SARI   : hic kesilmemis olabilir          -> {len(bos)} panel")
        print(f"\n  Ayrinti -> {OUT_SORUN}")
        print(f"  Silmek icin: o satirin 'sil' hucresine dosya adini yapistirip")
        print(f"      python tarama_haritasi.py --sil-isaretli")
    print(f"Haritalar -> {out}/")
    print(f"Liste     -> {OUT_CSV}")
    print("\nNASIL KULLANACAKSINIZ")
    print(f"  1) '{out}' klasöründen bir haritayı açın; her panelin köşesinde")
    print("     kırmızı kutu içinde sıra numarası var.")
    print(f"  2) '{OUT_CSV}' dosyasında aynı tarama + aynı sıra numarasının")
    print("     satırına defterdeki tarihi, baş harfi ve barkodu yazın.")
    print("  3) Numaralar soldan sağa, yukarıdan aşağı ilerler. Defteriniz")
    print("     başka bir sırada tutuluyorsa haritaya bakıp kendi sıranızı")
    print("     çıkarabilirsiniz -- harita gerçek görüntüyü gösterir.")


if __name__ == "__main__":
    main()
