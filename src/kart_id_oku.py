# -*- coding: utf-8 -*-
"""
kart_id_oku.py -- jel kartlarindaki BASILI ID numarasini otomatik okur
=======================================================================
NEDEN
------
Jellerde hasta kimligi el yazisiyla yazili (bas harfler, tarih) ve el
yazisi guvenilir sekilde okunamiyor. AMA her jel kartinin alt seridinde
BASILI bir numara var:

    ID 136950    Gel N° ___    Date 21/12/2016

Bu numara basili oldugu icin OCR ile yuksek dogrulukla okunuyor. Gercek
taramanizda test edildi: 4 kartin 4'u de dogru okundu.

NE YAPAR
---------
1) Her taramada kirpma koordinatlarindan KART yapisini cikarir.
   Bir kartta 2x2 = 4 hasta paneli vardir; tarama 2x2 ya da 3x2 kart
   icerebilir. Panel satir/sutunlari ikiserli gruplanarak kart bulunur.
2) Her kartin alt seridini OCR'a verir, "ID nnnnnn" desenini arar.
3) Her kirpmaya ait oldugu kartin ID'sini ve kart icindeki panel
   konumunu (IFE1..IFE4) yazar.

SONUC
------
    kirpma_dosyasi -> kart_id + panel_no

Defterinizde bu kart ID'si varsa, 3300 kaydi tek tek girmenize GEREK
KALMAZ: kart ID uzerinden otomatik eslesir, panel_no da karttaki 4
hastadan hangisi oldugunu soyler.

KURULUM
--------
    pip install pytesseract
    brew install tesseract          # Mac
    (Windows: tesseract'i kurup PATH'e ekleyin)

KULLANIM
---------
    python kart_id_oku.py --taramalar A4_Taramalar
    python kart_id_oku.py --taramalar A4_Taramalar --sadece 2018
    python kart_id_oku.py --taramalar A4_Taramalar --kontrol 3   # ilk 3 tarama
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MASTER = "annotations_master.csv"
OUT = "kart_id_eslestirme.csv"
OUT_KONTROL = "kart_id_kontrol"

RE_ID = re.compile(r"ID\s*[:.]?\s*([0-9]{4,8})")

# (ust mu, sol mu) -> IFE numarasi, taramanin DONDURULME durumuna gore.
# Normal kart: IFE3 sol ust, IFE4 sag ust, IFE1 sol alt, IFE2 sag alt.
# Bazi taramalar kartlar yan yatirilarak yapilmis (5100x7016, dikey). O
# zaman hem OCR icin goruntu dondurulur hem de IFE duzeni degisir:
#   duzeltme 270 (kart 90 CCW taranmis): IFE4 sol ust, IFE2 sag ust,
#                                        IFE3 sol alt, IFE1 sag alt
OKUMA_TO_IFE_BY_ROT = {
    0:   {(True, True): 3, (True, False): 4, (False, True): 1, (False, False): 2},
    90:  {(True, True): 1, (True, False): 3, (False, True): 2, (False, False): 4},
    180: {(True, True): 2, (True, False): 1, (False, True): 4, (False, False): 3},
    270: {(True, True): 4, (True, False): 2, (False, True): 3, (False, False): 1},
}


def geri_donustur(x2, y2, ang, W, H):
    """rotate(ang, expand=True) ile dondurulmus goruntudeki noktayi ORIJINAL
    tarama koordinatina cevirir (deneysel olarak dogrulandi)."""
    if ang == 0:
        return x2, y2
    if ang == 90:
        return W - 1 - y2, x2
    if ang == 180:
        return W - 1 - x2, H - 1 - y2
    return y2, H - 1 - x2   # 270


def ileri_donustur(x, y, ang, W, H):
    """ORIJINAL koordinati rotate(ang, expand=True) cercevesine cevirir."""
    if ang == 0:
        return x, y
    if ang == 90:
        return y, W - 1 - x
    if ang == 180:
        return W - 1 - x, H - 1 - y
    return H - 1 - y, x   # 270


def kart_seridi_oku(im_duz, kutu_duz, gw, gh):
    """
    Duzeltilmis cercevede, kart kutusunun HEMEN ALTINDAKI seridi hedefli
    okur (ID normal kartta orada). Sayfa geneli tarama kaciran kartlar
    icin yedek yontem. 2x/3x/4x buyutme ve iki sayfa modu dener.
    """
    import pytesseract
    from PIL import ImageOps
    x0, y0, x1, y1 = kutu_duz
    W, H = im_duz.size
    # Genis serit: kart kutusu kirpmalardan tahmin edildigi icin ID biraz
    # asagida/solda kalabilir. Sol taraf ve alt pay cömert tutulur.
    sx0 = max(0, min(W - 1, int(x0 - gw * 0.4)))
    sx1 = max(sx0 + 1, min(W, int(x0 + (x1 - x0) * 0.7)))
    sy0 = max(0, min(H - 1, int(y1 - gh * 0.3)))
    sy1 = max(sy0 + 1, min(H, int(y1 + gh * 2.2)))
    if sx1 - sx0 < 30 or sy1 - sy0 < 15:
        return ""
    serit = im_duz.crop((sx0, sy0, sx1, sy1))
    serit = ImageOps.autocontrast(serit)
    # GUVENLIK: OCR rakamlari karistirabilir (5<->8, 3<->8, 1<->7). Tek bir
    # okumaya guvenmek yanlis hastaya baglar. Farkli olceklerde okuyup
    # EN AZ IKI okumanin ayni sonucu vermesini sart kosuyoruz; anlasma
    # yoksa bos donup elle kontrole birakiyoruz.
    from collections import Counter
    oylar = Counter()
    for olcek in (2, 3, 4):
        s2 = serit.resize((serit.width * olcek, serit.height * olcek))
        for cfg in ("--psm 6", "--psm 11"):
            try:
                t = pytesseract.image_to_string(s2, config=cfg)
            except Exception:
                continue
            m = re.search(r"ID\s*[:.]?\s*([0-9]{5,7})", t)
            if m:
                oylar[m.group(1)] += 1
    if not oylar:
        return ""
    kid, oy = oylar.most_common(1)[0]
    return kid if oy >= 2 else ""


def idler_dondurerek(im):
    """
    Tarama dondurulmus olabilir. En-boy oranina gore once en olasi
    duzeltmeyi dener; ID bulamazsa alternatifi dener. Bulunan ID
    konumlarini orijinal cerceveye cevirip (idler, uygulanan_aci) dondurur.
    """
    W, H = im.size
    sira = [270, 90] if H > W else [0, 180]
    en_iyi, en_iyi_aci = [], sira[0]
    for ang in sira:
        r = im.rotate(ang, expand=True) if ang else im
        bul = sayfadaki_idler(r)
        if len(bul) > len(en_iyi):
            en_iyi, en_iyi_aci = bul, ang
        if len(bul) >= 2:
            break
    cevrilmis = []
    for k in en_iyi:
        x, y = geri_donustur(k["x"], k["y"], en_iyi_aci, W, H)
        cevrilmis.append({"kart_id": k["kart_id"], "x": x, "y": y})
    return cevrilmis, en_iyi_aci
RE_TARIH = re.compile(r"([0-3]?[0-9])\s*[/.\-]\s*([01]?[0-9])\s*[/.\-]\s*(20[0-9]{2})")

# Kartin alt seridi: ID satiri panellerin ALTINDA, kart kenarina yakin.
# Genis tutuyoruz cunku kart kutusu kirpma koordinatlarindan TAHMIN
# ediliyor; dar bir serit kaymalarda ID'yi kacirir.
ALT_SERIT = 0.30
# Kart kutusunu panellerin disina biraz tasir (serit panellerin altinda)
KART_PAY = 0.06


def kumele(v, tol):
    s = sorted(range(len(v)), key=lambda i: v[i])
    g = [0] * len(v)
    k = 0
    for i in range(1, len(s)):
        if v[s[i]] - v[s[i - 1]] > tol:
            k += 1
        g[s[i]] = k
    return g


def sayfadaki_idler(im):
    """
    Sayfayi BINDIRMELI KAROLARA bolup her karoyu 3x buyuterek OCR eder ve
    "ID nnnnnn" gecen her yeri konumuyla dondurur.

    Neden karo: sayfa geneli OCR kucuk yaziyi kaciriyor (6 kartli taramada
    6 ID'den 1'ini buldu). Karo + buyutme ile ayni taramada 5/6 bulundu;
    kalan 1'i taramadaki kirisiklik ID'nin ustunden gectigi icin okunamaz
    durumdaydi. Okunamayanlar raporlanir, elle tamamlanir.
    """
    import pytesseract
    from PIL import ImageOps
    W, H = im.size
    bul = {}
    nx, ny = 8, 8
    tw, th = W // nx, H // ny
    for j in range(ny * 2 - 1):
        for i in range(nx * 2 - 1):
            x0, y0 = i * tw // 2, j * th // 2
            k = im.crop((x0, y0, min(W, x0 + tw), min(H, y0 + th)))
            if k.width < 20 or k.height < 20:
                continue
            k = ImageOps.autocontrast(k).resize((k.width * 3, k.height * 3))
            try:
                t = pytesseract.image_to_string(k, config="--psm 6")
            except Exception:
                continue
            for m in re.finditer(r"ID\s*[:.]?\s*([0-9]{5,7})", t):
                v = m.group(1)
                if v not in bul:
                    bul[v] = {"kart_id": v, "x": x0 + tw // 2, "y": y0 + th // 2, "oy": 0}
                bul[v]["oy"] += 1
    # Bindirmeli karolarda gercek bir ID genelde 2+ karoda okunur; tek
    # karoda okunan buyuk olasilikla rakam hatasidir (or. 138574 -> 138874).
    return [k for k in bul.values() if k["oy"] >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=MASTER)
    ap.add_argument("--taramalar", default="A4_Taramalar")
    ap.add_argument("--sadece", default=None, help="Adinda bu gecen taramalar")
    ap.add_argument("--kontrol", type=int, default=0,
                    help="Ilk N tarama icin kart seritlerini resim olarak kaydet")
    ap.add_argument("--cikti", default=OUT)
    ap.add_argument("--guncelle", action="store_true",
                    help="Mevcut ciktiyi SILME; sadece islenen taramalarin satirlarini degistir")
    ap.add_argument("--sadece-dondurulmus", action="store_true",
                    help="Yalnizca dikey (dondurulmus) taramalari isle")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    df = df[df[["x", "y", "w", "h"]].notna().all(axis=1)].copy()
    if a.sadece:
        df = df[df["source_file"].astype(str).str.contains(a.sadece, case=False)]
    kok = Path(a.taramalar)
    if not kok.exists():
        raise SystemExit(f"'{kok}' klasoru yok.")
    if a.kontrol:
        Path(OUT_KONTROL).mkdir(exist_ok=True)

    satirlar = []
    okunan = okunamayan = 0
    taramalar = sorted(df["source_file"].astype(str).unique())
    if a.sadece_dondurulmus:
        secilen = []
        for t in taramalar:
            y = kok / t
            if not y.exists():
                adaylar = list(kok.glob(Path(t).stem + ".*"))
                y = adaylar[0] if adaylar else None
            if y and y.exists():
                try:
                    with Image.open(y) as im0:
                        if im0.height > im0.width:
                            secilen.append(t)
                except Exception:
                    pass
        taramalar = secilen
        print(f"Dikey (dondurulmus) tarama sayisi: {len(taramalar)}")
        for t in taramalar:
            print("   ", t)
    for n, tarama in enumerate(taramalar):
        g = df[df["source_file"].astype(str) == tarama].copy()
        yol = kok / tarama
        if not yol.exists():
            adaylar = list(kok.glob(Path(tarama).stem + ".*"))
            yol = adaylar[0] if adaylar else None
        if yol is None or not yol.exists():
            continue
        try:
            im = Image.open(yol).convert("L")
        except Exception as e:
            print(f"  ATLANDI ({tarama}): {e}")
            continue

        idler, aci = idler_dondurerek(im)
        OKUMA_TO_IFE = OKUMA_TO_IFE_BY_ROT[aci]
        if aci:
            print(f"  {tarama}: tarama dondurulmus, {aci} derece duzeltildi")
        g["cx"] = g["x"] + g["w"] / 2
        g["cy"] = g["y"] + g["h"] / 2
        gw, gh = float(g["w"].median()), float(g["h"].median())

        # 1) Panelleri GEOMETRIK olarak karta grupla (bir kart = 2x2 panel).
        #    Panel satir/sutunlarini ikiserli gruplayinca kart cikar.
        g["psatir"] = kumele(g["cy"].tolist(), gh * 0.5)
        g["psutun"] = kumele(g["cx"].tolist(), gw * 0.5)
        g["kart_grp"] = (g["psatir"] // 2).astype(str) + "_" + (g["psutun"] // 2).astype(str)

        # 2) Her ID'yi EN YAKIN karta ata (yonden bagimsiz). ID, normal
        #    kartta altta; dondurulmus taramada yanda olabilir. Bu yuzden
        #    "asagida ara" yerine her ID icin kart kutusuna uzakligi olculur
        #    ve ID en yakin karta gider. Bir karta birden fazla ID duserse
        #    en yakini alinir; hic dusmezse bos kalir (komsudan odunc ALINMAZ).
        kutular = {}
        for grp, kg in g.groupby("kart_grp"):
            kutular[grp] = (kg["x"].min(), kg["y"].min(),
                            (kg["x"] + kg["w"]).max(), (kg["y"] + kg["h"]).max())

        def kutu_uzaklik(px, py, kutu):
            x0, y0, x1, y1 = kutu
            dx = max(x0 - px, 0, px - x1)
            dy = max(y0 - py, 0, py - y1)
            return (dx * dx + dy * dy) ** 0.5

        atanan = {}   # grp -> (uzaklik, kart_id)
        for k in idler:
            if not kutular:
                break
            grp, d = min(((grp, kutu_uzaklik(k["x"], k["y"], kutu))
                          for grp, kutu in kutular.items()), key=lambda t: t[1])
            # Kart kutusundan bir panel boyundan fazla uzaksa ait sayilmaz
            if d > max(gw, gh) * 1.8:
                continue
            if grp not in atanan or d < atanan[grp][0]:
                atanan[grp] = (d, k["kart_id"])

        g["kart_id"] = ""
        g["ife_no"] = 0
        im_duz = im.rotate(aci, expand=True) if aci else im
        yedekle_bulunan = 0
        for grp, kg in g.groupby("kart_grp"):
            kid = atanan.get(grp, (None, ""))[1]
            if not kid:
                # YEDEK: kart kutusunu duzeltilmis cerceveye cevir, altini oku
                pts = [ileri_donustur(px, py, aci, im.width, im.height)
                       for px, py in [(kg["x"].min(), kg["y"].min()),
                                      ((kg["x"] + kg["w"]).max(), kg["y"].min()),
                                      (kg["x"].min(), (kg["y"] + kg["h"]).max()),
                                      ((kg["x"] + kg["w"]).max(), (kg["y"] + kg["h"]).max())]]
                xs_, ys_ = [q[0] for q in pts], [q[1] for q in pts]
                gw2, gh2 = (gh, gw) if aci in (90, 270) else (gw, gh)
                kid = kart_seridi_oku(im_duz, (min(xs_), min(ys_), max(xs_), max(ys_)), gw2, gh2)
                if kid:
                    yedekle_bulunan += 1
            g.loc[kg.index, "kart_id"] = kid
        if yedekle_bulunan:
            print(f"  {tarama}: yedek hedefli okuma ile {yedekle_bulunan} kart daha bulundu")

        # 3) Kart icinde IFE numarasi (dondurmeye gore; normalde 3/4 ust, 1/2 alt)
        for grp, kg in g.groupby("kart_grp"):
            ust = kg["cy"] < kg["cy"].median() if len(kg) > 1 else kg["cy"] * 0 == 0
            sol = kg["cx"] < kg["cx"].median() if len(kg) > 1 else kg["cx"] * 0 == 0
            for idx_ in kg.index:
                g.loc[idx_, "ife_no"] = OKUMA_TO_IFE[(bool(ust[idx_]), bool(sol[idx_]))]

        for _, r in g.iterrows():
            if str(r["kart_id"]).strip():
                okunan += 1
            else:
                okunamayan += 1
            satirlar.append({
                "kirpma_dosyasi": r["filename"],
                "kaynak_tarama": tarama,
                "kart_id": r["kart_id"],
                "ife_no": int(r["ife_no"]),
                "mevcut_patern": r.get("pattern", ""),
                "x": int(r["x"]), "y": int(r["y"]),
                # --- defterden dolduracaklariniz (kart_id ile eslesirse
                #     bunlari elle girmenize gerek kalmaz) ---
                "defter_hasta": "", "defter_tani": "", "defter_barkod": "",
            })
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(taramalar)} tarama islendi "
                  f"(okunan ID: {okunan}, okunamayan: {okunamayan})")

    son = pd.DataFrame(satirlar)
    if not len(son):
        raise SystemExit("Hic kayit islenemedi. --taramalar yolunu kontrol edin.")
    if a.guncelle and Path(a.cikti).exists():
        eski = pd.read_csv(a.cikti, dtype={"kart_id": str})
        islenen = set(son["kaynak_tarama"].astype(str))
        eski = eski[~eski["kaynak_tarama"].astype(str).isin(islenen)]
        son = pd.concat([eski, son], ignore_index=True)
        print(f"\nMevcut cikti guncellendi: {len(islenen)} taramanin satirlari yenilendi.")
    son.to_csv(a.cikti, index=False, encoding="utf-8-sig")

    idli = son[son["kart_id"].astype(str).str.strip() != ""]
    print(f"\n{len(son)} kirpma, {son['kaynak_tarama'].nunique()} tarama")
    print(f"ID okunabilen kirpma : {len(idli)} (%{100 * len(idli) / max(len(son),1):.0f})")
    print(f"Farkli kart ID       : {idli['kart_id'].nunique()}")
    if len(idli):
        panel_dagilim = idli.groupby("kart_id").size().value_counts().sort_index()
        print("\nKart basina panel sayisi (4 olmasi beklenir):")
        for k, n in panel_dagilim.items():
            isaret = "" if k == 4 else "   <-- kontrol edin"
            print(f"  {n:5d} kart -> {k} panel{isaret}")
    eksik = son[son["kart_id"].astype(str).str.strip() == ""]
    if len(eksik):
        print(f"\nID okunamayan {len(eksik)} kirpma, "
              f"{eksik['kaynak_tarama'].nunique()} taramada:")
        for sf, n in eksik["kaynak_tarama"].value_counts().head(6).items():
            print(f"  {sf:<26} {n}")
    print(f"\n-> {a.cikti}")
    if a.kontrol:
        print(f"Kontrol seritleri -> {OUT_KONTROL}/  "
              f"(OCR'in ne gordugunu gozle dogrulayin)")
    print("\nSIRADAKI ADIM")
    print("  Defterinizde bu kart ID'leri varsa, artik tek tek girmenize")
    print("  gerek yok: kart_id uzerinden eslestirilir, panel_no da karttaki")
    print("  4 hastadan hangisi oldugunu soyler (ife_no = defterdeki sira).")
    print("  Sonra:  python kart_defter_birlestir.py")


if __name__ == "__main__":
    main()
