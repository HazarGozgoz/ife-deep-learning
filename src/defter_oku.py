# -*- coding: utf-8 -*-
"""
defter_oku.py -- el yazisi defter fotograflarini otomatik Excel'e gecirir
==========================================================================
NEDEN TESSERACT DEGIL
----------------------
Defteriniz el yazisi. Tesseract basili metin icin tasarlandi; sizin
sayfalarinizda test edildi ve barkodlarin HICBIRINI okuyamadi (0/4).
Zorlamanin anlami yok.

Buna karsilik goruntu anlayan bir dil modeli bu sayfalari rahatlikla
okuyor. Bu script her defter fotografini Claude'a gonderip yapilandirilmis
JSON olarak geri alir ve Excel'e yazar.

DEFTER YAPISI (sizin sayfalarinizdan)
--------------------------------------
    142266            IFE              28.03.2018
    1) mehmet  korkus      16131478
       IgG-K  MGG
    2) serif mehmet  Cetin  16131624
       MGO  Ancak IgG'de SG
    ...
Sayfa basina 4 kayit -> jel kartinin 4 paneline birebir denk gelir.

KLASOR DUZENI
--------------
Defter fotograflari jel taramasiyla AYNI ADLI klasorde:
    Defter/2018_12/IMG_4897.HEIC ...   <-> tarama "2018_File 12"

KURULUM
--------
    pip install anthropic pillow
    # HEIC icin (Mac):  brew install imagemagick
    export ANTHROPIC_API_KEY="anahtariniz"

KULLANIM
---------
    python defter_oku.py --defter Defter --kontrol 2   # once 2 sayfa dene
    python defter_oku.py --defter Defter               # hepsi

GIZLILIK -- ONEMLI
-------------------
Bu script hasta adi iceren goruntuleri disaridaki bir servise gonderir.
Kurumunuzun veri politikasi ve etik kurul onayi buna izin veriyor mu,
BASLAMADAN once dogrulayin. Izin yoksa bu yolu kullanmayin.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUT_CSV = "defter_okunan.csv"
OUT_XLSX = "defter_okunan.xlsx"
MODEL = "claude-sonnet-5"

# MALIYET (Eylul 2026, Sonnet 5: $2 girdi / $10 cikti her 1M token)
#   Bir defter sayfasi ~3.000 girdi + ~300 cikti token  -> ~$0.009
#   825 sayfa                                           -> ~$7-8
# Abonelik yok; kullandiginiz kadar odersiniz. Yeni hesaplar on odemeli
# kredi ile baslar (en az $5). --kontrol ile once birkac sayfa deneyin.

ISTEM = """Bu bir Turk hastanesinin immunfiksasyon elektroforezi (IFE) defterinin el yazisi sayfasi.

Sayfa yapisi genelde soyle:
  <sayfa numarasi>        IFE        <tarih>
  1) <ad> <soyad>   <8 haneli barkod>
     <tani notu>
  2) ...

Sayfadaki HER kaydi oku ve SADECE su JSON'u dondur, baska hicbir sey yazma:

{
  "sayfa_no": "142266",
  "tarih": "28.03.2018",
  "kayitlar": [
    {"sira": 1, "ad": "mehmet", "soyad": "korkus", "barkod": "16131478",
     "tani_metni": "IgG-K MGG", "guven": "yuksek"}
  ]
}

Kurallar:
- "barkod" 8 haneli bir sayidir. Emin degilsen "" birak ve guven "dusuk" yaz.
- "tani_metni" defterde ne yaziyorsa aynen yaz (IgG-K, MGG, MGO, OGG, SG,
  "iki ayri bant halinde" gibi). Yorumlama, cevirme.
- Ad ve soyadi ayri alanlara yaz. Uc kelimeyse ilk kelime ad, kalani soyad.
- Okuyamadigin alani "" birak ve o kaydin guven degerini "dusuk" yap.
- guven: "yuksek" | "orta" | "dusuk"
- Tahmin YURUTME. Emin olmadigin yeri bos birak; sonradan elle bakilacak.
"""


ISTEM_BASHARF = """Bu bir immunfiksasyon jel seridinin kirpilmis goruntusu. Ust kisminda
el yazisiyla hastanin bas harfleri yazili (ornek: "Y.Y", "S.G", "E.Ozlen Fenerbak").
SADECE bu el yazisi metni oldugu gibi yaz, baska hicbir sey yazma.
Okuyamiyorsan sadece ? yaz."""


def kirpmalari_oku(client, master_csv: str, cikti: str, kontrol: int, cihaz: str = "", yeniden: bool = False):
    """Her kirpmanin ustundeki el yazisi bas harfleri okur -> kirpma_basharf.csv
    Daha once OKUNMUS kirpmalar atlanir, sonuc mevcut CSV'ye eklenir (--yeniden ile hepsi)."""
    import io
    m = pd.read_csv(master_csv)
    m = m[m["img_relpath"].notna()]
    if cihaz and "device" in m.columns:
        m = m[m["device"].astype(str).str.upper() == cihaz.upper()]
    eski = pd.DataFrame(columns=["kirpma_dosyasi", "jel_basharf", "not"])
    if Path(cikti).exists() and not yeniden:
        eski = pd.read_csv(cikti, dtype=str).fillna("")
        okunmus = set(eski.loc[eski["jel_basharf"] != "", "kirpma_dosyasi"])
        onceki = len(m); m = m[~m["filename"].isin(okunmus)]
        print(f"{onceki - len(m)} kirpma daha once okunmus, atlaniyor")
    if kontrol:
        m = m.head(kontrol)
    eski = eski[~eski["kirpma_dosyasi"].isin(m["filename"])]       # yeniden okunacaklarin eski bos satirlarini at
    if m.empty:
        raise SystemExit("Okunacak yeni kirpma yok.")
    print(f"{len(m)} kirpma okunacak  (tahmini maliyet ~${len(m) * 0.004:.2f})")
    satirlar = []
    for i, r in enumerate(m.itertuples(), 1):
        yol = Path("Cropped_Data_EN") / str(r.img_relpath)
        if not yol.exists():
            satirlar.append({"kirpma_dosyasi": r.filename, "jel_basharf": "", "not": "dosya yok"})
            continue
        try:
            im = Image.open(yol).convert("RGB")
            im.thumbnail((900, 900))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            b64 = base64.standard_b64encode(buf.getvalue()).decode()
            msg = client.messages.create(
                model=MODEL, max_tokens=400,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": ISTEM_BASHARF}]}])
            txt = "".join(p.text for p in msg.content if getattr(p, "type", "") == "text").strip()
            txt = txt.strip("`\"' \n")
            if not txt and getattr(msg, "stop_reason", "") == "max_tokens":
                satirlar.append({"kirpma_dosyasi": r.filename, "jel_basharf": "", "not": "cevap kesildi (max_tokens)"})
                continue
            satirlar.append({"kirpma_dosyasi": r.filename,
                             "jel_basharf": "" if txt in ("?", "") else txt[:40], "not": "" if txt else "bos cevap"})
        except Exception as e:
            satirlar.append({"kirpma_dosyasi": r.filename, "jel_basharf": "", "not": str(e)[:80]})
        if i % 50 == 0:
            print(f"  {i}/{len(m)}")
    df = pd.concat([eski, pd.DataFrame(satirlar)], ignore_index=True)
    df.to_csv(cikti, index=False, encoding="utf-8-sig")
    ok = sum(1 for s_ in satirlar if s_["jel_basharf"])
    print(f"\n{ok}/{len(satirlar)} yeni kirpmada bas harf okundu; toplam {len(df)} satir -> {cikti}")


ISTEM_HELENA_AD = (
    "Bu, bir laboratuvar jel kartinin kenar seridi (arsiv kaydi, transkripsiyon amacli). Uzerinde mavi kalemle "
    "el yazisi bir etiket var: bas harf + soyad bicimde (or. 'E. Senol', 'A. Kaplan', 'M. Cakir'). Yazi yan/ters olabilir. "
    "Gorevin sadece bu el yazisini transkribe etmek. SADECE yazili metni oldugu gibi yaz (Turkce karakter serbest), "
    "baska hicbir sey ekleme. Hic el yazisi yoksa sadece ? yaz.")
ISTEM_HELENA_TARIH = (
    "Bu serit bir jel kartinin ust kenari. El yazisiyla bir tarih (GG.AA.YYYY) ve daire icinde bir kart numarasi var. "
    "SADECE 'GG.AA.YYYY|N' bicimde yaz (or. 15.03.2021|1). Tarih okunmuyorsa ?|N, numara okunmuyorsa GG.AA.YYYY|?.")


def _sor(client, im, istem, max_tokens=200):
    import io
    im = im.convert("RGB"); im.thumbnail((1200, 1200))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
        {"type": "text", "text": istem}]}])
    txt = "".join(p.text for p in msg.content if getattr(p, "type", "") == "text").strip().strip("`\"' ")
    return txt


def helena_kenar_oku(client, master_csv: str, tarama_dir: str, cikti: str, kontrol: int):
    """HELENA: hasta adi kartin SOL/SAG kenarinda dikey yazili, tarih kartin USTUNDE.
    Her kirpma icin taramadan kenar seridini kesip okur -> kirpma_basharf.csv (jel_basharf=ad, jel_tarih)."""
    import glob as _glob
    from PIL import Image as _I
    _I.MAX_IMAGE_PIXELS = None
    m = pd.read_csv(master_csv)
    m = m[m["device"].astype(str).str.upper() == "HELENA"].copy()
    for c in ("x", "y", "w", "h"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["x", "y", "w", "h"])
    m_tum = m.copy()          # geometri (sol/sag, kart ustu) HER ZAMAN taramanin tum kirpmalarindan
    eski = pd.DataFrame(columns=["kirpma_dosyasi", "jel_basharf", "not", "jel_tarih", "jel_kart_no"])
    if Path(cikti).exists():
        eski = pd.read_csv(cikti, dtype=str).fillna("")
        for c in ("jel_tarih", "jel_kart_no"):
            if c not in eski.columns: eski[c] = ""
        okunmus = set(eski.loc[eski["jel_basharf"] != "", "kirpma_dosyasi"])
        m = m[~m["filename"].isin(okunmus)]
    if kontrol:
        m = m.head(kontrol)
    if m.empty:
        raise SystemExit("Okunacak Helena kirpmasi yok (hepsi okunmus).")
    eski = eski[~eski["kirpma_dosyasi"].isin(m["filename"])]
    taramalar = {os.path.basename(p): p for p in _glob.glob(os.path.join(tarama_dir, "**", "*.*"), recursive=True)}
    print(f"{len(m)} Helena kirpmasi icin kenar okunacak (tahmini ~${len(m) * 0.006:.2f})")

    def kume(v, esik):
        sirali = sorted(set(v)); etiket = {}; k = 0; onceki = None
        for x in sirali:
            if onceki is not None and x - onceki > esik: k += 1
            etiket[x] = k; onceki = x
        return {x: etiket[x] for x in v}

    satirlar = []; tarih_onbellek = {}; i = 0
    for sf, g in m.groupby("source_file"):
        yol = taramalar.get(str(sf))
        if yol is None:
            for _, r in g.iterrows():
                satirlar.append({"kirpma_dosyasi": r["filename"], "jel_basharf": "", "not": "tarama yok", "jel_tarih": "", "jel_kart_no": ""})
            continue
        im = _I.open(yol); W, H = im.size
        tum = m_tum[m_tum["source_file"] == sf]
        xs = sorted(set(tum["x"].astype(int).tolist())); blok = kume(xs, 2500)   # kart bloklari: iki sutun ~1700 px, kartlar arasi >2500
        ys_all = tum["y"].astype(int).tolist(); ykume = kume(ys_all, 2000)          # kart satirlari
        sol_x = {}
        for b_ in set(blok.values()):
            uyeler = sorted(x for x in xs if blok[x] == b_)
            sutun = kume(uyeler, 800)                       # blok icinde 2 sutun; +-10 px oynama ayni sutun
            for x in uyeler: sol_x[x] = (sutun[x] == 0)
        for _, r in g.iterrows():
            i += 1
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            try:
                sol = sol_x.get(int(r["x"]), True)
                if sol:
                    kutu = (max(0, x - 340), max(0, y - 60), max(0, x - 10), min(H, y + h + 60))
                    serit = im.crop(kutu).rotate(-90, expand=True)   # sol kenar: asagidan yukari yazi -> saat yonu
                else:
                    kutu = (min(W, x + w + 10), max(0, y - 60), min(W, x + w + 340), min(H, y + h + 60))
                    serit = im.crop(kutu).rotate(90, expand=True)    # sag kenar: yukaridan asagi yazi -> tersi
                if kontrol:                                        # deneme: seritleri diske yaz, goz kontrolu
                    Path("helena_kenar_ornek").mkdir(exist_ok=True)
                    serit.save(f"helena_kenar_ornek/{i:02d}_{'sol' if sol else 'sag'}_{r['filename'][:30]}.jpg", quality=85)
                ad = _sor(client, serit, ISTEM_HELENA_AD)
                if ad in ("?", ""):                               # kenar yazisi ters yonde yazilmis olabilir
                    ad = _sor(client, serit.rotate(180), ISTEM_HELENA_AD)
                # tarih: kartin ustu (kart = x blogu + y blogu); onbellekle
                yb = ykume[int(r["y"])]; xb = blok[int(r["x"])]
                anahtar = (sf, xb, yb)
                if anahtar not in tarih_onbellek:
                    kart_x = [xx for xx in xs if blok[xx] == xb]
                    kart_y = [yy for yy in ys_all if ykume[yy] == yb]
                    x0, y0 = min(kart_x), min(kart_y)
                    ust = im.crop((max(0, x0 - 500), max(0, y0 - 700), min(W, x0 + 2 * w + 500), max(0, y0 - 10)))
                    if kontrol:
                        Path("helena_kenar_ornek").mkdir(exist_ok=True)
                        ust.save(f"helena_kenar_ornek/{i:02d}_ust_{str(sf)[:20]}_x{xb}y{yb}.jpg", quality=85)
                    tarih_onbellek[anahtar] = _sor(client, ust, ISTEM_HELENA_TARIH)
                t = tarih_onbellek[anahtar]; tarih, _, kno = t.partition("|")
                satirlar.append({"kirpma_dosyasi": r["filename"], "jel_basharf": "" if ad in ("?", "") else ad[:40],
                                 "not": "helena kenar" if ad not in ("?", "") else "kenarda ad okunamadi",
                                 "jel_tarih": tarih.strip() if tarih.strip() != "?" else "", "jel_kart_no": kno.strip()})
            except Exception as e:
                satirlar.append({"kirpma_dosyasi": r["filename"], "jel_basharf": "", "not": str(e)[:80], "jel_tarih": "", "jel_kart_no": ""})
            if i % 25 == 0:
                print(f"  {i}/{len(m)}  son: {satirlar[-1]['jel_basharf']} | {satirlar[-1]['jel_tarih']}")
    df = pd.concat([eski, pd.DataFrame(satirlar)], ignore_index=True)
    df.to_csv(cikti, index=False, encoding="utf-8-sig")
    ok = sum(1 for s_ in satirlar if s_["jel_basharf"])
    print(f"\n{ok}/{len(satirlar)} Helena kirpmasinda ad okundu; toplam {len(df)} satir -> {cikti}")


def heic_to_jpg(src: Path, hedef: Path) -> bool:
    """HEIC -> JPEG (ImageMagick ile). Zaten jpg/png ise kopyalar."""
    if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
        im = Image.open(src)
        im.thumbnail((1600, 1600))
        im.convert("RGB").save(hedef, quality=88)
        return True
    for komut in (["magick", str(src), "-auto-orient", "-resize", "1600x", str(hedef)],
                  ["convert", str(src), "-auto-orient", "-resize", "1600x", str(hedef)]):
        try:
            r = subprocess.run(komut, capture_output=True, timeout=120)
            if r.returncode == 0 and hedef.exists():
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def sayfayi_oku(client, jpg: Path) -> dict:
    b64 = base64.standard_b64encode(jpg.read_bytes()).decode()
    mesaj = client.messages.create(
        model=MODEL, max_tokens=16000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": ISTEM},
        ]}],
    )
    metin = "".join(p.text for p in mesaj.content if getattr(p, "type", "") == "text")
    ham = metin
    metin = re.sub(r"```(?:json)?", "", metin).strip()
    i = metin.find("{")
    try:
        if i < 0:
            raise json.JSONDecodeError("JSON yok", metin, 0)
        obj, _ = json.JSONDecoder().raw_decode(metin[i:])   # ilk gecerli nesne; sonrasindaki fazla metin yok sayilir
        return obj
    except json.JSONDecodeError as e:
        print("    MODEL CEVABI (ilk 300 karakter):", ham[:300].replace("\n", " | "))
        if getattr(mesaj, "stop_reason", "") == "max_tokens":
            print("    -> cevap kesildi (max_tokens); MAX_TOKENS degerini buyutun")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defter", default="Defter",
                    help="Icinde tarama adli alt klasorler olan kok klasor")
    ap.add_argument("--kontrol", type=int, default=0,
                    help="Sadece ilk N sayfayi oku (once dogrulamak icin)")
    ap.add_argument("--kirpmalar", action="store_true",
                    help="Defter yerine KIRPMALARIN ustundeki el yazisi bas harfleri oku")
    ap.add_argument("--master", default="annotations_master.csv")
    ap.add_argument("--csv", default=OUT_CSV)
    ap.add_argument("--helena-kenar", action="store_true", help="HELENA: kart kenarindaki hasta adini ve ustteki tarihi TARAMADAN oku")
    ap.add_argument("--taramalar", default="A4_Taramalar", help="Ham tarama klasoru (--helena-kenar icin)")
    ap.add_argument("--cihaz", default="", help="--kirpmalar ile: sadece bu cihazin kirpmalari (HELENA/INTERLAB)")
    ap.add_argument("--yeniden", action="store_true", help="Mevcut CSV'yi yok say, HER SEYI yeniden oku (para harcar)")
    ap.add_argument("--xlsx", default=OUT_XLSX)
    a = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY tanimli degil.\n"
                         '  export ANTHROPIC_API_KEY="anahtariniz"')
    try:
        import anthropic
    except ImportError:
        raise SystemExit("anthropic kurulu degil:  pip install anthropic")
    client = anthropic.Anthropic()

    if a.helena_kenar:
        helena_kenar_oku(client, a.master, a.taramalar, "kirpma_basharf.csv", a.kontrol)
        return
    if a.kirpmalar:
        kirpmalari_oku(client, a.master, "kirpma_basharf.csv", a.kontrol, a.cihaz, a.yeniden)
        return

    kok = Path(a.defter)
    if not kok.exists():
        raise SystemExit(f"'{kok}' klasoru yok.")

    uzantilar = {".heic", ".heif", ".jpg", ".jpeg", ".png"}
    sayfalar = []
    for klasor in sorted([p for p in kok.iterdir() if p.is_dir()]):
        for f in sorted(klasor.iterdir()):
            if f.suffix.lower() in uzantilar:
                sayfalar.append((klasor.name, f))
    if not sayfalar:
        raise SystemExit(f"'{kok}' altinda fotograf bulunamadi.")
    # Daha once okunmus sayfalari atla (tarama_klasoru + fotograf adi). CSV'ye EKLENIR, silinmez.
    eski = pd.DataFrame()
    if Path(a.csv).exists() and not a.yeniden:
        eski = pd.read_csv(a.csv, dtype=str).fillna("")
        okunmus = set(zip(eski["tarama_klasoru"], eski["defter_fotograf"]))
        onceki = len(sayfalar)
        sayfalar = [(k, f) for k, f in sayfalar if (k, f.name) not in okunmus]
        print(f"{onceki - len(sayfalar)} sayfa daha once okunmus, atlaniyor "
              f"({eski['tarama_klasoru'].nunique()} klasor, {len(eski)} kayit {a.csv} icinde)")
    if not sayfalar:
        raise SystemExit("Okunacak yeni sayfa yok.")
    if a.kontrol:
        sayfalar = sayfalar[:a.kontrol]
    print(f"{len(sayfalar)} sayfa okunacak "
          f"({len(set(k for k, _ in sayfalar))} tarama klasoru)")
    print(f"Tahmini maliyet: ~${len(sayfalar) * 0.009:.2f} "
          f"(sayfa basina ~$0.009, Sonnet 5)")

    satirlar, hatalar = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (tarama, f) in enumerate(sayfalar, 1):
            jpg = Path(tmp) / (f.stem + ".jpg")
            if not heic_to_jpg(f, jpg):
                hatalar.append((f.name, "HEIC cevrilemedi (ImageMagick kurulu mu?)"))
                continue
            try:
                veri = sayfayi_oku(client, jpg)
            except Exception as e:
                hatalar.append((f.name, str(e)[:120]))
                continue
            for k in veri.get("kayitlar", []):
                satirlar.append({
                    "tarama_klasoru": tarama,
                    "defter_fotograf": f.name,
                    "sayfa_no": veri.get("sayfa_no", ""),
                    "sayfa_tarih": veri.get("tarih", ""),
                    "sira": k.get("sira", ""),
                    "ad": k.get("ad", ""),
                    "soyad": k.get("soyad", ""),
                    "barkod": k.get("barkod", ""),
                    "defter_tani": k.get("tani_metni", ""),
                    "guven": k.get("guven", ""),
                })
            print(f"  [{i}/{len(sayfalar)}] {tarama}/{f.name} -> "
                  f"{len(veri.get('kayitlar', []))} kayit")

    if not satirlar:
        for ad, hata in hatalar[:5]:
            print(f"  HATA {ad}: {hata}")
        raise SystemExit("Hic kayit okunamadi. (API anahtari dogru mu? Ustteki HATA satirlarina bakin.)")
    df = pd.DataFrame(satirlar)
    if len(eski):
        df = pd.concat([eski, df.astype(str)], ignore_index=True)
        print(f"Eski {len(eski)} kayit korundu, {len(satirlar)} yeni kayit eklendi -> toplam {len(df)}")
    df.to_csv(a.csv, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(a.xlsx, index=False)
        print(f"\n-> {a.xlsx}  (Excel'de dogrudan acilir)")
    except Exception as e:
        print(f"\nExcel yazilamadi ({e}); CSV kullanin.")
    print(f"-> {a.csv}")

    print(f"\n{len(df)} kayit, {df['defter_fotograf'].nunique()} sayfa")
    print("\nGuven dagilimi:")
    for g, n in df["guven"].value_counts().items():
        print(f"  {g:<8} {n}")
    bos = df[df["barkod"].astype(str).str.strip() == ""]
    if len(bos):
        print(f"\nBarkodu okunamayan {len(bos)} kayit -- bunlara elle bakin.")
    if hatalar:
        print(f"\n{len(hatalar)} sayfa islenemedi:")
        for ad, h in hatalar[:5]:
            print(f"  {ad}: {h}")

    print("\nSIRADAKI ADIM")
    print("  1) 'dusuk' guvenli ve barkodu bos kayitlari gozle kontrol edin.")
    print("  2) Kendi hasta listenizle barkod uzerinden eslestirin.")
    print("  3) Sayfadaki 'sira' (1-4) jel kartindaki panel sirasina denk gelir.")


if __name__ == "__main__":
    main()
