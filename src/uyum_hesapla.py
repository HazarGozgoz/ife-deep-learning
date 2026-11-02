# -*- coding: utf-8 -*-
"""
uyum_hesapla.py -- sizin taniniz vs defterdeki uzman tanisi (Cohen kappa)
==========================================================================
birlesik_kayitlar.xlsx dosyasindaki ESLESTI satirlarini alir, defterdeki
serbest metni kanonik paterne cevirir ve gozlemciler arasi uyumu hesaplar.

DEFTER KISALTMALARI (sizin defterinizden gordugum kadariyla)
-------------------------------------------------------------
    MGG          monoklonal gammopati (gorüldü)      -> MONOCLONAL
    MGØ / MGφ    monoklonal gammopati YOK            -> NO_MONOCLONAL_BAND
    ŞG           supheli gorunum / supheli bant      -> SUSPICIOUS_BAND
    OGG          oligoklonal gammopati               -> OLIGOCLONAL_BANDS
    "iki ayri bant halinde ... MGG"                  -> BICLONAL
    "MGØ ancak IgG'de ŞG"  = monoklonal yok, supheli var -> SUSPICIOUS_BAND

Bu eslemeleri DOGRULAYIN: OGG'nin ne anlama geldiginden emin degilim
(oligoklonal mi, "iki bant" mi?). Yanlissa asagidaki KISALTMA sozlugunu
duzeltin. Cevrilemeyen metinler ayri dosyaya yazilir, elle eslersiniz.

KULLANIM
---------
    python uyum_hesapla.py
    python uyum_hesapla.py --girdi birlesik_kayitlar.xlsx

IKI OKUYUCU (H.G. vs P.A.) -- referans standardinin cift okumasi
---------------------------------------------------------------
    1) Sablon:  python uyum_hesapla.py --sablon-olustur --hg-ilk annotations_goc_oncesi_yedek.csv
                Panel listesi annotations_master.csv'den (serum) gelir.
                H.G. etiketi: panel --hg-ilk dosyasinda varsa ORADAN (8 Mayis
                oncesi ham okuma; eski kodlar guncel taksonomiyle patern'e
                cevrilir), yoksa master'dan. Kaynak her panel icin yazilir.
       -> okuyucu_karsilastirma.xlsx. P.A.'nin tanilari 'PA_giris' sayfasina
          acilir listeden girilir. H.G. etiketleri GIZLI sayfadadir: giris
          yapan kisi kendi okumasini gormez (transkripsiyon korlugu).
    2) Hesap:   python uyum_hesapla.py --iki-okuyucu okuyucu_karsilastirma.xlsx
       -> hg_pa_uyum.xlsx (kappa + %95 GA, cihaz bazinda, capraz tablo,
          ayrisan paneller, M.O. hakemligi ozeti)

UCUNCU OKUYUCU (T.H.E.) + MODEL -- ayni paneller uzerinde
---------------------------------------------------------
    1) Sablon:  python uyum_hesapla.py --the-sablon      -> okuyucu_THE.xlsx
       (THE_giris sayfasi; referans/okuyucu etiketi ICERMEZ, giris kordur)
    2) Hesap:   python uyum_hesapla.py --uc-okuyucu okuyucu_THE.xlsx \
                    --pa-dosyasi okuyucu_karsilastirma.xlsx --havuz cv/havuz_tahminleri.npz
       T.H.E.'nin doldurdugu panellerde: H.G.-P.A., T.H.E.-referans, model-referans
       (referans = annotations_master.csv, uzlastirilmis son etiket)
"""
from __future__ import annotations

import argparse
import collections
import re
import unicodedata

import pandas as pd

import ife_taxonomy as T

OUT_UYUM = "gozlemciler_arasi_uyum.xlsx"
OUT_CEVRILEMEYEN = "defter_cevrilemeyen_tanilar.csv"

_TR = str.maketrans({"İ": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G", "ğ": "g",
                     "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C", "ç": "c",
                     "Ø": "0", "φ": "0", "ø": "0", "Φ": "0", "∅": "0", "ф": "0",
                     "Ф": "0", "θ": "0", "λ": "L", "Λ": "L"})


def sade(s) -> str:
    if not isinstance(s, str):
        return ""
    s = s.translate(_TR)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " " + re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() + " "


# Ozel kisaltmalar (sade() sonrasi hali). Sira ONEMLI: spesifik once.
KISALTMA: list[tuple[tuple[str, ...], str]] = [
    (("iki ayri bant",), "BICLONAL_GAMMOPATHY"),
    (("iki bant",), "BICLONAL_GAMMOPATHY"),
    (("biklonal",), "BICLONAL_GAMMOPATHY"),
    (("oligoklonal",), "OLIGOCLONAL_BANDS"),
    (("ogg",), "OLIGOCLONAL_BANDS"),
    (("yorumlanamad",), "UNINTERPRETABLE_TECHNICAL_ISSUE"),
    (("tekrar",), "UNINTERPRETABLE_TECHNICAL_ISSUE"),
    (("atipik",), "ATYPICAL_PATTERN"),
    # MG0 / MGO / MGO + supheli: monoklonal yok ama supheli bant var
    # (2017 defteri: "MGO" = "MGÖ" = monoklonal gammopati izlenmedi; "mG" = monoklonal gammopati;
    #  "SG" = supheli gorunum; "S.O.G" = supheli ... gorunum -> sade() sonrasi "s o g")
    (("mgo", "sg"), "SUSPICIOUS_BAND"),
    (("mgo", "s o g"), "SUSPICIOUS_BAND"),
    (("mgo", "suphe"), "SUSPICIOUS_BAND"),
    (("s o g",), "SUSPICIOUS_BAND"),
    (("mg0", "sg"), "SUSPICIOUS_BAND"),
    (("mg0", "5g"), "SUSPICIOUS_BAND"),
    (("mg 0", "sg"), "SUSPICIOUS_BAND"),
    (("mg0", "suphe"), "SUSPICIOUS_BAND"),
    (("sg",), "SUSPICIOUS_BAND"),
    (("5g",), "SUSPICIOUS_BAND"),      # OCR "ŞG" -> "5G"
    (("s g",), "SUSPICIOUS_BAND"),
    (("suphe",), "SUSPICIOUS_BAND"),
    (("mgg0",), "NO_MONOCLONAL_BAND"),
    (("mgg 0",), "NO_MONOCLONAL_BAND"),
    (("mgg",), "MONOCLONAL_GAMMOPATHY"),
    (("monoklonal",), "MONOCLONAL_GAMMOPATHY"),
    (("poliklonal",), "POLYCLONAL_HYPERGAMMAGLOBULINEMIA"),
    (("hipergama",), "POLYCLONAL_HYPERGAMMAGLOBULINEMIA"),
    (("mg0",), "NO_MONOCLONAL_BAND"),
    (("mg 0",), "NO_MONOCLONAL_BAND"),
    ((" mgo ",), "NO_MONOCLONAL_BAND"),
    ((" mgo",), "NO_MONOCLONAL_BAND"),
    (("normal",), "NO_MONOCLONAL_BAND"),
    (("negatif",), "NO_MONOCLONAL_BAND"),
    (("saptanmad",), "NO_MONOCLONAL_BAND"),
    (("izlenmedi",), "NO_MONOCLONAL_BAND"),
    # tek basina "mG" / "MG" = monoklonal gammopati (EN SONDA: mgo/mg0/mgg'yi yakalamasin)
    ((" mg ",), "MONOCLONAL_GAMMOPATHY"),
    ((" meg ",), "MONOCLONAL_GAMMOPATHY"),
]


BELIRSIZ = "BELIRSIZ_IZOTIPSIZ_MG"   # tek basina "MGG"/"mG": Ø dusmus olabilir -> karsilastirmadan dislanir


def cevir(metin) -> str:
    if isinstance(metin, str):
        # El yazisi "yok" isareti: Ø / ∅ / ø / φ / (-) / (−)  -> "0"  (sade() bunlari silerdi!)
        metin = re.sub(r"[ØøΦφ∅⌀]", "0", metin)
        metin = re.sub(r"\(\s*[-−–]\s*\)", "0", metin)
    s = sade(metin)
    if not s.strip():
        return ""
    izotip_var = bool(re.search(r"ig\s*[gam]|kappa|lambda|\bk\b|\bl\b", s))
    for anahtarlar, sinif in KISALTMA:
        if all(a in s for a in anahtarlar):
            if sinif == "MONOCLONAL_GAMMOPATHY" and not izotip_var and anahtarlar[0] in ("mgg", " mg ", " meg "):
                return BELIRSIZ
            return sinif
    if re.search(r"\bsup\b|\bsup\.|supheli|s o g|\bsg\b", s):     # "sup. gor.", "S.O.G", "SG" -> supheli
        return "SUSPICIOUS_BAND"
    # Sadece izotip yazilmis ("IgG-K", "IgA-lambda"): rutin raporda bu, o tipte monoklonal bant demektir
    if izotip_var and not re.search(r"sup|sg|s o g|yok|izlenmedi|0|olmayan|poliklonal|hipo|normal", s):
        return "MONOCLONAL_GAMMOPATHY"
    return ""


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    uyum = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    sans = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return (uyum - sans) / (1 - sans) if sans < 1 else float("nan")


def yorum(k):
    if k != k:
        return "hesaplanamadi"
    for esik, ad in ((0.81, "cok iyi"), (0.61, "iyi"), (0.41, "orta"), (0.21, "zayif"), (-1, "cok zayif")):
        if k >= esik:
            return ad
    return "?"


def _boot_kappa(a, b, tekrar=2000, seed=42):
    """Panel bazinda bootstrap ile kappa %95 GA."""
    import numpy as np
    a, b = list(a), list(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    ks = []
    for _ in range(tekrar):
        idx = rng.integers(0, n, n)
        ks.append(kappa([a[i] for i in idx], [b[i] for i in idx]))
    ks = [k for k in ks if k == k]
    return (np.percentile(ks, 2.5), np.percentile(ks, 97.5)) if ks else (float("nan"), float("nan"))


def ikinci_okuyucu_uyumu(master_yolu, ikinci_yolu, havuz_yolu=None, tekrar=2000, cikti="ikinci_okuyucu_uyum.xlsx"):
    """Gozlemciler arasi uyum: birinci okuyucu (master) vs kor ikinci okuyucu; istenirse model de."""
    m = pd.read_csv(master_yolu, dtype=str)
    r2 = pd.read_csv(ikinci_yolu, dtype=str)
    k = r2.merge(m[["filename", "pattern", "band_target_definite", "device"]].rename(
        columns={"pattern": "pattern_1", "band_target_definite": "def_1", "device": "device_m"}),
        on="filename", how="inner")
    k["def_2"] = k["band_target_definite"].fillna("")
    k["def_1"] = k["def_1"].fillna("")
    bin1 = (k["def_1"].str.strip() != "").astype(int).tolist()
    bin2 = (k["def_2"].str.strip() != "").astype(int).tolist()
    pat1, pat2 = k["pattern_1"].tolist(), k["pattern"].tolist()
    # supheli = pozitif duyarlilik analizi (patern SUSPICIOUS ya da kesin bant)
    poz = {"MONOCLONAL_GAMMOPATHY", "BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS", "SUSPICIOUS_BAND"}
    sp1 = [int(p in poz) for p in pat1]; sp2 = [int(p in poz) for p in pat2]
    print(f"\nIKINCI OKUYUCU UYUMU  (n = {len(k)} panel; ikinci okuyucu: {r2['annotator'].iloc[0] if len(r2) else '?'})")
    satirlar = []
    for ad, x, y in (("8 sinif patern", pat1, pat2), ("bant var/yok (kesin)", bin1, bin2),
                     ("bant var/yok (supheli dahil)", sp1, sp2)):
        kp = kappa(x, y); lo, hi = _boot_kappa(x, y, tekrar)
        ham = sum(p == q for p, q in zip(x, y)) / len(x)
        print(f"  {ad:<30} kappa {kp:.3f} (%95 GA {lo:.3f}-{hi:.3f})  ham uyum %{100 * ham:.1f}  ({yorum(kp)})")
        satirlar.append({"karsilastirma": "okuyucu1 vs okuyucu2", "olcut": ad, "n": len(x),
                         "kappa": round(kp, 3), "ga_alt": round(lo, 3), "ga_ust": round(hi, 3), "ham_uyum": round(ham, 3)})
    for cz, g in k.groupby(k["device_m"].astype(str).str.upper()):
        if len(g) < 20:
            continue
        gi = g.index
        print(f"    {cz}: n={len(g)}  kappa(8 sinif)={kappa([pat1[i] for i in gi], [pat2[i] for i in gi]):.3f}"
              f"  bant var/yok={kappa([bin1[i] for i in gi], [bin2[i] for i in gi]):.3f}")
    capraz = pd.crosstab(k["pattern_1"].map(lambda p: T.PATTERN_TR.get(p, p)).rename("Okuyucu 1 (calisma uzmani)"),
                         k["pattern"].map(lambda p: T.PATTERN_TR.get(p, p)).rename("Okuyucu 2 (kor)"))
    fark = k[k["pattern_1"] != k["pattern"]]
    if havuz_yolu:
        import numpy as np
        d = np.load(havuz_yolu, allow_pickle=True)
        fn = d["filename"].astype(str)
        konum = {f: i for i, f in enumerate(fn)}
        idx = [konum.get(f) for f in k["filename"].astype(str)]
        ok = [i for i in idx if i is not None]
        if len(ok) == len(idx):
            mp = [T.PATTERN_CLASSES[int(d["pattern_pred"][i])] for i in idx]
            mb = [int(d["band_pred"][i].any()) for i in idx]
            print(f"\n  MODEL (havuz tahminleri, ayni {len(k)} panel):")
            for ad, x, y, lab in (("8 sinif: model vs okuyucu1", mp, pat1, "model vs okuyucu1"),
                                  ("8 sinif: model vs okuyucu2", mp, pat2, "model vs okuyucu2"),
                                  ("bant var/yok: model vs okuyucu1", mb, bin1, "model vs okuyucu1"),
                                  ("bant var/yok: model vs okuyucu2", mb, bin2, "model vs okuyucu2")):
                kp = kappa(x, y); lo, hi = _boot_kappa(x, y, tekrar)
                print(f"  {ad:<34} kappa {kp:.3f} (%95 GA {lo:.3f}-{hi:.3f})")
                satirlar.append({"karsilastirma": lab, "olcut": ad.split(":")[0], "n": len(x), "kappa": round(kp, 3),
                                 "ga_alt": round(lo, 3), "ga_ust": round(hi, 3),
                                 "ham_uyum": round(sum(p == q for p, q in zip(x, y)) / len(x), 3)})
            k["model_pattern"] = mp
        else:
            print(f"  UYARI: havuzda bulunamayan panel: {len(idx) - len(ok)} -> model karsilastirmasi atlandi")
    with pd.ExcelWriter(cikti) as w:
        pd.DataFrame(satirlar).to_excel(w, sheet_name="ozet", index=False)
        capraz.to_excel(w, sheet_name="capraz_tablo")
        fark.to_excel(w, sheet_name="ayrisan_paneller", index=False)
    print(f"\n  -> {cikti}  (ozet + capraz tablo + ayrisan {len(fark)} panel)")
    if len(fark):
        print("  En sik ayrismalar:")
        for (x, y), n in fark.groupby(["pattern_1", "pattern"]).size().sort_values(ascending=False).head(5).items():
            print(f"   {n:3d}  okuyucu1: {T.PATTERN_TR.get(x, x)}  <->  okuyucu2: {T.PATTERN_TR.get(y, y)}")


# ===========================================================================
# IKI OKUYUCU: H.G. (birinci) vs P.A. (ikinci), M.O. hakemligi
# ===========================================================================
SABLON = "okuyucu_karsilastirma.xlsx"
_XESC = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _anahtar(s):
    """Excel, metindeki '_xHHHH_' dizilerini kacis kodu sanip tek karaktere cevirir
    (or. '..._5__x2187_y1672_...' -> '..._5_\u2187y1672_...'). Excel'den gelen
    ve CSV'den gelen dosya adlarini AYNI bicime getirir; eslestirme bununla yapilir."""
    s = unicodedata.normalize("NFC", str(s))
    return _XESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def _excel_guvenli(s):
    """Dosya adi sablona OLDUGU GIBI yazilir. Excel kaydederken '_xHHHH_' dizilerini
    bozabilir; okurken _anahtar() her iki tarafi ayni bicime getirdigi icin sorun olmaz."""
    return str(s)
BANT_VAR = {"MONOCLONAL_GAMMOPATHY", "BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS"}
BANT_VAR_SUPHELI = BANT_VAR | {"SUSPICIOUS_BAND"}


def _tr_to_kod(v):
    """Acilir listedeki Turkce adi (ya da dogrudan kodu) kanonik koda cevirir."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    v = str(v).strip()
    if not v:
        return ""
    if v in T.PATTERN_CLASSES:
        return v
    ters = {tr.strip().lower(): kod for kod, tr in T.PATTERN_TR.items()}
    return ters.get(v.lower(), "?" + v)


def _ilk_okuma_paternleri(yol):
    """Goc oncesi (eski arac) kayitlarindan filename -> patern kodu."""
    e = pd.read_csv(yol, dtype=str)
    if "sample_type" in e.columns:
        e = e[e["sample_type"].astype(str).str.upper() == "SERUM"]
    out = {}
    for fn, ad, pd_ in zip(e["filename"], e["all_diagnoses"].fillna(""), e["primary_diagnosis"].fillna("")):
        rec, _ = T.from_legacy(ad, pd_)
        out[str(fn)] = (rec.pattern, ";".join(sorted(rec.band_target_definite_only())))
    return out


def sablon_olustur(master_yolu, cikti=SABLON, hg_ilk_yolu=None):
    from pathlib import Path
    from openpyxl import load_workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    if Path(cikti).exists():
        raise SystemExit(f"'{cikti}' zaten var -- elle girilmis veriyi ezmemek icin DURDURULDU.\n"
                         f"Yeni sablon istiyorsaniz once eski dosyayi baska ada tasiyin.")
    m = pd.read_csv(master_yolu, dtype=str)
    if "sample_type" in m.columns:
        m = m[m["sample_type"].astype(str).str.upper() == "SERUM"]
    m = m.sort_values(["device", "source_file", "filename"]).reset_index(drop=True)
    giris = pd.DataFrame({"sira": range(1, len(m) + 1), "cihaz": m["device"].values,
                          "kaynak_jel": m["source_file"].values, "filename": m["filename"].map(_excel_guvenli).values,
                          "PA_patern": "", "PA_bant_serbest": "",
                          "MO_ye_gitti": "", "MO_karari": "", "HG_ilk_okuma_farkliysa": "", "not": ""})
    hg = pd.DataFrame({"filename": m["filename"].map(_excel_guvenli).values, "HG_patern_kod": m["pattern"].values,
                       "HG_bant_kesin": m.get("band_target_definite", pd.Series([""] * len(m))).fillna("").values,
                       "HG_kaynak": "master", "master_patern_kod": m["pattern"].values})
    if hg_ilk_yolu:
        ilk = _ilk_okuma_paternleri(hg_ilk_yolu)
        _ham = pd.Series(m["filename"].astype(str).values, index=hg.index)
        bulundu = _ham.isin(ilk)
        hg.loc[bulundu, "HG_patern_kod"] = [ilk[f][0] for f in _ham[bulundu]]
        hg.loc[bulundu, "HG_bant_kesin"] = [ilk[f][1] for f in _ham[bulundu]]
        hg.loc[bulundu, "HG_kaynak"] = "ilk_okuma"
        degisen = int((hg.loc[bulundu, "HG_patern_kod"] != hg.loc[bulundu, "master_patern_kod"]).sum())
        print(f"H.G. ilk okuma ({hg_ilk_yolu}): {int(bulundu.sum())} panel eslesti, "
              f"{int((~bulundu).sum())} panel master'dan alindi (sonradan eklenenler).")
        print(f"  Eslesen panellerde ilk okuma ile guncel master paterni farkli: {degisen} "
              f"(hakemlik + revizyonlar + kural degisiklikleri)")
        ilk_fn = set(ilk); kayip = len(ilk_fn - set(_ham))
        if kayip:
            print(f"  Bilgi: ilk okumadaki {kayip} panel guncel master'da yok (cikarilan kopyalar vb.)")
    hg["HG_patern"] = hg["HG_patern_kod"].map(lambda p: T.PATTERN_TR.get(p, p))
    aciklama = pd.DataFrame({"ACIKLAMA": [
        f"Panel listesi: {master_yolu}.  H.G. etiket kaynagi: "
        + (f"{hg_ilk_yolu} (ilk okuma), yoksa master" if hg_ilk_yolu else master_yolu),
        "PA_giris sayfasinda yalniz P.A.'nin ILK, BAGIMSIZ okumasini girin (tartisma oncesi).",
        "PA_patern: acilir listeden secin (8 patern). Bos birakilan panel hesaba girmez.",
        "PA_bant_serbest: istege bagli, ornek 'IgG-kappa'. Kappa hesabinda kullanilmaz.",
        "MO_ye_gitti: evet / hayir.  MO_karari: Murat Hoca'nin kesin karari (acilir liste).",
        "HG_ilk_okuma_farkliysa: YALNIZ sizin ilk okumaniz sonradan degistiyse (MO karari, revizyon),",
        "   ILK okumanizi buraya girin; hesapta gizli sayfadaki etiketin yerine bu kullanilir.",
        "HG sayfasi bilerek gizlidir; P.A. tanilarini girerken kendi etiketinizi gormeyin.",
        "Satir silmeyin. Siralama/filtre serbesttir (kaynak_jel ile filtreleyin); 'filename' hucresini DEGISTIRMEYIN, eslestirme anahtaridir."]})
    listeler = pd.DataFrame({"patern": [T.PATTERN_TR[c] for c in T.PATTERN_CLASSES]})
    with pd.ExcelWriter(cikti, engine="openpyxl") as w:
        giris.to_excel(w, sheet_name="PA_giris", index=False)
        aciklama.to_excel(w, sheet_name="aciklama", index=False)
        hg.to_excel(w, sheet_name="HG", index=False)
        listeler.to_excel(w, sheet_name="listeler", index=False)
    wb = load_workbook(cikti)
    ws = wb["PA_giris"]
    n = ws.max_row
    rng = f"=listeler!$A$2:$A${len(listeler) + 1}"
    for kol in ("E", "H", "I"):        # PA_patern, MO_karari, HG_ilk_okuma_farkliysa
        dv = DataValidation(type="list", formula1=rng, allow_blank=True,
                            showErrorMessage=True, errorTitle="Gecersiz", error="Listeden secin.")
        ws.add_data_validation(dv); dv.add(f"{kol}2:{kol}{n}")
    dv = DataValidation(type="list", formula1='"evet,hayir"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"G2:G{n}")
    ws.freeze_panes = "E2"
    for kol, gen in zip("ABCDEFGHIJ", (6, 10, 22, 34, 34, 16, 11, 34, 34, 20)):
        ws.column_dimensions[kol].width = gen
    wb["aciklama"].column_dimensions["A"].width = 110
    wb["HG"].sheet_state = "hidden"
    wb["listeler"].sheet_state = "hidden"
    wb.active = 0
    wb.save(cikti)
    print(f"Sablon -> {cikti}  ({len(giris)} panel)")
    print("P.A. tanilarini 'PA_giris' sayfasina girin; sonra:")
    print(f"   python uyum_hesapla.py --iki-okuyucu {cikti}")


def iki_okuyucu_uyumu(yol, tekrar=2000, cikti="hg_pa_uyum.xlsx", panel_listesi=None):
    g = pd.read_excel(yol, sheet_name="PA_giris", dtype=str)
    g["filename"] = g["filename"].map(_anahtar)
    if panel_listesi:
        # Ayni panellerde karsilastirma (or. ikinci okuyucunun 300 paneli)
        from pathlib import Path
        pl = pd.read_csv(panel_listesi, dtype=str)
        secili = set(pl["filename"].map(_anahtar))
        once = len(g)
        g = g[g["filename"].astype(str).isin(secili)].copy()
        print(f"Panel filtresi ({panel_listesi}): {len(g)} / {once} panel secildi "
              f"(listede {len(secili)} panel)")
        cikti = cikti.replace(".xlsx", f"_{Path(panel_listesi).stem}.xlsx")
    h = pd.read_excel(yol, sheet_name="HG", dtype=str)
    h["filename"] = h["filename"].map(_anahtar)
    k = g.merge(h[["filename", "HG_patern_kod", "HG_kaynak"]], on="filename", how="left")
    k["PA"] = k["PA_patern"].apply(_tr_to_kod)
    ilk = k["HG_ilk_okuma_farkliysa"].apply(_tr_to_kod)
    k["HG"] = [i if i else hgk for i, hgk in zip(ilk, k["HG_patern_kod"].fillna(""))]
    n_duzelt = int((ilk != "").sum())
    bos = int((k["PA"] == "").sum())
    hatali = k[k["PA"].str.startswith("?") | k["HG"].str.startswith("?")]
    if len(hatali):
        print(f"UYARI: taninamayan {len(hatali)} deger (listeden secilmemis) -> dislandi:")
        for v, c in pd.concat([hatali["PA_patern"], hatali["HG_ilk_okuma_farkliysa"]]).dropna().value_counts().head(8).items():
            print(f"   {c:3d}  '{v}'")
    k = k[(k["PA"] != "") & ~k["PA"].str.startswith("?") & (k["HG"] != "") & ~k["HG"].str.startswith("?")].copy()
    print(f"\nH.G. vs P.A.  (degerlendirilen n = {len(k)}; P.A. bos {bos}; "
          f"H.G. ilk okuma ile duzeltilen {n_duzelt})")
    if len(k) < 10:
        raise SystemExit("Yeterli giris yok.")
    hg, pa = k["HG"].tolist(), k["PA"].tolist()
    satirlar = []
    for ad, x, y in (("8 sinif patern", hg, pa),
                     ("bant var/yok (supheli = yok)", [int(p in BANT_VAR) for p in hg], [int(p in BANT_VAR) for p in pa]),
                     ("bant var/yok (supheli = var)", [int(p in BANT_VAR_SUPHELI) for p in hg],
                      [int(p in BANT_VAR_SUPHELI) for p in pa])):
        kp = kappa(x, y); lo, hi = _boot_kappa(x, y, tekrar)
        ham = sum(p == q for p, q in zip(x, y)) / len(x)
        print(f"  {ad:<30} kappa {kp:.3f} (%95 GA {lo:.3f}-{hi:.3f})  ham uyum %{100 * ham:.1f}  ({yorum(kp)})")
        satirlar.append({"kapsam": "tumu", "olcut": ad, "n": len(x), "kappa": round(kp, 3),
                         "ga_alt": round(lo, 3), "ga_ust": round(hi, 3), "ham_uyum": round(ham, 3)})
    for cz, gg in k.groupby(k["cihaz"].astype(str).str.upper()):
        if len(gg) < 20:
            continue
        x, y = gg["HG"].tolist(), gg["PA"].tolist()
        xb, yb = [int(p in BANT_VAR) for p in x], [int(p in BANT_VAR) for p in y]
        kp, kb = kappa(x, y), kappa(xb, yb)
        print(f"    {cz}: n={len(gg)}  kappa(8 sinif)={kp:.3f}  bant var/yok={kb:.3f}")
        satirlar += [{"kapsam": cz, "olcut": "8 sinif patern", "n": len(gg), "kappa": round(kp, 3)},
                     {"kapsam": cz, "olcut": "bant var/yok (supheli = yok)", "n": len(gg), "kappa": round(kb, 3)}]
    for kyn, gg in k.groupby(k["HG_kaynak"].fillna("?")):
        if len(gg) < 20 or k["HG_kaynak"].nunique() < 2:
            continue
        kp = kappa(gg["HG"].tolist(), gg["PA"].tolist())
        print(f"    H.G. kaynagi = {kyn}: n={len(gg)}  kappa(8 sinif)={kp:.3f}")
        satirlar.append({"kapsam": f"HG_kaynak={kyn}", "olcut": "8 sinif patern", "n": len(gg), "kappa": round(kp, 3)})
    fark = k[k["HG"] != k["PA"]].copy()
    print(f"  Ayrisan panel: {len(fark)} / {len(k)}")
    # M.O. hakemligi
    mo = k[k["MO_ye_gitti"].fillna("").str.strip().str.lower() == "evet"].copy()
    mo["MO"] = mo["MO_karari"].apply(_tr_to_kod)
    mo_ozet = {"ayrisan_panel": len(fark), "MO_ye_giden": len(mo),
               "MO_karari_girilmis": int((mo["MO"] != "").sum()),
               "karar_HG_ile_ayni": int(((mo["MO"] == mo["HG"]) & (mo["MO"] != mo["PA"])).sum()),
               "karar_PA_ile_ayni": int(((mo["MO"] == mo["PA"]) & (mo["MO"] != mo["HG"])).sum()),
               "karar_ucuncu_secenek": int(((mo["MO"] != "") & (mo["MO"] != mo["HG"]) & (mo["MO"] != mo["PA"])).sum()),
               "ayrisip_MO_ye_gitmeyen": int(((fark["MO_ye_gitti"].fillna("").str.strip().str.lower()) != "evet").sum())}
    print("  M.O. hakemligi: " + ", ".join(f"{a}={b}" for a, b in mo_ozet.items()))
    tr = lambda p: T.PATTERN_TR.get(p, p)
    capraz = pd.crosstab(k["HG"].map(tr).rename("H.G."), k["PA"].map(tr).rename("P.A."))
    with pd.ExcelWriter(cikti) as w:
        pd.DataFrame(satirlar).to_excel(w, sheet_name="ozet", index=False)
        capraz.to_excel(w, sheet_name="capraz_tablo")
        pd.DataFrame([mo_ozet]).T.rename(columns={0: "n"}).to_excel(w, sheet_name="MO_hakemligi")
        fark[["sira", "cihaz", "kaynak_jel", "filename", "HG", "PA", "MO_ye_gitti", "MO_karari", "not"]].to_excel(
            w, sheet_name="ayrisan_paneller", index=False)
        yy = k[["sira", "cihaz", "kaynak_jel", "filename", "HG_kaynak", "HG", "PA",
                "MO_ye_gitti", "MO_karari", "not"]].copy()
        yy.insert(5, "HG_patern", yy["HG"].map(tr)); yy.insert(6, "PA_patern", yy["PA"].map(tr))
        yy.insert(7, "ayni_mi", (yy["HG"] == yy["PA"]).map({True: "evet", False: "hayir"}))
        yy.insert(8, "HG_bant_var", yy["HG"].isin(BANT_VAR).map({True: 1, False: 0}))
        yy.insert(9, "PA_bant_var", yy["PA"].isin(BANT_VAR).map({True: 1, False: 0}))
        yy.assign(sira=pd.to_numeric(yy["sira"], errors="coerce")).drop(columns=["HG", "PA"]).sort_values("sira").to_excel(w, sheet_name="tum_paneller_yan_yana", index=False)
    print(f"\n  -> {cikti}")
    if len(fark):
        print("  En sik ayrismalar:")
        for (x, y), n in fark.groupby(["HG", "PA"]).size().sort_values(ascending=False).head(5).items():
            print(f"   {n:3d}  H.G.: {tr(x)}  <->  P.A.: {tr(y)}")


def the_sablon_olustur(master_yolu, cikti="okuyucu_THE.xlsx"):
    from pathlib import Path
    from openpyxl import load_workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    if Path(cikti).exists():
        raise SystemExit(f"'{cikti}' zaten var -- elle girilmis veriyi ezmemek icin DURDURULDU.")
    m = pd.read_csv(master_yolu, dtype=str)
    if "sample_type" in m.columns:
        m = m[m["sample_type"].astype(str).str.upper() == "SERUM"]
    m = m.sort_values(["device", "source_file", "filename"]).reset_index(drop=True)
    giris = pd.DataFrame({"sira": range(1, len(m) + 1), "cihaz": m["device"].values,
                          "kaynak_jel": m["source_file"].values, "filename": m["filename"].map(_excel_guvenli).values,
                          "THE_patern": "", "THE_bant_serbest": "", "not": ""})
    listeler = pd.DataFrame({"patern": [T.PATTERN_TR[c] for c in T.PATTERN_CLASSES]})
    with pd.ExcelWriter(cikti, engine="openpyxl") as w:
        giris.to_excel(w, sheet_name="THE_giris", index=False)
        listeler.to_excel(w, sheet_name="listeler", index=False)
    wb = load_workbook(cikti)
    ws = wb["THE_giris"]; n = ws.max_row
    dv = DataValidation(type="list", formula1=f"=listeler!$A$2:$A${len(listeler) + 1}", allow_blank=True,
                        showErrorMessage=True, errorTitle="Gecersiz", error="Listeden secin.")
    ws.add_data_validation(dv); dv.add(f"E2:E{n}")
    ws.freeze_panes = "E2"
    for kol, gen in zip("ABCDEFG", (6, 10, 22, 34, 34, 16, 24)):
        ws.column_dimensions[kol].width = gen
    wb["listeler"].sheet_state = "hidden"
    wb.save(cikti)
    print(f"T.H.E. sablonu -> {cikti}  ({len(giris)} panel; referans ve diger okuyucu etiketleri YOK)")
    print("Bos birakilan panel hesaba girmez; T.H.E.'nin okudugu panelleri doldurun.")


def uc_okuyucu_uyumu(the_yol, pa_yol, master_yolu, havuz_yolu=None, tekrar=2000, cikti="uc_okuyucu_uyum.xlsx"):
    _sayfalar = pd.ExcelFile(the_yol).sheet_names
    # T.H.E. ayri sablonda (THE_giris) ya da P.A. dosyasina eklenmis sutunda (PA_giris.THE_patern)
    th = pd.read_excel(the_yol, sheet_name="THE_giris" if "THE_giris" in _sayfalar else "PA_giris", dtype=str)
    if "THE_patern" not in th.columns:
        raise SystemExit(f"{the_yol}: THE_patern sutunu bulunamadi.")
    th["filename"] = th["filename"].map(_anahtar)
    th["THE"] = th["THE_patern"].apply(_tr_to_kod)
    hatali = th[th["THE"].str.startswith("?")]
    if len(hatali):
        print(f"UYARI: taninamayan {len(hatali)} T.H.E. degeri dislandi: "
              f"{hatali['THE_patern'].value_counts().head(5).to_dict()}")
    th = th[(th["THE"] != "") & ~th["THE"].str.startswith("?")]
    m = pd.read_csv(master_yolu, dtype=str)[["filename", "pattern"]].rename(columns={"pattern": "REF"})
    m["filename"] = m["filename"].map(_anahtar)
    g = pd.read_excel(pa_yol, sheet_name="PA_giris", dtype=str)
    h = pd.read_excel(pa_yol, sheet_name="HG", dtype=str)
    g["filename"] = g["filename"].map(_anahtar); h["filename"] = h["filename"].map(_anahtar)
    g = g.merge(h[["filename", "HG_patern_kod"]], on="filename", how="left")
    g["PA"] = g["PA_patern"].apply(_tr_to_kod)
    ilk = g["HG_ilk_okuma_farkliysa"].apply(_tr_to_kod) if "HG_ilk_okuma_farkliysa" in g else ""
    g["HG"] = [i if i else hgk for i, hgk in zip(ilk, g["HG_patern_kod"].fillna(""))]
    # BAGIMSIZ REFERANS: iki ilk okuma ayniysa o; ayrisiyorsa M.O. karari.
    # Modelin isaret ettigi panellerin sonradan yeniden okunmasindan ETKILENMEZ.
    g["MO"] = g["MO_karari"].apply(_tr_to_kod) if "MO_karari" in g else ""
    g["REF2"] = [hg if hg == pa else mo for hg, pa, mo in zip(g["HG"], g["PA"], g["MO"])]
    k = th[["filename", "cihaz", "THE"]].merge(m, on="filename", how="left").merge(
        g[["filename", "HG", "PA", "REF2"]], on="filename", how="left")
    eksik = k[k["REF"].isna() | k["HG"].isna() | k["PA"].isna()]
    if len(eksik):
        print(f"UYARI: eslesmeyen {len(eksik)} panel dislandi (referans yok: {int(k['REF'].isna().sum())}, "
              f"H.G./P.A. yok: {int((k['HG'].isna() | k['PA'].isna()).sum())}). Ornek: "
              f"{eksik['filename'].head(3).tolist()}")
    k = k.drop(eksik.index)
    k = k[~k["PA"].str.startswith("?") & (k["PA"] != "")].reset_index(drop=True)
    print(f"\nUC OKUYUCU + MODEL  (ortak panel n = {len(k)})")
    karsilastirmalar = [("H.G. vs P.A. (ilk okumalar)", k["HG"].tolist(), k["PA"].tolist()),
                        ("T.H.E. vs referans", k["THE"].tolist(), k["REF"].tolist())]
    if havuz_yolu:
        import numpy as np
        d = np.load(havuz_yolu, allow_pickle=True)
        konum = {_anahtar(f): i for i, f in enumerate(d["filename"].astype(str))}
        idx = [konum.get(f) for f in k["filename"].astype(str)]
        if any(i is None for i in idx):
            print(f"UYARI: havuzda olmayan {sum(i is None for i in idx)} panel -> model satiri bu panelleri icermez")
        kk = [j for j, i in enumerate(idx) if i is not None]
        mp = [T.PATTERN_CLASSES[int(d["pattern_pred"][idx[j]])] for j in kk]
        if len(kk) == len(k):
            karsilastirmalar.append(("Model vs referans", mp, k["REF"].tolist()))
            k["MODEL"] = mp
    ref2_ok = (k["REF2"].fillna("") != "") & ~k["REF2"].fillna("").str.startswith("?")
    if ref2_ok.all():
        karsilastirmalar.append(("T.H.E. vs bagimsiz referans", k["THE"].tolist(), k["REF2"].tolist()))
        if "MODEL" in k:
            karsilastirmalar.append(("Model vs bagimsiz referans", k["MODEL"].tolist(), k["REF2"].tolist()))
        n_fark = int((k["REF2"] != k["REF"]).sum())
        print(f"  Bagimsiz referans (H.G.=P.A. ise o, degilse M.O.) guncel referanstan {n_fark} panelde farkli")
    else:
        print(f"  UYARI: {int((~ref2_ok).sum())} ayrisan panelde M.O. karari yok -> bagimsiz referans hesaplanmadi")
    satirlar = []
    print(f"  {'':<30}{'8 sinif kappa (%95 GA)':<30}{'bant var/yok kappa (%95 GA)':<30}")
    for ad, x, y in karsilastirmalar:
        kp = kappa(x, y); lo, hi = _boot_kappa(x, y, tekrar)
        xb, yb = [int(p in BANT_VAR) for p in x], [int(p in BANT_VAR) for p in y]
        kb = kappa(xb, yb); lb, hb = _boot_kappa(xb, yb, tekrar)
        print(f"  {ad:<30}{kp:.3f} ({lo:.3f}-{hi:.3f})          {kb:.3f} ({lb:.3f}-{hb:.3f})")
        satirlar.append({"karsilastirma": ad, "n": len(x), "kappa_8sinif": round(kp, 3),
                         "ga_alt": round(lo, 3), "ga_ust": round(hi, 3),
                         "ham_uyum_8sinif": round(sum(p == q for p, q in zip(x, y)) / len(x), 3),
                         "kappa_bant_var_yok": round(kb, 3), "ga_alt_b": round(lb, 3), "ga_ust_b": round(hb, 3)})
    if "MODEL" in k:
        # Eslestirilmis bootstrap: ayni panellerde kappa(model) - kappa(T.H.E.)
        import numpy as np
        rng = np.random.default_rng(42)
        n = len(k)
        print("\n  Fark: kappa(model, ref) - kappa(T.H.E., ref)  [eslestirilmis bootstrap, %95 GA]")
        for refad, rc in (("referans", "REF"), ("bagimsiz referans", "REF2")):
            if rc == "REF2" and not ref2_ok.all():
                continue
            R, M, H = k[rc].tolist(), k["MODEL"].tolist(), k["THE"].tolist()
            for olcut, f in (("8 sinif", lambda v: v), ("bant var/yok", lambda v: int(v in BANT_VAR))):
                r2, m2, h2 = [f(v) for v in R], [f(v) for v in M], [f(v) for v in H]
                fark = kappa(m2, r2) - kappa(h2, r2)
                fs = []
                for _ in range(tekrar):
                    ix = rng.integers(0, n, n)
                    rr = [r2[i] for i in ix]
                    fs.append(kappa([m2[i] for i in ix], rr) - kappa([h2[i] for i in ix], rr))
                lo, hi = np.percentile(fs, [2.5, 97.5])
                print(f"    {refad:<18} {olcut:<13} fark {fark:+.3f} (%95 GA {lo:+.3f} - {hi:+.3f})")
                satirlar.append({"karsilastirma": f"FARK model - T.H.E. ({refad})", "n": n,
                                 "kappa_8sinif" if olcut == "8 sinif" else "kappa_bant_var_yok": round(fark, 3),
                                 "ga_alt" if olcut == "8 sinif" else "ga_alt_b": round(lo, 3),
                                 "ga_ust" if olcut == "8 sinif" else "ga_ust_b": round(hi, 3)})
    for cz, gg in k.groupby(k["cihaz"].astype(str).str.upper()):
        if len(gg) >= 20:
            print(f"    {cz}: n={len(gg)}  T.H.E. vs referans kappa(8 sinif)={kappa(gg['THE'].tolist(), gg['REF'].tolist()):.3f}")
    tr = lambda p: T.PATTERN_TR.get(p, p)
    with pd.ExcelWriter(cikti) as w:
        pd.DataFrame(satirlar).to_excel(w, sheet_name="ozet", index=False)
        pd.crosstab(k["REF"].map(tr).rename("Referans"), k["THE"].map(tr).rename("T.H.E.")).to_excel(
            w, sheet_name="THE_capraz_tablo")
        k[k["THE"] != k["REF"]].assign(REF_tr=lambda x: x["REF"].map(tr), THE_tr=lambda x: x["THE"].map(tr)).to_excel(
            w, sheet_name="THE_ayrisan", index=False)
    print(f"\n  -> {cikti}")


def bagimsiz_referans_yaz(pa_yol, master_yolu, hg_ilk_yolu, cikti="bagimsiz_referans.csv"):
    """BIRINCIL REFERANS: iki bagimsiz ilk okuma ayniysa o patern, ayrisiyorsa M.O. karari.
    Cift okumadan SONRA yapilan ve hakemlik disi olan patern degisiklikleri (model kaynakli
    yeniden okuma, tutarlilik denetimi) geri alinir: o panellerde H.G.'nin ilk okumasi
    (patern + bantlar) kullanilir. Diger tum paneller master ile aynidir.
    Cikti master bicimindedir: guven_araligi.py --etiket-kaynagi ile kullanilir."""
    g = pd.read_excel(pa_yol, sheet_name="PA_giris", dtype=str)
    h = pd.read_excel(pa_yol, sheet_name="HG", dtype=str)
    g["filename"] = g["filename"].map(_anahtar); h["filename"] = h["filename"].map(_anahtar)
    g = g.merge(h[["filename", "HG_patern_kod"]], on="filename", how="left")
    g["PA"] = g["PA_patern"].apply(_tr_to_kod)
    ilk = g["HG_ilk_okuma_farkliysa"].apply(_tr_to_kod) if "HG_ilk_okuma_farkliysa" in g else ""
    g["HG"] = [i if i else hgk for i, hgk in zip(ilk, g["HG_patern_kod"].fillna(""))]
    g["MO"] = g["MO_karari"].apply(_tr_to_kod)
    g["REF2"] = [hg if hg == pa else mo for hg, pa, mo in zip(g["HG"], g["PA"], g["MO"])]
    eksik = g[(g["REF2"] == "") | g["REF2"].str.startswith("?")]
    if len(eksik):
        raise SystemExit(f"{len(eksik)} panelde bagimsiz referans kurulamadi (ayrisma var ama M.O. karari yok).")
    ref2 = dict(zip(g["filename"], g["REF2"]))
    m = pd.read_csv(master_yolu, dtype=str)
    e = pd.read_csv(hg_ilk_yolu, dtype=str)
    if "sample_type" in e.columns:
        e = e[e["sample_type"].astype(str).str.upper() == "SERUM"]
    ilk_rec = {_anahtar(fn): T.from_legacy(ad, pdg)[0] for fn, ad, pdg in
               zip(e["filename"], e["all_diagnoses"].fillna(""), e["primary_diagnosis"].fillna(""))}
    out = m.copy()
    out["referans_kaynagi"] = ""
    serum = out["sample_type"].astype(str).str.upper().eq("SERUM") if "sample_type" in out else pd.Series(True, index=out.index)
    n_degis, n_bulunamadi = 0, 0
    for i in out.index[serum]:
        key = _anahtar(out.at[i, "filename"])
        r2 = ref2.get(key)
        if r2 is None:
            n_bulunamadi += 1
            continue
        if r2 == out.at[i, "pattern"]:
            out.at[i, "referans_kaynagi"] = "master (cift okuma ile uyumlu)"
            continue
        rec = ilk_rec.get(key)
        if rec is None or rec.pattern != r2:
            raise SystemExit(f"{out.at[i, 'filename']}: bagimsiz referans {r2}, ama H.G. ilk okumasi bulunamadi/uyusmuyor.")
        out.at[i, "pattern"] = r2
        out.at[i, "band_target_definite"] = ";".join(sorted(rec.band_target_definite_only()))
        out.at[i, "band_target"] = ";".join(sorted(rec.band_target()))
        out.at[i, "referans_kaynagi"] = "H.G. ilk okuma (= P.A.); sonraki revizyon geri alindi"
        n_degis += 1
    out.to_csv(cikti, index=False, encoding="utf-8-sig")
    print(f"Bagimsiz referans -> {cikti}")
    print(f"  Patern degistirilen (revizyon geri alinan) panel: {n_degis}")
    print(f"  Okuyucu dosyasinda bulunamayan serum paneli: {n_bulunamadi}")
    print("Sonraki adim (tum yayin metrikleri bu referansla):")
    print(f"   python guven_araligi.py --girdi cv/havuz_tahminleri.npz --etiket-kaynagi {cikti} --cikti yayin_ciktilari_bagimsiz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="birlesik_kayitlar.xlsx")
    ap.add_argument("--ikinci-okuyucu", default=None,
                    help="ikinci_okuyucu_annotations.csv: kor ikinci okuyucu ile gozlemciler arasi kappa (+GA)")
    ap.add_argument("--master", default="annotations_master.csv")
    ap.add_argument("--havuz", default=None, help="cv/havuz_tahminleri.npz: modeli de ayni panellerde karsilastir")
    ap.add_argument("--tekrar", type=int, default=2000)
    ap.add_argument("--sablon-olustur", action="store_true",
                    help="H.G. vs P.A. icin giris sablonu (okuyucu_karsilastirma.xlsx) uret")
    ap.add_argument("--bagimsiz-referans-yaz", default=None, metavar="OKUYUCU_XLSX",
                    help="doldurulmus okuyucu dosyasindan bagimsiz_referans.csv uret (--hg-ilk gerekli)")
    ap.add_argument("--the-sablon", action="store_true", help="T.H.E. icin kor giris sablonu (okuyucu_THE.xlsx)")
    ap.add_argument("--uc-okuyucu", default=None, help="doldurulmus okuyucu_THE.xlsx")
    ap.add_argument("--pa-dosyasi", default="okuyucu_karsilastirma.xlsx",
                    help="--uc-okuyucu ile: H.G./P.A. dosyasi")
    ap.add_argument("--panel-listesi", default=None,
                    help="--iki-okuyucu ile: yalniz bu CSV'deki (filename) panellerde hesapla")
    ap.add_argument("--hg-ilk", default=None,
                    help="H.G. ilk okuma kaynagi (or. annotations_goc_oncesi_yedek.csv)")
    ap.add_argument("--iki-okuyucu", default=None,
                    help="doldurulmus okuyucu_karsilastirma.xlsx: H.G. vs P.A. kappa (+GA)")
    a = ap.parse_args()
    if a.bagimsiz_referans_yaz:
        if not a.hg_ilk:
            raise SystemExit("--hg-ilk annotations_goc_oncesi_yedek.csv de verilmeli")
        bagimsiz_referans_yaz(a.bagimsiz_referans_yaz, a.master, a.hg_ilk)
        return
    if a.the_sablon:
        the_sablon_olustur(a.master)
        return
    if a.uc_okuyucu:
        uc_okuyucu_uyumu(a.uc_okuyucu, a.pa_dosyasi, a.master, a.havuz, a.tekrar)
        return
    if a.sablon_olustur:
        sablon_olustur(a.master, hg_ilk_yolu=a.hg_ilk)
        return
    if a.iki_okuyucu:
        iki_okuyucu_uyumu(a.iki_okuyucu, a.tekrar, panel_listesi=a.panel_listesi)
        return
    if a.ikinci_okuyucu:
        ikinci_okuyucu_uyumu(a.master, a.ikinci_okuyucu, a.havuz, a.tekrar)
        return

    b = pd.read_excel(a.girdi, dtype=str) if a.girdi.endswith(".xlsx") else pd.read_csv(a.girdi, dtype=str)
    # Yalniz guvenilir eslesmeler: bas harfi UYUSMAYAN ("KONTROL") paneller kappa'ya girmez
    kontrol_n = int(b["durum"].fillna("").str.startswith("KONTROL").sum())
    b = b[b["durum"].fillna("").str.startswith(("ESLESTI", "HELENA ESLESTI")) & ~b["durum"].fillna("").str.contains("kontrol")].copy()
    print(f"Bas harfi uyusmayan/kismen uyan {kontrol_n} panel uyum hesabindan DISLANDI")
    b = b.drop_duplicates("kirpma_dosyasi")
    print(f"Defterle eslesen kayit: {len(b)}")

    b["uzman2"] = b["defter_tani"].apply(cevir)
    cev = b[b["uzman2"].eq("") & b["defter_tani"].fillna("").str.strip().ne("")]
    if len(cev):
        (cev["defter_tani"].value_counts().rename_axis("defter_metni").reset_index(name="adet")
         .to_csv(OUT_CEVRILEMEYEN, index=False, encoding="utf-8-sig"))
        print(f"Cevrilemeyen {cev['defter_tani'].nunique()} farkli metin -> {OUT_CEVRILEMEYEN}")
        for t, n in cev["defter_tani"].value_counts().head(8).items():
            print(f"   {n:3d}  {t}")

    belirsiz_n = int(b["uzman2"].eq(BELIRSIZ).sum())
    print(f"Izotipsiz tek basina MGG/mG (Ø dusmus olabilir): {belirsiz_n} kayit karsilastirmadan DISLANDI")
    k = b[b["uzman2"].ne("") & b["uzman2"].ne(BELIRSIZ) & b["benim_patern"].notna()].copy()
    print(f"\nKarsilastirilabilir kayit: {len(k)}")
    if len(k) < 10:
        print("Yeterli veri yok.")
        return

    ben, o = k["benim_patern"].tolist(), k["uzman2"].tolist()
    kp = kappa(ben, o)
    print("\n=== GOZLEMCILER ARASI UYUM ===")
    print(f"  Ham uyum    : %{100 * sum(x == y for x, y in zip(ben, o)) / len(k):.1f}")
    print(f"  Cohen kappa : {kp:.3f}  ({yorum(kp)})")

    def ikili(p):
        return "BANT VAR" if p in ("MONOCLONAL_GAMMOPATHY", "BICLONAL_GAMMOPATHY",
                                   "OLIGOCLONAL_BANDS") else "BANT YOK"
    b2, o2 = [ikili(x) for x in ben], [ikili(y) for y in o]
    k2 = kappa(b2, o2)
    print(f"\n  Monoklonal bant var/yok:")
    print(f"    Ham uyum    : %{100 * sum(x == y for x, y in zip(b2, o2)) / len(k):.1f}")
    print(f"    Cohen kappa : {k2:.3f}  ({yorum(k2)})")

    # Cihaz bazinda (Interlab / Helena)
    k["_cihaz"] = k["kaynak_tarama"].astype(str).str.lower().map(lambda v: "HELENA" if "helena" in v else "INTERLAB") \
        if "kaynak_tarama" in k.columns else "INTERLAB"
    for cz, g in k.groupby("_cihaz"):
        if len(g) < 10: continue
        gb, go = g["benim_patern"].tolist(), g["uzman2"].tolist()
        gb2, go2 = [ikili(x) for x in gb], [ikili(y) for y in go]
        print(f"\n  {cz}: n={len(g)}  kappa(8 sinif)={kappa(gb, go):.3f}  ham=%{100 * sum(x == y for x, y in zip(gb, go)) / len(g):.1f}"
              f"  |  bant var/yok kappa={kappa(gb2, go2):.3f}  ham=%{100 * sum(x == y for x, y in zip(gb2, go2)) / len(g):.1f}")

    capraz = pd.crosstab(k["benim_patern"].map(lambda p: T.PATTERN_TR.get(p, p)).rename("Siz"),
                         k["uzman2"].map(lambda p: T.PATTERN_TR.get(p, p)).rename("Defter uzmani"))
    fark = k[k["benim_patern"] != k["uzman2"]]
    with pd.ExcelWriter(OUT_UYUM) as w:
        capraz.to_excel(w, sheet_name="capraz_tablo")
        fark[["kirpma_dosyasi", "kart_id", "ife_no", "ad", "soyad", "barkod",
              "defter_tani", "uzman2", "benim_patern", "benim_tani_tr"]].to_excel(
            w, sheet_name="ayrisan_kayitlar", index=False)
        pd.DataFrame({"olcut": ["n", "ham uyum", "kappa", "kappa (bant var/yok)"],
                      "deger": [len(k), round(sum(x == y for x, y in zip(ben, o)) / len(k), 3),
                                round(kp, 3), round(k2, 3)]}).to_excel(
            w, sheet_name="ozet", index=False)
    print(f"\n  -> {OUT_UYUM}  (capraz tablo + ayrisan {len(fark)} kayit)")
    if len(fark):
        print("\n  En sik ayrismalar:")
        for (x, y), n in fark.groupby(["benim_patern", "uzman2"]).size().sort_values(ascending=False).head(5).items():
            print(f"   {n:3d}  siz: {T.PATTERN_TR.get(x, x)}  <->  defter: {T.PATTERN_TR.get(y, y)}")


if __name__ == "__main__":
    main()
