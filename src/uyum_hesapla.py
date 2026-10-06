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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="birlesik_kayitlar.xlsx")
    ap.add_argument("--ikinci-okuyucu", default=None,
                    help="ikinci_okuyucu_annotations.csv: kor ikinci okuyucu ile gozlemciler arasi kappa (+GA)")
    ap.add_argument("--master", default="annotations_master.csv")
    ap.add_argument("--havuz", default=None, help="cv/havuz_tahminleri.npz: modeli de ayni panellerde karsilastir")
    ap.add_argument("--tekrar", type=int, default=2000)
    a = ap.parse_args()
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
