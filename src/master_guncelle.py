# -*- coding: utf-8 -*-
"""
master_guncelle.py -- mevcut master'i GUNCEL taksonomiye tasir
================================================================
etiket_gocu.py, annotations.csv'den master'i SIFIRDAN uretir. Helena
kayitlariniz artik sadece master'da oldugu icin onu tekrar calistirmak
YENI ISARETLEMELERI SILER. Bu script onun yerine gecer: master'i
yerinde gunceller, hicbir satiri silmez.

NE YAPAR
---------
1) Eski bayrak adlarini gunceller:
       POLYCLONAL_HYPERGAMMAGLOBULINEMIA -> POLYCLONAL_INCREASE
       POLYCLONAL_BACKGROUND             -> POLYCLONAL_INCREASE
   (Ikisi tek kavramda birlestirilmisti; eski satirlar hala eski adi
    tasidigi icin model hedefinde gorunmuyorlardi.)
2) Tureyen tum alanlari guncel taksonomiyle YENIDEN hesaplar:
   pattern, n_bands, n_clones, band_target, all_diagnoses_tr,
   review_reason. Boylece eski ve yeni kayitlar ayni kurallara uyar.

KULLANIM
---------
    python master_guncelle.py            # ne degisecegini gosterir
    python master_guncelle.py --uygula
"""
import argparse, re, shutil
from pathlib import Path
import pandas as pd

# Kirpma araci koordinatlari DOSYA ADINA da yaziyor:
#   2018_File 15__20260212150922250958_5__x2399_y2690_w932_h852.png
# Eski 9.3 surumu x/y/w/h sutunlarini bos birakmis ama dosya adi dolu.
RE_KOORD = re.compile(r"__x(\d+)_y(\d+)_w(\d+)_h(\d+)", re.I)

import ife_taxonomy as T

ESKI_BAYRAK = {"POLYCLONAL_HYPERGAMMAGLOBULINEMIA": "POLYCLONAL_INCREASE",
               "POLYCLONAL_BACKGROUND": "POLYCLONAL_INCREASE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="annotations_master.csv")
    ap.add_argument("--uygula", action="store_true")
    a = ap.parse_args()

    yol = Path(a.input)
    if not yol.exists():
        raise SystemExit(f"'{yol}' yok.")
    df = pd.read_csv(yol)
    for c in df.columns:
        df[c] = df[c].astype(object)
    print(f"{len(df)} kayit okundu.")

    # --- 0) KOORDINAT KURTARMA: bos x/y/w/h alanlarini dosya adindan doldur ---
    kurtarilan = 0
    if {"x", "y", "w", "h", "filename"} <= set(df.columns):
        for i, r in df.iterrows():
            if all(pd.notna(r.get(k)) for k in ("x", "y", "w", "h")):
                continue
            m = RE_KOORD.search(str(r.get("filename", "")))
            if not m:
                continue
            df.at[i, "x"] = int(m.group(1)); df.at[i, "y"] = int(m.group(2))
            df.at[i, "w"] = int(m.group(3)); df.at[i, "h"] = int(m.group(4))
            kurtarilan += 1
        kalan = int(df[["x", "y", "w", "h"]].isna().any(axis=1).sum())
        print(f"Koordinati dosya adindan kurtarilan : {kurtarilan}")
        print(f"Hala koordinatsiz kalan             : {kalan}")

    degisen_bayrak = 0
    yeni = {k: [] for k in ("pattern", "bands", "n_bands", "n_clones", "flags",
                            "poly_lanes", "band_target", "band_target_definite",
                            "has_suspicious", "all_diagnoses_tr",
                            "all_diagnoses_canonical", "review_reason")}
    for _, r in df.iterrows():
        bayraklar = {b for b in str(r.get("flags", "") or "").split("|") if b}
        yeni_b = {ESKI_BAYRAK.get(b, b) for b in bayraklar}
        yeni_b = {b for b in yeni_b if b in T.ALL_FLAGS}
        if yeni_b != bayraklar:
            degisen_bayrak += 1

        rec = T.IFERecord(bands=T.parse_bands(str(r.get("bands", "") or "")))
        rec.flags = yeni_b
        rec.poly_lanes = {l for l in str(r.get("poly_lanes", "") or "").split("|")
                          if l in T.POLY_LANE_CLASSES}
        rec.note = "" if pd.isna(r.get("detail_note")) else str(r.get("detail_note"))
        p = str(r.get("pattern", "") or "")
        if p in ("BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS"):
            rec.declared_pattern = p
        rec.resolve_pattern()

        legacy = ";".join(sorted(T.to_legacy_labels(rec)))
        tr = "; ".join([T.PATTERN_TR[rec.pattern]]
                       + [T.band_label(b.band_class, b.certainty) for b in rec.bands]
                       + [T.ALL_FLAG_TR.get(f, f) for f in sorted(rec.flags)])
        yeni["pattern"].append(rec.pattern)
        yeni["bands"].append(rec.bands_str())
        yeni["n_bands"].append(rec.n_bands())
        yeni["n_clones"].append(T.count_distinct_clones(rec.bands))
        yeni["flags"].append(rec.flags_str())
        yeni["poly_lanes"].append(rec.poly_lanes_str())
        yeni["band_target"].append(";".join(sorted(rec.band_target())))
        yeni["band_target_definite"].append(";".join(sorted(rec.band_target_definite_only())))
        yeni["has_suspicious"].append(int(rec.has_suspicious()))
        yeni["all_diagnoses_tr"].append(tr)
        yeni["all_diagnoses_canonical"].append(legacy)
        yeni["review_reason"].append(T.check_consistency(rec))

    eski_pat = df["pattern"].copy()
    for k, v in yeni.items():
        if k in df.columns:
            df[k] = v

    print(f"\nBayragi guncellenen kayit : {degisen_bayrak}")
    print(f"Paterni degisen kayit     : {int((eski_pat != df['pattern']).sum())}")
    print("\nGuncel bayrak dagilimi:")
    import collections
    c = collections.Counter()
    for f in df["flags"]:
        c.update([x for x in str(f or '').split('|') if x])
    for k in T.ALL_FLAGS:
        etiket = T.ALL_FLAG_TR.get(k, k)
        hedef = "" if k in T.FLAG_CLASSES else "  (model hedefi degil)"
        print(f"  {c.get(k,0):6d}  {etiket}{hedef}")
    print("\nGuncel patern dagilimi:")
    for k, n in df["pattern"].value_counts().items():
        print(f"  {n:6d}  {T.PATTERN_TR.get(k,k)}")

    if not a.uygula:
        print("\nKaydetmek icin:  python master_guncelle.py --uygula")
        return
    yedek = yol.with_name(yol.stem + "_guncelleme_oncesi_yedek.csv")
    shutil.copy2(yol, yedek)
    df.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"\nKaydedildi. Yedek: {yedek}")
    print("Simdi:  python veri_hazirlama.py")


if __name__ == "__main__":
    main()
