# -*- coding: utf-8 -*-
"""
veri_hazirlama.py -- annotations_master.csv -> train/validation/test
======================================================================
etiket_gocu.py'nin urettigi annotations_master.csv'yi okur ve modelin
kullanacagi UC bolmeyi uretir. Projedeki TEK bolme noktasi burasidir
(eskiden veri_hazirlama.py ve yayin.py birbirinden habersiz IKI AYRI bolme
uretiyordu).

TASARIM KARARLARI
------------------
1) GRUPLU BOLME (source_file'a gore). Ayni jel taramasindan kirpilan 15-20
   serit ayni hastanin/ayni jelin seritleridir; bunlar hem train hem test'e
   dagilirsa model "ezberleyip" sahte yuksek skor verir. GroupShuffleSplit
   ile bir kaynak jelin TUM seritleri tek bir bolmede kalir.

2) CIHAZ-DUYARLI BOLME. Helena ve Interlab ayri ayri bolunur, sonra
   birlestirilir. Boylece her bolmede iki cihazdan da ornek bulunur ve
   test setinde cihaz kirilimi raporlanabilir. Bir cihazda tek kaynak jel
   varsa bolunemez -- script bunu ACIKCA uyarir (su an Helena'da 1 jel var).

3) ETIKET KAYNAGI = all_diagnoses_canonical. Klasor adi (primary_diagnosis)
   sadece dosya yolu icin kullanilir (bkz. etiket_gocu.py 5/7. adim).

4) QC onerileri SESSIZCE UYGULANMAZ. qc_flag=1 satirlar sadece raporlanir.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ife_taxonomy import (ALL_FLAGS, ALL_FLAG_TR, BAND_CLASSES, BAND_TR,
                          FLAG_CLASSES, PATTERN_CLASSES,
                          PATTERN_TR, POLY_LANE_CLASSES)

MASTER_IN = "annotations_master.csv"
TRAIN_OUT = "train_annotations.csv"
VAL_OUT = "validation_annotations.csv"
TEST_OUT = "test_annotations.csv"
REPORT_SPLIT = "bolme_raporu.csv"

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
HOLDOUT_VAL_FRAC = 0.15   # --test-cihazi modunda kalan cihazin jellerinin val payi
UYARLAMA_TEST_JEL = 16    # --uyarlama-jel modunda test cihazinin sabit test jeli sayisi
SEED = 42


def grouped_split(df: pd.DataFrame, seed: int = SEED):
    """source_file'a gore 70/15/15 gruplu bolme (iki asamali)."""
    groups = df["source_file"].astype(str).values
    n_groups = len(set(groups))
    if n_groups < 3:
        return None  # bolunemez

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(VAL_FRAC + TEST_FRAC), random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    rest_df = df.iloc[rest_idx].copy()

    rest_groups = rest_df["source_file"].astype(str).values
    if len(set(rest_groups)) < 2:
        return train_df, rest_df.iloc[0:0].copy(), rest_df

    rel_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_groups))
    return train_df, rest_df.iloc[val_idx].copy(), rest_df.iloc[test_idx].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=MASTER_IN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--test-cihazi", default=None,
                    help="HOLDOUT modu: bu cihazin (orn. HELENA) TUM jelleri test setine gider; "
                         "kalan cihazlarin jelleri train/val'e bolunur. Zamansal + platform "
                         "dogrulama icin. Egitim bitince 'python veri_hazirlama.py' ile normal "
                         "bolmeyi GERI GETIRIN.")
    ap.add_argument("--uyarlama-jel", type=int, default=None,
                    help="(--test-cihazi ile) UYARLAMA EGRISI: test cihazinin jelleri karistirilir, "
                         f"{UYARLAMA_TEST_JEL} jel SABIT test seti olur, kalan havuzdan N jel egitime "
                         "eklenir (N=0 icin holdout npz'yi guven_araligi --jel-filtre ile sinirlayin).")
    args = ap.parse_args()
    if args.uyarlama_jel is not None and not args.test_cihazi:
        raise SystemExit("HATA: --uyarlama-jel yalniz --test-cihazi ile kullanilir.")

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(
            f"HATA: '{src}' yok. Once 'python etiket_gocu.py --input annotations.csv' calistirin.")

    df = pd.read_csv(src)
    print(f"[1/5] Okundu: {len(df)} kayit, {df['source_file'].nunique()} kaynak jel")

    # Guvenlik agi
    if "sample_type" in df.columns:
        before = len(df)
        df = df[df["sample_type"].astype(str).str.upper() == "SERUM"].copy()
        if len(df) < before:
            print(f"      {before - len(df)} idrar kaydi elendi (guvenlik agi).")
    if "device" not in df.columns:
        df["device"] = "BILINMIYOR"
        print("      UYARI: 'device' kolonu yok, hepsi BILINMIYOR sayildi.")

    # Üç eksenli alanlar (etiket_gocu.py / ife_annotator.py yazar)
    for col, default in (("pattern", "NO_ABNORMAL_BAND"), ("bands", ""),
                         ("flags", ""), ("poly_lanes", ""), ("band_target", ""),
                         ("band_target_definite", ""), ("review_reason", "")):
        if col not in df.columns:
            df[col] = default
    df["band_set"] = df["band_target"].fillna("").apply(
        lambda s: {t for t in str(s).split(";") if t})
    df["flag_set"] = df["flags"].fillna("").apply(
        lambda s: {t for t in str(s).split("|") if t})
    df["lane_set"] = df["poly_lanes"].fillna("").apply(
        lambda s: {t for t in str(s).split("|") if t})
    if "img_relpath" not in df.columns:
        df["img_relpath"] = df.apply(
            lambda r: f"{r['primary_diagnosis']}/{r['filename']}", axis=1)

    # fillna("") SART: bos hucreler CSV'den NaN gelir, str(NaN)=="nan"
    # olur ve HER kayit isaretli gorunur.
    n_rev = int((df["review_reason"].fillna("").astype(str).str.strip() != "").sum())
    if n_rev:
        print(f"      NOT: {n_rev} kayitta gozden gecirme onerisi duruyor "
              f"(elenmedi; 'ife_review_tool.py --only-flagged' ile bakabilirsiniz).")

    # --- Cihaz bazinda ayri ayri bol ---
    parts = {"Train": [], "Val": [], "Test": []}
    holdout = (args.test_cihazi or "").strip().upper()
    if holdout:
        dev_col = df["device"].astype(str).str.upper()
        if holdout not in set(dev_col):
            raise SystemExit(f"HATA: '{holdout}' cihazi yok. Mevcut: {sorted(set(dev_col))}")
        te = df[dev_col == holdout].copy()
        rest = df[dev_col != holdout].copy()
        if rest["source_file"].nunique() < 2:
            raise SystemExit("HATA: kalan cihazda train/val icin en az 2 kaynak jel gerekli.")
        gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_VAL_FRAC, random_state=args.seed)
        tr_idx, va_idx = next(gss.split(rest, groups=rest["source_file"].astype(str).values))
        tr, va = rest.iloc[tr_idx].copy(), rest.iloc[va_idx].copy()
        if args.uyarlama_jel is not None:
            N = args.uyarlama_jel
            jel_list = sorted(te["source_file"].astype(str).unique())
            import numpy as _np
            jel_list = list(_np.random.default_rng(args.seed).permutation(jel_list))
            test_jel, havuz = jel_list[:UYARLAMA_TEST_JEL], jel_list[UYARLAMA_TEST_JEL:]
            if N > len(havuz):
                raise SystemExit(f"HATA: havuzda {len(havuz)} jel var, N={N} olamaz.")
            egitim_jel = havuz[:N]
            te_all = te
            te = te_all[te_all["source_file"].astype(str).isin(test_jel)].copy()
            ek = te_all[te_all["source_file"].astype(str).isin(egitim_jel)].copy()
            tr = pd.concat([tr, ek], ignore_index=True)
            pd.DataFrame({"source_file": test_jel}).to_csv("uyarlama_test_jelleri.csv", index=False)
            print(f"\n      UYARLAMA EGRISI N={N}: {holdout} jelleri karistirildi (seed {args.seed}); "
                  f"sabit TEST {len(test_jel)} jel / {len(te)} panel (uyarlama_test_jelleri.csv); "
                  f"egitime eklenen {N} jel / {len(ek)} panel; kullanilmayan {len(havuz) - N} jel.")
        parts["Train"].append(tr); parts["Val"].append(va); parts["Test"].append(te)
        print(f"\n[2/5] HOLDOUT bolme: {holdout} jellerinin TAMAMI test; "
              f"kalan cihaz(lar) jel bazinda train/val ({int((1 - HOLDOUT_VAL_FRAC) * 100)}/"
              f"{int(HOLDOUT_VAL_FRAC * 100)}):")
        print(f"      train {len(tr)} kirpma ({tr['source_file'].nunique()} jel) | "
              f"val {len(va)} ({va['source_file'].nunique()}) | "
              f"TEST={holdout} {len(te)} ({te['source_file'].nunique()} jel)")
        print("      Not: model bu cihazi egitimde HIC gormeyecek; esikler kalan cihazin val jellerinde secilir.")
    else:
        print("\n[2/5] Cihaz-duyarli gruplu bolme:")
    for dev in ([] if holdout else sorted(df["device"].astype(str).unique())):
        sub = df[df["device"].astype(str) == dev].copy()
        n_g = sub["source_file"].nunique()
        res = grouped_split(sub, seed=args.seed)
        if res is None:
            print(f"      {dev:<10} {len(sub):5d} kirpma / {n_g} jel  "
                  f"-> BOLUNEMEDI (en az 3 farkli kaynak jel gerekli). "
                  f"Tamami TRAIN'e konuldu.")
            parts["Train"].append(sub)
            continue
        tr, va, te = res
        parts["Train"].append(tr); parts["Val"].append(va); parts["Test"].append(te)
        print(f"      {dev:<10} {len(sub):5d} kirpma / {n_g} jel  -> "
              f"train {len(tr)} ({tr['source_file'].nunique()} jel) | "
              f"val {len(va)} ({va['source_file'].nunique()}) | "
              f"test {len(te)} ({te['source_file'].nunique()})")

    train_df = pd.concat(parts["Train"], ignore_index=True) if parts["Train"] else pd.DataFrame()
    val_df = pd.concat(parts["Val"], ignore_index=True) if parts["Val"] else pd.DataFrame()
    test_df = pd.concat(parts["Test"], ignore_index=True) if parts["Test"] else pd.DataFrame()

    # --- Sizinti kontrolu ---
    print("\n[3/5] Sizinti kontrolu (ayni kaynak jel birden fazla bolmede mi?):")
    g_tr = set(train_df["source_file"]) if len(train_df) else set()
    g_va = set(val_df["source_file"]) if len(val_df) else set()
    g_te = set(test_df["source_file"]) if len(test_df) else set()
    leaks = (g_tr & g_va) | (g_tr & g_te) | (g_va & g_te)
    if leaks:
        print(f"      !!! SIZINTI: {sorted(leaks)}")
    else:
        print("      Temiz -- hicbir kaynak jel birden fazla bolmede degil.")

    # --- Sinif dagilimi ---
    print("\n[4/5] Bolme basina dagilim (uc eksen):")

    def _counts(d, col_set):
        c = collections.Counter()
        for x in d.get(col_set, []):
            c.update(x)
        return c

    rows = []
    hdr = f"      {'':<44}{'Train':>8}{'Val':>7}{'Test':>7}"

    print("\n      EKSEN A -- PATERN"); print(hdr)
    for cls in PATTERN_CLASSES:
        v = [int((d["pattern"] == cls).sum()) if len(d) else 0
             for d in (train_df, val_df, test_df)]
        w = "  <-- EGITIMDE YOK" if v[0] == 0 else ("  <-- TESTTE YOK" if v[2] == 0 else "")
        print(f"      {PATTERN_TR[cls]:<44}{v[0]:>8}{v[1]:>7}{v[2]:>7}{w}")
        rows.append({"eksen": "patern", "sinif": cls, "train": v[0], "val": v[1], "test": v[2]})

    print("\n      EKSEN B -- BANTLAR"); print(hdr)
    ct = [_counts(d, "band_set") for d in (train_df, val_df, test_df)]
    for cls in BAND_CLASSES:
        v = [c.get(cls, 0) for c in ct]
        w = ""
        if v[0] == 0:
            w = "  <-- EGITIMDE YOK"
        elif v[2] == 0:
            w = "  <-- TESTTE YOK"
        elif v[2] < 5:
            w = "  <-- test n<5, metrik oynak"
        print(f"      {BAND_TR[cls]:<44}{v[0]:>8}{v[1]:>7}{v[2]:>7}{w}")
        rows.append({"eksen": "bant", "sinif": cls, "train": v[0], "val": v[1], "test": v[2]})

    print("\n      EKSEN C/D -- BAYRAK + POLIKLONAL LANE"); print(hdr)
    cf = [_counts(d, "flag_set") for d in (train_df, val_df, test_df)]
    cl = [_counts(d, "lane_set") for d in (train_df, val_df, test_df)]
    for cls in ALL_FLAGS:
        v = [c.get(cls, 0) for c in cf]
        mark = "" if cls in FLAG_CLASSES else "  (model hedefi degil)"
        print(f"      {ALL_FLAG_TR.get(cls, cls):<44}{v[0]:>8}{v[1]:>7}{v[2]:>7}{mark}")
        rows.append({"eksen": "bayrak", "sinif": cls, "train": v[0], "val": v[1], "test": v[2]})
    for cls in POLY_LANE_CLASSES:
        v = [c.get(cls, 0) for c in cl]
        print(f"      {'poliklonal ' + cls:<44}{v[0]:>8}{v[1]:>7}{v[2]:>7}")
        rows.append({"eksen": "poliklonal_lane", "sinif": cls,
                     "train": v[0], "val": v[1], "test": v[2]})
    pd.DataFrame(rows).to_csv(REPORT_SPLIT, index=False, encoding="utf-8-sig")

    # --- Kaydet ---
    for name, d, path in (("Train", train_df, TRAIN_OUT),
                          ("Val", val_df, VAL_OUT),
                          ("Test", test_df, TEST_OUT)):
        out = d.drop(columns=["band_set", "flag_set", "lane_set"], errors="ignore")
        out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[5/5] Yazildi: {TRAIN_OUT} ({len(train_df)}) | {VAL_OUT} ({len(val_df)}) | "
          f"{TEST_OUT} ({len(test_df)})  + {REPORT_SPLIT}")
    if holdout:
        print(f"\n!!! Bu bolme {holdout} HOLDOUT icindir. Egitim ve degerlendirme bitince NORMAL "
              f"bolmeyi geri getirin:  python veri_hazirlama.py")


if __name__ == "__main__":
    main()
