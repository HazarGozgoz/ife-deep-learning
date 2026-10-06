# -*- coding: utf-8 -*-
"""
capraz_dogrulama.py -- 5 katli, jel bazinda gruplu, cihaz dengeli CV
=====================================================================
Tek bir 70/15/15 bolmesi "bu sonuc sans eseri mi" sorusuna cevap vermez.
5 katli CV'de her jel tam bir kez test setine girer; tum tahminler
havuzlanip metrikler + guven araliklari hesaplanir.

NASIL CALISIR
--------------
1) annotations_master.csv'den 5 kat uretir (GroupKFold: kaynak jel;
   cihaz oranlari korunur). Her katta: 4 kat egitim (icinden %15 dogrulama),
   1 kat test.
2) Her kat icin egitim scriptini AYNEN calistirir (ayni ayarlar).
3) Test tahminlerini havuzlar -> tum verinin uzerinde tek bir sonuc seti.

SURE: 5 x (tek egitim suresi). Gece calistirin:
    nohup python capraz_dogrulama.py > cv.log 2>&1 &

KULLANIM
---------
    python capraz_dogrulama.py                       # varsayilan ayarlar
    python capraz_dogrulama.py --ayar "--cozunurluk=384 --bant-hedefi=kesin"
    python capraz_dogrulama.py --sadece-topla        # egitim bitti, sadece havuzla

CIKTI
------
    cv/kat_1..5/          her katin yayin ciktilari
    cv/havuz_tahminleri.npz   tum test tahminleri (guven_araligi.py girdisi)
    cv/table_cv_summary.csv   kat basina + havuz metrikleri
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

MASTER = "annotations_master.csv"
CV_DIR = Path("cv")
K = 5
SEED = 42


def katlari_uret(df: pd.DataFrame):
    """Cihaz bazinda ayri ayri GroupKFold, sonra katlar birlestirilir."""
    df = df.copy()
    df["_kat"] = -1
    for dev, sub in df.groupby("device"):
        if sub["source_file"].nunique() < K:
            # Cok az jel: hepsi egitimde kalsin (kat = -1 -> her katta train)
            print(f"  {dev}: {sub['source_file'].nunique()} jel < {K}, tum katlarda egitimde")
            continue
        gkf = GroupKFold(n_splits=K)
        for k, (_, test_idx) in enumerate(gkf.split(sub, groups=sub["source_file"])):
            df.loc[sub.index[test_idx], "_kat"] = k
    return df


def kat_yaz(df: pd.DataFrame, k: int, hedef: Path):
    test = df[df["_kat"] == k]
    kalan = df[df["_kat"] != k]
    # Kalanin icinden %15 dogrulama (jel bazinda)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + k)
    tr_i, va_i = next(gss.split(kalan, groups=kalan["source_file"]))
    train, val = kalan.iloc[tr_i], kalan.iloc[va_i]
    for ad, d in (("train", train), ("validation", val), ("test", test)):
        d.drop(columns=["_kat"]).to_csv(hedef / f"{ad}_annotations.csv",
                                        index=False, encoding="utf-8-sig")
    return len(train), len(val), len(test), test["source_file"].nunique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ayar", default="", help="Egitim scriptine gecirilecek ek argumanlar")
    ap.add_argument("--sadece-topla", action="store_true")
    ap.add_argument("--esik-modu", default="dengeli", choices=["dengeli", "tarama"])
    a = ap.parse_args()
    CV_DIR.mkdir(exist_ok=True)

    if not a.sadece_topla:
        df = pd.read_csv(MASTER)
        if "sample_type" in df.columns:
            df = df[df["sample_type"].astype(str).str.upper() == "SERUM"]
        df = katlari_uret(df)
        print(f"{len(df)} kayit, {df['source_file'].nunique()} jel -> {K} kat")

        for k in range(K):
            kat = CV_DIR / f"kat_{k + 1}"
            kat.mkdir(exist_ok=True)
            ntr, nva, nte, nj = kat_yaz(df, k, kat)
            print(f"\n=== KAT {k + 1}/{K}: train {ntr} | val {nva} | test {nte} ({nj} jel) ===")
            # Egitim scripti sabit dosya adlari okur: kat dosyalarini yerine koy
            for ad in ("train", "validation", "test"):
                shutil.copy(kat / f"{ad}_annotations.csv", f"{ad}_annotations.csv")
            cmd = [sys.executable, "egitim_ve_degerlendirme.py", f"--esik-modu={a.esik_modu}",
                   "--dil=en"] + a.ayar.split()
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"Kat {k + 1} egitimi basarisiz (kod {r.returncode}).")
            # Ciktilari kata tasi
            for p in Path(".").glob("yayin_ciktilari_*"):
                if p.is_dir():
                    hedef = kat / p.name
                    if hedef.exists():
                        shutil.rmtree(hedef)
                    shutil.move(str(p), str(hedef))
            for p in Path(".").glob("best_ife_*.pth"):
                shutil.move(str(p), str(kat / p.name))
            print(f"Kat {k + 1} tamamlandi -> {kat}/")

    # --- HAVUZLA ---
    print("\n=== TAHMINLER HAVUZLANIYOR ===")
    parcalar = []
    for k in range(K):
        adaylar = list((CV_DIR / f"kat_{k + 1}").glob("yayin_ciktilari_*/test_tahminleri.npz"))
        if not adaylar:
            print(f"  kat {k + 1}: tahmin dosyasi yok, atlandi")
            continue
        d = np.load(adaylar[0], allow_pickle=True)
        parcalar.append({kk: d[kk] for kk in d.files})
        print(f"  kat {k + 1}: {len(d['pattern_true'])} tahmin")
    if not parcalar:
        raise SystemExit("Havuzlanacak tahmin yok.")

    havuz = {}
    for kk in parcalar[0]:
        if kk in ("pattern_classes", "band_classes", "band_thr"):
            havuz[kk] = parcalar[0][kk]
        else:
            havuz[kk] = np.concatenate([p[kk] for p in parcalar])
    havuz["fold"] = np.concatenate([np.full(len(p["pattern_true"]), i + 1) for i, p in enumerate(parcalar)])
    np.savez(CV_DIR / "havuz_tahminleri.npz", **havuz)

    # Kat basina + havuz ozeti
    from sklearn.metrics import f1_score
    sat = []
    for i, p in enumerate(parcalar):
        sat.append({"fold": i + 1, "n": len(p["pattern_true"]),
                    "pattern_acc": round(float((p["pattern_true"] == p["pattern_pred"]).mean()), 3),
                    "band_macro_f1": round(float(f1_score(p["band_true"], p["band_pred"],
                                                          average="macro", zero_division=0)), 3)})
    sat.append({"fold": "POOLED", "n": len(havuz["pattern_true"]),
                "pattern_acc": round(float((havuz["pattern_true"] == havuz["pattern_pred"]).mean()), 3),
                "band_macro_f1": round(float(f1_score(havuz["band_true"], havuz["band_pred"],
                                                      average="macro", zero_division=0)), 3)})
    s = pd.DataFrame(sat)
    s.to_csv(CV_DIR / "table_cv_summary.csv", index=False)
    print("\n" + s.to_string(index=False))
    acc = s[s["fold"] != "POOLED"]["pattern_acc"]
    print(f"\nKatlar arasi patern dogrulugu: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"\n-> {CV_DIR}/havuz_tahminleri.npz")
    print("Guven araligi icin:  python guven_araligi.py --girdi cv/havuz_tahminleri.npz")


if __name__ == "__main__":
    main()
