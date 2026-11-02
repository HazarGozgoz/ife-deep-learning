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

OMURGA KARSILASTIRMASI (yayindaki cv/ sonuclarina DOKUNMAZ)
    python capraz_dogrulama.py --omurga=resnet50
      -> cv/kat_1..5 icindeki AYNI kat dosyalarini kullanir (katlar yeniden
         uretilmez), ciktilari cv_resnet50/ klasorune yazar.
    Secenekler: resnet50 | efficientnet_b0 | densenet121 | swin_t
    Elle: --cv-klasoru=<klasor> --katlari-kullan=<eski cv klasoru>

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
    ap.add_argument("--omurga", default=None,
                    help="resnet50 | efficientnet_b0 | densenet121 | swin_t (karsilastirma kosusu)")
    ap.add_argument("--cv-klasoru", default=None, help="ciktilarin yazilacagi klasor (varsayilan cv)")
    ap.add_argument("--katlari-kullan", default=None,
                    help="katlari yeniden uretme; bu klasordeki kat_1..5 CSV'lerini kullan (or. cv)")
    ap.add_argument("--uzerine-yaz", action="store_true",
                    help="klasorde onceki sonuc varsa yine de ez (dikkat)")
    a = ap.parse_args()

    global CV_DIR
    if a.omurga:
        a.ayar = (a.ayar + f" --omurga={a.omurga}").strip()
        a.cv_klasoru = a.cv_klasoru or f"cv_{a.omurga}"
        a.katlari_kullan = a.katlari_kullan or "cv"
    CV_DIR = Path(a.cv_klasoru or "cv")
    if (CV_DIR / "havuz_tahminleri.npz").exists() and not (a.sadece_topla or a.uzerine_yaz):
        raise SystemExit(f"{CV_DIR}/ icinde onceki sonuclar var; ezilmesin diye durduruldu.\n"
                         f"Bilerek yeniden kosmak icin --uzerine-yaz ekleyin.")
    CV_DIR.mkdir(exist_ok=True)
    print(f"Cikti klasoru: {CV_DIR}/  |  egitim ayarlari: {a.ayar or '(varsayilan)'}")

    if not a.sadece_topla:
        if a.katlari_kullan:
            kaynak = Path(a.katlari_kullan)
            for k in range(K):
                for ad in ("train", "validation", "test"):
                    if not (kaynak / f"kat_{k + 1}" / f"{ad}_annotations.csv").exists():
                        raise SystemExit(f"{kaynak}/kat_{k + 1}/{ad}_annotations.csv yok.")
            print(f"Katlar yeniden URETILMIYOR: {kaynak}/kat_1..{K} dosyalari kullaniliyor")
        else:
            df = pd.read_csv(MASTER)
            if "sample_type" in df.columns:
                df = df[df["sample_type"].astype(str).str.upper() == "SERUM"]
            df = katlari_uret(df)
            print(f"{len(df)} kayit, {df['source_file'].nunique()} jel -> {K} kat")

        for k in range(K):
            kat = CV_DIR / f"kat_{k + 1}"
            kat.mkdir(exist_ok=True)
            if a.katlari_kullan:
                for ad in ("train", "validation", "test"):
                    shutil.copy(Path(a.katlari_kullan) / f"kat_{k + 1}" / f"{ad}_annotations.csv",
                                kat / f"{ad}_annotations.csv")
                _d = {ad: pd.read_csv(kat / f"{ad}_annotations.csv") for ad in ("train", "validation", "test")}
                ntr, nva, nte = len(_d["train"]), len(_d["validation"]), len(_d["test"])
                nj = _d["test"]["source_file"].nunique()
            else:
                ntr, nva, nte, nj = kat_yaz(df, k, kat)
            print(f"\n=== KAT {k + 1}/{K}: train {ntr} | val {nva} | test {nte} ({nj} jel) ===")
            # Egitim scripti sabit dosya adlari okur: kat dosyalarini yerine koy
            for ad in ("train", "validation", "test"):
                shutil.copy(kat / f"{ad}_annotations.csv", f"{ad}_annotations.csv")
            # Yalniz BU katin urettigi klasor/model tasinsin: kok dizinde baska
            # yayin_ciktilari_* klasorleri (bagimsiz referans, holdout...) varsa dokunulmaz.
            onceki_klasor = {p.name for p in Path(".").glob("yayin_ciktilari_*") if p.is_dir()}
            onceki_pth = {p.name for p in Path(".").glob("best_ife_*.pth")}
            cmd = [sys.executable, "egitim_ve_degerlendirme.py", f"--esik-modu={a.esik_modu}",
                   "--dil=en"] + a.ayar.split()
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"Kat {k + 1} egitimi basarisiz (kod {r.returncode}).")
            # Ciktilari kata tasi
            for p in Path(".").glob("yayin_ciktilari_*"):
                if p.is_dir() and p.name not in onceki_klasor:
                    hedef = kat / p.name
                    if hedef.exists():
                        shutil.rmtree(hedef)
                    shutil.move(str(p), str(hedef))
            for p in Path(".").glob("best_ife_*.pth"):
                if p.name in onceki_pth:
                    continue
                shutil.move(str(p), str(kat / p.name))
            print(f"Kat {k + 1} tamamlandi -> {kat}/")

    # --- HAVUZLA ---
    print("\n=== TAHMINLER HAVUZLANIYOR ===")
    parcalar = []
    for k in range(K):
        adaylar = list((CV_DIR / f"kat_{k + 1}").glob("yayin_ciktilari_*/test_tahminleri.npz"))
        # Kat klasorune yanlislikla tasinmis baska analiz klasorleri (holdout, uyarlama,
        # bagimsiz referans) havuza KARISMASIN: yalniz bu kosunun ciktisi secilir.
        if a.omurga:
            adaylar = [x for x in adaylar if f"_{a.omurga}" in x.parent.name]
        else:
            yabanci = ("resnet50", "efficientnet_b0", "densenet121", "swin_t", "holdout", "uyarlama", "bagimsiz")
            adaylar = [x for x in adaylar if not any(y in x.parent.name for y in yabanci)]
        if len(adaylar) > 1:
            raise SystemExit(f"kat {k + 1}: birden fazla aday tahmin dosyasi var, hangisi oldugu belirsiz:\n  "
                             + "\n  ".join(str(x) for x in adaylar))
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
    print(f"Guven araligi icin:  python guven_araligi.py --girdi {CV_DIR}/havuz_tahminleri.npz")
    print("UNUTMAYIN: normal bolmeyi geri getirmek icin  python veri_hazirlama.py")


if __name__ == "__main__":
    main()
