# -*- coding: utf-8 -*-
"""
guven_araligi.py -- yayin tablolari ve sekilleri, %95 guven araligi ile
=======================================================================
Girdi: havuzlanmis CV tahminleri (cv/havuz_tahminleri.npz) veya tek test
seti (yayin_ciktilari*/test_tahminleri.npz). Butun guven araliklari KAYNAK
JELE GORE kumelenmis bootstrap ile hesaplanir (ayni jelin seritleri bagimli;
serit bazli bootstrap araligi yapay daraltir).

CIKTILAR (yayin_ciktilari/ altina)
  tablo5_guven_araliklari.csv   ozet metrikler + cihaz bazli (GA ile)
  table3_bands_ci.csv           11 bant sinifi + gruplar: sens/spec/PPV/F1/AUC (GA ile)
  table_pattern_confusion.csv   8x8 patern karisiklik matrisi (satir=uzman)
  table_pattern_per_class.csv   patern sinifi basina recall/precision (GA ile)
  table_suspicious_analysis.csv supheli etiketli bantlar pozitif sayilinca duyarlilik
                                (cv/kat_*/test_annotations.csv gerekir; yoksa atlanir)
  Figure1_pattern_confusion.tiff / .png    Ingilizce, 1000 dpi LZW
  Figure2_band_ROC.tiff / .png
  Figure3_platform.tiff / .png

KULLANIM
  python guven_araligi.py --girdi cv/havuz_tahminleri.npz
  python guven_araligi.py --girdi cv/havuz_tahminleri.npz --tekrar 5000
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

BANT_VAR = {"MONOCLONAL_GAMMOPATHY", "BICLONAL_GAMMOPATHY", "OLIGOCLONAL_BANDS"}
OUT = Path("yayin_ciktilari")

PATTERN_EN = {"NO_MONOCLONAL_BAND": "No monoclonal band",
              "POLYCLONAL_HYPERGAMMAGLOBULINEMIA": "Polyclonal increase",
              "SUSPICIOUS_BAND": "Suspicious band",
              "MONOCLONAL_GAMMOPATHY": "Monoclonal",
              "BICLONAL_GAMMOPATHY": "Biclonal",
              "OLIGOCLONAL_BANDS": "Oligoclonal",
              "ATYPICAL_PATTERN": "Atypical",
              "UNINTERPRETABLE_TECHNICAL_ISSUE": "Uninterpretable"}
BAND_EN = {"IGG_KAPPA": "IgG κ", "IGG_LAMBDA": "IgG λ", "IGA_KAPPA": "IgA κ", "IGA_LAMBDA": "IgA λ",
           "IGM_KAPPA": "IgM κ", "IGM_LAMBDA": "IgM λ", "IGG_NO_LIGHT_CHAIN": "IgG heavy chain only",
           "IGA_NO_LIGHT_CHAIN": "IgA heavy chain only", "IGM_NO_LIGHT_CHAIN": "IgM heavy chain only",
           "FREE_KAPPA": "Free κ", "FREE_LAMBDA": "Free λ"}
DEVICE_EN = {"INTERLAB": "Interlab", "HELENA": "Helena"}


def _tiff(fig, ad, dpi):
    fig.savefig(OUT / f"{ad}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{ad}.tiff", dpi=dpi, format="tiff", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print(f"-> {OUT / ad}.tiff ({dpi} dpi, LZW) + .png")


def _ss(t, p):
    tp = int(((t == 1) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
    tn = int(((t == 0) & (p == 0)).sum()); fp = int(((t == 0) & (p == 1)).sum())
    return {"sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def _auc(t, s):
    try:
        return float(roc_auc_score(t, s)) if 0 < t.sum() < len(t) else np.nan
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# Metrik kumeleri (hepsi idx alt-kumesi uzerinde; bootstrap icin)
# ----------------------------------------------------------------------------
def band_altkume(d, idx):
    bv = d["band_valid"][idx] if "band_valid" in d.files else np.ones(len(idx), bool)
    return d["band_true"][idx][bv], d["band_pred"][idx][bv], d["band_prob"][idx][bv], bv


def metrikler(d, idx):
    pt, pp = d["pattern_true"][idx], d["pattern_pred"][idx]
    bt, bp, bpr, _ = band_altkume(d, idx)
    pc = list(d["pattern_classes"])
    out = {"patern_dogruluk": float((pt == pp).mean())}
    var_idx = [i for i, c in enumerate(pc) if c in BANT_VAR]
    yt = np.isin(pt, var_idx).astype(int); yp = np.isin(pp, var_idx).astype(int)
    m = _ss(yt, yp)
    out.update({"bant_var_duyarlilik": m["sens"], "bant_var_ozgulluk": m["spec"],
                "bant_var_ppv": m["ppv"], "bant_var_npv": m["npv"]})
    sup = pc.index("SUSPICIOUS_BAND") if "SUSPICIOUS_BAND" in pc else -1
    yp2 = (np.isin(pp, var_idx) | (pp == sup)).astype(int)
    out["bant_var_duyarlilik_supheli_uyari_dahil"] = float(((yt == 1) & (yp2 == 1)).sum() / max(yt.sum(), 1))
    out["bant_macro_f1"] = float(f1_score(bt, bp, average="macro", zero_division=0))
    out["bant_micro_recall"] = float(recall_score(bt, bp, average="micro", zero_division=0))
    out["bant_micro_precision"] = float(precision_score(bt, bp, average="micro", zero_division=0))
    ya, pa = bt.max(1), bp.max(1)
    m = _ss(ya, pa)
    out.update({"herhangi_bant_duyarlilik": m["sens"], "herhangi_bant_ozgulluk": m["spec"],
                "herhangi_bant_ppv": m["ppv"], "herhangi_bant_auc": _auc(ya, bpr.max(1))})
    return out


def sinif_metrikleri(d, idx):
    """Bant sinifi basina + grup basina sens/spec/ppv/f1/auc."""
    bt, bp, bpr, _ = band_altkume(d, idx)
    bc = list(d["band_classes"])
    gruplar = {"Clone-defined band (Ig-κ/λ)": [i for i, c in enumerate(bc) if ("KAPPA" in c or "LAMBDA" in c) and not c.startswith("FREE")],
               "Heavy chain without light chain": [i for i, c in enumerate(bc) if "NO_LIGHT" in c],
               "Free light chain": [i for i, c in enumerate(bc) if c.startswith("FREE")],
               "Any monoclonal band": list(range(len(bc)))}
    out = {}
    for i, c in enumerate(bc):
        m = _ss(bt[:, i], bp[:, i]); m["auc"] = _auc(bt[:, i], bpr[:, i]); m["n"] = int(bt[:, i].sum())
        out[BAND_EN.get(c, c)] = m
    for g, ii in gruplar.items():
        t, p, s = bt[:, ii].max(1), bp[:, ii].max(1), bpr[:, ii].max(1)
        m = _ss(t, p); m["auc"] = _auc(t, s); m["n"] = int(t.sum())
        out[g] = m
    return out


def patern_sinif_metrikleri(d, idx):
    pt, pp = d["pattern_true"][idx], d["pattern_pred"][idx]
    out = {}
    for i, c in enumerate(d["pattern_classes"]):
        t, p = (pt == i).astype(int), (pp == i).astype(int)
        m = _ss(t, p); m["n"] = int(t.sum())
        out[PATTERN_EN.get(c, c)] = {"recall": m["sens"], "precision": m["ppv"], "n": m["n"]}
    return out


def bootstrap(fn, d, idx_all, jeller, rng, tekrar):
    """fn(d, idx) -> dict[str, float] veya dict[str, dict[str, float]]. Jel-bazli kume bootstrap."""
    benzersiz = np.unique(jeller[idx_all])
    jel_idx = {j: idx_all[jeller[idx_all] == j] for j in benzersiz}
    nokta = fn(d, idx_all)
    ornek = []
    for _ in range(tekrar):
        sec = rng.choice(benzersiz, size=len(benzersiz), replace=True)
        ornek.append(fn(d, np.concatenate([jel_idx[j] for j in sec])))
    return nokta, ornek, len(benzersiz)


def ga(ornek, *anahtar):
    v = np.array([o[anahtar[0]] if len(anahtar) == 1 else o[anahtar[0]][anahtar[1]] for o in ornek], float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))


# ----------------------------------------------------------------------------
def etiketleri_yenile(d, master_yol):
    """Havuzdaki her panelin gercek etiketini master'dan (dosya adiyla) yeniden turetir.
    Olasiliklar degismez. Master'da olmayan paneller havuzdan cikarilir."""
    m = pd.read_csv(master_yol)
    if "sample_type" in m.columns:
        m = m[m["sample_type"].astype(str).str.upper() == "SERUM"]
    m = m.drop_duplicates("filename").set_index("filename")
    pc, bc = list(d["pattern_classes"]), list(d["band_classes"])
    fn = d["filename"].astype(str)
    var = np.array([f in m.index for f in fn])
    out = {k: d[k][var] for k in d.files if hasattr(d[k], "shape") and d[k].shape and d[k].shape[0] == len(fn)}
    for k in d.files:
        if k not in out: out[k] = d[k]
    fn = fn[var]
    pat = np.array([pc.index(m.at[f, "pattern"]) if m.at[f, "pattern"] in pc else 0 for f in fn])
    bt = np.zeros((len(fn), len(bc)), np.float32)
    kol = "band_target_definite" if "band_target_definite" in m.columns else "band_target"
    for i, f in enumerate(fn):
        for t in str(m.at[f, kol] if pd.notna(m.at[f, kol]) else "").split(";"):
            if t in bc: bt[i, bc.index(t)] = 1
    degisen = int((pat != out["pattern_true"]).sum() + (np.abs(bt - out["band_true"]).sum(1) > 0).sum())
    out["pattern_true"] = pat; out["band_true"] = bt
    out["band_valid"] = pat != pc.index("UNINTERPRETABLE_TECHNICAL_ISSUE")
    print(f"Etiketler master'dan yenilendi: {int((~var).sum())} panel master'da yok (cikarildi), "
          f"{degisen} panelde etiket degisti.")
    return out, m


def havuzu_tamamla(d, m, kat_dizin="cv"):
    """Master'da olup havuzda olmayan panelleri, jelinin test setinde oldugu katin modeliyle tahmin edip ekler."""
    import os, torch
    from gradcam import IFEMultiHead, make_eval_tf, IMAGE_DIR, DEVICE
    from PIL import Image
    pc, bc = list(d["pattern_classes"]), list(d["band_classes"])
    mevcut = set(d["filename"].astype(str))
    eksik = m[~m.index.isin(mevcut)].copy()
    eksik = eksik[eksik["img_relpath"].apply(lambda p: os.path.exists(os.path.join(IMAGE_DIR, str(p))))]
    if eksik.empty:
        print("Havuzda eksik panel yok."); return d
    jel_kat = {}
    for csv in sorted(glob.glob(os.path.join(kat_dizin, "kat_*", "test_annotations.csv"))):
        kat = Path(csv).parent
        for sf in pd.read_csv(csv)["source_file"].astype(str).unique():
            jel_kat[sf] = kat
    eksik["kat"] = eksik["source_file"].astype(str).map(jel_kat)
    if eksik["kat"].isna().any():
        print("UYARI: kat bulunamayan jel:", eksik[eksik["kat"].isna()]["source_file"].unique())
        eksik = eksik.dropna(subset=["kat"])
    ek = {k: [] for k in ("pattern_prob", "band_prob", "flag_prob", "lane_prob", "band_pred", "pattern_pred",
                          "pattern_true", "band_true", "band_valid", "device", "source_file", "filename", "fold")}
    for kat, g in eksik.groupby("kat"):
        pth = sorted(Path(kat).glob("best_ife_*.pth"))[0]
        thr = np.load(sorted(Path(kat).glob("yayin_ciktilari*/test_tahminleri.npz"))[0], allow_pickle=True)["band_thr"]
        model = IFEMultiHead(); model.load_state_dict(torch.load(pth, map_location="cpu")); model.to(DEVICE).eval()
        tf = make_eval_tf(384)
        with torch.no_grad():
            for f, r in g.iterrows():
                x = tf(Image.open(os.path.join(IMAGE_DIR, str(r["img_relpath"]))).convert("RGB")).unsqueeze(0).to(DEVICE)
                op, ob, of, ol = model(x)
                pp = torch.softmax(op, 1).cpu().numpy()[0]; bp = torch.sigmoid(ob).cpu().numpy()[0]
                ek["pattern_prob"].append(pp); ek["band_prob"].append(bp)
                ek["flag_prob"].append(torch.sigmoid(of).cpu().numpy()[0]); ek["lane_prob"].append(torch.sigmoid(ol).cpu().numpy()[0])
                ek["band_pred"].append((bp >= thr).astype(int)); ek["pattern_pred"].append(int(pp.argmax()))
                pat = pc.index(r["pattern"]) if r["pattern"] in pc else 0
                bt = np.zeros(len(bc), np.float32)
                kol = "band_target_definite" if "band_target_definite" in m.columns else "band_target"
                for t in str(r[kol] if pd.notna(r[kol]) else "").split(";"):
                    if t in bc: bt[bc.index(t)] = 1
                ek["pattern_true"].append(pat); ek["band_true"].append(bt)
                ek["band_valid"].append(pat != pc.index("UNINTERPRETABLE_TECHNICAL_ISSUE"))
                ek["device"].append(str(r.get("device", "INTERLAB")).upper()); ek["source_file"].append(str(r["source_file"]))
                ek["filename"].append(str(f)); ek["fold"].append(int(str(Path(kat).name).split("_")[-1]))
        print(f"  {Path(kat).name}: {len(g)} panel tahmin edildi")
    out = dict(d)
    for k, v in ek.items():
        v = np.array(v); out[k] = np.concatenate([np.asarray(d[k]), v.astype(np.asarray(d[k]).dtype) if np.asarray(d[k]).dtype != object else v])
    for k in ("flag_true", "lane_true"):   # eksik panellerde bilinmiyor -> NaN ile doldur
        if k in out:
            out[k] = np.concatenate([np.asarray(d[k]), np.full((len(ek["filename"]), np.asarray(d[k]).shape[1]), np.nan, np.float32)])
    print(f"Havuza eklenen panel: {len(ek['filename'])} -> toplam {len(out['filename'])}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="cv/havuz_tahminleri.npz")
    ap.add_argument("--tekrar", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kat-dizin", default="cv", help="Supheli analiz icin kat test CSV'leri")
    ap.add_argument("--tiff-dpi", type=int, default=1000)
    ap.add_argument("--etiket-kaynagi", default=None, help="annotations_master.csv: gercek etiketleri buradan yenile")
    ap.add_argument("--tamamla", action="store_true", help="Master'da olup havuzda olmayan panelleri kat modeliyle tahmin edip ekle")
    ap.add_argument("--jel-filtre", default=None,
                    help="CSV (source_file sutunu): yalniz bu jellerin panelleri degerlendirilir "
                         "(uyarlama egrisinde N=0 karsilastirmasi icin).")
    ap.add_argument("--cikti", default="yayin_ciktilari",
                    help="Cikti klasoru. Holdout gibi ikinci bir degerlendirme icin AYRI klasor verin; "
                         "yoksa yayin tablolari/figurleri ezilir.")
    a = ap.parse_args()
    global OUT
    OUT = Path(a.cikti)
    OUT.mkdir(exist_ok=True)

    d = np.load(a.girdi, allow_pickle=True)
    if a.etiket_kaynagi:
        d, m = etiketleri_yenile(d, a.etiket_kaynagi)
        if a.tamamla:
            d = havuzu_tamamla(d, m, a.kat_dizin)
        gpath = Path("cv/havuz_tahminleri_guncel.npz") if a.cikti == "yayin_ciktilari" else OUT / "tahminler_guncel.npz"
        np.savez(gpath, **d)
        print(f"-> {gpath} (orijinal korunuyor)")
        class _D(dict):
            files = property(lambda self: list(self.keys()))
        d = _D(d)
    if a.jel_filtre:
        keep = set(pd.read_csv(a.jel_filtre)["source_file"].astype(str))
        n0 = len(d["pattern_true"]); sf0 = d["source_file"].astype(str); mask = np.isin(sf0, list(keep))
        dd = {k: (d[k][mask] if (np.ndim(d[k]) >= 1 and np.shape(d[k])[0] == n0) else d[k]) for k in d.files}
        class _DF(dict):
            files = property(lambda self: list(self.keys()))
        d = _DF(dd)
        print(f"Jel filtresi {a.jel_filtre}: {int(mask.sum())}/{n0} kirpma, {len(np.unique(sf0[mask]))} jel")
    n = len(d["pattern_true"]); jeller = d["source_file"].astype(str); dev = d["device"].astype(str)
    idx_all = np.arange(n); rng = np.random.default_rng(a.seed)
    print(f"Girdi: {n} kirpma, {len(np.unique(jeller))} jel | bootstrap {a.tekrar} tekrar (jel bazinda)")

    # ---------- Tablo 5: ozet + cihaz ----------
    satirlar = []
    nokta, ornek, nj = bootstrap(metrikler, d, idx_all, jeller, rng, a.tekrar)
    print(f"\n{'METRIK':<44}{'nokta':>8}{'%95 GA':>18}")
    for k, v in nokta.items():
        lo, hi = ga(ornek, k)
        satirlar.append({"metrik": k, "nokta": round(v, 3), "ga_alt": round(lo, 3), "ga_ust": round(hi, 3),
                         "n_kirpma": n, "n_jel": nj})
        print(f"{k:<44}{v:>8.3f}   [{lo:.3f} - {hi:.3f}]")
    for dv in np.unique(dev):
        idx_d = idx_all[dev == dv]
        if len(np.unique(jeller[idx_d])) < 2:
            continue
        nd, od, njd = bootstrap(metrikler, d, idx_d, jeller, rng, a.tekrar)
        for k in ("patern_dogruluk", "bant_var_duyarlilik", "bant_var_ozgulluk", "herhangi_bant_duyarlilik",
                  "herhangi_bant_ozgulluk", "herhangi_bant_auc", "bant_macro_f1"):
            lo, hi = ga(od, k)
            satirlar.append({"metrik": f"{k}_{dv}", "nokta": round(nd[k], 3), "ga_alt": round(lo, 3),
                             "ga_ust": round(hi, 3), "n_kirpma": len(idx_d), "n_jel": njd})
            print(f"{k + '_' + dv:<44}{nd[k]:>8.3f}   [{lo:.3f} - {hi:.3f}]   ({njd} jel)")
    pd.DataFrame(satirlar).to_csv(OUT / "tablo5_guven_araliklari.csv", index=False, encoding="utf-8-sig")
    print(f"-> {OUT}/tablo5_guven_araliklari.csv")

    # ---------- Tablo 3: bant siniflari + gruplar ----------
    nokta, ornek, _ = bootstrap(sinif_metrikleri, d, idx_all, jeller, rng, a.tekrar)
    rows = []
    for c, m in nokta.items():
        r = {"class": c, "n_positive": m["n"]}
        for k in ("sens", "spec", "ppv", "f1", "auc"):
            lo, hi = ga(ornek, c, k)
            r[k] = round(m[k], 3); r[f"{k}_ci"] = f"{lo:.3f}–{hi:.3f}"
        rows.append(r)
    t3 = pd.DataFrame(rows); t3.to_csv(OUT / "table3_bands_ci.csv", index=False, encoding="utf-8-sig")
    print(f"\n-> {OUT}/table3_bands_ci.csv")
    print(t3[["class", "n_positive", "sens", "sens_ci", "spec", "ppv", "f1", "auc"]].to_string(index=False))

    # ---------- Patern: karisiklik + sinif basina ----------
    pc = list(d["pattern_classes"]); pen = [PATTERN_EN.get(c, c) for c in pc]
    cm = confusion_matrix(d["pattern_true"], d["pattern_pred"], labels=range(len(pc)))
    pd.DataFrame(cm, index=[f"Expert: {c}" for c in pen], columns=[f"Model: {c}" for c in pen]).to_csv(
        OUT / "table_pattern_confusion.csv", encoding="utf-8-sig")
    nokta, ornek, _ = bootstrap(patern_sinif_metrikleri, d, idx_all, jeller, rng, a.tekrar)
    rows = []
    for c, m in nokta.items():
        lo, hi = ga(ornek, c, "recall"); lo2, hi2 = ga(ornek, c, "precision")
        rows.append({"pattern": c, "n": m["n"], "recall": round(m["recall"], 3), "recall_ci": f"{lo:.3f}–{hi:.3f}",
                     "precision": round(m["precision"], 3), "precision_ci": f"{lo2:.3f}–{hi2:.3f}"})
    pd.DataFrame(rows).to_csv(OUT / "table_pattern_per_class.csv", index=False, encoding="utf-8-sig")
    print(f"-> {OUT}/table_pattern_confusion.csv, table_pattern_per_class.csv")

    # Figure 1: confusion matrix (satir-normalize renk, ham sayi yazi)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pc))); ax.set_yticks(range(len(pc)))
    ax.set_xticklabels(pen, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(pen, fontsize=8)
    ax.set_xlabel("Model prediction"); ax.set_ylabel("Expert interpretation")
    for i in range(len(pc)):
        for j in range(len(pc)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised proportion")
    fig.tight_layout(); _tiff(fig, "Figure1_pattern_confusion", a.tiff_dpi)

    # Figure 2: ROC per band class (2 panel: clone-defined / single-lane)
    bt, bp, bpr, _ = band_altkume(d, idx_all); bc = list(d["band_classes"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, sec, baslik in ((axes[0], [c for c in bc if ("KAPPA" in c or "LAMBDA" in c) and not c.startswith("FREE")],
                             "Clone-defined bands"),
                            (axes[1], [c for c in bc if "NO_LIGHT" in c or c.startswith("FREE")],
                             "Single-lane bands")):
        for c in sec:
            i = bc.index(c)
            if 0 < bt[:, i].sum() < len(bt):
                fpr, tpr, _ = roc_curve(bt[:, i], bpr[:, i])
                ax.plot(fpr, tpr, lw=1.4, label=f"{BAND_EN[c]} (AUC {_auc(bt[:, i], bpr[:, i]):.2f}, n={int(bt[:, i].sum())})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
        ax.text(-0.12, 1.04, "A" if ax is axes[0] else "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right", title=baslik, title_fontsize=8)
    fig.tight_layout(); _tiff(fig, "Figure2_band_ROC", a.tiff_dpi)

    # Figure 3: platform comparison (any-band sens/spec, macro-F1, pattern acc; GA ile)
    cih = sorted([c for c in np.unique(dev) if (dev == c).sum() > 0], key=lambda c: c != "INTERLAB")
    olc = [("herhangi_bant_duyarlilik", "Band sensitivity"), ("herhangi_bant_ozgulluk", "Band specificity"),
           ("bant_macro_f1", "Band macro-F1"), ("patern_dogruluk", "Pattern accuracy")]
    df5 = pd.DataFrame(satirlar).set_index("metrik")
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(cih)
    for k, c in enumerate(cih):
        vals = [df5.loc[f"{m}_{c}", "nokta"] if f"{m}_{c}" in df5.index else np.nan for m, _ in olc]
        err = [[df5.loc[f"{m}_{c}", "nokta"] - df5.loc[f"{m}_{c}", "ga_alt"] if f"{m}_{c}" in df5.index else 0 for m, _ in olc],
               [df5.loc[f"{m}_{c}", "ga_ust"] - df5.loc[f"{m}_{c}", "nokta"] if f"{m}_{c}" in df5.index else 0 for m, _ in olc]]
        x = np.arange(len(olc)) + (k - (len(cih) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=err, capsize=3, label=DEVICE_EN.get(c, c))
    ax.set_xticks(range(len(olc))); ax.set_xticklabels([l for _, l in olc]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value (95% CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(cih), frameon=False)
    fig.tight_layout(); _tiff(fig, "Figure3_platform", a.tiff_dpi)

    # ---------- Supheli etiket analizi (kat CSV'leri gerekli) ----------
    csvs = sorted(glob.glob(os.path.join(a.kat_dizin, "kat_*", "test_annotations.csv")))
    if a.etiket_kaynagi:
        csvs = [a.etiket_kaynagi]          # kat CSV'leri eski; supheli bantlar da master'dan
    if csvs and "filename" in d.files:
        tab = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        tab = tab.drop_duplicates("filename").set_index("filename")
        fn = pd.Series(d["filename"].astype(str))
        def _set(v): return {t for t in str(v).split(";") if t} if isinstance(v, str) else set()
        hepsi = fn.map(lambda f: _set(tab["band_target"].get(f, ""))).values
        kesin = fn.map(lambda f: _set(tab["band_target_definite"].get(f, "")) if "band_target_definite" in tab else set()).values
        supheli_var = np.array([len(h - k) > 0 for h, k in zip(hepsi, kesin)])
        sup_i = pc.index("SUSPICIOUS_BAND") if "SUSPICIOUS_BAND" in pc else -1
        var_idx = [i for i, c in enumerate(pc) if c in BANT_VAR]
        bv = d["band_valid"] if "band_valid" in d.files else np.ones(n, bool)

        def sup_metrik(_, idx):
            idx = idx[bv[idx]]
            # Gercek pozitif: kesin VEYA supheli bant (uzman). Uyari: bant basligi pozitif VEYA patern supheli/monoklonal
            yt = (d["band_true"][idx].max(1) == 1) | supheli_var[idx]
            uyari = (d["band_pred"][idx].max(1) == 1) | np.isin(d["pattern_pred"][idx], var_idx + [sup_i])
            sadece_sup = supheli_var[idx] & ~(d["band_true"][idx].max(1) == 1)
            return {"sens_definite_or_suspicious_truth": float((uyari[yt]).mean()) if yt.sum() else np.nan,
                    "spec_definite_or_suspicious_truth": float((~uyari[~yt]).mean()) if (~yt).sum() else np.nan,
                    "alert_rate_suspicious_only_panels": float(uyari[sadece_sup].mean()) if sadece_sup.sum() else np.nan,
                    "n_suspicious_only_panels": float(sadece_sup.sum()),
                    "n_positive_definite_or_suspicious": float(yt.sum())}
        nokta, ornek, _ = bootstrap(sup_metrik, d, idx_all, jeller, rng, a.tekrar)
        rows = []
        print("\nSupheli etiketli bantlar pozitif sayilinca (uzman supheli = gercek pozitif; model uyarisi = bant veya supheli/monoklonal patern):")
        for k, v in nokta.items():
            lo, hi = ga(ornek, k)
            rows.append({"metrik": k, "nokta": round(v, 3), "ga_alt": round(lo, 3), "ga_ust": round(hi, 3)})
            print(f"{k:<44}{v:>8.3f}   [{lo:.3f} - {hi:.3f}]")
        pd.DataFrame(rows).to_csv(OUT / "table_suspicious_analysis.csv", index=False, encoding="utf-8-sig")
        print(f"-> {OUT}/table_suspicious_analysis.csv")
    else:
        print("\nSupheli analiz atlandi: cv/kat_*/test_annotations.csv veya npz'de filename yok.")

    print("\nNOT: Helena'da az jel oldugu icin guven araliklari genis; makalede acikca yazilmali.")


if __name__ == "__main__":
    main()
