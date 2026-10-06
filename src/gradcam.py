# -*- coding: utf-8 -*-
"""
gradcam.py -- modelin jelde NEREYE baktigini gosterir (aciklanabilirlik)
=========================================================================
Q1 dergilerde tibbi YZ icin aciklanabilirlik artik neredeyse zorunlu.
Hakemin sorusu: "Model bandi mi ogrendi, yoksa jel kenarindaki bir
artefakti mi?" Grad-CAM buna gorsel cevap verir: modelin karara en cok
katkida bulunan bolgeleri isi haritasi olarak cizer.

NE URETIR
----------
  yayin_ciktilari/gradcam/
    dogru_IGG_KAPPA_01.png      model dogru bildi -- neye bakti?
    yanlis_IGG_KAPPA_01.png     model yanildi -- nereye bakti?
    sekil4_gradcam_paneli.png   makale icin 2x4 ozet panel

Her gorsel: sol = orijinal serit, sag = isi haritasi bindirilmis.
Isi haritasi ilgili LANE'lerde yogunlasmali (IgG-kappa icin IgG ve kappa
lane'leri). Kenarlarda/etikette yogunlasiyorsa model artefakt ogrenmis
demektir -- bunu bilmek zorunlu.

KULLANIM
---------
    python gradcam.py
    python gradcam.py --sinif IGG_KAPPA --adet 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

from ife_taxonomy import BAND_CLASSES, BAND_TR, FLAG_CLASSES, PATTERN_CLASSES, POLY_LANE_CLASSES

BAND_EN = {"IGG_KAPPA": "IgG κ", "IGG_LAMBDA": "IgG λ", "IGA_KAPPA": "IgA κ", "IGA_LAMBDA": "IgA λ",
           "IGM_KAPPA": "IgM κ", "IGM_LAMBDA": "IgM λ", "IGG_NO_LIGHT_CHAIN": "IgG heavy chain only",
           "IGA_NO_LIGHT_CHAIN": "IgA heavy chain only", "IGM_NO_LIGHT_CHAIN": "IgM heavy chain only",
           "FREE_KAPPA": "Free κ", "FREE_LAMBDA": "Free λ"}
DIL = "en"                               # --dil tr|en (sekil yazilari)
MODEL_PATH = "best_ife_multihead.pth"   # --model ile degistirilir; yoksa best_ife_*.pth aranir
COZUNURLUK = 384                         # --cozunurluk; egitimle ayni olmali
IMAGE_DIR = "Cropped_Data_EN"
OUT = "yayin_ciktilari/gradcam"
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))


class IFEMultiHead(nn.Module):
    """egitim_ve_degerlendirme.py ile BIREBIR ayni mimari."""
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        n = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
        self.backbone = base
        self.head_pattern = nn.Linear(n, len(PATTERN_CLASSES))
        self.head_bands = nn.Linear(n, len(BAND_CLASSES))
        self.head_flags = nn.Linear(n, len(FLAG_CLASSES))
        self.head_lanes = nn.Linear(n, len(POLY_LANE_CLASSES))

    def forward(self, x):
        z = self.backbone(x)
        return self.head_pattern(z), self.head_bands(z), self.head_flags(z), self.head_lanes(z)


def make_eval_tf(coz):
    return transforms.Compose([
        transforms.Resize((coz, coz)), transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


eval_tf = make_eval_tf(COZUNURLUK)


class GradCAM:
    """ConvNeXt'in son evrisim asamasina kanca takar."""
    def __init__(self, model):
        self.model = model
        self.act, self.grad = None, None
        hedef = model.backbone.features[-1]          # son asama
        hedef.register_forward_hook(self._fwd)
        hedef.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.act = o.detach()

    def _bwd(self, m, gi, go):
        self.grad = go[0].detach()

    def __call__(self, x, head: str, cls_idx: int):
        self.model.zero_grad()
        op, ob, of, ol = self.model(x)
        skor = {"pattern": op, "bands": ob, "flags": of, "lanes": ol}[head][0, cls_idx]
        skor.backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)          # kanal agirliklari
        cam = (w * self.act).sum(dim=1, keepdim=True).relu()   # [1,1,h,w]
        cam = torch.nn.functional.interpolate(cam, size=(COZUNURLUK, COZUNURLUK), mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), float(torch.sigmoid(skor).item() if head != "pattern"
                                        else torch.softmax(op, 1)[0, cls_idx].item())


def ad(cls):
    return BAND_EN[cls] if DIL == "en" else BAND_TR[cls]


def bindir(ax, img, cam):
    """Isi haritasini goruntunun uzerine, ayni en-boy oraninda gerer.
    img: orijinal oranli gri dizi (H x W); cam: COZ x COZ kare. extent ile cam
    goruntu boyutuna esleniyor; aksi halde kare cam ile dikdortgen goruntu kayar."""
    h, w = img.shape
    ax.imshow(img, cmap="gray", extent=(0, w, h, 0))
    ax.imshow(cam, cmap="jet", alpha=0.45, extent=(0, w, h, 0), interpolation="bilinear")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


def ciz(img, cam, baslik, yol):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Strip" if DIL == "en" else "Serit"); ax[0].axis("off")
    bindir(ax[1], img, cam)
    ax[1].set_title("Grad-CAM"); ax[1].axis("off")
    fig.suptitle(baslik, fontsize=10)
    fig.tight_layout(); fig.savefig(yol, dpi=200); plt.close(fig)


def elp_maskele(img, oran, bolge="elp"):
    """bolge='elp': sol %oran seridi (ELP/SPE lane) zemin rengine boyar.
    bolge='alt': ayni oranda ALT seridi (lane etiketleri, bant yok) boyar -- kontrol:
    ayni buyuklukte bir 'yabanci' dikdortgen tek basina performansi dusuruyor mu?"""
    g = np.array(img.convert("L"))
    zemin = int(np.median(g))
    out = img.copy()
    from PIL import ImageDraw
    if bolge == "alt":
        h = max(1, int(round(img.height * oran)))
        ImageDraw.Draw(out).rectangle([0, img.height - h, img.width, img.height], fill=(zemin, zemin, zemin))
    else:
        w = max(1, int(round(img.width * oran)))
        ImageDraw.Draw(out).rectangle([0, 0, w, img.height], fill=(zemin, zemin, zemin))
    return out


def ablasyon_elp(a):
    """ELP lane maskelenince bant saptama degisiyor mu? 5 kat modeli, kendi test
    setleri, kendi esikleri; sadece cikarim. Cikti: yayin_ciktilari/table_elp_ablation.csv"""
    from sklearn.metrics import f1_score, roc_auc_score
    katlar = sorted(Path("cv").glob("kat_*"))
    if not katlar:
        raise SystemExit("cv/kat_* yok; once capraz_dogrulama.py.")
    Path("yayin_ciktilari").mkdir(exist_ok=True)
    kosullar = [("intact", 0.0, "elp"), ("ELP masked", a.maske_oran, "elp"),
                ("control (bottom masked)", a.maske_oran, "alt")]
    havuz = {k: {"bt": [], "bp": [], "bpr": [], "pt": [], "pp": [], "sf": [], "sfb": []} for k, _, _ in kosullar}
    satirlar = []
    ornek_yazildi = set()
    for kat in katlar:
        pth = sorted(kat.glob("best_ife_*.pth")); npz = sorted(kat.glob("yayin_ciktilari*/test_tahminleri.npz"))
        csv = kat / "test_annotations.csv"
        if not (pth and npz and csv.exists()):
            print(f"{kat}: eksik dosya, atlandi"); continue
        thr = np.load(npz[0], allow_pickle=True)["band_thr"]
        model = IFEMultiHead(); model.load_state_dict(torch.load(pth[0], map_location="cpu")); model.to(DEVICE).eval()
        df = pd.read_csv(csv)
        if a.etiket_kaynagi:                      # guncel etiketler + eksik paneller: katin jellerini master'dan al
            m = pd.read_csv(a.etiket_kaynagi)
            if "sample_type" in m.columns:
                m = m[m["sample_type"].astype(str).str.upper() == "SERUM"]
            df = m[m["source_file"].astype(str).isin(df["source_file"].astype(str).unique())].copy()
        df["img_path"] = df["img_relpath"].apply(lambda q: os.path.join(IMAGE_DIR, str(q)))
        df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
        gecerli = df["pattern"].astype(str) != "UNINTERPRETABLE_TECHNICAL_ISSUE"     # STARD
        bt = np.zeros((len(df), len(BAND_CLASSES)), np.float32)
        for i, v in enumerate(df.get("band_target_definite", pd.Series([""] * len(df))).fillna("")):
            for t in str(v).split(";"):
                if t in BAND_CLASSES: bt[i, BAND_CLASSES.index(t)] = 1
        pt = df["pattern"].apply(lambda q: PATTERN_CLASSES.index(q) if q in PATTERN_CLASSES else 0).values
        for kosul, oran, bolge in kosullar:
            bpr, ppr = [], []
            with torch.no_grad():
                for b0 in range(0, len(df), 16):
                    xs = []
                    for _, r in df.iloc[b0:b0 + 16].iterrows():
                        img = Image.open(r["img_path"]).convert("RGB")
                        if oran > 0:
                            img = elp_maskele(img, oran, bolge)
                            anahtar = (bolge, str(r.get("device", "")))
                            if anahtar not in ornek_yazildi:      # her cihaz x bolge icin bir ornek
                                img.save(f"yayin_ciktilari/maske_ornek_{bolge}_{r.get('device', 'x')}.png")
                                ornek_yazildi.add(anahtar)
                        xs.append(eval_tf(img))
                    op, ob, _, _ = model(torch.stack(xs).to(DEVICE))
                    bpr.append(torch.sigmoid(ob).cpu().numpy()); ppr.append(torch.softmax(op, 1).cpu().numpy())
            bpr = np.vstack(bpr); pp = np.vstack(ppr).argmax(1); bp = (bpr >= thr).astype(int)
            h = havuz[kosul]
            h["bt"].append(bt[gecerli]); h["bp"].append(bp[gecerli]); h["bpr"].append(bpr[gecerli])
            h["pt"].append(pt); h["pp"].append(pp)
            sfv = df["source_file"].astype(str).values if "source_file" in df else np.arange(len(df)).astype(str)
            h["sf"].append(sfv); h["sfb"].append(sfv[gecerli.values])
            satirlar.append(dict(fold=kat.name, condition=kosul, **_ablasyon_metrik(bt[gecerli], bp[gecerli], bpr[gecerli], pt, pp)))
            print(f"{kat.name:6s} {kosul:24s} " + "  ".join(f"{k}={v:.3f}" for k, v in satirlar[-1].items() if isinstance(v, float)))
    for kosul, _, _ in kosullar:
        h = havuz[kosul]
        m = _ablasyon_metrik(np.vstack(h["bt"]), np.vstack(h["bp"]), np.vstack(h["bpr"]),
                             np.concatenate(h["pt"]), np.concatenate(h["pp"]))
        satirlar.append(dict(fold="POOLED", condition=kosul, **m))
        print(f"POOLED {kosul:24s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    pd.DataFrame(satirlar).to_csv("yayin_ciktilari/table_elp_ablation.csv", index=False, encoding="utf-8-sig")
    # Panel bazli tahminleri sakla (eslestirilmis test icin) ve istatistik
    H = {k: {kk: (np.vstack(v) if v and v[0].ndim == 2 else np.concatenate(v)) for kk, v in h.items()}
         for k, h in havuz.items()}
    np.savez("yayin_ciktilari/elp_ablation_tahminler.npz",
             **{f"{k.replace(' ', '_')}__{kk}": v for k, h in H.items() for kk, v in h.items()})
    _ablasyon_istatistik(H, kosullar)
    print("-> yayin_ciktilari/table_elp_ablation.csv\n-> yayin_ciktilari/maske_ornek_*.png (maskenin ELP'yi kapattigini KONTROL EDIN)")


def _ablasyon_istatistik(H, kosullar, B=2000, seed=42):
    """Eslestirilmis karsilastirma: McNemar (kesin binom) + jel-bazli kume bootstrap fark GA'si."""
    from scipy.stats import binomtest
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    ref = "intact"
    # sf: patern icin tum kayitlar; bant icin gecerli kayitlar -> ayni sirada olduklarindan
    # bant satirlarina denk gelen source_file'i yeniden turetmek icin gecerli maskesini sakliyoruz;
    # basitlik icin her iki bootstrap da 'jel' kumesini bant-gecerli kayit sayisina gore olusturur.
    rows = []
    for kosul, _, _ in kosullar:
        if kosul == ref: continue
        for karsi in ([ref] + (["control (bottom masked)"] if kosul.startswith("ELP") else [])):
            A, Bk = H[karsi], H[kosul]
            # --- herhangi bant duyarliligi: McNemar ---
            yt = A["bt"].max(1) == 1
            a_hit = (A["bp"].max(1) == 1)[yt]; b_hit = (Bk["bp"].max(1) == 1)[yt]
            n01 = int((a_hit & ~b_hit).sum()); n10 = int((~a_hit & b_hit).sum())
            p_sens = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            # --- patern dogrulugu: McNemar ---
            a_ok = A["pt"] == A["pp"]; b_ok = Bk["pt"] == Bk["pp"]
            m01 = int((a_ok & ~b_ok).sum()); m10 = int((~a_ok & b_ok).sum())
            p_pat = binomtest(min(m01, m10), m01 + m10, 0.5).pvalue if m01 + m10 else 1.0
            # --- kume bootstrap (jel) fark GA ---
            sf_all, sf_b = A["sf"], A["sfb"]; jel = np.unique(sf_all)
            idx_p = {j: np.where(sf_all == j)[0] for j in jel}
            idx_b = {j: np.where(sf_b == j)[0] for j in jel}
            d_sens, d_f1, d_pat = [], [], []
            for _ in range(B):
                pick = rng.choice(jel, len(jel), replace=True)
                ip = np.concatenate([idx_p[j] for j in pick])
                ib = np.concatenate([idx_b[j] for j in pick])
                ta, tb_ = A["bt"][ib], A["bp"][ib]; tbb = Bk["bp"][ib]
                pos = ta.max(1) == 1
                if pos.sum():
                    d_sens.append((tbb.max(1)[pos] == 1).mean() - (tb_.max(1)[pos] == 1).mean())
                d_f1.append(f1_score(ta, tbb, average="macro", zero_division=0) -
                            f1_score(ta, tb_, average="macro", zero_division=0))
                d_pat.append((Bk["pt"][ip] == Bk["pp"][ip]).mean() - (A["pt"][ip] == A["pp"][ip]).mean())
            ci = lambda v: tuple(np.round(np.percentile(v, [2.5, 97.5]), 3)) if len(v) else (np.nan, np.nan)
            rows.append(dict(condition=kosul, versus=karsi,
                             d_any_band_sens=round(b_hit.mean() - a_hit.mean(), 3), d_sens_ci=ci(d_sens),
                             mcnemar_sens_discordant=f"{n01}/{n10}", mcnemar_sens_p=p_sens,
                             d_band_macro_f1=round(float(np.mean(d_f1)), 3), d_f1_ci=ci(d_f1),
                             d_pattern_acc=round(b_ok.mean() - a_ok.mean(), 3), d_pat_ci=ci(d_pat),
                             mcnemar_pat_discordant=f"{m01}/{m10}", mcnemar_pat_p=p_pat))
            r = rows[-1]
            print(f"\n{kosul} vs {karsi}:")
            print(f"  any-band sens  diff {r['d_any_band_sens']:+.3f}  95% CI {r['d_sens_ci']}  "
                  f"McNemar discordant {r['mcnemar_sens_discordant']}  p={p_sens:.2e}")
            print(f"  band macro-F1  diff {r['d_band_macro_f1']:+.3f}  95% CI {r['d_f1_ci']}")
            print(f"  pattern acc    diff {r['d_pattern_acc']:+.3f}  95% CI {r['d_pat_ci']}  "
                  f"McNemar discordant {r['mcnemar_pat_discordant']}  p={p_pat:.2e}")
    pd.DataFrame(rows).to_csv("yayin_ciktilari/table_elp_ablation_ci.csv", index=False, encoding="utf-8-sig")
    print("-> yayin_ciktilari/table_elp_ablation_ci.csv")


def _ablasyon_metrik(bt, bp, bpr, pt, pp):
    from sklearn.metrics import f1_score, roc_auc_score
    yt, yp = bt.max(1), bp.max(1)
    tp = ((yt == 1) & (yp == 1)).sum(); fn = ((yt == 1) & (yp == 0)).sum()
    tn = ((yt == 0) & (yp == 0)).sum(); fp = ((yt == 0) & (yp == 1)).sum()
    tek = [i for i, c in enumerate(BAND_CLASSES) if "NO_LIGHT" in c or c.startswith("FREE")]
    yt1, yp1 = bt[:, tek].max(1), bp[:, tek].max(1)
    return {"any_band_sens": float(tp / max(tp + fn, 1)), "any_band_spec": float(tn / max(tn + fp, 1)),
            "any_band_auc": float(roc_auc_score(yt, bpr.max(1))) if 0 < yt.sum() < len(yt) else np.nan,
            "single_lane_sens": float(((yt1 == 1) & (yp1 == 1)).sum() / max(yt1.sum(), 1)),
            "band_macro_f1": float(f1_score(bt, bp, average="macro", zero_division=0)),
            "pattern_acc": float((pt == pp).mean())}


def main():
    global MODEL_PATH, COZUNURLUK, eval_tf, DIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="test_annotations.csv")
    ap.add_argument("--sinif", default=None, help="Sadece bu bant sinifi (or. IGG_KAPPA)")
    ap.add_argument("--adet", type=int, default=4, help="Sinif basina dogru + yanlis ornek")
    ap.add_argument("--model", default=None, help="Agirlik dosyasi (.pth); verilmezse best_ife_*.pth aranir")
    ap.add_argument("--esik", default="thresholds.npz", help="Bant esikleri (egitimin yazdigi npz)")
    ap.add_argument("--cozunurluk", type=int, default=COZUNURLUK, help="Egitimdeki cozunurluk (384)")
    ap.add_argument("--dil", default="en", choices=["tr", "en"], help="Sekil yazilari")
    ap.add_argument("--tiff-dpi", type=int, default=1000, help="Yayin TIFF cozunurlugu")
    ap.add_argument("--ablasyon-elp", action="store_true", help="ELP lane maskeleme ablasyonu (5 kat, sadece cikarim)")
    ap.add_argument("--maske-oran", type=float, default=0.17, help="ELP maskesi: genisligin orani (sol serit)")
    ap.add_argument("--etiket-kaynagi", default=None, help="Ablasyonda gercek etiketleri annotations_master.csv'den al")
    ap.add_argument("--ornek", action="append", default=[],
                    help="Panel hucresini elle sec: SINIF:CIHAZ=dosyaadi (or. IGA_KAPPA:HELENA=2017_File48__....png). Tekrarlanabilir.")
    a = ap.parse_args()

    if a.ablasyon_elp:
        COZUNURLUK = a.cozunurluk; eval_tf = make_eval_tf(COZUNURLUK)
        return ablasyon_elp(a)
    if a.model:
        MODEL_PATH = a.model
    elif not Path(MODEL_PATH).exists():
        adaylar = sorted(Path(".").glob("best_ife_*.pth")) or sorted(Path("cv").glob("kat_*/best_ife_*.pth"))
        if not adaylar:
            raise SystemExit("Agirlik dosyasi bulunamadi (best_ife_*.pth ne kokte ne cv/kat_*/ altinda). "
                             "--model ile yol verin.")
        MODEL_PATH = str(adaylar[0])
    COZUNURLUK = a.cozunurluk
    DIL = a.dil
    eval_tf = make_eval_tf(COZUNURLUK)
    if a.test == "test_annotations.csv":                       # --test verilmediyse
        kat_dir = Path(MODEL_PATH).parent
        if kat_dir.name.startswith("kat_") and (kat_dir / "test_annotations.csv").exists():
            a.test = str(kat_dir / "test_annotations.csv")     # sizinti olmasin: katin kendi test seti
    print(f"Model: {MODEL_PATH} | cozunurluk {COZUNURLUK} | test {a.test} | esik {a.esik}")
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"{MODEL_PATH} yok.")
    Path(OUT).mkdir(parents=True, exist_ok=True)

    model = IFEMultiHead()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(DEVICE).eval()
    cam_fn = GradCAM(model)

    thr_path = a.esik
    if a.esik == "thresholds.npz":                            # varsayilan: katin kendi esikleri
        kat_dir = Path(MODEL_PATH).parent
        adaylar = sorted(kat_dir.glob("yayin_ciktilari*/test_tahminleri.npz"))
        if kat_dir.name.startswith("kat_") and adaylar:
            thr_path = str(adaylar[0])
    if Path(thr_path).exists():
        _z = np.load(thr_path, allow_pickle=True)
        thr = _z["band"] if "band" in _z.files else _z["band_thr"]
        print(f"Esikler: {thr_path}")
    else:
        thr = np.full(len(BAND_CLASSES), 0.5); print("Esik dosyasi yok, 0.5 kullaniliyor")

    df = pd.read_csv(a.test)
    df["img_path"] = df["img_relpath"].apply(lambda p: os.path.join(IMAGE_DIR, str(p)))
    df = df[df["img_path"].apply(os.path.exists)]
    df["bant_set"] = df["band_target"].fillna("").apply(lambda v: {t for t in str(v).split(";") if t})
    if "device" not in df.columns:
        df["device"] = df["img_relpath"].astype(str).str.lower().map(lambda v: "HELENA" if "helena" in v else "INTERLAB")
    df["device"] = df["device"].astype(str).str.upper()
    CIHAZLAR = ["INTERLAB", "HELENA"]

    siniflar = [a.sinif] if a.sinif else ["IGG_KAPPA", "IGG_LAMBDA", "IGA_KAPPA", "FREE_LAMBDA"]
    elle = {}                                            # (sinif, cihaz) -> dosyaadi
    for o in a.ornek:
        anahtar, _, dosya = o.partition("=")
        cls_, _, cihaz_ = anahtar.partition(":")
        elle[(cls_.upper(), cihaz_.upper())] = dosya
    panel, panel_yanlis = [], []
    for cls in siniflar:
        ci = BAND_CLASSES.index(cls)
        pozitif = df[df["bant_set"].apply(lambda s: cls in s)]
        dogru, yanlis = [], []
        ilk_dogru, ilk_yanlis = {}, {}                   # cihaz -> panel ornegi (dogru / kacirilan)
        for _, r in pozitif.iterrows():
            img = Image.open(r["img_path"]).convert("RGB")
            x = eval_tf(img).unsqueeze(0).to(DEVICE).requires_grad_(False)
            cam, p = cam_fn(x, "bands", ci)
            gw = COZUNURLUK; gh = max(1, round(img.height * gw / img.width))   # orijinal en-boy orani
            gri = np.array(img.convert("L").resize((gw, gh)))
            kayit = (gri, cam, p, r["filename"])
            secili = elle.get((cls, r["device"]))
            if p >= thr[ci]:
                dogru.append(kayit)
                # Panel icin: elle secilmisse o, degilse en yuksek olasilikli
                if secili == r["filename"] or (secili is None and
                        (r["device"] not in ilk_dogru or p > ilk_dogru[r["device"]][2])):
                    ilk_dogru[r["device"]] = kayit
            else:
                yanlis.append(kayit); ilk_yanlis.setdefault(r["device"], kayit)
            if len(dogru) >= a.adet and len(yanlis) >= a.adet and all(c in ilk_dogru for c in CIHAZLAR) \
                    and not any(k[0] == cls for k in elle):
                break
        for etiket, liste in (("dogru", dogru), ("yanlis", yanlis)):
            for i, (gri, cam, p, fn) in enumerate(liste[:a.adet], 1):
                yol = f"{OUT}/{etiket}_{cls}_{i:02d}.png"
                ciz(gri, cam, f"{ad(cls)}  |  p = {p:.2f}  |  {etiket}\n{fn}", yol)
        if dogru:
            panel.append((cls, ilk_dogru)); panel_yanlis.append((cls, ilk_yanlis))
        print(f"{ad(cls):<40} dogru {len(dogru[:a.adet])}  yanlis {len(yanlis[:a.adet])}")
        for cihaz, (_, _, p, fn) in ilk_dogru.items():
            print(f"    panel {cihaz:<9} p={p:.2f}  {fn}")
        for _, _, p, fn in dogru[:a.adet]:
            print(f"    aday  p={p:.2f}  {fn}")

    def panel_ciz(veri, png, tiff):
        """satir = cihaz (Interlab, Helena), sutun = bant sinifi, hucre = Grad-CAM bindirmesi."""
        cihaz_ad = {"INTERLAB": "Interlab", "HELENA": "Helena"}
        fig, axes = plt.subplots(len(CIHAZLAR), len(veri),
                                 figsize=(3.2 * len(veri), 2.6 * len(CIHAZLAR)))
        axes = np.array(axes).reshape(len(CIHAZLAR), len(veri))
        for i, cihaz in enumerate(CIHAZLAR):
            for j, (cls, ornekler) in enumerate(veri):
                ax = axes[i, j]; ax.axis("off")
                if cihaz not in ornekler:
                    ax.set_title(f"{ad(cls)} — no example", fontsize=9); continue
                gri, cam, p, fn = ornekler[cihaz]
                bindir(ax, gri, cam)
                ax.set_title(f"{ad(cls)}  (p = {p:.2f})", fontsize=9)
            axes[i, 0].text(-0.04, 0.5, cihaz_ad.get(cihaz, cihaz), transform=axes[i, 0].transAxes,
                            rotation=90, va="center", ha="right", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(png, dpi=300)                                              # hizli bakis
        fig.savefig(tiff, dpi=a.tiff_dpi, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
        print(f"-> {png}\n-> {tiff} ({a.tiff_dpi} dpi, LZW)")

    # Makale paneli (dogru tahminler)
    if panel:
        panel_ciz(panel, "yayin_ciktilari/sekil4_gradcam_paneli.png", "yayin_ciktilari/Figure4_GradCAM.tiff")
    # Ek materyal: kacirilan bantlar (model esigin altinda kaldi) — hata modu gosterimi
    if any(v for _, v in panel_yanlis):
        panel_ciz(panel_yanlis, "yayin_ciktilari/ek_sekil_gradcam_kacirilan.png",
                  "yayin_ciktilari/SupplFigure_GradCAM_missed.tiff")
        print("\n-> yayin_ciktilari/sekil4_gradcam_paneli.png")
    print(f"-> {OUT}/")
    print("\nKONTROL: isi haritasi ilgili LANE'lerde mi yogunlasiyor (IgG-kappa icin")
    print("IgG ve kappa seritleri)? Kenarda/etikette yogunlasiyorsa model artefakt")
    print("ogrenmis demektir; bunu tartisma bolumunde belirtin.")


if __name__ == "__main__":
    main()
