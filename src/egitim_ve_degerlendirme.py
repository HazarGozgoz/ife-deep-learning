# -*- coding: utf-8 -*-
"""
egitim_ve_degerlendirme.py -- ÇOK BAŞLI (multi-head) IFE modeli
=================================================================
Taksonomi üç eksene ayrıldığı için model de dört ayrı başlık öğrenir:

  A) pattern     : 7 sınıf, TEK etiket   (softmax + CrossEntropy)
  B) bands       : 11 sınıf, ÇOK etiket  (sigmoid + BCE, sınıf ağırlıklı)
  C) flags       : 2 sınıf, ÇOK etiket   (hipogama, poliklonal zemin)
  D) poly_lanes  : 5 sınıf, ÇOK etiket   (poliklonal artışın lane'leri)

Neden tek düz 21 sınıflı çıktı değil: hipogamaglobulinemi 848 kaydın
424'ünde BİR BANTLA BİRLİKTE (immünoparezi + M-protein). Bunlar birbirini
dışlamayan ayrı eksenlerdir; tek softmax'a sıkıştırmak modele yanlış bir
"ya o ya bu" varsayımı dayatır.

KULLANICI KARARLARI (bu çalışmaya özgü)
  * Şüpheli bantlar, bant hedefinde kesin bantlarla AYNI sayılır
    (duyarlılık öncelikli). 'band_target_definite' kolonu sayesinde
    "sadece kesin bantlar" duyarlılık analizi de üretilir.
  * Poliklonal lane bilgisi AYRI bir çıktı ekseni olarak tahmin edilir.

Korunanlar: ConvNeXt-Tiny, katmanlı öğrenme oranı + ısınma, cosine
annealing warm restarts, gradyan kırpma, PR-eğrisi tabanlı eşik
optimizasyonu (kritik bantlarda ≥%70 duyarlılık tabanı), cihaz
genellemesi için gri tonlama + ağır fotometrik augmentasyon, YZ
çift-kontrol etiket hatası avcısı, yayın tablo/şekilleri.
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (auc, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights

from ife_taxonomy import (BAND_CLASSES, BAND_EN, BAND_EN_SHORT, BAND_TR, FLAG_CLASSES,
                          FLAG_EN, FLAG_TR, PATTERN_EN, band_label,
                          PATTERN_CLASSES, PATTERN_TR, POLY_LANE_CLASSES, POLY_LANE_TR)

# --- KONFİGÜRASYON ---
TRAIN_CSV, VAL_CSV, TEST_CSV = "train_annotations.csv", "validation_annotations.csv", "test_annotations.csv"
IMAGE_DIR = "Cropped_Data_EN"
FIG_DIR = "yayin_ciktilari"
ERROR_REPORT = "ai_suspected_errors.csv"
MODEL_PATH = "best_ife_multihead.pth"   # asagida deney etiketiyle guncellenir

BATCH_SIZE, EPOCHS, PATIENCE, NUM_WORKERS, SEED = 32, 30, 7, 0, 42
# 384 px'te batch 32 MacBook belleğine sığmıyor, diske takas yapıp 17x
# yavaşlıyordu (epoch 2317 s). Cozunurluk buyudukce batch otomatik kuculur;
# --batch=N ile elle de verilebilir.
WARMUP_EPOCHS = 2
HEAD_LR, BACKBONE_LR = 1e-4, 3e-5
T0, TMULT, ETA_MIN = 10, 1, 1e-6
# Başlıkların kayıp ağırlıkları. Bant başlığı klinik olarak en kritik olan.
LOSS_W = {"pattern": 1.0, "bands": 1.5, "flags": 0.5, "lanes": 0.5}
GRAYSCALE_DEVICE_INVARIANCE = True
# Hata avcisi marji: model dusuk esikle cok isaretledigi icin 0.10 ile
# binlerce sahte "celiski" cikiyordu. Marj yukseltildi; sadece modelin
# GERCEKTEN emin olup insanla celistigi kayitlar listelenir.
MARGIN_CRIT, MARGIN_NONCRIT = 0.35, 0.45

N_PAT, N_BAND, N_FLAG, N_LANE = (len(PATTERN_CLASSES), len(BAND_CLASSES),
                                 len(FLAG_CLASSES), len(POLY_LANE_CLASSES))


# --sadece-degerlendir : egitimi atlar, kaydedilmis modeli yukleyip
# esikleri, test metriklerini, yayin ciktilarini ve hata avcisini yeniden
# uretir. Egitim tamamlandiktan sonra sadece raporlama kismini tekrar
# calistirmak icin (saatlerce yeniden egitmeye gerek kalmaz).
SKIP_TRAIN = "--sadece-degerlendir" in sys.argv

# ESIK MODU -- ayni model, farkli calisma noktasi. Yeniden egitim GEREKMEZ.
#   tarama : yuksek duyarlilik (>=%70 recall tabani). Kacirmamak oncelikli.
#            Precision dusuk cikar (~0.2): model cok isaretler.
#   dengeli: precision ve recall dengede (max F1). Yayinda ikisini de
#            "iki calisma noktasi" olarak verin; hakem ikisini de sorar.
ESIK_MODU = "dengeli" if "--esik-modu=dengeli" in sys.argv else "tarama"


def _arg(ad, varsayilan, tip=str):
    for a_ in sys.argv:
        if a_.startswith(f"--{ad}="):
            return tip(a_.split("=", 1)[1])
    return varsayilan


# ---- DENEY AYARLARI (metrikleri dusuren uc supheliyi hedefler) ----
# 1) Bant hedefi: "kesin" -> supheli bantlar kayiptan MASKELENIR (ne pozitif
#    ne negatif); model sadece kesin bantlardan ogrenir. "hepsi" -> eski
#    davranis (supheli = pozitif; egitim sinyalini kirletiyor).
BANT_HEDEFI = _arg("bant-hedefi", "kesin")
# 2) Giris cozunurlugu. 224'te lane ~37 px, ince bant kaybolur. 384 onerilir.
COZUNURLUK = _arg("cozunurluk", 384, int)
# 3) Sinif agirligi tavani. 50 nadir siniflari asiri itiyordu.
POS_WEIGHT_MAX = _arg("pos-weight-max", 10.0, float)
# 4) Model secimi: recall@0.25 terimi cok-isaretleyen modeli oduullendiriyordu.
#    "f1" -> (patern-acc + bant-macroF1)/2 ; "eski" -> onceki hibrit
SECIM_OLCUTU = _arg("secim", "f1")
# 5) Cikti dili (sekil/tablo basliklari): tr | en
DIL = _arg("dil", "en")
_b = _arg("batch", None, int)
if _b:
    BATCH_SIZE = _b
elif COZUNURLUK >= 384:
    BATCH_SIZE = 8
elif COZUNURLUK >= 320:
    BATCH_SIZE = 12
CIKTI_ETIKETI = f"{BANT_HEDEFI}_{COZUNURLUK}px_pw{int(POS_WEIGHT_MAX)}"
# --etiket=<ad> : model dosyasi ve cikti klasoru adina eklenir; ayni ayarla
# ikinci bir deney (orn. helena_holdout) onceki ciktilari EZMESIN diye.
_ETIKET_EK = _arg("etiket", "").strip()
if _ETIKET_EK:
    CIKTI_ETIKETI += f"_{_ETIKET_EK}"
# Sekil/tablo etiketleri: yayin Ingilizce ise DIL=en
L = {
    "tr": dict(cm_title="Patern ekseni — test seti", cm_x="Model", cm_y="Uzman referansı",
               roc_title="Bant ekseni — ROC (n≥10)", roc_x="1 - Özgüllük", roc_y="Duyarlılık",
               dev_title="Cihazlar arası performans", dev_y="Bant macro-F1",
               band="Bant", pattern="Patern", n="Test n", thr="Eşik", prec="Precision",
               rec="Recall", f1="F1", auc="ROC-AUC", prauc="PR-AUC", note="Not"),
    "en": dict(cm_title="Pattern axis — test set", cm_x="Model prediction", cm_y="Expert reference",
               roc_title="Band axis — ROC (n ≥ 10)", roc_x="1 − Specificity", roc_y="Sensitivity",
               dev_title="Cross-platform performance", dev_y="Band macro-F1",
               band="Band", pattern="Pattern", n="Test n", thr="Threshold", prec="Precision",
               rec="Recall", f1="F1", auc="ROC-AUC", prauc="PR-AUC", note="Note"),
}[DIL]
PAT_L = PATTERN_EN if DIL == "en" else PATTERN_TR
BAND_L = BAND_EN_SHORT if DIL == "en" else BAND_TR
FLAG_L = FLAG_EN if DIL == "en" else FLAG_TR
LANE_L = ({"IGG": "IgG", "IGA": "IgA", "IGM": "IgM", "KAPPA": "κ", "LAMBDA": "λ"}
          if DIL == "en" else POLY_LANE_TR)
if "--sadece-degerlendir" not in sys.argv:
    print(f"DENEY: bant_hedefi={BANT_HEDEFI} cozunurluk={COZUNURLUK} batch={BATCH_SIZE} "
          f"pos_weight_max={POS_WEIGHT_MAX} secim={SECIM_OLCUTU} dil={DIL}")
MODEL_PATH = f"best_ife_{CIKTI_ETIKETI}.pth"
FIG_DIR = f"{FIG_DIR}_{CIKTI_ETIKETI}" + ("_dengeli" if ESIK_MODU == "dengeli" else "")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


set_seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Aktif hızlandırıcı: {DEVICE}")

# ===========================================================================
# 1) VERİ
# ===========================================================================
for f in (TRAIN_CSV, VAL_CSV, TEST_CSV):
    if not Path(f).exists():
        raise FileNotFoundError(f"'{f}' yok. Önce 'python veri_hazirlama.py' çalıştırın.")


def _split_set(v, sep):
    """
    DİKKAT: pandas boş hücreleri NaN (float) olarak verir ve float('nan')
    Python'da DOĞRU sayılır. Bu yüzden `str(v or "")` ifadesi NaN için
    "nan" üretir ve her boş alan hayalet bir 'nan' etiketi doğurur.
    Tek-sıcak vektörlerde zararsız görünür ('nan' sınıf listesinde yok)
    ama "bu alan boş mu?" sorusunu sessizce bozar -- lane maskesi tam
    bu yüzden hiçbir kaydı maskelemiyordu.
    """
    if v is None or (isinstance(v, float) and v != v):
        return set()
    return {t for t in str(v).split(sep) if t and t.lower() != "nan"}


def prep(path, name):
    d = pd.read_csv(path)
    d["Split"] = name
    if "sample_type" in d.columns:
        d = d[d["sample_type"].astype(str).str.upper() == "SERUM"].copy()
    if "device" not in d.columns:
        d["device"] = "BILINMIYOR"
    d["img_path"] = d["img_relpath"].apply(lambda p: os.path.join(IMAGE_DIR, str(p)))
    miss = ~d["img_path"].apply(os.path.exists)
    if miss.any():
        print(f"  DİKKAT: {name} — {int(miss.sum())} görüntü diskte yok, çıkarıldı.")
    d = d[~miss].copy()
    d["y_pattern"] = d["pattern"].apply(
        lambda p: PATTERN_CLASSES.index(p) if p in PATTERN_CLASSES else 0)
    d["y_bands_all"] = d["band_target"].apply(lambda v: _split_set(v, ";"))
    d["y_bands_def"] = d.get("band_target_definite", "").apply(lambda v: _split_set(v, ";"))
    # Egitim hedefi: "kesin" modda kesin bantlar pozitif, supheli bantlar
    # MASKELENIR (asagida band_mask). Degerlendirme her iki kumeyle yapilir.
    d["y_bands"] = d["y_bands_def"] if BANT_HEDEFI == "kesin" else d["y_bands_all"]
    d["y_bands_susp"] = [a - b for a, b in zip(d["y_bands_all"], d["y_bands_def"])]
    # Yalnızca KLİNİK bayraklar model hedefidir. Kayıt işaretleri
    # (UNTYPED_SUSPICIOUS_BAND, BAND_COUNT_INCOMPLETE) ve paterni belirleyen
    # seçimler (ATYPICAL, UNINTERPRETABLE -- patern başlığı zaten tahmin
    # ediyor) hedefe SOKULMAZ.
    d["y_flags"] = d["flags"].apply(
        lambda v: _split_set(v, "|") & set(FLAG_CLASSES))
    d["y_lanes"] = d["poly_lanes"].apply(lambda v: _split_set(v, "|"))

    # LANE MASKESİ -- kısmi etiketli veri için.
    # Poliklonal işaretli ama hangi lane olduğu YAZILMAMIŞ kayıtlarda lane
    # etiketi "hepsi sıfır" değil, BİLİNMİYOR'dur. Maskelenmezse model o
    # kayıtlardan beş kez yanlış negatif öğrenir (poliklonal kayıtların
    # ~%21'i bu durumda). Maske, bu örnekleri lane kaybından dışlar;
    # patern ve bant başlıklarına katkıları aynen sürer.
    # SADECE hipergammaglobulinemi lane bekler. POLYCLONAL_BACKGROUND
    # (poliklonal zemin) lane gerektirmez -- onu da yakalayan genis bir
    # arama 76 kaydi bosuna maskeliyordu.
    _poly = d["flags"].fillna("").astype(str).str.contains("POLYCLONAL_INCREASE")
    _has_lane = d["y_lanes"].apply(bool)
    d["lane_known"] = (~_poly | _has_lane).astype("float32")  # MPS icin float32
    n_unknown = int((d["lane_known"] == 0).sum())
    if n_unknown:
        print(f"  {name}: {n_unknown} kayıtta poliklonal var ama lane bilinmiyor "
              f"-> lane kaybından dışlandı (patern/bant eğitimi etkilenmiyor).")
    return d.reset_index(drop=True)


train_df, val_df, test_df = (prep(TRAIN_CSV, "Train"), prep(VAL_CSV, "Val"), prep(TEST_CSV, "Test"))
df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")


def onehot(sets, classes):
    m = np.zeros((len(sets), len(classes)), dtype=np.float32)
    for i, s in enumerate(sets):
        for j, c in enumerate(classes):
            if c in s:
                m[i, j] = 1.0
    return m


# ===========================================================================
# 2) DATASET
# ===========================================================================
class IFEDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.t = transform
        self.yp = self.df["y_pattern"].to_numpy()
        self.yb = onehot(self.df["y_bands"], BAND_CLASSES)
        # 1 = bu sinif icin etiket guvenilir; 0 = supheli, kayiptan disla
        self.ybm = 1.0 - onehot(self.df["y_bands_susp"], BAND_CLASSES) \
            if BANT_HEDEFI == "kesin" else np.ones_like(self.yb)
        # YORUMLANAMAYAN jellerde bant etiketi anlamsizdir (uzman "bant var
        # mi" diyememis). Bant kaybindan TAMAMEN maskelenir; patern kaybi
        # kalir -- model bozuk jeli bayraklamayi ogrenmeli.
        _uninterp = (self.df["pattern"].astype(str) == "UNINTERPRETABLE_TECHNICAL_ISSUE").to_numpy()
        self.ybm[_uninterp] = 0.0
        self.ybm = self.ybm.astype(np.float32)
        self.yf = onehot(self.df["y_flags"], FLAG_CLASSES)
        self.yl = onehot(self.df["y_lanes"], POLY_LANE_CLASSES)
        self.lm = self.df["lane_known"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["img_path"]).convert("RGB")
        return (self.t(img), int(self.yp[i]),
                torch.from_numpy(self.yb[i]), torch.from_numpy(self.yf[i]),
                torch.from_numpy(self.ybm[i]),
                # MPS float64 desteklemez; Python float'i DataLoader float64
                # tensore cevirir ve .to("mps") patlar. Bastan float32 veriyoruz.
                torch.from_numpy(self.yl[i]),
                torch.tensor(self.lm[i], dtype=torch.float32),
                str(r["filename"]), str(r["img_path"]),
                str(r["Split"]), str(r["device"]))


# Cihaz genellemesi: Helena ile Interlab arasındaki en büyük fark bant deseni
# değil, BOYAMA RENGİ ve kontrasttır. Renk kanalını atmak ve parlaklık/kontrastı
# geniş aralıkta oynatmak, modelin "hangi cihaz" yerine "hangi bant" sorusunu
# çözmesini zorlar. Geometrik augmentasyon bilerek HAFİF: IFE'de bandın dikey
# konumu tanısal bilginin kendisidir.
_gray = [transforms.Grayscale(num_output_channels=3)] if GRAYSCALE_DEVICE_INVARIANCE else []
train_tf = transforms.Compose(
    [transforms.Resize((COZUNURLUK + 12, COZUNURLUK + 12)), transforms.RandomCrop(COZUNURLUK),
     transforms.RandomRotation(degrees=4)] + _gray +
    [transforms.ColorJitter(brightness=0.35, contrast=0.35),
     transforms.RandomAdjustSharpness(1.5, p=0.3), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     transforms.RandomErasing(p=0.15, scale=(0.02, 0.08))])
eval_tf = transforms.Compose(
    [transforms.Resize((COZUNURLUK, COZUNURLUK))] + _gray +
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_ds, val_ds, test_ds = (IFEDataset(train_df, train_tf), IFEDataset(val_df, eval_tf),
                             IFEDataset(test_df, eval_tf))
full_ds = IFEDataset(df_all, eval_tf)
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
full_ld = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ===========================================================================
# 3) MODEL -- tek gövde, dört başlık
# ===========================================================================
class IFEMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.n_feat = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
        self.backbone = base
        self.head_pattern = nn.Linear(self.n_feat, N_PAT)
        self.head_bands = nn.Linear(self.n_feat, N_BAND)
        self.head_flags = nn.Linear(self.n_feat, N_FLAG)
        self.head_lanes = nn.Linear(self.n_feat, N_LANE)

    def forward(self, x):
        z = self.backbone(x)
        return (self.head_pattern(z), self.head_bands(z),
                self.head_flags(z), self.head_lanes(z))


model = IFEMultiHead().to(DEVICE)
HEAD_KEYS = ("head_pattern", "head_bands", "head_flags", "head_lanes")
bb_params = [p for n, p in model.named_parameters() if not any(k in n for k in HEAD_KEYS)]
hd_params = [p for n, p in model.named_parameters() if any(k in n for k in HEAD_KEYS)]
optimizer = optim.AdamW([{"params": bb_params, "lr": BACKBONE_LR},
                         {"params": hd_params, "lr": HEAD_LR}], weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=TMULT, eta_min=ETA_MIN)


def pos_weight_for(mat):
    pos = mat.sum(0)
    neg = mat.shape[0] - pos
    return torch.tensor(np.clip(neg / (pos + 1e-6), 1.0, POS_WEIGHT_MAX), dtype=torch.float32).to(DEVICE)


Yb_tr, Yf_tr, Yl_tr = train_ds.yb, train_ds.yf, train_ds.yl
crit_pattern = nn.CrossEntropyLoss()
crit_bands = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(Yb_tr), reduction="none")


def band_loss(logits, target, mask):
    """Supheli bantlar (mask=0) kayiptan dislanir."""
    per = crit_bands(logits, target) * mask
    return per.sum() / mask.sum().clamp(min=1.0)
crit_flags = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(Yf_tr))
# reduction="none": maskeyi örnek bazında uygulayabilmek için
crit_lanes = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for(Yl_tr), reduction="none")


def lane_loss(logits, target, mask):
    """Lane'i bilinmeyen örnekleri kayıptan dışlar."""
    per = crit_lanes(logits, target).mean(dim=1)     # örnek başına
    denom = mask.sum()
    if denom < 1:
        return logits.sum() * 0.0
    return (per * mask).sum() / denom

for cls, n in zip(BAND_CLASSES, Yb_tr.sum(0).astype(int)):
    if n == 0:
        print(f"  UYARI: '{BAND_TR[cls]}' eğitim setinde hiç yok — öğrenilemez.")


def set_backbone_trainable(flag):
    for n, p in model.named_parameters():
        if not any(k in n for k in HEAD_KEYS):
            p.requires_grad = flag


set_backbone_trainable(False)
print(f"Isınma: ilk {WARMUP_EPOCHS} epoch omurga donduruldu.")


def predict(loader):
    model.eval()
    P, B, F, L = [], [], [], []
    yp, yb, yf, yl, lmm = [], [], [], [], []
    with torch.no_grad():
        for x, tp, tb, tf, tbm, tl, lm, *_ in loader:
            op, ob, of, ol = model(x.to(DEVICE))
            P.append(torch.softmax(op, 1).cpu().numpy())
            B.append(torch.sigmoid(ob).cpu().numpy())
            F.append(torch.sigmoid(of).cpu().numpy())
            L.append(torch.sigmoid(ol).cpu().numpy())
            yp.append(tp.numpy()); yb.append(tb.numpy())
            yf.append(tf.numpy()); yl.append(tl.numpy())
            lmm.append(lm.detach().cpu().numpy().astype(np.float32).reshape(-1))
    return (np.vstack(P), np.vstack(B), np.vstack(F), np.vstack(L),
            np.concatenate(yp), np.vstack(yb), np.vstack(yf), np.vstack(yl),
            np.concatenate(lmm))


# ===========================================================================
# 4) EĞİTİM
# ===========================================================================
best, no_imp = -1.0, 0
if SKIP_TRAIN:
    if not Path(MODEL_PATH).exists():
        raise SystemExit(f"'{MODEL_PATH}' yok -- once egitimi calistirin.")
    print(f"\n--- Egitim ATLANDI (--sadece-degerlendir) ---")
    print(f"    Kaydedilmis model yukleniyor: {MODEL_PATH}")
else:
    print("\n--- Eğitim ---")
for ep in ([] if SKIP_TRAIN else range(EPOCHS)):
    t0 = time.time()
    if ep == WARMUP_EPOCHS:
        set_backbone_trainable(True)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = False
        optimizer.param_groups[0]["lr"] = BACKBONE_LR * 0.3
        optimizer.param_groups[1]["lr"] = HEAD_LR
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=TMULT, eta_min=ETA_MIN)
        print("[!] Isınma bitti, omurga çözüldü.")

    model.train()
    tot = 0.0
    for i, (x, tp, tb, tf, tbm, tl, lm, *_) in enumerate(train_ld):
        x, tp = x.to(DEVICE), tp.to(DEVICE)
        tb, tf, tl = tb.to(DEVICE), tf.to(DEVICE), tl.to(DEVICE)
        tbm = tbm.float().to(DEVICE)   # supheli bant maskesi
        lm = lm.float().to(DEVICE)
        optimizer.zero_grad()
        op, ob, of, ol = model(x)
        loss = (LOSS_W["pattern"] * crit_pattern(op, tp)
                + LOSS_W["bands"] * band_loss(ob, tb, tbm)
                + LOSS_W["flags"] * crit_flags(of, tf)
                + LOSS_W["lanes"] * lane_loss(ol, tl, lm))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(ep + i / len(train_ld))
        tot += loss.item() * x.size(0)

    vP, vB, vF, vL, ayp, ayb, ayf, ayl, alm = predict(val_ld)
    pat_acc = float((vP.argmax(1) == ayp).mean())
    band_f1 = f1_score(ayb, (vB >= 0.5).astype(int), average="macro", zero_division=0)
    band_rec = recall_score(ayb, (vB >= 0.25).astype(int), average="micro", zero_division=0)
    score = (pat_acc + band_f1 + band_rec) / 3.0
    print(f"Epoch {ep+1:02d}/{EPOCHS} | loss {tot/len(train_ds):.4f} | "
          f"patern-acc {pat_acc:.4f} | bant-F1 {band_f1:.4f} | "
          f"bant-recall@.25 {band_rec:.4f} | hibrit {score:.4f} | {time.time()-t0:.0f}s")

    if score > best + 1e-4:
        best, no_imp = score, 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        no_imp += 1
    if no_imp >= PATIENCE:
        print(f"[!] Early stopping (en iyi hibrit={best:.4f})")
        break

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(DEVICE)

# ===========================================================================
# 5) BANT EŞİKLERİ (VAL üzerinde PR-eğrisi geometrik optimizasyonu)
# ===========================================================================
vP, vB, vF, vL, ayp, ayb, ayf, ayl, alm = predict(val_ld)
band_thr = np.full(N_BAND, 0.5)
for i in range(N_BAND):
    y, p = ayb[:, i].astype(int), vB[:, i]
    if y.sum() == 0:
        continue
    pr, rc, th = precision_recall_curve(y, p)
    pr, rc = pr[:-1], rc[:-1]
    # AZ ORNEKLI SINIF KORUMASI: dogrulama setinde <8 pozitif varsa esik
    # aramasi dejenere sonuc verir (or. 0.001 -> her seye pozitif, precision
    # 0.01). Bu sinifta sabit 0.5 kullan ve ustunde esik aramasi yapma.
    if y.sum() < 8:
        band_thr[i] = 0.5
        continue
    if ESIK_MODU == "dengeli":
        # Max F1 noktasi; ama esik makul bir tabanin altina inemez
        f1v = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
        gecerli = th >= 0.05
        f1v = np.where(gecerli, f1v, -1)
        band_thr[i] = float(th[np.argmax(f1v)])
    else:
        # Tarama: yeterli ornek varsa %70 duyarlilik tabani, sonra geometrik
        if y.sum() >= 15:
            ok = rc >= 0.70
            if ok.any():
                pr, rc, th = pr[ok], rc[ok], th[ok]
        band_thr[i] = float(th[np.argmin(np.sqrt((1 - pr) ** 2 + (1 - rc) ** 2))])
np.savez("thresholds.npz", band=band_thr, band_classes=BAND_CLASSES)

# ===========================================================================
# 6) TEST + YAYIN ÇIKTILARI
# ===========================================================================
os.makedirs(FIG_DIR, exist_ok=True)
tP, tB, tF, tL, typ, tyb, tyf, tyl, tlm = predict(test_ld)
pred_band = (tB >= band_thr[None, :]).astype(int)
pred_pat = tP.argmax(1)

print(f"\n=== TEST SONUÇLARI  (esik modu: {ESIK_MODU}) ===")
print(f"Patern doğruluk (7 sınıf) : {(pred_pat == typ).mean():.4f}")
print(f"Bant macro-F1 (11 sınıf)  : {f1_score(tyb, pred_band, average='macro', zero_division=0):.4f}")
print(f"Bant micro-recall         : {recall_score(tyb, pred_band, average='micro', zero_division=0):.4f}")
print(f"Bant micro-precision      : {precision_score(tyb, pred_band, average='micro', zero_division=0):.4f}")

# TABLO 1 -- kohort (STARD akış şeması)
pd.DataFrame([
    {"Partition": nm, "Platform": dv.title(), "Panels": int((d["device"] == dv).sum()),
     "Source sheets": d.loc[d["device"] == dv, "source_file"].nunique()}
    for nm, d in (("Training", train_df), ("Validation", val_df), ("Test", test_df))
    for dv in sorted(df_all["device"].unique()) if (d["device"] == dv).any()
]).to_csv(f"{FIG_DIR}/table1_cohort.csv", index=False, encoding="utf-8-sig")

# TABLO 2 -- patern ekseni
pd.DataFrame(classification_report(
    typ, pred_pat, labels=list(range(N_PAT)),
    target_names=[PAT_L[c] for c in PATTERN_CLASSES],
    zero_division=0, output_dict=True)).transpose().to_csv(
        f"{FIG_DIR}/table2_pattern.csv", encoding="utf-8-sig")

# TABLO 3 -- bant ekseni (+ sadece-kesin duyarlılık analizi)
tyb_def = onehot(test_df["y_bands_def"], BAND_CLASSES)
tyb_all = onehot(test_df["y_bands_all"], BAND_CLASSES)

# STARD: yorumlanamayan sonuclar duyarlilik/ozgulluk hesabindan DISLANIR,
# sayisi ayrica raporlanir. Patern metrikleri (confusion matrix) bu sinifi
# icermeye devam eder; bant metrikleri yorumlanabilir kayitlarla hesaplanir.
_yorumlanabilir = (test_df["pattern"].astype(str) != "UNINTERPRETABLE_TECHNICAL_ISSUE").to_numpy()
N_UNINTERP_TEST = int((~_yorumlanabilir).sum())
tB_full, tyb_full, pred_band_full = tB.copy(), tyb.copy(), pred_band.copy()
tB, tyb, tyb_def, tyb_all, pred_band = (tB[_yorumlanabilir], tyb[_yorumlanabilir],
                                        tyb_def[_yorumlanabilir], tyb_all[_yorumlanabilir],
                                        pred_band[_yorumlanabilir])
test_dev_bands = test_df["device"].astype(str).to_numpy()[_yorumlanabilir]
print(f"Bant metrikleri: {int(_yorumlanabilir.sum())} yorumlanabilir kayit "
      f"({N_UNINTERP_TEST} yorumlanamayan STARD geregi dislandi)")
rows = []
for i, c in enumerate(BAND_CLASSES):
    n = int(tyb[:, i].sum())
    if n == 0:
        rows.append({L["band"]: BAND_L[c], L["n"]: 0,
                     L["note"]: "no test examples" if DIL == "en" else "test setinde örnek yok"})
        continue
    try:
        rocv = round(float(roc_auc_score(tyb[:, i], tB[:, i])), 3)
    except ValueError:
        rocv = None
    r = {"Bant": BAND_TR[c], "Band (EN)": BAND_EN[c],
         "Şüpheli karşılığı": band_label(c, "suspicious"), "Test n": n, "Eşik": round(float(band_thr[i]), 3),
         "Precision": round(float(precision_score(tyb[:, i], pred_band[:, i], zero_division=0)), 3),
         "Recall": round(float(recall_score(tyb[:, i], pred_band[:, i], zero_division=0)), 3),
         "F1": round(float(f1_score(tyb[:, i], pred_band[:, i], zero_division=0)), 3),
         "ROC-AUC": rocv,
         "PR-AUC": round(float(average_precision_score(tyb[:, i], tB[:, i])), 3),
         "Not": "n<10, güven aralığı geniş" if n < 10 else ""}
    nd = int(tyb_def[:, i].sum())
    r["Kesin-only n"] = nd
    r["Kesin-only Recall"] = (round(float(recall_score(tyb_def[:, i], pred_band[:, i],
                                                       zero_division=0)), 3) if nd else None)
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{FIG_DIR}/table3_bands.csv", index=False, encoding="utf-8-sig")

# TABLO 3b -- GRUPLANMIS bant sonuclari (yeniden egitim GEREKMEZ)
# Tek-lane bantlar tek tek az orneklidir. Klinik ayrimi koruyarak iki gruba
# toplanir: "agir zincir tek basina" ve "serbest hafif zincir". Bir de tum
# klon-tanimli bantlar tek grup olarak verilir. Tahmin: grup uyelerinden
# HERHANGI biri pozitifse grup pozitif (OR).
GRUPLAR = {
    ("Clone-defined band (Ig-κ/λ)" if DIL == "en" else "Klon tanimli bant (Ig-kappa/lambda)"): ["IGG_KAPPA", "IGG_LAMBDA", "IGA_KAPPA",
                                            "IGA_LAMBDA", "IGM_KAPPA", "IGM_LAMBDA"],
    ("Heavy chain without light chain" if DIL == "en" else "Agir zincir tek basina (hafif zincir yok)"): ["IGG_NO_LIGHT_CHAIN", "IGA_NO_LIGHT_CHAIN",
                                                  "IGM_NO_LIGHT_CHAIN"],
    ("Free light chain" if DIL == "en" else "Serbest hafif zincir"): ["FREE_KAPPA", "FREE_LAMBDA"],
    ("Any monoclonal band" if DIL == "en" else "Herhangi bir monoklonal bant"): BAND_CLASSES,
}
rows = []
for ad, uyeler in GRUPLAR.items():
    idx = [BAND_CLASSES.index(c) for c in uyeler]
    y_true = tyb[:, idx].max(axis=1)
    y_pred = pred_band[:, idx].max(axis=1)
    y_prob = tB[:, idx].max(axis=1)
    n = int(y_true.sum())
    if n == 0:
        rows.append({"Group": ad, "Test n": 0, "Note": "no test examples"})
        continue
    try:
        rocv = round(float(roc_auc_score(y_true, y_prob)), 3)
    except ValueError:
        rocv = None
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    rows.append({"Group": ad, "Test n": n,
                 "Sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
                 "Specificity": round(tn / max(tn + fp, 1), 3),
                 "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
                 "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
                 "ROC-AUC": rocv})
pd.DataFrame(rows).to_csv(f"{FIG_DIR}/table3b_grouped_bands.csv", index=False, encoding="utf-8-sig")
print("\n--- Gruplanmis bant sonuclari ---")
for r in rows:
    if r.get("Test n"):
        print(f"  {r['Group']:<44} n={r['Test n']:3d}  sens={r['Sensitivity']:.2f}  "
              f"spec={r['Specificity']:.2f}  F1={r['F1']:.2f}  AUC={r['ROC-AUC']}")

# TABLO 4 -- bayrak + poliklonal lane eksenleri
rows = []
for mat, prob, classes, tr, ax in ((tyf, tF, FLAG_CLASSES, FLAG_L, "Finding" if DIL == "en" else "Bulgu"),
                                   (tyl[tlm > 0], tL[tlm > 0], POLY_LANE_CLASSES,
                                    LANE_L, "Polyclonal lane" if DIL == "en" else "Poliklonal lane")):
    # Lane metrikleri yalnızca lane'i BİLİNEN test örnekleri üzerinden.
    pred = (prob >= 0.5).astype(int)
    for i, c in enumerate(classes):
        n = int(mat[:, i].sum())
        rows.append({"Eksen": ax, "Sınıf": tr[c], "Test n": n,
                     "Recall": round(float(recall_score(mat[:, i], pred[:, i], zero_division=0)), 3) if n else None,
                     "Precision": round(float(precision_score(mat[:, i], pred[:, i], zero_division=0)), 3) if n else None,
                     "PR-AUC": round(float(average_precision_score(mat[:, i], prob[:, i])), 3) if n else None})
pd.DataFrame(rows).to_csv(f"{FIG_DIR}/table4_findings_lanes.csv", index=False, encoding="utf-8-sig")

# ŞEKİL 1 -- patern confusion matrix
cm = confusion_matrix(typ, pred_pat, labels=list(range(N_PAT)))
lab = [PAT_L[c] for c in PATTERN_CLASSES]
fig, ax = plt.subplots(figsize=(9, 8))
cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(N_PAT)); ax.set_yticks(range(N_PAT))
ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(lab, fontsize=8)
ax.set_xlabel(L["cm_x"]); ax.set_ylabel(L["cm_y"])
ax.set_title(L["cm_title"])
for r in range(N_PAT):
    for c in range(N_PAT):
        if cm[r, c]:
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=7,
                    color="white" if cmn[r, c] > .5 else "black")
fig.colorbar(im, ax=ax, fraction=.046); fig.tight_layout()
fig.savefig(f"{FIG_DIR}/figure1_pattern_confusion.png", dpi=300); plt.close(fig)

# ŞEKİL 2 -- bant ROC
fig, ax = plt.subplots(figsize=(7, 7))
for i, c in enumerate(BAND_CLASSES):
    if tyb[:, i].sum() < 10:
        continue
    fpr, tpr, _ = roc_curve(tyb[:, i], tB[:, i])
    ax.plot(fpr, tpr, lw=1.3, label=f"{BAND_L[c]} (AUC = {auc(fpr,tpr):.2f}, n = {int(tyb[:,i].sum())})")
ax.plot([0, 1], [0, 1], "k--", lw=.8)
ax.set_xlabel(L["roc_x"]); ax.set_ylabel(L["roc_y"])
ax.set_title(L["roc_title"]); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/figure2_band_roc.png", dpi=300); plt.close(fig)

# ŞEKİL 3 -- cihaz karşılaştırması
tdev = set(test_df["device"].astype(str))
for d in sorted(set(train_df["device"].astype(str)) - tdev):
    print(f"\n[!] UYARI: '{d}' cihazından test setinde HİÇ örnek yok. Bu cihaz için "
          f"genelleme iddiası kurulamaz; yayında belirtin ya da bu cihazdan daha "
          f"fazla KAYNAK JEL etiketleyin.")
if len(tdev) > 1:
    devs = sorted(tdev)
    vals = [f1_score(tyb[test_dev_bands == d],
                     pred_band[test_dev_bands == d],
                     average="macro", zero_division=0) for d in devs]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(devs, vals, color=["#4472c4", "#ed7d31"][:len(devs)])
    ax.set_ylabel(L["dev_y"]); ax.set_ylim(0, 1)
    ax.set_title(L["dev_title"])
    ax.set_xticks(range(len(devs))); ax.set_xticklabels([d.title() for d in devs])
    for i, v in enumerate(vals):
        ax.text(i, v + .02, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/figure3_platform.png", dpi=300); plt.close(fig)
else:
    print("  Şekil 3 atlandı: test setinde tek cihaz var.")

# Test seti tahminlerini KAYDET -- guven araligi (bootstrap) ve
# aciklanabilirlik analizleri bu dosyadan calisir, yeniden egitim gerekmez.
np.savez(f"{FIG_DIR}/test_tahminleri.npz",
         pattern_prob=tP, pattern_true=typ, pattern_pred=pred_pat,
         band_prob=tB_full, band_true=tyb_full, band_pred=pred_band_full, band_thr=band_thr,
         band_valid=_yorumlanabilir,   # STARD: bant metrikleri bu maskeyle hesaplanir
         flag_prob=tF, flag_true=tyf, lane_prob=tL, lane_true=tyl,
         device=test_df["device"].astype(str).to_numpy(),
         source_file=test_df["source_file"].astype(str).to_numpy(),
         filename=test_df["filename"].astype(str).to_numpy(),
         pattern_classes=np.array(PATTERN_CLASSES), band_classes=np.array(BAND_CLASSES))
print(f"Test tahminleri kaydedildi -> {FIG_DIR}/test_tahminleri.npz")
print(f"\nYayın çıktıları -> {FIG_DIR}/")

# ===========================================================================
# 7) YZ ÇİFT-KONTROL (etiket hatası avcısı)
# ===========================================================================
print("\n--- Etiket hatası avcısı ---")
hits = []
model.eval()
with torch.no_grad():
    for x, tp, tb, tf, tbm, tl, lm, names, paths, splits, devs in full_ld:
        _, ob, _, _ = model(x.to(DEVICE))
        prob = torch.sigmoid(ob).cpu().numpy()
        act = tb.numpy()
        for i in range(len(names)):
            found = [f"{BAND_TR[c]} (%{prob[i][j]*100:.0f})"
                     for j, c in enumerate(BAND_CLASSES) if prob[i][j] >= band_thr[j]]
            summary = " + ".join(found) if found else "TEMİZ (bant bulunamadı)"
            for j, c in enumerate(BAND_CLASSES):
                p, y, thr = float(prob[i][j]), int(act[i][j]), float(band_thr[j])
                margin = MARGIN_CRIT
                if p >= thr + margin and y == 0:
                    hits.append({"Dosya_Adi": names[i], "Yol": paths[i], "Split": splits[i],
                                 "Cihaz": devs[i], "Bulgu": c, "Bant_TR": BAND_TR[c],
                                 "Durum": "YZ var dedi / İnsan yok dedi",
                                 "YZ_Tam_Tahmini": summary, "Model_Prob": p,
                                 "Threshold": thr, "ScoreNorm": (p - thr) / margin})
                elif p <= thr - margin and y == 1:
                    hits.append({"Dosya_Adi": names[i], "Yol": paths[i], "Split": splits[i],
                                 "Cihaz": devs[i], "Bulgu": c, "Bant_TR": BAND_TR[c],
                                 "Durum": "İnsan var dedi / YZ yok dedi",
                                 "YZ_Tam_Tahmini": summary, "Model_Prob": p,
                                 "Threshold": thr, "ScoreNorm": (thr - p) / margin})
if hits:
    pd.DataFrame(hits).sort_values("ScoreNorm", ascending=False).to_csv(
        ERROR_REPORT, index=False, encoding="utf-8-sig")
    print(f"{len(hits)} çelişki -> {ERROR_REPORT} (hata_ayiklayici.py ile inceleyin)")
else:
    print("Çelişki bulunamadı.")
