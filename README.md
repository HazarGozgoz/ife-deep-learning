# Deep learning for serum immunofixation electrophoresis (IFE)

Code, trained model weights, anonymised panel-level labels and predictions for the manuscript

> Gözgöz H, Ecevit TH, Akan Konuk B, Akan P. *Deep learning for serum immunofixation electrophoresis: band-level detection, typing and platform adaptation on 3,596 gel panels.* (submitted)

A ConvNeXt-Tiny network with four output heads (eight-class pattern, 11-class band list, additional findings, polyclonal lanes) was trained and evaluated on 3,596 serum IFE panels from 920 archived gels (Interlab G26 and Helena SAS-1) with five-fold cross-validation grouped by scanned page, cluster-bootstrap confidence intervals, an SPE-lane masking experiment, a platform-transfer/adaptation experiment and a blinded independent-reader sub-study.

## Repository layout

```
src/        Python scripts (annotation tool, data preparation, training, cross-validation,
            confidence intervals, Grad-CAM, agreement and laboratory-value analyses)
data/       anonymised panel-level reference labels and pooled test-fold predictions (CSV),
            class names and band-head thresholds (JSON)
results/    aggregated result tables as reported in the manuscript and supplement
weights/    see weights/README.md (trained fold weights are distributed as release assets)
```

Script names are in Turkish (the working language of the laboratory); their roles:

| Script | Role |
|---|---|
| `ife_taxonomy.py` | Three-axis structured labelling scheme (pattern, bands with certainty, flags, polyclonal lanes); deterministic pattern derivation; consistency checks |
| `ife_annotator.py`, `ife_dialog.py` | PyQt6 annotation tool used to crop panels from scanned pages and record structured labels |
| `ife_review_tool.py` | Review tool for blinded re-reading (`--kor-okuyucu` = blinded independent-reader mode) |
| `master_guncelle.py` | Builds/updates the master annotation table |
| `veri_hazirlama.py` | Train/validation/test split grouped by scanned page and stratified by platform; `--test-cihazi HELENA` for the platform holdout; `--uyarlama-jel N` for the adaptation curve |
| `egitim_ve_degerlendirme.py` | Model definition, training loop (AdamW, differential learning rates, warm-up, cosine restarts, early stopping), threshold selection and evaluation |
| `capraz_dogrulama.py` | Five-fold cross-validation driver (calls the training script per fold, pools test-fold predictions) |
| `guven_araligi.py` | Cluster-bootstrap confidence intervals over scanned pages, McNemar tests, result tables and figures |
| `gradcam.py` | Grad-CAM attention maps; `--ablasyon-elp` runs the SPE-lane masking and control-strip experiment |
| `uyum_hesapla.py` | Cohen's κ for routine-report and independent-reader agreement (`--ikinci-okuyucu`) |
| `lab_birlestir.py` | Matching to laboratory information system values; `--karsilastir` runs the Mann–Whitney comparisons |
| `kart_id_oku.py`, `kart_defter_birlestir.py`, `defter_oku.py`, `tarama_haritasi.py`, `bitti_isaretle.py` | Helper scripts for gel-card identifiers, notebook matching, notebook transcription (vision-language model via API; no credentials are stored) and bookkeeping |

## Environment

Python 3.13; PyTorch 2.9.0; torchvision 0.24.0; scikit-learn 1.7.2 (see `requirements.txt`). Training was run on Apple Silicon (MPS backend); CUDA works without changes.

## Reproducing the analysis

The gel images cannot be shared (gel cards carry handwritten patient identifiers). With your own cropped panels and a master annotation table in the format of `data/panel_labels_and_predictions.csv` (one row per panel, grouped by scanned page):

```bash
python veri_hazirlama.py                     # grouped split
python capraz_dogrulama.py                   # five-fold CV
python guven_araligi.py --tamamla            # CIs, tables, figures
python gradcam.py --ablasyon-elp             # SPE-lane masking experiment
python veri_hazirlama.py --test-cihazi HELENA && python egitim_ve_degerlendirme.py --etiket=helena_holdout
```

All manuscript numbers can be recomputed from `data/panel_labels_and_predictions.csv` (pooled test-fold predictions of the five cross-validation models with the final reference labels) and `data/class_names_and_thresholds.json` without the images.

## Training details

ConvNeXt-Tiny backbone (ImageNet-pretrained), four parallel heads. Loss weights: bands 1.5, pattern 1.0, additional findings 0.5, polyclonal lanes 0.5; class-balanced positive weights capped at 10. AdamW with differential learning rates (backbone 3e-5, heads 1e-4), two-epoch warm-up with frozen backbone, cosine annealing with warm restarts, gradient clipping, early stopping on a validation composite score (patience 7, max 30 epochs). Inputs 384 × 384 grayscale. Augmentation: brightness/contrast jitter ±35 %, rotation ±4°, random erasing, sharpness variation. Suspicious bands masked from the band loss; panels with polyclonal increase but unrecorded lanes masked from the lane loss. Band-head thresholds chosen per class on the validation fold at the maximum-F1 point.

## Data availability

- `data/panel_labels_and_predictions.csv`: 3,596 panels; anonymised panel and page identifiers, platform, cross-validation fold, reference pattern and band labels, model predictions and class probabilities. No patient identifiers, image file names or coordinates are included.
- Gel images, laboratory notebook transcriptions and laboratory values are not shared.

## Licence and citation

Code: MIT License. Data in `data/` and `results/`: CC BY 4.0. Please cite the manuscript above.

## Acknowledgement

Parts of the analysis code were written with the assistance of a large language model (Claude, Anthropic) and reviewed and verified by the authors.
