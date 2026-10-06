# data/

## panel_labels_and_predictions.csv
One row per serum IFE panel (n = 3,596). Pooled test-fold predictions of the five cross-validation models, scored against the final reference labels.

| column | meaning |
|---|---|
| panel_id | anonymised panel identifier (P0001–P3596) |
| page_id | anonymised scanned-page identifier (S001–S176); grouping unit for cross-validation and cluster bootstrap |
| platform | INTERLAB or HELENA |
| fold | cross-validation test fold (1–5) in which the panel was predicted |
| pattern_true, pattern_pred | reference and predicted eight-class pattern |
| p_pattern_* | softmax probability per pattern class |
| band_valid | 1 = included in band-level metrics (0 = uninterpretable panel) |
| band_true_*, band_pred_*, p_band_* | reference label, thresholded prediction and sigmoid probability per band class (11 classes) |
| flag_true_*, p_flag_* | hypogammaglobulinemia and polyclonal-increase labels and probabilities |
| lane_true_*, p_lane_* | polyclonal-lane labels and probabilities (IgG, IgA, IgM, κ, λ) |

Suspicious bands are not part of `band_true_*` (the band head was trained and scored on definite bands); the reference pattern class SUSPICIOUS_BAND identifies panels whose only finding was a suspicious band.

## class_names_and_thresholds.json
Class order used in the CSV and the per-class band-head decision thresholds (maximum-F1 point on the validation fold, averaged over folds).
