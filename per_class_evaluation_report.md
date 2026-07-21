# NIDS Multi-Class Model — Per-Class Evaluation Report

Generated with `src/evaluate.py`, using the exact train/test split from `src/train.py`
(19 numeric + 1 categorical feature, 70/30 random stratified split, `random_state=42`)
against the currently serialized `nids_multi_pipeline.pkl`.

## Aggregate Numbers

| Metric | Value |
|---|---|
| Accuracy | 80.95% |
| Macro Precision | 0.633 |
| Macro Recall | 0.596 |
| Macro F1 | 0.608 |
| Weighted F1 | 0.817 |

## Per-Class Breakdown

| Class | Precision | Recall | F1 | Test Support | Total in Dataset |
|---|---|---|---|---|---|
| Analysis | 0.29 | 0.16 | 0.20 | 600 | 2,000 |
| Backdoor | 0.13 | 0.15 | 0.14 | 524 | 1,746 |
| DoS | 0.34 | 0.53 | 0.42 | 3,679 | 12,264 |
| Exploits | 0.71 | 0.67 | 0.69 | 10,018 | 33,393 |
| Fuzzers | 0.74 | 0.75 | 0.74 | 5,455 | 18,184 |
| Generic | 1.00 | 0.98 | 0.99 | 12,000 | 40,000 |
| Normal | 0.94 | 0.91 | 0.93 | 16,800 | 56,000 |
| Reconnaissance | 0.91 | 0.73 | 0.81 | 3,148 | 10,491 |
| Shellcode | 0.63 | 0.62 | 0.62 | 340 | 1,133 |
| Worms | 0.64 | 0.46 | 0.54 | 39 | 130 |

Full confusion matrix (raw counts and row-normalized recall) saved to
`confusion_matrix_multi_annotated.png`.

## Reading the Gap

Accuracy (80.95%) sits far above macro F1 (0.608) because the dataset is heavily
imbalanced — Normal and Generic together make up roughly 87,000 of the 175,341 rows
(about 55%), and the model is scored near-perfectly on both (F1 0.93 and 0.99).
Aggregate accuracy is dominated by these two classes.

The classes actually failing are the smallest ones:

- **Backdoor** — F1 0.14, worst in the model, on only 1,746 total examples.
- **Analysis** — F1 0.20, on 2,000 total examples.
- **DoS** — F1 0.42, surprisingly weak despite 12,264 examples. Its confusion is mostly
  with Exploits and Generic (see confusion matrix), suggesting real feature-level
  overlap between DoS and Exploits traffic in UNSW-NB15, not just a sample-size problem.
- **Worms** — F1 0.54, but only 130 examples exist in the *entire* dataset (39 in test),
  so this number is high-variance and not very trustworthy on its own.

`class_weight='balanced'` is already applied in `train.py`, and it is visibly not
enough on its own for classes under ~2,000 examples — the model still defaults to the
majority classes on ambiguous flows.

## What Would Move Macro F1

1. **Targeted resampling** (SMOTE or similar) on Backdoor, Analysis, and Worms
   specifically — these are the true bottom of the distribution, not DoS.
2. **Investigate DoS/Exploits confusion directly** — since DoS has decent support but
   still underperforms, this is a feature-separability problem, not a
   data-quantity problem. Worth an ablation checking whether current features
   distinguish the two attack types at all.
3. **Report macro F1 alongside accuracy going forward** — accuracy alone is
   misleading on this dataset and should not be quoted without this table attached.

## Caveat: Split Methodology

This split is random and stratified by `attack_cat`, not session- or time-based.
UNSW-NB15 is known to contain near-duplicate flows across sessions; a random split can
let near-identical rows land in both train and test, inflating all of the above numbers
to an unknown degree. This has not been corrected for — flagging it here so it's not
presented as settled.
