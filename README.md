# Home Credit Default Risk — Credit Scorecard

## Problem
Predicting the probability that a loan applicant will default, using
Home Credit Group's application data (Kaggle competition dataset).
Many applicants in this dataset are thin-file or unscored borrowers —
people with limited traditional credit history — making this a realistic
proxy for underwriting decisions at NBFCs and digital lenders in markets
like India, where a large share of loan applicants fall into this category.

## Business Context
Credit risk models like this sit at the core of lending decisions: approve
or reject, and at what interest rate. The core challenge isn't just
predicting risk accurately — it's balancing two costs that don't weigh the
same: approving someone who defaults (bad debt) versus rejecting someone
who would have repaid (lost revenue, and a real person denied credit).
This project treats that tradeoff explicitly rather than optimizing for
raw accuracy, which is misleading on an imbalanced dataset like this one.

## Data
Source: [Home Credit Default Risk, Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)
Not included in this repo due to size (~2.5GB). To reproduce:
1. Download from the Kaggle link above (requires accepting competition rules)
2. Place `application_train.csv` in `data/raw/`

## Approach
- **Data cleaning:** Dropped 17 columns with >60% missing values (mostly
  property/building characteristics), with `EXT_SOURCE_1` explicitly
  protected despite 56% missingness due to its known predictive value.
  Full reasoning in `decisions_log.md`.
- **Imputation:** Median for numeric columns, explicit "Missing" category
  for categorical columns — chosen over mean/mode to avoid distorting
  skewed distributions or inventing false majority answers.
- **Encoding:** One-hot encoding for categorical features (105 → 216 columns).
- **Split:** 80/20 train/test, stratified on target to preserve the ~8%
  default rate in both sets.
- **Model:** Logistic regression baseline, with `class_weight="balanced"`
  to address class imbalance (features standardized via `StandardScaler`,
  fit on train only to avoid data leakage).
- **Evaluation metric:** AUC-ROC, not accuracy — a model that predicts
  "no default" for every applicant would score ~92% accuracy on this
  dataset while being completely useless. AUC-ROC measures ranking quality
  across all classification thresholds, which is the standard metric for
  credit scoring.

## Results

| Model | AUC-ROC | Recall (default) | Precision (default) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7486 | 0.68 | 0.16 |

The model catches roughly two-thirds of actual defaulters, at the cost of
a high false-positive rate — a direct consequence of prioritizing recall
via `class_weight="balanced"`. In a real underwriting context, this
tradeoff would be tuned against the actual cost ratio between missed
defaults and false alarms, rather than left at the default 0.5 threshold.

## Repo Structure
├── data/raw/          # Raw CSVs (gitignored, download instructions above)
├── src/
│   ├── data_prep.py   # Loading, cleaning, imputation, encoding, split
│   └── model.py        # Training and evaluation
├── decisions_log.md    # Reasoning behind every modeling decision
├── requirements.txt


## Setup
```bash
git clone https://github.com/RahulGanesh822/home-credit-default-risk.git
cd home-credit-default-risk
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Next Steps
- Threshold tuning based on an assumed cost ratio (false negative vs.
  false positive), rather than the default 0.5 cutoff
- Gradient boosting comparison (XGBoost/LightGBM) — expected to outperform
  logistic regression by roughly 3-5 AUC points on this dataset
- Feature importance analysis
- SQL layer for the related Early Warning System (EWS) prototype
