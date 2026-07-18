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
- **Anomaly handling:** `DAYS_EMPLOYED` contained a placeholder value
  (365243, ~1000 years) representing "not currently employed" rather than
  a real day count — a known issue in this dataset. Replaced with NaN
  (handled by standard imputation) and preserved as a binary
  `DAYS_EMPLOYED_ANOMALY` flag. Found via feature importance review: the
  placeholder was distorting the logistic regression coefficient (7.47,
  more than double the next-highest feature) despite barely affecting
  AUC — see decisions_log.md for full detail.

## Results
| Model | AUC-ROC | Threshold | Recall (default) | Precision (default) |
| Logistic Regression (baseline, default threshold) | 0.7484 | 0.50 | 0.68 | 0.16 |
| Logistic Regression (F1-tuned threshold) | 0.7484 | 0.65 | 0.43 | 0.23 |
| XGBoost (untuned) | 0.7578 | 0.50 | — | — |

XGBoost improved AUC-ROC by 0.0094 over logistic regression — a real but
modest gain, smaller than typically reported for this dataset. Likely
explanations: hyperparameters were not tuned, and this uses only the main
application table — most published high-performing solutions incorporate
bureau.csv and previous_application.csv for richer credit history
features, which is deferred to a future iteration.

## Repo Structure
├── data/raw/          # Raw CSVs (gitignored, download instructions above)
├── src/
│   ├── data_prep.py   # Loading, cleaning, imputation, encoding, split
│   └── model.py        # Training and evaluation
├── decisions_log.md    # Reasoning behind every modeling decision
├── requirements.txt


## Setup
```bashgit clone https://github.com/RahulGanesh822/home-credit-default-risk.git
cd home-credit-default-risk
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## Next Steps
- Cost-ratio-based threshold tuning using real cost-per-bad-loan and
  cost-per-review figures, once available — current threshold (0.65) is
  F1-optimized only, as a neutral stand-in for actual business costs- Feature importance analysis
- SQL layer for the related Early Warning System (EWS) prototype
