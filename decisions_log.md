# Decisions Log

Running record of modeling and engineering decisions, with reasoning.
Kept so choices are traceable later — both for me and for anyone reviewing this repo.

---

## 2026-07-16 — Dropped columns with >60% missing values

**Decision:** Dropped 17 columns from `application_train.csv` where missing
values exceeded 60% of rows (122 → 105 columns). Mostly property/building
characteristics (COMMONAREA_*, NONLIVINGAPARTMENTS_*, LIVINGAPARTMENTS_*,
FLOORSMIN_*, YEARS_BUILD_*, OWN_CAR_AGE, LANDAREA_*).

**Reasoning:** At 60-70% missing, imputation would mostly be guessing rather
than recovering real signal. Missingness in these columns appears clustered
by category (not random), suggesting it correlates with property type or
data collection process — genuinely informative, but not worth the added
complexity before a baseline model exists.

**Alternative considered:** Convert these to binary "was_missing" flag
features instead of dropping outright, since missingness itself may be
predictive of default risk.

**Deferred to v2:** Once a baseline model is trained, check whether dropped
columns would likely have mattered (e.g. via domain reasoning or a quick
model-with-flags comparison). Revisit only if there's a concrete reason to
believe they'd move the needle — not by default.


## 2026-07-17 — Median imputation applied to EXT_SOURCE_1 despite 56% missingness

**Note:** Because EXT_SOURCE_1 was protected from dropping but still 56%
missing, median imputation means over half its values are now a single
constant (0.505998), flattening real variance in this otherwise high-signal
column. Accepted for baseline; a smarter treatment (e.g. model-based
imputation using EXT_SOURCE_2/3 as predictors, or a missing-flag + median
combo) is a stronger v2 candidate specifically for this column.

## 2026-07-17 — Baseline logistic regression trained and evaluated

**Result:** AUC-ROC = 0.7486 on held-out test set (20%, stratified).
Recall on default class = 0.68, precision = 0.16 (class_weight="balanced").

**Interpretation:** Model catches ~68% of actual defaulters at the cost of
a high false-positive rate (17,436 safe applicants flagged as risky out of
56,538). This is the direct effect of class_weight="balanced" prioritizing
recall over precision — appropriate as a starting point since missing a
defaulter is typically costlier than a false alarm, but the 0.5 default
classification threshold is not tuned to any real cost structure.

**v2 candidates:** (1) Threshold tuning based on assumed cost ratio between
false positives and false negatives, rather than default 0.5 cutoff.
(2) Try gradient boosting (XGBoost/LightGBM) for comparison — known to
outperform logistic regression on this dataset by roughly 3-5 AUC points.
(3) Feature importance analysis to identify which engineered/raw features
drive predictions, and whether EXT_SOURCE_1's flattened imputation (see
earlier entry) is hurting performance.


## 2026-07-18 — Threshold tuning: selected 0.65 via F1 optimization

**Decision:** Swept classification thresholds from 0.10 to 0.90 in 0.05
increments. Selected 0.65 as the operating threshold — this maximizes F1
(0.2980) across the sweep, versus F1 = 0.2605 at the default 0.50 cutoff.
At 0.65: precision = 0.2277, recall = 0.4312.

**Reasoning:** No real cost data exists for this project to weight false
negatives (missed defaults) against false positives (unnecessary manual
review) explicitly, so F1 was used as a neutral balance point rather than
guessing a cost ratio. This is an explicit simplification — a production
deployment would need actual cost-per-bad-loan and cost-per-review figures
to set the threshold correctly, likely favoring recall over this F1-optimal
point given that missed defaults are typically far costlier than false
alarms in real lending.

**Observation:** The precision-recall tradeoff is steep and the ceiling is
a property of the model's separability, not something threshold tuning
can overcome — no threshold achieves both high precision and high recall.
This motivates trying a stronger model (XGBoost) as the next step, since
threshold tuning alone has diminishing returns on a linear baseline.
