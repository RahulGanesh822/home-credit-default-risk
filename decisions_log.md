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