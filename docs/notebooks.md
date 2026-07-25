# Notebooks — purpose and data flow

Each notebook reads the output of the one before it. Never skip ahead, never
read a raw file from a notebook other than 01–03.

## 01_EDA_Basic.ipynb
**Reads:** `train.csv`, `test.csv`
**Writes:** nothing (read-only exploration)
**Purpose:** understand the data before touching it — shape, dtypes, target
class balance, missing value counts, duplicate rows, numeric range sanity
checks, categorical value sanity checks, missing-value pattern vs. target
(MNAR check).

## 02_Research_Questions.ipynb
**Reads:** `train.csv`, `test.csv`
**Writes:** nothing (read-only exploration)
**Purpose:** answer the 7 guiding research questions with charts and
statistics (sleep distribution, stress vs. sleep, sleep quality vs. heart
rate, diet vs. activity, exercise vs. calories, smoking/alcohol vs. health
condition, numeric correlations). Feeds ideas into `05_Feature_Engineering`.

## 03_Preprocessing.ipynb
**Reads:** `train.csv`, `test.csv`
**Writes:** `processed_train.csv`, `processed_test.csv`, `feature_importance_rf.csv`
**Purpose:**
- Impute missing values (median for numeric, most-frequent for categorical) —
  fit on train, transform-only on test (never re-fit on test).
- Encode categoricals (ordinal for stress_level/sleep_quality/
  physical_activity_level, one-hot for diet_type/smoking_alcohol/gender).
- Map target to integers (fit:0, at-risk:1, unhealthy:2).
- Feature selection: drop near-zero-variance columns, flag highly correlated
  (>0.9) column pairs for manual review.

## 04_Modeling.ipynb
**Reads:** `processed_train.csv`, `processed_test.csv`
**Writes:** `submission/checkpoint_<model>.csv`
**Purpose:** train and compare RandomForest, CatBoost, LightGBM with
5-fold Stratified CV, scored on **balanced accuracy** (matches the
competition metric). All models use class weighting to counter target
imbalance. Exports a checkpoint submission purely to confirm the CV score
is in the same range as the public leaderboard — not the final submission.
