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

Reads: processed_train.csv, processed_test.csv Writes: submission/checkpoint_<model>.csv, submission/checkpoint_ensemble.csv Purpose: compare RandomForest, CatBoost, LightGBM with 5-fold Stratified CV, scored on balanced accuracy. All models use class weighting. Tried custom weight grids and probability-averaging ensembles — neither beat CatBoost alone (see decisions.md #4). Decision: CatBoost alone, no ensemble. OOF balanced accuracy 0.9086, leaderboard 0.9059.

## 05_Feature_Engineering.ipynb

Reads: processed_train.csv, processed_test.csv, train.csv/test.csv (raw, for missing-value masks), feature_importance_rf.csv Writes: engineered_train.csv, engineered_test.csv Purpose: test candidate features one at a time against the CatBoost baseline (0.9086), keep only if OOF balanced accuracy improves by >0.001. Winning features: _is_missing flags for stress_level, sleep_duration, sleep_quality, and the remaining columns — jumped OOF balanced accuracy from 0.9086 to 0.9492. See decisions.md #5 for why this worked. Interaction terms, BMI bucketing, and the composite lifestyle score were all tested and dropped (no improvement).

## 06_Final_Submission.ipynb

Reads: engineered_train.csv, engineered_test.csv Writes: submission/submission_final.csv Purpose: confirm the engineered feature set on full CV, light hyperparameter tuning pass (depth/learning_rate), train CatBoost on 100% of training data, export the official submission. Final CV: ~0.953 balanced accuracy. Public leaderboard: 0.949. See decisions.md #6 for the CV-vs-leaderboard gap discussion.
