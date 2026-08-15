# Decision log

Chronological record of what changed, and why — mainly useful for explaining
the reasoning behind commits later, and so future-me doesn't repeat the same
debugging.

---

## 1. Preprocessing bug: imputer fit on test set

**Problem:** `X_test` was imputed with `imputer.fit_transform(X_test[...])`
instead of `imputer.transform(X_test[...])`. This re-fits the imputer's
median/mode from the test set itself instead of reusing what was learned
from train.

**Why it matters:** the test set must never be used to learn any parameter.
Fitting separately on test broke train/test distribution consistency.

**Fix:** `fit_transform` on train, `transform` only on test, for both the
numeric and categorical imputers.

---

## 2. Preprocessing bug: `id` leaking into features

**Problem:** `train.csv`/`test.csv` were read without `index_col="id"`, so
`id` was treated as a normal feature column and flowed through the whole
pipeline into the model.

**Fix:** read both CSVs with `index_col="id"`.

---

## 3. Metric mismatch: accuracy vs. balanced accuracy

**Problem:** CV was scored with plain `accuracy_score`, giving ~0.965,
while the Kaggle public leaderboard showed ~0.855 for the same model.

**Investigation:** target class balance is `at-risk` ~86%, `unhealthy`
~8.4%, `fit` ~5.8%. Macro-average recall was ~0.86, almost exactly matching
the leaderboard. Confirmed via the Kaggle "Evaluation" tab: **the
competition scores submissions on Balanced Accuracy**, not plain accuracy.

**Fix:** switched all CV scoring in `04_Modeling.ipynb` from
`accuracy_score` to `balanced_accuracy_score`.

---

## 4. Class weighting, custom weight search, and ensemble — what worked and what didn't

**Class weighting:** confusion matrix showed most errors were minority
classes (`fit`, `unhealthy`) being misclassified as the majority class
(`at-risk`). Added `class_weight='balanced'` (RandomForest, LightGBM) /
`auto_class_weights='Balanced'` (CatBoost). **Result: OOF balanced accuracy
improved from ~0.86 to ~0.9086.**

**Custom weight grid search:** tried several `{0: w0, 1: 1, 2: w2}` weight
combos on LightGBM via CV — none beat the default `'balanced'` weighting by
a meaningful margin.

**Ensemble (RandomForest + CatBoost + LightGBM):** tried simple average and
OOF-score-weighted average of `predict_proba` across all 3 models, and a
2-way CatBoost+LightGBM average. **All versions scored lower than CatBoost
alone** (weighted ensemble 0.9083 vs. CatBoost alone 0.9086). Reason: all 3
models are tree-based on the identical feature set, so their errors are
correlated (they fail on the same borderline `at-risk` cases) — averaging
just dilutes the best model instead of correcting it.

**Decision: use CatBoost alone, no ensemble, no custom weights beyond
`'balanced'`.** Leaderboard confirmed: `checkpoint_catboost.csv` scored
0.9059, `checkpoint_ensemble.csv` scored 0.9046.

---

## 5. Feature engineering: missing-value flags were the single biggest win

**Hypothesis (from `01_EDA_Basic`):** `stress_level`, `sleep_duration`, and
`sleep_quality` are self-reported columns with 12%/11%/8.5% missing rates —
suspected MNAR (missing not at random), i.e. students who are more
stressed/sleep-deprived may be more likely to skip those questions, meaning
missingness itself could carry signal about `health_condition`.

**Test:** added `_is_missing` (0/1) flags for those 3 columns, computed from
the raw (pre-imputation) train/test files, tested via CV against the
CatBoost baseline (0.9086).

**Result: OOF balanced accuracy jumped from 0.9086 to 0.9461** (+3.75
points) — by far the largest single improvement in the whole project.
Recall on the minority classes went from 0.70/0.69 to 0.94/0.96
(fit/unhealthy). Extending the same `_is_missing` flag to the remaining
columns (calorie_expenditure, water_intake, physical_activity_level,
smoking_alcohol, gender, step_count, bmi, heart_rate, exercise_duration,
diet_type) added another +0.31 points, to **0.9492**.

**Tested and dropped (no improvement over the running best score):**
- `stress_level × sleep_duration` interaction (ratio and product)
- `bmi_category` (WHO-style bucketing)
- composite `lifestyle_score` (z-score combination of sleep/steps/activity/stress)
- `calorie_per_exercise_min`

**Takeaway:** for this dataset, *whether* a value is missing carried more
predictive signal than the numeric values themselves for several features —
worth checking for MNAR early in any project with a lot of self-reported
columns, before assuming missingness is just noise to impute away.

---

## 6. Final result: CV vs. leaderboard gap (0.9493 vs. 0.949)

06_Final_Submission confirmed the engineered feature set on full CV, ran a
light hyperparameter pass (depth/learning_rate), and trained CatBoost on
100% of the training data. Final CV balanced accuracy: **0.94930**. Public
leaderboard score after submission: **0.949** (later improved to 0.94984
with seed averaging, see #9).

Why the gap, and why it's expected here (not a bug):
- CV score is measured on train, split into folds — leaderboard is measured
  on genuinely unseen test data. Some gap between the two is always normal.
- The final model in 06 was trained on 100% of train data (no held-out
  fold), while the CV estimate is necessarily based on models trained on
  ~80% of the data per fold — the CV number is a slightly
  pessimistic-but-close estimate, not a promise.
- A ~0.004 point gap (0.9493 → 0.949) is an EXTREMELY tight match — much
  closer than typical, and nowhere near the earlier 0.965-vs-0.855 gap
  that turned out to be a real metric mismatch bug (see #3). No further
  debugging needed for a gap this size — if anything, this tight a match
  confirms the pipeline has no leakage left.

Final pipeline summary:

| Stage | Balanced accuracy (CV) | Leaderboard |
|---|---|---|
| Baseline preprocessing only | 0.859 (majority-class floor) | — |
| + class weighting (CatBoost) | 0.9086 | 0.9059 |
| + missing-value flags (all columns) | 0.9492 | — |
| + light hyperparameter tuning (final) | 0.94930 | 0.949 |
| + seed averaging (N=3) | — | **0.94984** |
---

## 7. Re-tested dropped features on the new baseline — still no improvement

Re-tested `stress_sleep_interaction`, `bmi_category`, `lifestyle_score` on
the 0.9492 baseline (after all missing flags), plus a new
`total_missing_count` feature. None improved OOF balanced accuracy beyond
+0.001.

**Takeaway:** tree-based models (CatBoost) already learn feature
interactions automatically through successive splits — manually engineered
interaction/composite features add little once the base signal (here: the
missing-value pattern) is already captured. Manual feature engineering has
diminishing returns once a tree model has strong signal to split on.

---

## 8. Cross-family ensemble — also didn't beat CatBoost alone

Tried CatBoost + Logistic Regression + MLP (genuinely different model
families, unlike the same-family tree ensemble in #4). OOF-weighted
(power-4) combination scored 0.94711 vs. CatBoost alone 0.94910 — still
lower.

**Why:** the dominant signal in the engineered feature set is the binary
`_is_missing` flags — a near-linear signal that Logistic Regression and MLP
pick up on just as well as CatBoost, so their errors ended up correlated
with CatBoost's anyway (ensemble weights came out nearly balanced: 0.39 /
0.33 / 0.28, meaning all 3 models scored similarly on this data). Model-
family diversity only helps when the dominant signal actually needs
different model mechanics to capture — that wasn't the case here.

**Decision: CatBoost alone remains final.** Kept in `06_Final_Submission.ipynb`
section 7 for documentation, clearly marked as an experiment that wasn't
used for the official submission.

---

## 9. Seed averaging — small final gain, project wrap-up

Trained the tuned CatBoost model with multiple random seeds and averaged
`predict_proba` across them (reduces variance from a single initialization).

| N_SEEDS | Public leaderboard |
|---|---|
| 3 | 0.94984 |
| 7 | 0.94979 |

Difference between 3 and 7 seeds is within noise (0.00005) — **kept
`N_SEEDS = 3`**, cheaper for virtually identical result.

**Project wrap-up — final pipeline summary:**
| Stage | Balanced accuracy (CV) | Leaderboard |
|---|---|---|
| Baseline preprocessing only | 0.859 (majority-class floor) | — |
| + class weighting (CatBoost) | 0.9086 | 0.9059 |
| + missing-value flags (all columns) | 0.9492 | — |
| + light hyperparameter tuning | ~0.953 | 0.949 |
| + seed averaging (N=3), final | — | **0.94984** |

**What was tried and explicitly ruled out (documented, not just skipped):**
- Custom class weight grids beyond `'balanced'` (#4)
- Same-family tree ensemble — RandomForest + CatBoost + LightGBM (#4)
- Manual interaction/composite features on the post-missing-flag baseline (#7)
- Cross-family ensemble — CatBoost + Logistic Regression + MLP (#8)

**Project considered complete at this point.** Remaining possible gains
(deeper Optuna search, probability threshold tuning, stacking, target
encoding) are lower-ROI and left as "next steps" below rather than pursued
further — diminishing returns relative to time invested, and the
project's main goal (a documented, defensible ML pipeline for a portfolio)
is met.

---

## Next steps (not pursued further, left for reference)

- Threshold/probability adjustment on `predict_proba` instead of raw `argmax`
- Deeper automated hyperparameter search (Optuna/random search over more
  trials — only a small manual grid was tried)
- Stacking (meta-model over CatBoost/Logistic/MLP outputs) instead of a
  fixed-weight average
- K-fold target encoding for categoricals instead of one-hot, as an
  alternative to the current encoding in `03_Preprocessing`
