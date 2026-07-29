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
In production, a model can't "peek" at the full incoming batch to compute a
median before predicting on it — the same rule applies here. Fitting
separately on test broke train/test distribution consistency.

**Fix:** `fit_transform` on train, `transform` only on test, for both the
numeric and categorical imputers.

---

## 2. Preprocessing bug: `id` leaking into features

**Problem:** `train.csv`/`test.csv` were read without `index_col="id"`, so
`id` was treated as a normal feature column and flowed through the whole
pipeline into the model.

**Why it matters:** with synthetic competition data, `id` can carry a
spurious correlation with the target if rows were generated in blocks by
class. A model can latch onto that pattern in-sample without it generalizing
to the real test set.

**Fix:** read both CSVs with `index_col="id"`.

---

## 3. Metric mismatch: accuracy vs. balanced accuracy

**Problem:** CV was scored with plain `accuracy_score`, giving ~0.965,
while the Kaggle public leaderboard showed ~0.855 for the same model — a
gap large enough to suspect a serious bug or data leak.

**Investigation:** checked the target class balance
(`at-risk` ~86%, `unhealthy` ~8.4%, `fit` ~5.8%) and the per-class
classification report — macro-average recall was ~0.86, almost exactly
matching the leaderboard score. Confirmed via the Kaggle "Evaluation" tab:
**the competition scores submissions on Balanced Accuracy**, not plain
accuracy.

**Why it matters:** with this much class imbalance, plain accuracy is
misleading — a model can score high just by favoring the majority class
(`at-risk`) without actually learning to separate the minority classes.

**Fix:** switched all CV scoring in `04_Modeling.ipynb` from
`accuracy_score` to `balanced_accuracy_score`, matching what Kaggle actually
grades.

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

## 6. Final result: CV vs. leaderboard gap (0.953 vs. 0.949)

**06_Final_Submission** confirmed the engineered feature set on full CV,
ran a light hyperparameter pass (depth/learning_rate), and trained CatBoost
on 100% of the training data. Final CV balanced accuracy: ~0.953. Public
leaderboard score after submission: **0.949**.

**Why the gap, and why it's expected here (not a bug):**
- CV score is measured on train, split into folds — leaderboard is measured
  on genuinely unseen test data. Some gap between the two is always normal.
- The final model in `06` was trained on 100% of train data (no held-out
  fold), while the CV estimate in section 4 is necessarily based on models
  trained on ~80% of the data per fold — the CV number is a slightly
  pessimistic-but-close estimate, not a promise.
- A ~0.4 point gap (0.953 → 0.949) is a normal and healthy amount of
  variance for a ~690k/~296k train/test split — nowhere near the earlier
  0.965-vs-0.855 gap that turned out to be a real metric mismatch bug (see
  #3). No further debugging needed for a gap this size.

**Final pipeline summary:**
| Stage | Balanced accuracy (CV) | Leaderboard |
|---|---|---|
| Baseline preprocessing only | 0.859 (majority-class floor) | — |
| + class weighting (CatBoost) | 0.9086 | 0.9059 |
| + missing-value flags (all columns) | 0.9492 | — |
| + light hyperparameter tuning (final) | ~0.953 | **0.949** |

---

## Next steps under consideration

- Threshold/probability adjustment on `predict_proba` instead of raw `argmax`
- Deeper hyperparameter search (currently only a small manual grid was tried)
- Revisit whether `stress_level × sleep_duration` or other interactions help
  now that the missing-flag features are in the mix (they were tested
  before the flags were added, so the combined effect wasn't checked)
