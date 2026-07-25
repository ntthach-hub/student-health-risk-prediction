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

## 4. Class weighting to address minority-class underprediction

**Problem:** confusion matrix showed most errors were minority classes
(`fit`, `unhealthy`) being misclassified as the majority class (`at-risk`):
~20% of true `fit` rows and ~21.5% of true `unhealthy` rows were predicted
as `at-risk`. Direct confusion between `fit` and `unhealthy` was almost
nonexistent, suggesting the underlying signal is real — the issue was the
model defaulting to the majority class on borderline cases.

**Fix:** added `class_weight='balanced'` (RandomForest, LightGBM) /
`auto_class_weights='Balanced'` (CatBoost) to upweight minority classes
during training.

**Result:** OOF balanced accuracy improved from ~0.86 to ~0.90.

**Trade-off to watch:** class weighting can push some borderline `at-risk`
cases the other way (misclassified as `fit`/`unhealthy` instead). Net effect
on balanced accuracy was positive here, but worth re-checking the confusion
matrix after any further weight tuning.

---

## Next steps under consideration

- Tune custom class weights (not just `'balanced'`) via CV grid search
- Feature engineering targeted at the `at-risk` boundary (composite health
  score, `stress_level` x `sleep_duration` interaction, missing-value flags
  for `stress_level`/`sleep_duration`/`sleep_quality` given the suspected
  MNAR pattern found in `01_EDA_Basic`)
- Threshold/probability adjustment on `predict_proba` instead of raw
  `argmax`
- Ensemble of RandomForest + CatBoost + LightGBM predictions
