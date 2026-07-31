# Predicting Student Health Risk

Kaggle Playground Series S6E7: https://www.kaggle.com/competitions/playground-series-s6e7

Multi-class classification — predict `health_condition` (`fit` / `at-risk` / `unhealthy`)
for a student from lifestyle and health metrics.

**Competition metric: Balanced Accuracy** (mean recall across the 3 classes — not
plain accuracy). Target is imbalanced: `at-risk` ~86%, `unhealthy` ~8.4%, `fit` ~5.8%.

## Current status — project complete

Pipeline complete, 01 through 06. Final public leaderboard score: **0.94984**
(balanced accuracy), up from an unweighted/unengineered baseline of ~0.859.
See [`docs/decisions.md`](./docs/decisions.md) for the full story, including
2 preprocessing bugs found and fixed, the accuracy-vs-balanced-accuracy
metric mismatch, why missing-value flags were the single biggest
improvement (+3.75 points), and 3 things that were tried and explicitly
ruled out (custom class weights, same-family tree ensemble, cross-family
ensemble) with the reasoning for each.

## Folder structure

```
.
├── README.md
├── data/
│   ├── train.csv, test.csv, sample_submission.csv   # raw data, never modified
│   └── processed/
│       ├── processed_train.csv, processed_test.csv   # output of 03_Preprocessing
│       ├── feature_importance_rf.csv                 # output of 03_Preprocessing
│       └── engineered_train.csv, engineered_test.csv  # output of 05_Feature_Engineering
├── notebooks/
│   ├── 01_EDA_Basic.ipynb
│   ├── 02_Research_Questions.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Modeling.ipynb
│   ├── 05_Feature_Engineering.ipynb
│   └── 06_Final_Submission.ipynb
├── submission/
│   └── checkpoint_*.csv, submission_final.csv
└── docs/
    ├── notebooks.md      # what each notebook reads/writes/does
    └── decisions.md       # log of key decisions, bugs found, and why things changed
```

## Rule for the pipeline

Each notebook only reads the **output of the notebook right before it** (by
number) — never the raw data directly (except 01–03), and never a notebook
further down the chain. See `docs/notebooks.md` for the exact read/write map.

## Quick links

- [What each notebook does](./docs/notebooks.md)
- [Decision log — bugs found, metric fix, why things changed](./docs/decisions.md)
