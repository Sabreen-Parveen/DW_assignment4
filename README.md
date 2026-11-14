# Bias mitigation with Reweighing on Adult, German and COMPAS

INFOMDW Assignment 4 (Task 2). Evaluates the **Reweighing** pre-processing algorithm
(F. Kamiran and T. Calders, "Data preprocessing techniques for classification without
discrimination", *Knowledge and Information Systems* 33(1), 2012) using the AIF360
implementation `aif360.algorithms.preprocessing.Reweighing`.

## How to run

Python 3.9 to 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook reweighing_pipeline.ipynb   # then Kernel > Restart & Run All
```

or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace reweighing_pipeline.ipynb --ExecutePreprocessor.timeout=1800
```

A full run takes about 15 to 20 minutes on a laptop (most of it is the Random Forest and the
consistency measure on Adult). The datasets are downloaded automatically into `data/` on the
first run. If a download is blocked, place the files there by hand:

| File in `data/` | Source |
|---|---|
| `adult.data`, `adult.test` | https://archive.ics.uci.edu/dataset/2/adult |
| `german.data` | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| `compas-scores-two-years.csv` | https://github.com/propublica/compas-analysis |

## Parameter settings

| Setting | Value |
|---|---|
| Protected attribute (privileged group) | Adult: sex (Male); German: age (> 25); COMPAS: race (Caucasian, only African-American and Caucasian kept) |
| Favourable label | Adult: income > 50K; German: good credit; COMPAS: no recidivism within 2 years |
| Removed columns | Adult: `fnlwgt`; COMPAS: all COMPAS scores (only demographics, juvenile/prior counts, charge degree used) |
| Encoding | One-hot for categorical, standardisation for numeric, fitted on the training split only |
| Split | 70 / 30, stratified on label × protected group |
| Seeds | 0, 1, 2, 3, 4 (results are mean ± std) |
| Protected attribute as model input | Both settings, as in the paper: scenario A = not used (assumption A2), scenario B = used |
| Classifiers | Logistic regression (`max_iter=2000`, `C=1.0`); Random forest (`n_estimators=200`, `min_samples_leaf=5`) |
| Mitigation | AIF360 `Reweighing`, weights passed as `sample_weight` to the classifier |
| Consistency | AIF360 `consistency`, k = 5 nearest neighbours |
| Metrics | DP_Diff, DIR, consistency, accuracy, balanced accuracy, F1 |

All settings are in the first code cell of the notebook.

## Output

* `results/original_labels.csv`: fairness of the original labels (whole dataset, before any model)
* `results/test_metrics_mean_std.csv`: test-set metrics, baseline vs Reweighing
* `results/before_after.csv`: original labels vs baseline vs Reweighing
* `results/all_runs.csv`: every seed
* `figures/fairness_before_after.png`
