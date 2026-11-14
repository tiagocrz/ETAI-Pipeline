# Baseline Predictive Pipeline -- ETAI

This repository contains the codebase and implementation developed in class for the Exploratory Topics in Artifitial Intelligence course that is part of the MDSAA @NOVAIMS

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary.

| Field | Details |
| :--- | :--- |
| **Author** | Tiago da Cruz |
| **Student ID** | 20231682 |
| **Institution** | NOVA IMS |
| **Program** | Master's in Data Science and Advanced Analytics with a specialization in Artificial Intelligence and Computational Systems |
| **Curricular Unit** | Exploratory Topics in Artificial Intelligence (ETAI) |


## Weekly Results
### Week 02:
**Current best model:**
Logistic Regression <br>
**Parameters**:<br>
  *max_iter*: 5000<br>
  *solver*: "liblinear"<br>

This model achieves better accuracy than a Decision Tree; The DT has a tendency to overfit, even when defining a max_depth of 5.
The max_iter parameter was increased to 5000, as with 1000 the model was not converging.<br>
This version also has better accuracy than a LR with the default solver, and with the newton-cholesky solver. The reason for this I do not yet know.
<br>

**Results:**<br>
Train accuracy: 0.679<br>
Test accuracy:  0.681

### Week 03:
**Current best model:**
Decision Tree
**Parameters:**
  *criterion*: "gini"
  *splitter*: "best"
  *max_depth*: 5
  *imputer*: KNNImputer(n_neighbors=5)

#### Results Comparison

| Pipeline Configuration | Model | Imputation Strategy | Train Acc | Test Acc | Gap | Macro F1 | 
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | 
| **Week 3 (Class Standard)** | Logistic Regression | Median / Mode | 0.675 | 0.658 | +0.018 | 0.64 | 
| **Week 3 (Class Standard)** | Decision Tree (depth=5) | Median / Mode | 0.684 | 0.665 | +0.020 | 0.65 | 
| **Week 3 (KNN Imputer)** | Logistic Regression | KNN (k=5) / Mode | 0.671 | 0.652 | +0.018 | 0.64 | 
| **Week 3 (KNN Imputer)** | Decision Tree (depth=5) | KNN (k=5) / Mode | **0.685** | **0.683** | **+0.003** | **0.68** | 

#### Key Comparisons

* **With vs. Without Preprocessing:** Week 2 test accuracy (0.681) was artificially inflated due to `dropna()` complete-case truncation, which discarded records with non-random missingness (MNAR) tied to `age_cat`. Evaluating on the full distribution of 7,214 cleaned records establishes a realistic, unbiased test accuracy baseline of 0.658 for Logistic Regression.
* **Logistic Regression vs. Decision Tree:** Constraining Decision Tree depth to 5 resolved the overfitting observed in Week 2. Decision Trees outperformed Logistic Regression across all Week 3 runs by directly capturing non-linear interactions between offense count variables and target-encoded categoricals.
* **Class Standard (Median) vs. KNN Imputation:**
  * *Logistic Regression:* KNN imputation slightly decreased test accuracy (0.658 $\to$ 0.652) and Macro F1 (0.64 $\to$ 0.63) by introducing localized feature variance into linear coefficient estimation.
  * *Decision Tree:* KNN imputation produced the top pipeline performance, achieving 0.683 test accuracy, 0.68 Macro F1, and reducing the generalization gap to +0.003.


## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA & Leak-Safe Preprocessing | Besides the changes seen in class (statistical missingness tests, domain rule checks, duplicate filtering, and multicollinearity removal (`src/data_diagnostics.py`); a leak-safe `ColumnTransformer` with target encoding, scaling, MNAR indicators); I added `KNNImputer` (`src/preprocessing.py`). Decision Tree with KNN imputation achieved the highest accuracy (0.683). |
| 4 | Preprocessing inside the pipeline + cross-validation | Merged the class changes: locked 20% test set, stratified 5-fold CV of the whole pipeline (mean ± std and train–validation gap), cross-fitted `TargetEncoder` + robust scaling, `"nan"` placeholder fix, training-only de-duplication, dummy floor and random forest. *(Added late, 30 Oct 2026.)* |
| 5 | Hyperparameter tuning without leakage | Merged the class changes: Optuna tuning (`src/tuning.py`), nested CV as the honest estimate, best-trial score shown as an optimism check -- see "Hyperparameter tuning" below. *(Added late, 30 Oct 2026; scored with the W7 pipeline, ROC-AUC.)* |
| 6 | Ensembles: random forest, oblique random forest, AdaBoost | Merged the class changes: `oblique_random_forest` and `adaboost` in `src/model.py` with their search spaces, `corrected_paired_test()` -- see "Ensembles" below. *(Added late, 30 Oct 2026; scored with the W7 pipeline, ROC-AUC.)* |
| 7 | Gradient boosting + evaluating with probabilities | Merged the class changes: `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes`; 8 metrics with ROC-AUC as the selection metric; out-of-fold probabilities and confusion matrix; scoreboard in `results/runs.csv`. My addition: a search space for logistic regression (`C`, log scale), so every model except the dummy is tuned. Best: gradient boosting, nested CV ROC-AUC 0.728 ± 0.014. |
| 8 | XGBoost + feature selection | Merged the class changes: `xgboost` model (`XGBClassifier`) with its search space (incl. the new brakes `gamma`, `reg_alpha`, `min_child_weight`); feature selection as a learned Pipeline step (`VoteSelector`: a random forest, gradient boosting and an L1 logistic regression vote, features with ≥ `min_votes` votes are kept), re-fit in every fold, with `min_votes` tuned alongside the model and assessed by nested CV; cached preprocessing/selection during tuning; kept features printed; `feature_selection` column on the scoreboard. My addition: `compare_runs.py`, which lists the scoreboard and compares any two runs with the corrected paired t-test from the command line. Result: selection keeps 2 of 10 features with no significant loss -- see "Feature selection" below. |
| 9 | Beyond COMPAS: multiclass target | Option B: predict COMPAS's own risk category `score_text` (Low / Medium / High) with its own config, `config_multiclass.yaml` (`python main.py config_multiclass.yaml`). Removed every binary-only assumption so that the same code runs both tasks (the binary run gives exactly the same results): labels mapped to 0..k-1, full probability matrix with argmax predictions, macro / one-vs-rest metrics, a k × k confusion matrix with per-class AUC, a multiclass target encoder, a one-vs-rest L1 voter in `VoteSelector`, a config path argument in `main.py` and `--config` in `compare_runs.py`. Result: logistic regression with mild class weights reaches all three targets (ROC-AUC OvR 0.761, accuracy 0.632, macro F1 0.548) -- see "Beyond COMPAS" below. |

## Model evaluation

All runs: development set only (5,771 rows; the locked test set is never used), stratified 5-fold CV (seed 42), target encoding + robust scaling, median imputation, 30 Optuna trials. `class_weight: balanced` for logistic regression, decision tree, random forest and HistGradientBoosting.

### Selection metric

**ROC-AUC.** It measures how well the model ranks people by risk without fixing a 0.5 threshold, so it can be compared directly with COMPAS's own 1–10 score. Accuracy is misleading here: the dummy scores 0.549 accuracy by predicting that nobody reoffends, yet its ROC-AUC is 0.500.

### Scoreboard (ROC-AUC, from `results/runs.csv`)

| Model | Untuned: CV mean ± std | Tuned: nested CV mean ± std | Floor |
|---|---|---|---|
| XGBoost | 0.724 ± 0.017 | **0.729 ± 0.014** | 0.500 |
| Gradient boosting | 0.727 ± 0.015 | 0.728 ± 0.014 | 0.500 |
| HistGradientBoosting | 0.717 ± 0.020 | 0.728 ± 0.015 | 0.500 |
| Oblique random forest | 0.686 ± 0.018 | 0.727 ± 0.015 | 0.500 |
| AdaBoost | 0.723 ± 0.016 | 0.727 ± 0.014 | 0.500 |
| Random forest | 0.683 ± 0.018 | 0.726 ± 0.013 | 0.500 |
| XGBoost + feature selection | 0.721 ± 0.013 | 0.725 ± 0.012 | 0.500 |
| Logistic regression | 0.719 ± 0.014 | 0.718 ± 0.014 | 0.500 |
| Decision tree | 0.605 ± 0.013 | 0.718 ± 0.015 | 0.500 |
| Naive Bayes | 0.685 ± 0.011 | 0.705 ± 0.012 | 0.500 |
| Dummy (floor) | 0.500 ± 0.000 | – | 0.500 |
 
The top six models are within 0.003 of each other, well below the fold-to-fold std (~0.014), so no single one is clearly best.

**Comparing two runs.** `results/` isn't tracked in git, so the scoreboard (`results/runs.csv`) only exists locally, on the machine that ran the pipeline. `python compare_runs.py` lists the runs in it with their ids; `python compare_runs.py <idA> <idB>` compares two of them fold by fold with `corrected_paired_test()` (Nadeau & Bengio's corrected paired t-test), and only calls a winner when p < 0.05. Both runs must use the same data, metric and CV folds.


### Hyperparameter tuning

With `tuning.enabled: true`, Optuna (seeded TPE sampler) runs 30 trials over each model's search space in `config.yaml` (`tuning.search_spaces`). Each trial is scored by 5-fold CV of the **whole** pipeline, so preprocessing is re-fit inside every fold. The honest estimate is **nested CV**: for each of the 5 outer folds, the whole study runs on the outer-training rows only, and the winner is scored on the outer fold it never saw. The final hyperparameters come from one more study on all development rows. No validation row ever influences its own score, and the test set stays locked, so nothing leaks.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.605 ± 0.013 | 0.718 ± 0.015 | 0.719 | 0.000 | `max_depth=9, min_samples_leaf=100, criterion=gini` |
| Logistic regression | 0.719 ± 0.014 | 0.718 ± 0.014 | 0.719 | +0.001 | `C=0.0022` |
| Random forest | 0.683 ± 0.018 | 0.726 ± 0.013 | 0.728 | 0.000 | `max_depth=9, min_samples_leaf=42, max_features=0.5` |

*Optimism = mean inner best-trial score − nested CV score, as reported by the pipeline.*

The tree models gained most from tuning: the decision tree by +0.113 and the random forest by +0.043, both far beyond the fold-to-fold std, mainly by curbing overfitting (train–validation gap +0.109 → +0.014 and +0.120 → +0.018). Logistic regression did not gain (0.719 → 0.718), because its default is already near its optimum. Tuning moved the random forest from second to first, although its lead over logistic regression (0.008) is smaller than the std. I report the nested CV score, not the best trial, because the best trial is the maximum of many noisy estimates.

### Ensembles

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.719 ± 0.014 (+0.002) | 0.718 ± 0.014 (+0.002) | 0.719 | `C=0.0022` |
| Decision tree | 0.605 ± 0.013 (+0.109) | 0.718 ± 0.015 (+0.014) | 0.719 | `max_depth=9, min_samples_leaf=100, criterion=gini` |
| Random forest | 0.683 ± 0.018 (+0.120) | 0.726 ± 0.013 (+0.018) | 0.728 | `max_depth=9, min_samples_leaf=42, max_features=0.5` |
| Oblique random forest | 0.686 ± 0.018 (+0.149) | 0.727 ± 0.015 (+0.021) | 0.728 | `max_depth=13, min_samples_leaf=91, n_components=8` |
| AdaBoost | 0.723 ± 0.016 (+0.005) | 0.727 ± 0.014 (+0.015) | 0.729 | `max_depth=4, n_estimators=55, learning_rate=0.167` |

I would choose AdaBoost: it ties the forests (lead ≤ 0.001, far below the ~0.014 std) with 55 shallow trees instead of 200 deeper ones. The hyperparameters match how each model works. The forests keep fairly deep trees (depth 9–13), because averaging many trees cancels their variance. AdaBoost picks shallow trees (depth 4), because boosting reduces bias by adding weak learners in sequence, and deep trees would overfit the reweighted rows.


### Feature selection

XGBoost tuned twice (30 trials, nested CV), with and without the `VoteSelector` step; everything else identical.
 
| Run | Untuned: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | `min_votes` chosen | Features kept |
|---|---|---|---|---|
| XGBoost, no selection | 0.724 ± 0.017 (+0.034) | **0.729 ± 0.014** (+0.010) | – | all 10 |
| XGBoost + selection | 0.721 ± 0.013 (+0.027) | 0.725 ± 0.012 (+0.005) | 1 (outer folds: 1, 3, 2, 3, 2) | 2: `age`, `priors_count` |
 
*Untuned run with selection uses `min_votes: 2` from `config.yaml`.*
 
**Corrected paired test** (`python compare_runs.py <idA> <idB>`, with A = the run with selection and B = the run without, same 5 folds): mean difference −0.0038, selection better in 2/5 folds, t = −1.14, p = 0.317 → no significant difference.
 
The smaller model is as good: with only `age` and `priors_count`, XGBoost scores 0.725 against 0.729 with all 10 features, a difference smaller than the fold-to-fold std and not significant (p = 0.32). Selection also slightly narrows the train–validation gap (+0.005 vs +0.010). I would keep selection on: a model that uses two features is much easier to explain, and it drops `sex` and the juvenile counts at no measurable cost in ROC-AUC. The chosen `min_votes` varies across outer folds, so the exact threshold isn't stable, but age and prior offences clearly are relevant.



## Beyond COMPAS

Option B: same data, same cleaning, a new target. Instead of `two_year_recid`, the pipeline predicts **COMPAS's own risk category**, `score_text`: **Low** (54%), **Medium** (27%) or **High** (20%).

### How to run it

```bash
python main.py config_multiclass.yaml     # the multiclass task
python main.py                            # the binary task (config.yaml, unchanged)
python compare_runs.py --config config_multiclass.yaml            # list the multiclass runs
python compare_runs.py --config config_multiclass.yaml <idA> <idB>   # compare two of them
```

`config_multiclass.yaml` is a copy of `config.yaml` with these changes:
- `target: score_text`, and `class_labels: [Low, Medium, High]`, which fixes the class order;
- `decile_score` and `two_year_recid` are dropped from the features (see the table below);
- the metrics are `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro` and `roc_auc_ovr`, with `roc_auc_ovr` as the selection metric;
- its own results go to `results/multiclass/` and `results/runs_multiclass.csv`, so they never mix with the binary scoreboard (both tasks read the same data file);
- the model is logistic regression with mild class weights (see "Results").

### What broke, why, and how it was fixed

Every fix checks the number of classes: with 2 it follows the old binary path, so `python main.py` gives exactly the same results as before.

| What broke | Why | Fix |
|---|---|---|
| Leaky features | `decile_score` is the same score as a number (1–10), so it gives the answer away. `two_year_recid` is measured two years **after** the score was given, so it wasn't known when the score was given. | Both are in `drop_columns` in `config_multiclass.yaml`. |
| Text labels | XGBoost needs the labels to be 0, 1, 2, not `Low` / `Medium` / `High`. | `main.py` maps the target to 0..k-1 in the order of `data.class_labels` (the default is the sorted values, so binary 0/1 stays 0/1). |
| Probabilities | `predict_proba(X)[:, 1]` kept one column; with 3 classes that is P(Medium) only. | The out-of-fold array keeps the whole (rows × classes) matrix in `cross_validate_pipeline()` and `nested_cross_validate()`. `predict_from_proba()` applies the 0.5 threshold with 2 classes and takes the most probable class (argmax) with more. |
| Metrics | `precision`, `recall`, `f1` and `roc_auc` assume two classes. | The macro / one-vs-rest versions in the config (`get_scorer` already knows them, so `build_scorers` is unchanged). `fbeta_2` stays a binary-only metric. |
| Out-of-fold report | The 2 × 2 confusion matrix had hard-coded `0` / `1` labels. | A k × k matrix and classification report labelled with the class names, plus each class's one-vs-rest ROC-AUC. |
| Fairness report | The false positive rate needs one "positive" class, and its COMPAS comparison would compare COMPAS with itself (COMPAS's score is now the target). | Kept binary-only: skipped for the multiclass task, with a line in the report. With 3 classes the natural definition would be "people not truly High, but predicted High", per group. |
| Target encoder | `TargetEncoder(target_type="binary")` crashes on 3 classes. | `target_type="auto"`: one column per category for a binary target, one per category **per class** otherwise (3 categorical features → 9 columns, 16 features in all). The target encoder stays on for both tasks. |
| `VoteSelector` | The L1 voter read `lasso.coef_[0]`, but with 3 classes there is one row of coefficients per class (row 0 = Low only), and `liblinear` refuses 3 classes outright (sklearn 1.9). | With 3 classes the same L1 model is fitted one class vs the rest (`OneVsRestClassifier`). A feature gets the vote if it is non-zero for **at least one** class, i.e. it helps tell some class apart from the others. Binary keeps the direct fit. |
| Config path | `main.py` always loaded `config.yaml`. | `python main.py [CONFIG]` (default `config.yaml`), and `compare_runs.py --config`. The report header now shows the config file, the target and the classes. |

The class-imbalance setting needs care too: `class_weight` works for 3 classes, but XGBoost's `scale_pos_weight` (negatives / positives) is binary-only, so XGBoost runs here without any class weighting.

### Results

Development set only (5,771 rows; the locked test set is never used), stratified 5-fold CV, target encoding + robust scaling, 30 Optuna trials with nested CV for the tuned runs. Targets, all three in the same run: **ROC-AUC (OvR) ≥ 0.75, accuracy ≥ 0.63, macro F1 ≥ 0.53**.

| Model | Class weights | Tuned | ROC-AUC (OvR) | Accuracy | Macro F1 | All 3 targets |
|---|---|---|---|---|---|---|
| Dummy (always Low) | – | no | 0.500 ± 0.000 | 0.540 ± 0.001 | 0.234 ± 0.000 | |
| Logistic regression | balanced | no | 0.757 ± 0.008 | 0.594 ± 0.012 | 0.550 ± 0.009 | |
| Logistic regression | balanced | yes | 0.757 ± 0.008 | 0.593 ± 0.014 | 0.550 ± 0.014 | |
| Random forest | balanced | no | 0.724 ± 0.012 | 0.585 ± 0.011 | 0.527 ± 0.012 | |
| Random forest | balanced | yes | 0.754 ± 0.009 | 0.598 ± 0.014 | 0.548 ± 0.013 | |
| HistGradientBoosting | balanced | no | 0.727 ± 0.013 | 0.579 ± 0.008 | 0.527 ± 0.013 | |
| HistGradientBoosting | balanced | yes | 0.755 ± 0.009 | 0.593 ± 0.010 | **0.553 ± 0.010** | |
| XGBoost | none | no | 0.756 ± 0.011 | 0.625 ± 0.007 | 0.514 ± 0.011 | |
| XGBoost | none | yes | **0.764 ± 0.009** | 0.629 ± 0.004 | 0.526 ± 0.010 | |
| **Logistic regression** | **mild (1 : 1.3 : 1.5)** | no | 0.760 ± 0.008 | **0.635 ± 0.010** | 0.549 ± 0.013 | ✔ |
| **Logistic regression** | **mild (1 : 1.3 : 1.5)** | yes | 0.761 ± 0.008 | 0.632 ± 0.008 | 0.548 ± 0.011 | ✔ (`python main.py config_multiclass.yaml`) |

Every model clears the ROC-AUC target; the hard part is accuracy and macro F1 **together**. `balanced` weights raise macro F1 but drop accuracy to ~0.59, while no weights keep accuracy but rarely predict Medium. Mild weights (1 : 1.3 : 1.5) sit in between, and logistic regression with them reaches all three targets. The weights were picked on the same development folds, so the scores are slightly optimistic.

Tuned XGBoost has the best ROC-AUC (significantly better than balanced logistic regression and tuned HistGradientBoosting, p < 0.02), but only by ~0.01. Plain logistic regression is almost as good.

### Which class is hardest, and why?

**Medium**, in every model: ROC-AUC 0.61–0.67 and F1 0.28–0.40, against ~0.80 AUC for Low and High. Age and prior offences push risk in one direction, so they separate the two ends well, but nothing points specifically to the middle. Medium overlaps with both neighbours and is a minority class, so uncertain cases fall back to Low (773 of 1,531 truly Medium people in the selected model).

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── compare_runs.py          # compares two runs on the scoreboard with the corrected paired t-test (week 8)
├── config.yaml               # all tunable settings live here
├── config_multiclass.yaml    # the same settings for the multiclass task (target: score_text) (week 9)
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction (incl. the oblique random forest and AdaBoost builders, week 6; XGBoost, week 8) + VoteSelector, the feature-selection step (week 8) + build_pipeline(): preprocessing + selection + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs; multiclass runs in results/multiclass/ + runs_multiclass.csv (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```


## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run.

**When `requirements.txt` changes** (a week adds a package -- e.g. week 5 adds `optuna`), reinstall once, with the venv active: `pip install -r requirements.txt`. Otherwise `python main.py` fails with `ModuleNotFoundError`. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Installing XGBoost

`xgboost` is in `requirements.txt`, so `pip install -r requirements.txt` (venv active) installs it. On its own: `pip install xgboost`. Check it worked, with the venv active:
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

**Windows:** nothing extra: `pip install xgboost` installs a ready-made version. If it fails to build or install, update pip first (`python -m pip install --upgrade pip`) and make sure your Python is 64-bit (`python -c "import struct; print(struct.calcsize('P') * 8)"` should print 64).

**macOS:** XGBoost needs the **OpenMP** library (for running on several CPU cores), which macOS doesn't ship. If `import xgboost` fails with `XGBoostError`, `Library not loaded: libomp.dylib` or similar:
1. install [Homebrew](https://brew.sh) if you don't have it (the command is on its home page);
2. run `brew install libomp`;
3. open a **new** terminal, activate the venv, and try the import again.

Works on both Apple Silicon (M1/M2/M3/M4) and Intel Macs. If you use conda instead of a venv, `conda install -c conda-forge xgboost` installs XGBoost together with its OpenMP library.

**Linux:** on most distributions `pip install xgboost` just works. Two things can go wrong:
- it downloads a very large extra package (`nvidia-nccl-...`, a few hundred MB) for NVIDIA GPUs. If you have no NVIDIA GPU, or the download is too slow, install the CPU-only version instead: `pip install xgboost-cpu` (the code is identical: still `import xgboost`). If you do this, replace `xgboost` with `xgboost-cpu` in your copy of `requirements.txt`, so a reinstall doesn't fetch the large package again.
- a very old distribution may have no ready-made version and try to compile it. Update pip first (`python -m pip install --upgrade pip`); recent releases need glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+).

**In a notebook (Jupyter / VS Code): `No module named 'xgboost'`, although you installed it.** The notebook runs on a different Python from the one `pip` installed into (e.g. you installed into the venv, but the notebook uses the system Python). Either:
- install from inside the notebook, into the Python it actually uses: run `%pip install xgboost` in a cell, then **restart the kernel**; or
- make your venv available as a kernel, once, with the venv active:
  ```bash
  pip install ipykernel
  python -m ipykernel install --user --name etai --display-name "ETAI (venv)"
  ```
  then choose **ETAI (venv)** as the notebook's kernel. The notebook then sees every package in `requirements.txt`.

To check which Python a notebook is using: `import sys; print(sys.executable)`. In the terminal: `which python` (macOS/Linux) or `where python` (Windows). They should point into the same `venv` folder.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py                            # binary task: config.yaml
python main.py config_multiclass.yaml     # multiclass task (week 9 -- see "Beyond COMPAS" above)
```

This loads `config.yaml` (or the config you pass), diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles), and prints, in order:
- the model, the selection metric and the CV design; the size of the development set and of the locked test set (which is **not** used)
- **Default hyperparameters**: the cross-validation score on the selection metric, the train-validation gap (how much better the model looks on rows it trained on: a large gap means it memorised) and the no-model floor
- with tuning on, **Tuning**: the nested cross-validation score (the honest score of the tuned model) and the chosen hyperparameters
- **All metrics**: mean ± std of every metric
- **Scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows), the features it keeps (with feature selection on) and where the full report was saved

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.