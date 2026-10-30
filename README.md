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




## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction (incl. the oblique random forest and AdaBoost builders, week 6) + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

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

## Model evaluation

All runs: development set only (5,771 rows; the locked test set is never used), stratified 5-fold CV (seed 42), target encoding + robust scaling, median imputation, 30 Optuna trials. `class_weight: balanced` for logistic regression, decision tree, random forest and HistGradientBoosting.

### Selection metric

**ROC-AUC.** It measures how well the model ranks people by risk without fixing a 0.5 threshold, so it can be compared directly with COMPAS's own 1–10 score. Accuracy is misleading here: the dummy scores 0.549 accuracy by predicting that nobody reoffends, yet its ROC-AUC is 0.500.

### Scoreboard (ROC-AUC, from `results/runs.csv`)

| Model | Untuned: CV mean ± std | Tuned: nested CV mean ± std | Floor |
|---|---|---|---|
| Gradient boosting | 0.727 ± 0.015 | **0.728 ± 0.014** | 0.500 |
| HistGradientBoosting | 0.717 ± 0.020 | 0.728 ± 0.015 | 0.500 |
| Oblique random forest | 0.686 ± 0.018 | 0.727 ± 0.015 | 0.500 |
| AdaBoost | 0.723 ± 0.016 | 0.727 ± 0.014 | 0.500 |
| Random forest | 0.683 ± 0.018 | 0.726 ± 0.013 | 0.500 |
| Logistic regression | 0.719 ± 0.014 | 0.718 ± 0.014 | 0.500 |
| Decision tree | 0.605 ± 0.013 | 0.718 ± 0.015 | 0.500 |
| Naive Bayes | 0.685 ± 0.011 | 0.705 ± 0.012 | 0.500 |
| Dummy (floor) | 0.500 ± 0.000 | – | 0.500 |

The top five models are within 0.002 of each other, well below the fold-to-fold std (~0.014), so no single one is clearly best.

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
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles), and prints, in order:
- the model, the selection metric and the CV design; the size of the development set and of the locked test set (which is **not** used)
- **Default hyperparameters**: the cross-validation score on the selection metric, the train-validation gap (how much better the model looks on rows it trained on: a large gap means it memorised) and the no-model floor
- with tuning on, **Tuning**: the nested cross-validation score (the honest score of the tuned model) and the chosen hyperparameters
- **All metrics**: mean ± std of every metric
- **Scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows) and where the full report was saved

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.