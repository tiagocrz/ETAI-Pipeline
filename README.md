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
Decision Tree[cite: 13]
**Parameters:**
  *criterion*: "gini"[cite: 13]
  *splitter*: "best"[cite: 13]
  *max_depth*: 5[cite: 13]
  *imputer*: KNNImputer(n_neighbors=5)[cite: 13]

#### Results Comparison

| Pipeline Configuration | Model | Imputation Strategy | Train Acc | Test Acc | Gap | Macro F1 | FPR (Afr-Am / Cauc) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Week 2 (Naive Baseline)** | Logistic Regression | Drop NA (Complete Case) | 0.679 | 0.681 | -0.002 | — | —[cite: 9] |
| **Week 3 (Class Standard)** | Logistic Regression | Median / Mode | 0.675 | 0.658 | +0.018 | 0.64 | 0.28 / 0.14[cite: 10] |
| **Week 3 (Class Standard)** | Decision Tree (depth=5) | Median / Mode | 0.684 | 0.665 | +0.020 | 0.65 | 0.28 / 0.16[cite: 11] |
| **Week 3 (KNN Imputer)** | Logistic Regression | KNN (k=5) / Mode | 0.671 | 0.652 | +0.018 | 0.64 | 0.28 / 0.16[cite: 12] |
| **Week 3 (KNN Imputer)** | Decision Tree (depth=5) | KNN (k=5) / Mode | **0.685** | **0.683** | **+0.003** | **0.68** | 0.34 / 0.18[cite: 13] |

#### Key Comparisons

* **With vs. Without Preprocessing:** Week 2 test accuracy (0.681) was artificially inflated due to `dropna()` complete-case truncation, which discarded records with non-random missingness (MNAR) tied to `age_cat`. Evaluating on the full distribution of 7,214 cleaned records establishes a realistic, unbiased test accuracy baseline of 0.658 for Logistic Regression.
* **Logistic Regression vs. Decision Tree:** Constraining Decision Tree depth to 5 resolved the overfitting observed in Week 2. Decision Trees outperformed Logistic Regression across all Week 3 runs by directly capturing non-linear interactions between offense count variables and target-encoded categoricals.
* **Class Standard (Median) vs. KNN Imputation:**
  * *Logistic Regression:* KNN imputation slightly decreased test accuracy (0.658 $\to$ 0.652) and Macro F1 (0.64 $\to$ 0.63) by introducing localized feature variance into linear coefficient estimation[cite: 10, 12].
  * *Decision Tree:* KNN imputation produced the top pipeline performance, achieving 0.683 test accuracy, 0.68 Macro F1, and reducing the generalization gap to +0.003.




## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA & Leak-Safe Preprocessing | Besides the changes seen in class (statistical missingness tests, domain rule checks, duplicate filtering, and multicollinearity removal (`src/data_diagnostics.py`); a leak-safe `ColumnTransformer` with target encoding, scaling, MNAR indicators); I added `KNNImputer` (`src/preprocessing.py`). Decision Tree with KNN imputation achieved the highest accuracy (0.683). |

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

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
