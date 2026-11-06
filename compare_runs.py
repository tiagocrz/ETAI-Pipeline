"""
Compare two runs from the scoreboard (results/runs.csv) with the corrected paired t-test.

Usage, from the project root with the venv active:
    python compare_runs.py                    # list the runs on the scoreboard
    python compare_runs.py RUN_A RUN_B        # compare two runs by their run id

Both runs must use the same data, metric and CV folds (the same cv section in config.yaml),
otherwise their per-fold scores can't be paired.
"""
import sys

import pandas as pd
import yaml

from src.evaluate import corrected_paired_test


def load_runs(config: dict) -> pd.DataFrame:
    path = config.get("output", {}).get("scoreboard", "results/runs.csv")
    return pd.read_csv(path, dtype={"run": str})


def describe(run: pd.Series) -> str:
    selection = run.get("feature_selection")
    selection = selection if isinstance(selection, str) else "off"
    return (f"{run['run']}  {run['model']}  tuned={run['tuned']}  feature selection: {selection}  "
            f"{run['metric']} = {run['mean']:.3f} ± {run['std']:.3f}")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    runs = load_runs(config)

    if len(sys.argv) != 3:
        print("Runs on the scoreboard (best first):\n")
        for _, run in runs.sort_values("mean", ascending=False).iterrows():
            print("  " + describe(run))
        print("\nUsage: python compare_runs.py RUN_A RUN_B")
        return

    run_a, run_b = sys.argv[1], sys.argv[2]
    missing = [r for r in (run_a, run_b) if r not in set(runs["run"])]
    if missing:
        raise SystemExit(f"Run(s) not in the scoreboard: {missing}. Run without arguments to list them.")
    a = runs[runs["run"] == run_a].iloc[0]
    b = runs[runs["run"] == run_b].iloc[0]
    if a["data"] != b["data"] or a["metric"] != b["metric"]:
        raise SystemExit("The two runs used different data or metrics: their fold scores can't be paired.")

    folds = [c for c in runs.columns if c.startswith("fold_")]
    n_splits = config["cv"]["n_splits"]
    result = corrected_paired_test(a[folds], b[folds], test_train_ratio=1 / (n_splits - 1))

    print(f"A: {describe(a)}")
    print(f"B: {describe(b)}\n")
    print(f"Corrected paired t-test ({n_splits} folds, A - B):")
    print(f"  mean difference  {result['mean_diff']:+.4f}")
    print(f"  folds A wins     {result['wins']}")
    print(f"  t                {result['t']:.2f}")
    print(f"  p-value          {result['p_value']:.3f}")
    verdict = ("the difference is significant (p < 0.05)" if result["p_value"] < 0.05
               else "no significant difference (p >= 0.05): don't call a winner")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()