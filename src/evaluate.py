"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline
on several metrics, out-of-fold probabilities and reports, a fairness check and a paired
comparison of two models. The locked test set is never used here.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer,
                             roc_auc_score)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold class probabilities, shape (n_rows, n_classes), for every row,
        from the fold model that did not train on it. `cv` must be a partition (each row
        validated exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty((len(X), pd.Series(y).nunique()))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])
    return fold_scores, oof_proba


def predict_from_proba(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Class predictions from a (n_rows, n_classes) probability matrix: with two classes, class 1
    if P(1) >= `threshold` (a threshold you can move); with more, the most probable class.
    """
    if proba.shape[1] == 2:
        return (proba[:, 1] >= threshold).astype(int)
    return proba.argmax(axis=1)


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, oof_proba, class_names: list = None, threshold: float = 0.5) -> str:
    """
    Confusion matrix and classification report of the out-of-fold predictions, as text. With
    more than two classes, predictions are the most probable class (no threshold) and each
    class's one-vs-rest ROC AUC is added.
    """
    n_classes = oof_proba.shape[1]
    class_names = class_names or [str(c) for c in range(n_classes)]
    y_true = np.asarray(y_true)
    y_pred = predict_from_proba(oof_proba, threshold)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(n_classes)),
                      index=[f"true {name}" for name in class_names],
                      columns=[f"predicted {name}" for name in class_names])
    header = (f"threshold {threshold}" if n_classes == 2 else "prediction = most probable class")
    text = (f"Out-of-fold predictions, development set ({header}):\n\n{cm.to_string()}\n\n"
            f"{classification_report(y_true, y_pred, labels=range(n_classes), target_names=class_names, zero_division=0)}")
    if n_classes > 2:
        text += "\nOne-vs-rest ROC AUC per class (that class vs all the others):\n"
        for c, name in enumerate(class_names):
            text += f"  {name:<12s} {roc_auc_score(y_true == c, oof_proba[:, c]):.3f}\n"
    return text


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5) -> str:
    """
    Binary targets only. False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = predict_from_proba(oof_proba, threshold)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)
