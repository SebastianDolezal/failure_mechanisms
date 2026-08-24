"""Regression-based analyses: continuous semantic-mechanistic correspondence
(Section 32), incremental held-out prediction of semantic type from D_i
(Section 33), and the functional-validity model for intervention transfer
(Section 39).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, log_loss


def continuous_correspondence_regression(
    C_mech: np.ndarray,
    S_sem: np.ndarray,
    X_covariates: pd.DataFrame,
    cluster_groups: np.ndarray,
) -> dict:
    """C_ij ~ beta0 + beta1 * S_ij^sem + beta2 * X_ij, cluster-robust SEs by
    `cluster_groups` (e.g. the pair's underlying-problem id) since rows
    sharing a problem are not independent (Section 32)."""
    import statsmodels.api as sm

    df = X_covariates.copy()
    df["S_sem"] = S_sem
    df["C_mech"] = C_mech
    df = df.dropna()

    X = sm.add_constant(df.drop(columns=["C_mech"]))
    model = sm.OLS(df["C_mech"], X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups[df.index]})

    return {
        "beta_sem": float(result.params.get("S_sem", np.nan)),
        "se_sem": float(result.bse.get("S_sem", np.nan)),
        "p_sem": float(result.pvalues.get("S_sem", np.nan)),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
        "summary": result.summary().as_text(),
        "params": result.params.to_dict(),
    }


def incremental_prediction_test(
    X_covariates: np.ndarray,
    D_signatures: np.ndarray,
    labels: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Compares P(T_i | X_i) vs P(T_i | X_i, D_i) under stratified k-fold CV
    with multinomial logistic regression (Section 33). Reports macro-F1,
    balanced accuracy, and held-out cross-entropy for both models."""
    y = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def _cv_scores(X):
        f1s, baccs, ces = [], [], []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx])
            f1s.append(f1_score(y[test_idx], pred, average="macro"))
            baccs.append(balanced_accuracy_score(y[test_idx], pred))
            ces.append(log_loss(y[test_idx], proba, labels=clf.classes_))
        return {"macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(baccs)),
                "cross_entropy": float(np.mean(ces))}

    baseline = _cv_scores(X_covariates)
    mechanistic = _cv_scores(np.concatenate([X_covariates, D_signatures], axis=1))

    return {
        "baseline": baseline,
        "mechanistic": mechanistic,
        "delta_macro_f1": mechanistic["macro_f1"] - baseline["macro_f1"],
        "delta_balanced_accuracy": mechanistic["balanced_accuracy"] - baseline["balanced_accuracy"],
        "delta_cross_entropy": baseline["cross_entropy"] - mechanistic["cross_entropy"],
    }


def functional_validity_model(
    R_transfer: np.ndarray,
    C_AB: np.ndarray,
    same_category: np.ndarray,
    X_AB: pd.DataFrame,
    cluster_groups: np.ndarray,
) -> dict:
    """R_A->B ~ beta0 + beta1*C_AB + beta2*1[T_A=T_B] + beta3*X_AB
    (Section 39)."""
    import statsmodels.api as sm

    df = X_AB.copy()
    df["C_AB"] = C_AB
    df["same_category"] = same_category.astype(float)
    df["R_transfer"] = R_transfer
    df = df.dropna()

    X = sm.add_constant(df.drop(columns=["R_transfer"]))
    model = sm.OLS(df["R_transfer"], X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups[df.index]})

    return {
        "beta_C_AB": float(result.params.get("C_AB", np.nan)),
        "p_C_AB": float(result.pvalues.get("C_AB", np.nan)),
        "beta_same_category": float(result.params.get("same_category", np.nan)),
        "p_same_category": float(result.pvalues.get("same_category", np.nan)),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
        "summary": result.summary().as_text(),
    }
