"""Cross-validation helpers that give honest performance estimates on small data."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from happiness.config import CV_REPEATS, CV_SPLITS, RANDOM_STATE
from happiness.models import Candidate

SCORING = ("accuracy", "balanced_accuracy", "roc_auc", "f1")


def repeated_cv(random_state: int = RANDOM_STATE) -> RepeatedStratifiedKFold:
    """Outer evaluation splits: CV_SPLITS folds, repeated CV_REPEATS times."""
    return RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=random_state
    )


def inner_cv(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """Tuning splits used inside each outer training fold."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=random_state)


def tuned(
    candidate: Candidate, encoding: str = "ordinal", random_state: int = RANDOM_STATE
) -> BaseEstimator:
    """An estimator that picks its own hyperparameters with an inner CV when fitted."""
    pipeline = candidate.pipeline(encoding)
    if not candidate.param_grid:
        return pipeline
    return GridSearchCV(
        pipeline, candidate.search_grid, cv=inner_cv(random_state), scoring="accuracy"
    )


def fold_scores(cv_results: dict) -> pd.DataFrame:
    """Test scores from sklearn's cross_validate output, one row per fold."""
    return pd.DataFrame({metric: cv_results[f"test_{metric}"] for metric in SCORING})


def nested_cv_scores(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str = "ordinal",
    cv=None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Score the full tune-then-fit procedure on outer folds it never saw.

    Hyperparameters are chosen inside each outer training fold, so the scores are not
    inflated by tuning on the evaluation data. Returns one row per outer fold.
    """
    results = cross_validate(
        tuned(candidate, encoding),
        X,
        y,
        cv=cv if cv is not None else repeated_cv(),
        scoring=SCORING,
        n_jobs=n_jobs,
    )
    return fold_scores(results)


def summarize(scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean and standard deviation of every metric per model, best accuracy first."""
    summary = pd.concat(
        {name: df.agg(["mean", "std"]).unstack() for name, df in scores.items()}, axis=1
    ).T
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values("accuracy_mean", ascending=False)


def corrected_ttest(
    scores_a: np.ndarray, scores_b: np.ndarray, n_splits: int = CV_SPLITS
) -> tuple[float, float]:
    """Compare two models evaluated on the same repeated k-fold splits.

    Uses the Nadeau & Bengio correction: CV training sets overlap, so a plain t-test
    on fold scores is overconfident. Returns (mean difference a - b, two-sided p-value).
    """
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    n = len(diff)
    mean, var = diff.mean(), diff.var(ddof=1)
    if var == 0:
        return float(mean), 1.0 if mean == 0 else 0.0
    test_train_ratio = 1 / (n_splits - 1)
    t_stat = mean / np.sqrt(var * (1 / n + test_train_ratio))
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return float(mean), float(p_value)
