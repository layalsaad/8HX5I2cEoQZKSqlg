import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold

from happiness.config import FEATURES, TARGET
from happiness.evaluation import SCORING, corrected_ttest, nested_cv_scores, summarize
from happiness.models import get_candidates


@pytest.fixture
def small_cv():
    return RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)


def test_nested_cv_scores_one_row_per_outer_fold(survey_df, small_cv):
    candidate = get_candidates()["decision_tree"]

    scores = nested_cv_scores(
        candidate, survey_df[FEATURES], survey_df[TARGET], cv=small_cv, n_jobs=1
    )

    assert list(scores.columns) == list(SCORING)
    assert len(scores) == 6
    assert scores["accuracy"].between(0, 1).all()


def test_nested_cv_learns_signal_beyond_baseline(survey_df, small_cv):
    X, y = survey_df[FEATURES], survey_df[TARGET]
    candidates = get_candidates()

    model = nested_cv_scores(candidates["logistic_regression"], X, y, cv=small_cv, n_jobs=1)
    baseline = nested_cv_scores(candidates["baseline"], X, y, cv=small_cv, n_jobs=1)

    assert model["accuracy"].mean() > baseline["accuracy"].mean()


def test_summarize_sorts_by_mean_accuracy():
    scores = {
        "weak": pd.DataFrame({"accuracy": [0.5, 0.6], "f1": [0.4, 0.5]}),
        "strong": pd.DataFrame({"accuracy": [0.8, 0.7], "f1": [0.7, 0.8]}),
    }

    summary = summarize(scores)

    assert list(summary.index) == ["strong", "weak"]
    assert summary.loc["strong", "accuracy_mean"] == pytest.approx(0.75)
    assert {"accuracy_std", "f1_mean", "f1_std"} <= set(summary.columns)


def test_corrected_ttest_detects_consistent_improvement():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.55, 0.05, 50)

    diff, p_value = corrected_ttest(baseline + 0.15, baseline)

    assert diff == pytest.approx(0.15)
    assert p_value < 0.01


def test_corrected_ttest_is_more_conservative_than_plain_ttest():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.62, 0.08, 50), rng.normal(0.60, 0.08, 50)

    _, p_value = corrected_ttest(a, b)

    assert p_value > ttest_rel(a, b).pvalue


def test_corrected_ttest_identical_scores():
    scores = np.full(50, 0.6)

    assert corrected_ttest(scores, scores) == (0.0, 1.0)
