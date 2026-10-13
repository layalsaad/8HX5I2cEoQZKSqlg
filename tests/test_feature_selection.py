import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from happiness.config import FEATURES, TARGET
from happiness.feature_selection import (
    all_subsets,
    fewest_questions_within_one_se,
    nested_subset_selection,
    permutation_importance_cv,
    selection_frequency,
    selection_models,
    subset_search,
)
from happiness.models import Candidate

SUBSETS = [("X1",), ("X3",), ("X1", "X5"), ("X1", "X3", "X5")]


@pytest.fixture
def candidate():
    return Candidate(LogisticRegression(), {"C": [0.1, 1.0]}, scale=True)


def cv_results(subsets, means, stds):
    return {
        "params": [{"select__columns": subset} for subset in subsets],
        "mean_test_score": np.array(means),
        "std_test_score": np.array(stds),
    }


@pytest.mark.parametrize("name", ["decision_tree", "logistic_regression"])
def test_selection_models_fit_on_a_question_subset(survey_df, name):
    candidate, encoding = selection_models()[name]
    X, y = survey_df[FEATURES], survey_df[TARGET]

    model = candidate.pipeline(encoding, columns=("X1", "X5")).fit(X, y)

    assert len(model.predict(X)) == len(X)


def test_all_subsets_lists_every_combination_smallest_first():
    subsets = all_subsets()

    assert len(subsets) == 2 ** len(FEATURES) - 1
    assert subsets[0] == ("X1",)
    assert subsets[-1] == tuple(FEATURES)
    assert [len(s) for s in subsets] == sorted(len(s) for s in subsets)


def test_one_se_rule_prefers_fewer_questions_within_one_standard_error():
    # Best: 0.70 with std 0.05 over 5 folds -> threshold 0.70 - 0.05 / sqrt(5) = 0.678.
    results = cv_results(
        [("X1", "X3", "X5"), ("X1", "X5"), ("X1",), ("X5",)],
        means=[0.70, 0.695, 0.68, 0.69],
        stds=[0.05, 0.05, 0.05, 0.05],
    )

    assert fewest_questions_within_one_se(results) == 3  # ("X5",): smallest size, higher score


def test_one_se_rule_keeps_best_when_nothing_simpler_is_close():
    results = cv_results([("X1",), ("X1", "X5")], means=[0.60, 0.70], stds=[0.01, 0.01])

    assert fewest_questions_within_one_se(results) == 1


def test_subset_search_selects_from_given_subsets_and_predicts_on_full_frame(survey_df, candidate):
    X, y = survey_df[FEATURES], survey_df[TARGET]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    search = subset_search(candidate, subsets=SUBSETS, cv=cv).fit(X, y)

    assert search.best_params_["select__columns"] in SUBSETS
    assert len(search.predict(X)) == len(X)


def test_nested_subset_selection_reports_chosen_questions_per_fold(survey_df, candidate):
    X, y = survey_df[FEATURES], survey_df[TARGET]
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)

    scores = nested_subset_selection(candidate, X, y, subsets=SUBSETS, cv=cv, n_jobs=1)

    assert len(scores) == 3
    assert scores["questions"].isin(["+".join(s) for s in SUBSETS]).all()
    assert scores["accuracy"].between(0, 1).all()


def test_selection_frequency_counts_folds_containing_each_question():
    questions = pd.Series(["X1", "X1+X5", "X5+X6"])

    frequency = selection_frequency(questions)

    assert list(frequency.index) == FEATURES
    assert frequency["X1"] == pytest.approx(2 / 3)
    assert frequency["X6"] == pytest.approx(1 / 3)
    assert frequency["X2"] == 0


def test_permutation_importance_ranks_informative_questions_higher(survey_df, candidate):
    X, y = survey_df[FEATURES], survey_df[TARGET]
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=0)

    importance = permutation_importance_cv(candidate.pipeline(), X, y, cv=cv, n_repeats=5, n_jobs=1)

    assert list(importance.columns) == FEATURES
    assert len(importance) == 6
    signal = importance[["X1", "X5"]].mean().mean()
    noise = importance[["X2", "X3", "X4", "X6"]].mean().mean()
    assert signal > noise
