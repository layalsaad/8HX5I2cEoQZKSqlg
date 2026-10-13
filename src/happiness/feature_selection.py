"""Which survey questions are worth asking: subset search and question importance."""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.parallel import Parallel, delayed

from happiness.config import CV_SPLITS, FEATURES, RANDOM_STATE
from happiness.evaluation import SCORING, fold_scores, inner_cv, repeated_cv
from happiness.models import Candidate


def selection_models(random_state: int = RANDOM_STATE) -> dict[str, tuple[Candidate, str]]:
    """Models (with their answer encoding) used to choose which questions to keep.

    Hyperparameters are fixed so the search only chooses questions: it keeps the 63-subset
    search affordable and avoids tuning many knobs on ~100 rows. The shallow tree can use
    thresholds and one interaction between questions; logistic regression on top-box
    answers is an additive cross-check.
    """
    return {
        "decision_tree": (
            Candidate(
                DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=random_state)
            ),
            "ordinal",
        ),
        "logistic_regression": (Candidate(LogisticRegression(), scale=True), "top_box"),
    }


def all_subsets(features: list[str] = FEATURES) -> list[tuple[str, ...]]:
    """Every non-empty combination of questions, smallest first."""
    return [
        subset for size in range(1, len(features) + 1) for subset in combinations(features, size)
    ]


def fewest_questions_within_one_se(cv_results: dict) -> int:
    """GridSearchCV refit rule: the candidate with the fewest questions whose mean score is
    within one standard error of the best (ties go to the higher score).

    The standard error uses CV_SPLITS folds even for repeated CV, because repeats reuse the
    same customers and are not independent evidence.
    """
    means = np.asarray(cv_results["mean_test_score"])
    best = means.argmax()
    threshold = means[best] - cv_results["std_test_score"][best] / np.sqrt(CV_SPLITS)
    sizes = np.array([len(params["select__columns"]) for params in cv_results["params"]])
    eligible = np.flatnonzero(means >= threshold)
    smallest = eligible[sizes[eligible] == sizes[eligible].min()]
    return int(smallest[means[smallest].argmax()])


def subset_search(
    candidate: Candidate,
    encoding: str = "ordinal",
    subsets: list[tuple[str, ...]] | None = None,
    cv=None,
    n_jobs: int | None = None,
) -> GridSearchCV:
    """An estimator that, when fitted, picks the question subset and hyperparameters by
    cross-validation, preferring fewer questions (one-standard-error rule)."""
    grid = {
        "select__columns": subsets if subsets is not None else all_subsets(),
        **candidate.search_grid,
    }
    return GridSearchCV(
        candidate.pipeline(encoding),
        grid,
        cv=cv if cv is not None else inner_cv(),
        scoring="accuracy",
        refit=fewest_questions_within_one_se,
        n_jobs=n_jobs,
    )


def score_subsets(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str = "ordinal",
    n_jobs: int | None = -1,
) -> tuple[pd.DataFrame, GridSearchCV]:
    """Repeated-CV accuracy of every question subset, with the best hyperparameters for each.

    Returns the per-subset table (best first) and the fitted search, whose best_params_ hold
    the one-standard-error choice. Picking the top of 63 subsets is itself a fit to these
    folds, so the top scores are optimistic; nested_subset_selection gives the honest number.
    """
    search = subset_search(candidate, encoding, cv=repeated_cv(), n_jobs=n_jobs).fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    results["questions"] = results["param_select__columns"].map("+".join)
    best_per_subset = results.loc[results.groupby("questions")["mean_test_score"].idxmax()]
    table = pd.DataFrame(
        {
            "questions": best_per_subset["questions"],
            "n_questions": best_per_subset["param_select__columns"].map(len),
            "accuracy_mean": best_per_subset["mean_test_score"],
            "accuracy_std": best_per_subset["std_test_score"],
        }
    )
    # On ties, list the subset with fewer questions first: extra questions the model ignores
    # produce identical scores.
    table = table.sort_values(["accuracy_mean", "n_questions"], ascending=[False, True])
    return table.reset_index(drop=True), search


def nested_subset_selection(
    candidate: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    encoding: str = "ordinal",
    subsets: list[tuple[str, ...]] | None = None,
    cv=None,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Honest estimate of the whole choose-questions-then-train procedure.

    Each outer training fold runs its own subset search; the outer test fold scores the
    result. Returns one row per outer fold: test scores and the questions that fold chose.
    """
    results = cross_validate(
        subset_search(candidate, encoding, subsets),
        X,
        y,
        cv=cv if cv is not None else repeated_cv(),
        scoring=SCORING,
        n_jobs=n_jobs,
        return_estimator=True,
    )
    scores = fold_scores(results)
    scores["questions"] = [
        "+".join(search.best_params_["select__columns"]) for search in results["estimator"]
    ]
    return scores


def selection_frequency(questions: pd.Series, features: list[str] = FEATURES) -> pd.Series:
    """Share of folds whose chosen subset includes each question."""
    return questions.str.get_dummies(sep="+").reindex(columns=features, fill_value=0).mean()


def _fold_importance(estimator, X, y, train, test, n_repeats, random_state) -> np.ndarray:
    model = clone(estimator).fit(X.iloc[train], y.iloc[train])
    result = permutation_importance(
        model,
        X.iloc[test],
        y.iloc[test],
        scoring="accuracy",
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return result.importances_mean


def permutation_importance_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = -1,
) -> pd.DataFrame:
    """Drop in held-out accuracy when one question's answers are shuffled.

    The model is fitted on each training fold and probed on the matching test fold, so this
    measures what each question contributes to predicting new customers. Returns one row
    per fold and one column per question.
    """
    cv = cv if cv is not None else repeated_cv()
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fold_importance)(estimator, X, y, train, test, n_repeats, random_state)
        for train, test in cv.split(X, y)
    )
    return pd.DataFrame(rows, columns=X.columns)
