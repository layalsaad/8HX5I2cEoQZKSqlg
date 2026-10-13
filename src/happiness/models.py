"""Candidate classifiers, answer encodings and hyperparameter grids."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from happiness.config import MAX_ANSWER, RANDOM_STATE

ENCODINGS = ("ordinal", "one_hot", "top_box")


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Keep only some DataFrame columns, so the question subset can be tuned like any
    other hyperparameter. None keeps every column."""

    def __init__(self, columns: tuple[str, ...] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.columns is None else X[list(self.columns)]


def top_box(X) -> np.ndarray:
    """1 where the customer gave the highest possible answer, else 0."""
    return (np.asarray(X) == MAX_ANSWER).astype(int)


def make_encoder(encoding: str):
    """How answers are presented to the model.

    ordinal: the raw 1-5 scale. one_hot: one indicator per answer level, so each level
    can have its own effect. top_box: whether the answer was the maximum.
    """
    if encoding == "ordinal":
        return "passthrough"
    if encoding == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if encoding == "top_box":
        return FunctionTransformer(top_box, feature_names_out="one-to-one")
    raise ValueError(f"Unknown encoding {encoding!r}, expected one of {ENCODINGS}")


@dataclass(frozen=True)
class Candidate:
    """A model, the hyperparameters worth tuning, and whether it needs scaled inputs."""

    model: BaseEstimator
    param_grid: dict[str, list] = field(default_factory=dict)
    scale: bool = False

    def pipeline(
        self, encoding: str = "ordinal", columns: tuple[str, ...] | None = None
    ) -> Pipeline:
        return Pipeline(
            [
                ("select", ColumnSelector(columns)),
                ("encode", make_encoder(encoding)),
                ("scale", StandardScaler() if self.scale else "passthrough"),
                ("model", clone(self.model)),
            ]
        )

    @property
    def search_grid(self) -> dict[str, list]:
        """param_grid keyed for the pipeline, ready for GridSearchCV."""
        return {f"model__{name}": values for name, values in self.param_grid.items()}


def get_candidates(random_state: int = RANDOM_STATE) -> dict[str, Candidate]:
    """Models compared in the modeling notebook, from simplest to most flexible.

    Grids are deliberately small: with ~100 training rows per fold, large searches
    mostly fit noise.
    """
    return {
        "baseline": Candidate(DummyClassifier(strategy="most_frequent")),
        "naive_bayes": Candidate(
            # min_categories covers answer codes 0-5, so unseen levels in a fold cannot fail.
            CategoricalNB(min_categories=MAX_ANSWER + 1),
            {"alpha": [0.5, 1.0, 2.0]},
        ),
        "logistic_regression": Candidate(
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1.0, 10.0]},
            scale=True,
        ),
        "knn": Candidate(
            KNeighborsClassifier(),
            {"n_neighbors": [5, 11, 21], "weights": ["uniform", "distance"]},
            scale=True,
        ),
        "svm": Candidate(
            SVC(random_state=random_state),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1]},
            scale=True,
        ),
        "decision_tree": Candidate(
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [1, 2, 3, 4], "min_samples_leaf": [1, 5]},
        ),
        "random_forest": Candidate(
            RandomForestClassifier(n_estimators=50, random_state=random_state),
            {"max_depth": [3, None], "min_samples_leaf": [1, 5]},
        ),
        "gradient_boosting": Candidate(
            GradientBoostingClassifier(random_state=random_state),
            {"max_depth": [1, 2], "learning_rate": [0.05, 0.1]},
        ),
    }
