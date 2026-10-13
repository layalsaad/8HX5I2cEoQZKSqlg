import numpy as np
import pytest

from happiness.config import FEATURES, TARGET
from happiness.models import ENCODINGS, ColumnSelector, get_candidates, make_encoder, top_box

CANDIDATES = get_candidates()


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("name", CANDIDATES)
def test_every_candidate_fits_and_predicts(survey_df, name, encoding):
    X, y = survey_df[FEATURES], survey_df[TARGET]

    predictions = CANDIDATES[name].pipeline(encoding).fit(X, y).predict(X)

    assert set(predictions) <= {0, 1}
    assert len(predictions) == len(X)


def test_search_grid_targets_the_model_step():
    grid = CANDIDATES["logistic_regression"].search_grid

    assert list(grid) == ["model__C"]


def test_pipeline_returns_fresh_model_each_time():
    candidate = CANDIDATES["decision_tree"]

    assert candidate.pipeline()["model"] is not candidate.pipeline()["model"]


def test_pipeline_uses_only_selected_columns(survey_df):
    X, y = survey_df[FEATURES], survey_df[TARGET]
    model = CANDIDATES["logistic_regression"].pipeline(columns=("X1", "X5")).fit(X, y)

    shuffled = X.assign(X2=X["X2"].sample(frac=1, random_state=0).to_numpy(), X3=1)

    np.testing.assert_array_equal(model.predict(X), model.predict(shuffled))


def test_column_selector_keeps_all_columns_by_default(survey_df):
    X = survey_df[FEATURES]

    assert ColumnSelector().fit(X).transform(X) is X


def test_top_box_flags_only_the_maximum_answer():
    answers = np.array([[1, 4, 5], [5, 5, 3]])

    np.testing.assert_array_equal(top_box(answers), [[0, 0, 1], [1, 1, 0]])


def test_unknown_encoding_raises():
    with pytest.raises(ValueError, match="Unknown encoding"):
        make_encoder("binary")
