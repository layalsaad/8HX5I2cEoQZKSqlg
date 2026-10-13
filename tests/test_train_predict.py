import json
import logging

import numpy as np
import pandas as pd
import pytest

from happiness.config import FEATURES, SELECTED_QUESTIONS, TARGET
from happiness.predict import load_model, predict
from happiness.predict import main as predict_main
from happiness.train import build_model, metadata_path, train
from happiness.train import main as train_main


@pytest.fixture
def model_path(survey_csv, tmp_path):
    path = tmp_path / "models" / "model.joblib"
    train(survey_csv, path)
    return path


def test_build_model_uses_selected_questions():
    assert build_model()["select"].columns == SELECTED_QUESTIONS


def test_train_saves_model_and_metadata(model_path, survey_df):
    metadata = json.loads(metadata_path(model_path).read_text())

    assert model_path.exists()
    assert metadata["questions"] == list(SELECTED_QUESTIONS)
    assert metadata["training_rows"] == len(survey_df)
    assert metadata["cross_validation"]["folds"] == 50
    assert 0 <= metadata["cross_validation"]["accuracy"]["mean"] <= 1
    assert len(metadata["data_sha256"]) == 64


def test_predict_adds_predictions_and_keeps_other_columns(model_path, survey_df):
    df = survey_df.drop(columns=TARGET).assign(customer_id=range(len(survey_df)))

    scored = predict(load_model(model_path), df)

    assert list(scored.columns) == [*df.columns, "prediction", "probability_happy"]
    assert scored["customer_id"].tolist() == df["customer_id"].tolist()
    assert set(scored["prediction"]) <= {0, 1}
    assert scored["probability_happy"].between(0, 1).all()
    assert ((scored["probability_happy"] > 0.5) == (scored["prediction"] == 1)).all()


def test_predict_needs_only_the_questions_the_model_uses(model_path, survey_df):
    shorter_survey = survey_df[list(SELECTED_QUESTIONS)]

    scored = predict(load_model(model_path), shorter_survey)

    assert len(scored) == len(shorter_survey)


def test_predict_ignores_an_empty_target_column(model_path, survey_df):
    df = survey_df.assign(**{TARGET: np.nan})

    assert len(predict(load_model(model_path), df)) == len(df)


def test_predict_rejects_invalid_answers(model_path, survey_df):
    survey_df.loc[0, SELECTED_QUESTIONS[0]] = 7

    with pytest.raises(ValueError, match="outside 1-5"):
        predict(load_model(model_path), survey_df)


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="happiness-train"):
        load_model(tmp_path / "missing.joblib")


def test_predict_command_exits_cleanly_on_invalid_input(model_path, tmp_path, capsys):
    bad_input = tmp_path / "bad.csv"
    pd.DataFrame({SELECTED_QUESTIONS[0]: [9]}).to_csv(bad_input, index=False)

    with pytest.raises(SystemExit) as exit_info:
        predict_main([str(bad_input), "--model", str(model_path)])

    assert exit_info.value.code == 1
    assert "error: Invalid survey data" in capsys.readouterr().err


def test_train_command_exits_cleanly_when_data_is_missing(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        train_main(["--data", str(tmp_path / "missing.csv")])

    assert exit_info.value.code == 1
    assert "error: Data file not found" in capsys.readouterr().err


def test_command_line_round_trip_reports_accuracy(survey_csv, tmp_path, caplog):
    model_file = tmp_path / "model.joblib"
    output = tmp_path / "predictions.csv"

    with caplog.at_level(logging.INFO):
        train_main(["--data", str(survey_csv), "--output", str(model_file)])
        predict_main([str(survey_csv), "--model", str(model_file), "--output", str(output)])

    predictions = pd.read_csv(output)
    assert len(predictions) == len(pd.read_csv(survey_csv))
    assert {*FEATURES, TARGET, "prediction", "probability_happy"} <= set(predictions.columns)
    assert "Cross-validated accuracy" in caplog.text
    assert "Accuracy on" in caplog.text
