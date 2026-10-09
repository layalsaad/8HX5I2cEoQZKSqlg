import numpy as np
import pytest

from happiness.config import FEATURES, TARGET
from happiness.data import load_data, validate


def test_load_data_returns_validated_frame(survey_csv, survey_df):
    df = load_data(survey_csv)

    assert list(df.columns) == [*FEATURES, TARGET]
    assert len(df) == len(survey_df)


def test_load_data_handles_byte_order_mark(survey_df, tmp_path):
    path = tmp_path / "with_bom.csv"
    survey_df.to_csv(path, index=False, encoding="utf-8-sig")

    assert TARGET in load_data(path).columns


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="see README"):
        load_data(tmp_path / "missing.csv")


def test_validate_drops_extra_columns(survey_df):
    survey_df["comment"] = "n/a"

    assert "comment" not in validate(survey_df).columns


def test_validate_missing_column_raises(survey_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(survey_df.drop(columns="X3"))


def test_validate_target_optional_for_prediction(survey_df):
    df = validate(survey_df.drop(columns=TARGET), require_target=False)

    assert list(df.columns) == FEATURES


def test_validate_keeps_target_when_present_and_not_required(survey_df):
    assert TARGET in validate(survey_df, require_target=False).columns


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("X2", np.nan, "missing values"),
        ("X4", 2.5, "non-integer"),
        ("X6", 0, "outside 1-5"),
        ("X1", 6, "outside 1-5"),
        (TARGET, 2, "only 0 and 1"),
    ],
)
def test_validate_rejects_bad_values(survey_df, column, value, message):
    if isinstance(value, float):
        survey_df[column] = survey_df[column].astype(float)
    survey_df.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate(survey_df)
