import numpy as np
import pandas as pd
import pytest

from happiness.config import FEATURES, TARGET


@pytest.fixture
def survey_df() -> pd.DataFrame:
    """Synthetic survey answers where happiness loosely follows X1 and X5."""
    rng = np.random.default_rng(0)
    n_rows = 80
    df = pd.DataFrame(rng.integers(1, 6, size=(n_rows, len(FEATURES))), columns=FEATURES)
    score = df["X1"] + df["X5"] + rng.normal(0, 1.5, n_rows)
    df.insert(0, TARGET, (score > score.median()).astype(int))
    return df


@pytest.fixture
def survey_csv(survey_df, tmp_path):
    path = tmp_path / "survey.csv"
    survey_df.to_csv(path, index=False)
    return path
