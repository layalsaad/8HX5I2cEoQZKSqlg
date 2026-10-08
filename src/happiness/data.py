from pathlib import Path
import pandas as pd
from happiness.config import DATA_PATH, FEATURES, MAX_ANSWER, MIN_ANSWER, TARGET


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the labelled survey CSV and return it validated."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "Download ACME-HappinessSurvey2020.csv and place it there (see README)."
        )
    # utf-8-sig transparently strips a byte-order mark if the export added one.
    return validate(pd.read_csv(path, encoding="utf-8-sig"))


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check schema and value ranges of survey data.

    Returns a copy restricted to the feature columns, plus the target when present.
    Raises ValueError listing every problem found.
    """
    required = [*FEATURES, TARGET] if require_target else FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    columns = [*FEATURES, TARGET] if TARGET in df.columns else FEATURES
    df = df[columns].copy()

    errors = []
    with_nulls = [col for col in columns if df[col].isna().any()]
    if with_nulls:
        errors.append(f"missing values in {with_nulls}")

    complete = [col for col in columns if col not in with_nulls]
    non_integer = [col for col in complete if not pd.api.types.is_integer_dtype(df[col])]
    if non_integer:
        errors.append(f"non-integer values in {non_integer}")

    # Range checks only make sense on complete integer columns.
    clean = [col for col in complete if col not in non_integer]
    out_of_range = [
        col for col in clean if col != TARGET and not df[col].between(MIN_ANSWER, MAX_ANSWER).all()
    ]
    if out_of_range:
        errors.append(f"answers outside {MIN_ANSWER}-{MAX_ANSWER} in {out_of_range}")

    if TARGET in clean and not df[TARGET].isin([0, 1]).all():
        errors.append(f"{TARGET} must contain only 0 and 1")

    if errors:
        raise ValueError("Invalid survey data: " + "; ".join(errors))
    return df
