"""Score survey responses with a trained model.

Usage: happiness-predict INPUT [--model PATH] [--output PATH]

Writes the input rows with two extra columns: `prediction` (1 = happy) and
`probability_happy`. If the input also has the target column, accuracy is reported.
Only load model files you created yourself: joblib files can execute code when loaded.
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from happiness.config import FEATURES, MODEL_PATH, TARGET
from happiness.data import validate

logger = logging.getLogger(__name__)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run happiness-train first.")
    return joblib.load(path)


def predict(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with `prediction` and `probability_happy` columns added.

    Only the questions the model uses must be present and valid; other columns (such as
    customer IDs, or an empty target column) are passed through unchanged.
    """
    questions = list(model["select"].columns or FEATURES)
    answers = validate(df.drop(columns=TARGET, errors="ignore"), False, questions)
    happy_column = list(model.classes_).index(1)
    return df.assign(
        prediction=model.predict(answers),
        probability_happy=model.predict_proba(answers)[:, happy_column].round(4),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict customer happiness from survey answers.")
    parser.add_argument("input", type=Path, help="CSV with survey answers")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="trained model file")
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.csv"), help="where to write predictions"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        scored = predict(load_model(args.model), df)
        labels = (
            validate(df, features=[])[TARGET]
            if TARGET in df.columns and df[TARGET].notna().all()
            else None
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    scored.to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(scored), args.output)
    if labels is not None:
        accuracy = (scored["prediction"] == labels).mean()
        logger.info("Accuracy on %d labelled rows: %.1f%%", len(labels), 100 * accuracy)


if __name__ == "__main__":
    main()
