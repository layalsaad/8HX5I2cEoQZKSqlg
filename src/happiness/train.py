"""Train the production model on all labelled data and save it with its metadata.

Usage: happiness-train [--data PATH] [--output PATH]
"""

import argparse
import hashlib
import json
import logging
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from happiness.config import DATA_PATH, FEATURES, MODEL_PATH, SELECTED_QUESTIONS, TARGET
from happiness.data import load_data
from happiness.evaluation import SCORING, fold_scores, repeated_cv
from happiness.feature_selection import selection_models

logger = logging.getLogger(__name__)


def build_model(questions: tuple[str, ...] = SELECTED_QUESTIONS) -> Pipeline:
    """The model chosen in notebooks/03: the shallow selection tree on the chosen questions."""
    candidate, encoding = selection_models()["decision_tree"]
    return candidate.pipeline(encoding, columns=questions)


def metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    """Fit the model on every labelled customer, save it, and return its metadata.

    The cross-validated scores describe this exact configuration. The questions themselves
    were chosen on the same data, so treat the estimate as slightly optimistic (the nested
    estimate in notebooks/03 is the strict one).
    """
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]
    model = build_model()

    scores = fold_scores(cross_validate(model, X, y, cv=repeated_cv(), scoring=SCORING, n_jobs=-1))
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": list(model["select"].columns),
        "model": repr(model["model"]),
        "training_rows": len(df),
        "happy_share": round(float(y.mean()), 4),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "cross_validation": {
            "folds": len(scores),
            **{
                metric: {
                    "mean": round(scores[metric].mean(), 4),
                    "std": round(scores[metric].std(), 4),
                }
                for metric in SCORING
            },
        },
        "versions": {
            "happiness": version("happiness"),
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
        },
    }
    metadata_path(model_path).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the customer happiness model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="labelled survey CSV")
    parser.add_argument("--output", type=Path, default=MODEL_PATH, help="where to save the model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        metadata = train(args.data, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    accuracy = metadata["cross_validation"]["accuracy"]
    logger.info(
        "Trained on %d customers using %s",
        metadata["training_rows"],
        "+".join(metadata["questions"]),
    )
    logger.info(
        "Cross-validated accuracy: %.1f%% ± %.1f%% (%d folds)",
        100 * accuracy["mean"],
        100 * accuracy["std"],
        metadata["cross_validation"]["folds"],
    )
    logger.info("Saved %s and %s", args.output, metadata_path(args.output))


if __name__ == "__main__":
    main()
