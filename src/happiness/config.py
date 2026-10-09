"""Project-wide constants: file locations and the survey schema."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ACME-HappinessSurvey2020.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "Y"
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6"]
QUESTIONS = {
    "X1": "My order was delivered on time",
    "X2": "Contents of my order was as I expected",
    "X3": "I ordered everything I wanted to order",
    "X4": "I paid a good price for my order",
    "X5": "I am satisfied with my courier",
    "X6": "The app makes ordering easy for me",
}

MIN_ANSWER, MAX_ANSWER = 1, 5
