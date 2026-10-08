# Customer Happiness Prediction

Predict whether a customer is happy (`Y = 1`) or unhappy (`Y = 0`) from their answers to a
six-question delivery survey, and identify which questions matter most.

| Column | Survey question (answers 1 = least, 5 = most) |
| ------ | --------------------------------------------- |
| `X1`   | My order was delivered on time                |
| `X2`   | Contents of my order was as I expected        |
| `X3`   | I ordered everything I wanted to order        |
| `X4`   | I paid a good price for my order              |
| `X5`   | I am satisfied with my courier                |
| `X6`   | The app makes ordering easy for me            |

**Goals**

- Predict happiness with at least 73% accuracy.
- Find the minimal set of questions that preserves predictive power, and whether any question
  can be removed from the next survey.

## Project structure

```
data/raw/            # input CSV
notebooks/           # EDA, modeling, feature selection
src/happiness/       # config, data loading/validation, modeling
tests/               # unit tests
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv` on macOS).

```bash
uv sync                      # creates .venv with Python 3.12 and all dependencies
```

Place the dataset at `data/raw/ACME-HappinessSurvey2020.csv`.

## Development

```bash
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run jupyter lab           # open the notebooks
```
