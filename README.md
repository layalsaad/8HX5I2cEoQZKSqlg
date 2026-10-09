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
data/raw/            # input CSV (not versioned, see Setup)
notebooks/           # analysis narrative: EDA, modeling, feature selection
reports/figures/     # charts saved by the notebooks
src/happiness/       # reusable package: config, data loading/validation, modeling
tests/               # unit tests (use synthetic data, no CSV needed)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/), which also installs the right Python version.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # install uv (or: brew install uv)
uv sync                                            # create .venv with Python 3.12 + dependencies
```

Place the dataset at `data/raw/ACME-HappinessSurvey2020.csv`. The data is not committed to the
repository.

## Development

```bash
uv run pytest                # run tests
uv run ruff check .          # lint
uv run ruff format .         # format
uv run jupyter lab           # open the notebooks
```
