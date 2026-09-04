"""
Production inference for the Adult Income (Census) classifier.

Loads the single canonical artifact produced by Day 5, Task 1
(model + calibration + tuned decision threshold, saved together so a
caller can never accidentally apply the wrong threshold to the model).

Usage
-----
    from inference import predict_income
    import pandas as pd

    new_rows = pd.read_csv("new_applicants.csv")   # same raw schema as training data
    result = predict_income(new_rows)
    print(result)

Or from the command line:

    python inference.py --input new_applicants.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

DEFAULT_ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "final_model.joblib"
)

REQUIRED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]


def load_artifact(artifact_path: str = DEFAULT_ARTIFACT_PATH) -> dict:
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"No artifact at {artifact_path}. Run the Day 5 notebook's Task 1 "
            "section first to produce final_model.joblib."
        )
    return joblib.load(artifact_path)


def predict_income(
    new_data: pd.DataFrame,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
) -> pd.DataFrame:
    """
    Score new, raw-schema rows with the saved production pipeline.

    Preprocessing (feature engineering, imputation, scaling, one-hot
    encoding) is entirely contained in the saved pipeline — this function
    does not, and must not, repeat any of it manually.

    Parameters
    ----------
    new_data : pd.DataFrame
        Raw rows with the same columns as the original training data.
        Missing values and the literal string "?" are handled by the
        pipeline's own imputers; do not pre-clean beyond that.
    artifact_path : str
        Path to the joblib artifact saved by the Day 5 notebook.

    Returns
    -------
    pd.DataFrame
        Same index as `new_data`, with `probability`, `prediction`
        (0/1), and `prediction_label` (">50K" / "<=50K") columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in new_data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    threshold = artifact["threshold"]

    clean_data = new_data.replace("?", np.nan)

    probabilities = model.predict_proba(clean_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return pd.DataFrame(
        {
            "probability": probabilities,
            "prediction": predictions,
            "prediction_label": np.where(predictions == 1, ">50K", "<=50K"),
        },
        index=new_data.index,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Score new rows with the final Adult Income model.")
    parser.add_argument("--input", required=True, help="CSV of new rows, raw schema.")
    parser.add_argument("--output", required=True, help="Where to write predictions CSV.")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT_PATH, help="Path to final_model.joblib.")
    args = parser.parse_args()

    new_data = pd.read_csv(args.input)
    result = predict_income(new_data, artifact_path=args.artifact)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    _main()
