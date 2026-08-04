
# ----------------------------------------------------
# Read .csv or .xlsx dataset & request models to run
# ----------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from error_handler import UserError
from model_library.models_importer import (
    decision_tree,
    k_means,
)


# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\Lenovo\Desktop\Final-Year-Project\sample_dataset\Iris.csv"
)

# This is for testing dataset purpose
TARGET_COLUMN = "Species"
TRAIN_PERCENT = 70
RANDOM_SEED = 42

# ----------------------------------------------------
# Read Dataset
# ----------------------------------------------------

# Confirm that the dataset exists before trying to read it
# ONLY accept .xlsx or .csv file format. Reject other file formats
def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UserError(f"The dataset does not exist: {path}")

    # Choose the correct Pandas reader based on the file extension
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False,)

        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, engine="openpyxl",)

    # Catch CSV, Excel and file-reading problems
    except Exception as error:
        raise UserError(
            f"The dataset could not be read: {error}"
        ) from error

    # Reject unsupported formats such as JSON or TXT
    raise UserError(
        "Only CSV and XLSX datasets are supported."
    )

# ----------------------------------------------------
# Terminal Output
# ----------------------------------------------------

# Display the dataset summary in the terminal
def print_dataset_summary(frame: pd.DataFrame,) -> None:
    print("\n" + "=" * 10)
    print("DATASET SUMMARY")
    print("=" * 10)
    print(f"Dataset path : {DATASET_PATH}")
    print(f"Rows         : {len(frame):,}")
    print(f"Columns      : {len(frame.columns):,}")
    print(f"Column names : {list(frame.columns)}")
    print(f"Target       : {TARGET_COLUMN}")
    print("\nFirst five rows:")
    print(frame.head().to_string(index=False))

# Print model result in the terminal
def print_model_result(result: dict[str, Any],) -> None:
    print("\n" + "=" * 70)
    print(result["model"].upper())
    print("=" * 70)
    print(json.dumps(result, indent=2,)    )

# ----------------------------------------------------
# Run Dataset Models
# ----------------------------------------------------

# Load input dataset and run both models (Decision Tree & K-Mean)
def run_dataset() -> dict[str, Any]:
    frame = read_dataset(DATASET_PATH)

    print_dataset_summary(frame)

    # ----------------------------------------------------
    # Run Decision Tree
    # ----------------------------------------------------
    decision_tree_result = decision_tree(
        frame,
        target_column=TARGET_COLUMN,
        train_percent=TRAIN_PERCENT,
        seed=RANDOM_SEED,
    )

    print_model_result(decision_tree_result)

    # ----------------------------------------------------
    # Run K-Means
    # ----------------------------------------------------
    kmeans_result = k_means(
        frame,
        target_column=TARGET_COLUMN,
        seed=RANDOM_SEED,
    )

    print_model_result(kmeans_result)

    print("\n" + "=" * 70)
    print("MODELLING COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    # Combine dataset, Decision Tree and K-Means results
    return {
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(frame),
            "columns": len(frame.columns),
            "target": TARGET_COLUMN,
        },
        "decision_tree": decision_tree_result,
        "kmeans": kmeans_result,
    }
