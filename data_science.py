
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
from model_library.model_helpers import safe_json_value


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
# Inspect Dataset Before Modelling
# ----------------------------------------------------

# Read an uploaded dataset and return information needed by the webpage
# Models do not run yet because the user has not selected a target column
def inspect_dataset(
    dataset_path: Path,
    original_filename: str,
) -> dict[str, Any]:
    frame = read_dataset(dataset_path)

    if frame.empty or len(frame.columns) == 0:
        raise UserError(
            "The uploaded dataset has no usable rows or columns."
        )

    # Convert every column name to text so Flask and Jinja can display it
    frame.columns = [
        str(column)
        for column in frame.columns
    ]

    # Convert the first five rows into values safe for webpage display
    preview = [
        {
            str(column): safe_json_value(value)
            for column, value in record.items()
        }
        for record in frame.head(5).to_dict(
            orient="records"
        )
    ]

    return {
        "filename": original_filename,
        # Give the webpage only the generated filename, not a full computer path.
        "stored_filename": dataset_path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": list(frame.columns),
        "preview": preview,
    }

# ----------------------------------------------------
# Terminal Output
# ----------------------------------------------------

# Display the dataset summary in the terminal
def print_dataset_summary(frame: pd.DataFrame, dataset_path: Path, target_column: str,) -> None:
    print("\n" + "=" * 10)
    print("DATASET SUMMARY")
    print("=" * 10)
    print(f"Dataset path : {dataset_path}")
    print(f"Rows         : {len(frame):,}")
    print(f"Columns      : {len(frame.columns):,}")
    print(f"Column names : {list(frame.columns)}")
    print(f"Target       : {target_column}")
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
# Decision Tree stays in decision_tree.py and K-Means stays in k_mean.py
def run_dataset(dataset_path: Path, target_column: str, train_percent: int = 70, seed: int = 42,) -> dict[str, Any]:
    frame = read_dataset(dataset_path)
    print_dataset_summary(frame, dataset_path, target_column,)

    # ----------------------------------------------------
    # Run Decision Tree
    # ----------------------------------------------------
    decision_tree_result = decision_tree(frame, target_column=target_column, train_percent=train_percent, seed=seed,)
    print_model_result(decision_tree_result)

    # ----------------------------------------------------
    # Run K-Means
    # ----------------------------------------------------
    kmeans_result = k_means(frame, target_column=target_column, seed=seed,)
    print_model_result(kmeans_result)

    print("\n" + "=" * 70)
    print("MODELLING COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    # Combine dataset, Decision Tree and K-Means results
    return {
        "dataset": {
            "path": str(dataset_path),
            "rows": len(frame),
            "columns": len(frame.columns),
            "target": target_column,
        },
        "decision_tree": decision_tree_result,
        "kmeans": kmeans_result,
    }
