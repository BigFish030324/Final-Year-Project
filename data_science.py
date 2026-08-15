
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
from model_library.model_helpers import (
    detect_model_type,
    safe_json_value,
)


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
    selected_target: str | None = None,
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

    column_names = list(frame.columns)
    suggested_target = next(
        (
            column
            for column in reversed(column_names)
            if column.lower() not in {"id", "index"}
        ),
        column_names[-1],
    )
    target_column = selected_target or suggested_target

    if target_column not in frame.columns:
        raise UserError(
            "Choose a prediction target that exists in the dataset."
        )

    if selected_target:
        target_reason = "You manually selected this prediction target."
    else:
        target_reason = (
            "Suggested because it is the final non-identifier column, "
            "which is a common dataset format."
        )

    try:
        detected_task, task_reason = detect_model_type(
            frame[target_column]
        )
    except UserError as error:
        detected_task = "unknown"
        task_reason = str(error)

    missing_cells = int(frame.isna().sum().sum())
    total_cells = len(frame) * len(frame.columns)
    missing_percentage = (
        round(missing_cells / total_cells * 100, 2)
        if total_cells
        else 0.0
    )

    column_summaries = []
    for column_name in column_names:
        column = frame[column_name]
        clean_column = column.dropna()

        if pd.api.types.is_numeric_dtype(column):
            data_type = "Numeric"
            summary_value = (
                safe_json_value(clean_column.mean())
                if not clean_column.empty
                else None
            )
        elif pd.api.types.is_datetime64_any_dtype(column):
            data_type = "Date / Time"
            summary_value = safe_json_value(
                clean_column.iloc[0]
                if not clean_column.empty
                else None
            )
        elif pd.api.types.is_bool_dtype(column):
            data_type = "Boolean"
            summary_value = safe_json_value(
                clean_column.mode().iloc[0]
                if not clean_column.empty
                else None
            )
        else:
            data_type = "Text / Category"
            summary_value = safe_json_value(
                clean_column.mode().iloc[0]
                if not clean_column.empty
                else None
            )

        column_summaries.append(
            {
                "name": column_name,
                "data_type": data_type,
                "missing_values": int(column.isna().sum()),
                "missing_percentage": round(
                    column.isna().mean() * 100,
                    2,
                ),
                "unique_values": int(column.nunique(dropna=True)),
                "mean_or_common_value": summary_value,
            }
        )

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
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "duplicate_rows": int(frame.duplicated().sum()),
        "column_names": column_names,
        "column_summaries": column_summaries,
        "preview": preview,
        "target": target_column,
        "target_reason": target_reason,
        "task": detected_task,
        "task_reason": task_reason,
        "default_input_columns": [
            column
            for column in column_names
            if (
                column != target_column
                and column.lower() not in {"id", "index"}
            )
        ],
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
def run_dataset(
    dataset_path: Path,
    target_column: str,
    training_percentage: int = 70,
    number_of_clusters: int = 3,
    selected_data_models: list[str] | tuple[str, ...] = (
        "decision_tree",
        "kmeans",
    ),
    decision_tree_parameters: dict[str, Any] | None = None,
    kmeans_parameters: dict[str, Any] | None = None,
    seed: int = 42,
    selected_input_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    frame = read_dataset(dataset_path)
    print_dataset_summary(frame, dataset_path, target_column,)

    allowed_data_models = {
        "decision_tree",
        "kmeans",
    }
    selected_data_model_names = set(selected_data_models)

    if not selected_data_model_names:
        raise UserError(
            "Choose at least one model to run."
        )

    if not selected_data_model_names.issubset(allowed_data_models):
        raise UserError(
            "The selected model is not available."
        )

    decision_tree_result = None
    kmeans_result = None

    # ----------------------------------------------------
    # Decision Tree
    # ----------------------------------------------------
    if "decision_tree" in selected_data_model_names:
        decision_tree_result = decision_tree(
            frame,
            target_column=target_column,
            selected_input_columns=selected_input_columns,
            training_percentage=training_percentage,
            parameters=decision_tree_parameters,
            seed=seed,
        )
        print_model_result(decision_tree_result)

    # ----------------------------------------------------
    # Run K-Means
    # ----------------------------------------------------
    if "kmeans" in selected_data_model_names:
        kmeans_result = k_means(
            frame,
            target_column=target_column,
            selected_input_columns=selected_input_columns,
            number_of_clusters=number_of_clusters,
            parameters=kmeans_parameters,
            seed=seed,
        )
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
            "input_columns": list(selected_input_columns or []),
        },
        "data_models_run": list(selected_data_models),
        "decision_tree": decision_tree_result,
        "kmeans": kmeans_result,
    }
