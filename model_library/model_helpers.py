
# ----------------------------------------------------
# Preprocessing & model type detection
# ----------------------------------------------------

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from error_handler import UserError


# ----------------------------------------------------
# JSON Conversion
# ----------------------------------------------------

# Convert Pandas and NumPy values into JSON-compatible Python values
def safe_json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)

        return (
            None
            if math.isnan(number) or math.isinf(number)
            else number
        )

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ----------------------------------------------------
# Model Type Detection
# ----------------------------------------------------

# Detect and filter input data into the correct model type
# Example: Classification or Regression
def detect_model_type(target: pd.Series) -> tuple[str, str]:
    # Remove missing target values so they do not affect model type detection
    clean_target = target.dropna()

    # Count how many different outcomes exist in the target column
    unique_values = int(clean_target.nunique())

    # A model cannot learn a useful outcome when every row has the same value
    if unique_values < 2:
        raise UserError(
            "The target must contain at least two different values."
        )

    # Check whether the target column contains non-numeric data
    # If the content is not numeric, treat the task as classification
    if not pd.api.types.is_numeric_dtype(clean_target):
        return (
            "classification",
            "The target contains text or category values.",
        )

    # Calculate the square root of the row count
    # Never allow the result below 12 or above 50
    classification_limit = max(
        12,
        min(
            50,
            int(math.sqrt(len(clean_target))),
        ),
    )

    # Determine what percentage of target values are unique
    unique_ratio = unique_values / len(clean_target)

    # Treat a numeric target as classification when it contains a relatively small number of repeated outcomes
    if (
        unique_values <= classification_limit
        and unique_ratio <= 0.2
    ):
        return (
            "classification",
            "The numeric target contains a small number of repeated outcomes.",
        )

    # Treat the numeric target as a continuous regression target
    return (
        "regression",
        "The numeric target contains many distinct values.",
    )


# ----------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------

# Numeric columns:
# - Fill missing values with the median
# - Standardize their scale
#
# Categorical columns:
# - Fill missing values with the most common value
# - Convert categories into one-hot columns
def build_preprocessor(features: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        StandardScaler,
    )

    # Find numeric columns containing numbers or Boolean values
    numeric_columns = list(
        features.select_dtypes(
            include=[np.number, "bool"]
        ).columns
    )

    # If a column is not numeric, treat it as categorical
    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    # Store numeric and categorical processing instructions
    transformers = []

    # Numeric pipeline
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                # Fill missing numeric values using the median
                (
                    "fill_missing",
                    SimpleImputer(strategy="median"),
                ),

                # Standardize the scale so mean = 0 and SD = 1
                (
                    "scale_numbers",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    # Categorical pipeline
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                # Fill missing categories using the most frequent value
                (
                    "fill_missing",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                # Convert text categories into numeric one-hot columns
                (
                    "encode_categories",
                    OneHotEncoder(
                        # Do not crash if test data contains a new category
                        handle_unknown="ignore",
                        sparse_output=False,
                        max_categories=50,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    # If there are no usable columns, nothing can be provided to a model
    if not transformers:
        raise UserError(
            "The dataset has no usable input columns."
        )

    return ColumnTransformer(
        transformers=transformers,

        # Remove columns not assigned to any transformer
        remainder="drop",
    )
