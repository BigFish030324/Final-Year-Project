from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
# Configuration
# ----------------------------------------------------

# Create a User Error exception
class UserError(ValueError):
    pass

# Convert Pandas and NumPy values into JSON-compatible Python values
def safe_json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number

    if isinstance(value, np.bool_):
        return bool(value)

    return value

# Confirm that the dataset exists before trying to read it
def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UserError(f"The dataset does not exist: {path}")

# Try read in .csv first, if can't run then try .xlsx format. If the file is not either one of the format, then catch error
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)

        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")

    except Exception as error:
        raise UserError(
            f"The dataset could not be read: {error}"
        ) from error

    raise UserError("Only CSV and XLSX datasets are supported.")

# Detect and filter input data into correct model type (Example: Classification / Regression)
def detect_model_type(target: pd.Series) -> tuple[str, str]:
    # Remove missing target values so they do not affect model type detection.
    clean_target = target.dropna()

    # Count how many different outcomes exist in the target column.
    unique_values = int(clean_target.nunique())

    # Model can't learn useful outcome when every row has the same value.
    if unique_values < 2:
        raise UserError("The target must contain at least two different values.")

    # Check if target column contains non-numberic data
    # If content is not numeric, then run the return context
    if not pd.api.types.is_numeric_dtype(clean_target):
        return (
            "classification",
            "The target contains text or category values.",
        )

    # Calculate square root of row count and never allow result >12 or <50
    classification_limit = max(
        12,
        min(50, int(math.sqrt(len(clean_target)))),
    )

    # Determine what percentage of target values are unique.
    unique_ratio = unique_values / len(clean_target)

    # Treat a numeric target as classification when it contains a relatively small number of repeated values.
    if unique_values <= classification_limit and unique_ratio <= 0.2:
        return (
            "classification",
            "The numeric target contains a small number of repeated outcomes.",
        )

    # Treat the numeric target as a continuous regression target.
    return (
        "regression",
        "The numeric target contains many distinct values.",
    )


# ----------------------------------------------------
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx
# ----------------------------------------------------

# Numeric columns:
# - Fill missing values with the median
# - Standardize their scale

# Categorical columns:
# - Fill missing values with the most common value
# - Convert categories into one-hot columns

def build_preprocessor(features: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Find numeric columns -> Contains numbers / booleans
    numeric_columns = list(
        features.select_dtypes(include=[np.number, "bool"]).columns
    )

    # Find categorical columns -> If not numeric, then categorical
    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    # Create a transformer list to store numeric & categorical processing instructions
    transformers = []

    # Numeric pipleline
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                # Fill missing values as median
                ("fill_missing", SimpleImputer(strategy="median")),

                # Standardize the scale -> Mean = 0, SD = 1
                ("scale_numbers", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    # Categorical pipeline
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    # Fill missing categories with most frequent
                    "fill_missing",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encode_categories",
                    OneHotEncoder(
                        # If test data contains category that is not in training data, don't crash
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

    # If dataset don't have numeric / categorical colums, provice nothing to the model
    if not transformers:
        raise UserError("The dataset has no usable input columns.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

# Decision Tree
def decision_tree(
    frame: pd.DataFrame,
    target_column: str,
    train_percent: int = 70,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.tree import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
    )

    if target_column not in frame.columns:
        raise UserError(
            f"Target column '{target_column}' was not found."
        )

    usable_data = frame.dropna(subset=[target_column]).copy()

    if len(usable_data) < 10:
        raise UserError(
            "At least 10 usable rows are required."
        )

    feature_columns = [
    column
    for column in usable_data.columns
    if column not in {target_column, "Id"}
    ]

    if not feature_columns:
        raise UserError(
            "At least one input column is required."
        )

    features = usable_data[feature_columns]
    target = usable_data[target_column]

    task, task_reason = detect_model_type(target)
    preprocessor = build_preprocessor(features)

    if task == "classification":
        estimator = DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
        )

        stratify = (
            target
            if target.value_counts().min() >= 2
            else None
        )

    else:
        estimator = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=seed,
        )

        stratify = None

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        features,
        target,
        train_size=train_percent / 100,
        random_state=seed,
        stratify=stratify,
    )

    model_pipeline = Pipeline(
        steps=[
            ("prepare_data", preprocessor),
            ("decision_tree", estimator),
        ]
    )

    model_pipeline.fit(features_train, target_train)
    predictions = model_pipeline.predict(features_test)

    if task == "classification":
        labels = sorted(
            set(target_test.astype(str))
            | set(pd.Series(predictions).astype(str))
        )

        metrics = {
            "accuracy": round(
                float(accuracy_score(target_test, predictions)),
                5,
            ),
            "precision": round(
                float(
                    precision_score(
                        target_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                5,
            ),
            "recall": round(
                float(
                    recall_score(
                        target_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                5,
            ),
            "f1_score": round(
                float(
                    f1_score(
                        target_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                5,
            ),
        }

        result_details = {
            "labels": labels,
            "confusion_matrix": confusion_matrix(
                target_test.astype(str),
                pd.Series(predictions).astype(str),
                labels=labels,
            ).tolist(),
        }

    else:
        mean_squared = mean_squared_error(
            target_test,
            predictions,
        )

        metrics = {
            "mae": round(
                float(
                    mean_absolute_error(
                        target_test,
                        predictions,
                    )
                ),
                5,
            ),
            "mse": round(float(mean_squared), 5),
            "rmse": round(float(math.sqrt(mean_squared)), 5),
            "r2": round(
                float(r2_score(target_test, predictions)),
                5,
            ),
        }

        result_details = {
            "actual": [
                safe_json_value(value)
                for value in target_test.head(10)
            ],
            "predicted": [
                safe_json_value(value)
                for value in predictions[:10]
            ],
        }

    return {
        "model": "Decision Tree",
        "task": task,
        "task_reason": task_reason,
        "target": target_column,
        "feature_columns": feature_columns,
        "train_rows": len(features_train),
        "test_rows": len(features_test),
        "metrics": metrics,
        "details": result_details,
    }

# K-Mean
def k_means(
    frame: pd.DataFrame,
    target_column: str,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        silhouette_score,
    )

    feature_columns = [
    column
    for column in frame.columns
    if column not in {target_column, "Id"}
    ]

    if not feature_columns:
        raise UserError(
            "K-Means requires at least one input column."
        )

    features = frame[feature_columns].copy()
    preprocessor = build_preprocessor(features)

    prepared_features = preprocessor.fit_transform(features)

    number_of_clusters = 3

    model = KMeans(
        n_clusters=number_of_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=seed,
    )

    cluster_labels = model.fit_predict(prepared_features)

    metrics = {
        "inertia": round(float(model.inertia_), 5),
        "silhouette_score": round(
            float(
                silhouette_score(
                    prepared_features,
                    cluster_labels,
                )
            ),
            5,
        ),
    }

    # Additional diagnostic
    if (
        target_column in frame.columns
        and frame[target_column].notna().all()
        and frame[target_column].nunique() > 1
    ):
        encoded_target = pd.factorize(
            frame[target_column].astype(str)
        )[0]

        metrics["adjusted_rand_index"] = round(
            float(
                adjusted_rand_score(
                    encoded_target,
                    cluster_labels,
                )
            ),
            5,
        )

    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "model": "K-Means",
        "task": "clustering",
        "target_used_for_training": False,
        "feature_columns": feature_columns,
        "rows_used": len(features),
        "number_of_clusters": number_of_clusters,
        "metrics": metrics,
        "cluster_counts": {
            f"Cluster {cluster_number}": int(count)
            for cluster_number, count in cluster_counts.items()
        },
        "note": (
            "K-Means discovers groups. Its metrics should not be directly "
            "compared with Decision Tree accuracy."
        ),
    }

# Display in Terminal
def print_dataset_summary(frame: pd.DataFrame) -> None:
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

# Print model result in Terminal
def print_model_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print(result["model"].upper())
    print("=" * 70)
    print(json.dumps(result, indent=2))

# Load input dataset and run both models (Decision Tree & K-Mean)
def run_dataset() -> dict[str, Any]:
    frame = read_dataset(DATASET_PATH)
    print_dataset_summary(frame)
    decision_tree_result = decision_tree(
        frame,
        target_column=TARGET_COLUMN,
        train_percent=TRAIN_PERCENT,
        seed=RANDOM_SEED,
    )

    print_model_result(decision_tree_result)

    kmeans_result = k_means(
        frame,
        target_column=TARGET_COLUMN,
        seed=RANDOM_SEED,
    )
    print_model_result(kmeans_result)

    print("\n" + "=" * 70)
    print("MODELLING COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

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
