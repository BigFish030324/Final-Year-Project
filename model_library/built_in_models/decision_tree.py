
# ----------------------------------------------------
# Prepare, train & evaluate Decision Tree
# ----------------------------------------------------

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from error_handler import UserError
from model_library.model_helpers import (
    build_preprocessor,
    detect_model_type,
    safe_json_value,
)


# ----------------------------------------------------
# Decision Tree
# ----------------------------------------------------

def decision_tree(
    frame: pd.DataFrame,
    target_column: str,
    train_percent: int = 70,
    parameters: dict[str, Any] | None = None,
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

    # Confirm that the target column exists in the dataset
    if target_column not in frame.columns:
        raise UserError(
            f"Target column '{target_column}' was not found."
        )

    # Remove rows where the prediction target is missing
    usable_data = frame.dropna(
        subset=[target_column]
    ).copy()

    # Require enough rows to create useful training and testing groups
    if len(usable_data) < 10:
        raise UserError(
            "At least 10 usable rows are required."
        )

    # Both groups need a percentage. Extreme values are allowed so the user
    # can experiment and observe why an unbalanced split may perform poorly.
    if train_percent < 1 or train_percent > 99:
        raise UserError(
            "The Decision Tree training percentage must be between 1 and 99."
        )

    estimated_train_rows = math.floor(
        len(usable_data) * train_percent / 100
    )
    estimated_test_rows = len(usable_data) - estimated_train_rows

    if estimated_train_rows < 1 or estimated_test_rows < 1:
        raise UserError(
            "This dataset is too small for the selected percentage. "
            "The split must create at least one training row and one testing row."
        )

    # Use every column except the prediction target and identifier
    feature_columns = [
        column
        for column in usable_data.columns
        if column not in {target_column, "Id"}
    ]

    if not feature_columns:
        raise UserError(
            "At least one input column is required."
        )

    # Separate model inputs from the prediction target
    features = usable_data[feature_columns]
    target = usable_data[target_column]

    # Detect whether Decision Tree should use classification or regression
    task, task_reason = detect_model_type(target)

    # Start with the same Decision Tree defaults as the test environment,
    # then replace them with any values chosen in the green parameter panel.
    model_parameters = {
        "criterion": "auto",
        "splitter": "best",
        "max_depth": 8,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",
    }
    model_parameters.update(parameters or {})

    criterion = model_parameters["criterion"]
    if criterion == "auto":
        criterion = "gini" if task == "classification" else "squared_error"

    allowed_criteria = (
        {"gini", "entropy", "log_loss"}
        if task == "classification"
        else {"squared_error", "friedman_mse", "absolute_error", "poisson"}
    )
    if criterion not in allowed_criteria:
        raise UserError(
            f"Criterion '{criterion}' cannot be used for Decision Tree {task}. "
            "Choose Auto or a criterion that matches the detected task."
        )

    max_features = model_parameters["max_features"]
    if max_features in {"auto", "none"}:
        max_features = None

    # Create the shared numeric and categorical preprocessor
    preprocessor = build_preprocessor(features)

    # Create a Decision Tree classifier for category targets
    if task == "classification":
        estimator = DecisionTreeClassifier(
            criterion=criterion,
            splitter=model_parameters["splitter"],
            max_depth=model_parameters["max_depth"],
            min_samples_split=model_parameters["min_samples_split"],
            min_samples_leaf=model_parameters["min_samples_leaf"],
            max_features=max_features,
            random_state=seed,
        )

        # Keep class proportions similar in training and testing data
        # when every class has enough rows
        class_count = target.nunique()
        stratify = (
            target
            if (
                target.value_counts().min() >= 2
                and estimated_train_rows >= class_count
                and estimated_test_rows >= class_count
            )
            else None
        )

    # Create a Decision Tree regressor for continuous numeric targets
    else:
        estimator = DecisionTreeRegressor(
            criterion=criterion,
            splitter=model_parameters["splitter"],
            max_depth=model_parameters["max_depth"],
            min_samples_split=model_parameters["min_samples_split"],
            min_samples_leaf=model_parameters["min_samples_leaf"],
            max_features=max_features,
            random_state=seed,
        )

        stratify = None

    # Divide the dataset into training and testing groups
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

    # First prepare the data, then train the Decision Tree
    model_pipeline = Pipeline(
        steps=[
            ("prepare_data", preprocessor),
            ("decision_tree", estimator),
        ]
    )

    model_pipeline.fit(
        features_train,
        target_train,
    )

    predictions = model_pipeline.predict(
        features_test
    )

    # Calculate classification metrics
    if task == "classification":
        labels = sorted(
            set(target.astype(str))
            | set(pd.Series(predictions).astype(str))
        )

        metrics = {
            "accuracy": round(
                float(
                    accuracy_score(
                        target_test,
                        predictions,
                    )
                ),
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

    # Calculate regression metrics
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
            "mse": round(
                float(mean_squared),
                5,
            ),
            "rmse": round(
                float(math.sqrt(mean_squared)),
                5,
            ),
            "r2": round(
                float(
                    r2_score(
                        target_test,
                        predictions,
                    )
                ),
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

    # Return a standard result dictionary for Flask and terminal output
    return {
        "model": "Decision Tree",
        "task": task,
        "task_reason": task_reason,
        "target": target_column,
        "train_percent": train_percent,
        "parameters": model_parameters,
        "feature_columns": feature_columns,
        "train_rows": len(features_train),
        "test_rows": len(features_test),
        "metrics": metrics,
        "details": result_details,
    }
