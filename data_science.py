# --------------------
# Data Science Workflow
# This file handles dataset profiling, cleaning and data modelling for the website.
# --------------------

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAX_TRAINING_ROWS = 200_000
PREVIEW_ROWS = 12
PROCESSING_MODES: dict[str, dict[str, Any]] = {
    "economy": {
        "label": "Economy",
        "row_limit": 10_000,
        "description": "A quick, low-resource first pass using up to 10,000 rows.",
        "tradeoff": "Fastest and lightest, but uncommon patterns may be missed.",
    },
    "balanced": {
        "label": "Balanced",
        "row_limit": 50_000,
        "description": "A broader comparison using up to 50,000 rows.",
        "tradeoff": "More representative than Economy, with moderate time and memory use.",
    },
    "full": {
        "label": "Full",
        "row_limit": None,
        "description": "Use every available row for the final comparison.",
        "tradeoff": "Most complete, but may require substantially more time and memory.",
    },
}
TARGET_HINTS = (
    "target", "label", "class", "outcome", "result", "response", "dependent",
    "price", "sales", "score", "churn", "species", "diagnosis", "status",
)
ID_HINTS = ("id", "uuid", "guid", "index", "serial", "code")


MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "linear_regression", "name": "Linear Regression", "family": "Regression",
        "tasks": ["regression"], "version": "scikit-learn",
        "summary": "Fits a straight-line relationship to predict a continuous number.",
        "best_for": "Numeric targets with roughly linear relationships and a clear baseline.",
        "defaults": {"fit_intercept": True, "positive": False}, "factor": 0.35,
    },
    {
        "id": "decision_tree", "name": "Decision Tree", "family": "Tree-based",
        "tasks": ["classification", "regression"], "version": "scikit-learn",
        "summary": "Learns easy-to-follow if/then decision rules from the data.",
        "best_for": "Mixed patterns and beginners who want an interpretable model.",
        "defaults": {
            "criterion": "auto", "splitter": "best", "max_depth": 8,
            "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "auto",
        }, "factor": 0.55,
    },
    {
        "id": "random_forest", "name": "Random Forest", "family": "Ensemble",
        "tasks": ["classification", "regression"], "version": "scikit-learn",
        "summary": "Combines many decision trees to make a more stable prediction.",
        "best_for": "Tabular data with non-linear relationships and mixed feature importance.",
        "defaults": {
            "n_estimators": 120, "criterion": "auto", "max_depth": 12,
            "min_samples_split": 2, "min_samples_leaf": 1,
            "max_features": "sqrt", "bootstrap": True,
        }, "factor": 2.2,
    },
    {
        "id": "naive_bayes", "name": "Naive Bayes", "family": "Probabilistic",
        "tasks": ["classification"], "version": "scikit-learn",
        "summary": "Uses probability to decide which class an example most likely belongs to.",
        "best_for": "Fast classification baselines, especially smaller or high-dimensional data.",
        "defaults": {"var_smoothing": 1e-9}, "factor": 0.25,
    },
    {
        "id": "logistic_regression", "name": "Logistic Regression", "family": "Classification",
        "tasks": ["classification"], "version": "scikit-learn",
        "summary": "Estimates the probability that a row belongs to a category.",
        "best_for": "Binary or multi-class targets and an explainable baseline.",
        "defaults": {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "class_weight": "none", "tol": 0.0001}, "factor": 0.65,
    },
    {
        "id": "neural_network", "name": "Neural Network", "family": "Deep learning",
        "tasks": ["classification", "regression"], "version": "scikit-learn MLP",
        "summary": "Learns layered non-linear relationships between inputs and outcomes.",
        "best_for": "Scaled numeric data with enough rows and more complex patterns.",
        "defaults": {
            "hidden_layer_sizes": [64, 32], "activation": "relu", "solver": "adam",
            "learning_rate_init": 0.001, "max_iter": 300, "alpha": 0.0001,
        },
        "factor": 3.2,
    },
    {
        "id": "arima", "name": "ARIMA", "family": "Time series",
        "tasks": ["time_series"], "version": "statsmodels",
        "summary": "Forecasts future numeric values from their earlier time-ordered values.",
        "best_for": "A numeric target with a meaningful date/time column and regular observations.",
        "defaults": {"p": 1, "d": 1, "q": 1}, "factor": 4.0,
    },
    {
        "id": "knn", "name": "K-Nearest Neighbours", "family": "Instance-based",
        "tasks": ["classification", "regression"], "version": "scikit-learn",
        "summary": "Predicts using the most similar examples already present in the data.",
        "best_for": "Smaller, scaled datasets where nearby rows tend to have similar outcomes.",
        "defaults": {"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2}, "factor": 1.1,
    },
    {
        "id": "gradient_boosting", "name": "Gradient Boosting", "family": "Ensemble",
        "tasks": ["classification", "regression"], "version": "scikit-learn",
        "summary": "Builds trees in sequence so each new tree corrects earlier mistakes.",
        "best_for": "Structured data where predictive performance matters more than simplicity.",
        "defaults": {
            "loss": "auto", "n_estimators": 100, "learning_rate": 0.1,
            "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "subsample": 1.0,
        },
        "factor": 2.7,
    },
    {
        "id": "kmeans", "name": "K-Means", "family": "Clustering",
        "tasks": ["clustering"], "version": "scikit-learn",
        "summary": "Groups similar rows into clusters without needing a prediction target.",
        "best_for": "Exploring natural groups in numeric data; it is not a supervised predictor.",
        "defaults": {"n_clusters": 3, "init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 0.0001}, "factor": 0.9,
    },
]

MODEL_BY_ID = {item["id"]: item for item in MODEL_CATALOG}


class UserFacingError(ValueError):
    # --------------------
    # User-Facing Error
    # This error keeps the message simple enough to display on the website.
    # --------------------
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def records_for_json(frame: pd.DataFrame, limit: int = PREVIEW_ROWS) -> list[dict[str, Any]]:
    return [
        {str(k): safe_json_value(v) for k, v in record.items()}
        for record in frame.head(limit).to_dict(orient="records")
    ]


def read_dataset(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        if suffix == ".xlsx":
            return pd.read_excel(path, nrows=nrows, engine="openpyxl")
    except UnicodeDecodeError as exc:
        raise UserFacingError("The CSV text encoding could not be read. Save it as UTF-8 and try again.") from exc
    except Exception as exc:
        raise UserFacingError(f"The dataset could not be read: {exc}") from exc
    raise UserFacingError("Only CSV and XLSX datasets are supported.")


def count_csv_rows(path: Path) -> int:
    """Count CSV records without holding the complete file in memory."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        count = sum(1 for _ in reader)
    return max(0, count - 1)


def estimate_rows(path: Path, sample: pd.DataFrame) -> int:
    if path.suffix.lower() == ".csv" and path.stat().st_size > 80 * 1024 * 1024:
        return count_csv_rows(path)
    if len(sample) < MAX_TRAINING_ROWS:
        return len(sample)
    if path.suffix.lower() == ".csv":
        return count_csv_rows(path)
    # XLSX row counting is expensive and cannot be streamed reliably.  Reading
    # the worksheet once is the most truthful option for the local application.
    return len(read_dataset(path))


def _normalised_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def detect_datetime_columns(frame: pd.DataFrame) -> list[str]:
    found: list[str] = []
    for column in frame.columns:
        series = frame[column]
        name = _normalised_name(str(column))
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(str(column))
            continue
        if any(token in name for token in ("date", "time", "year", "month")):
            converted = pd.to_datetime(series.dropna().head(200), errors="coerce")
            if len(converted) and converted.notna().mean() >= 0.8:
                found.append(str(column))
    return found


def rank_target_columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    row_count = max(1, len(frame))
    for index, column in enumerate(frame.columns):
        series = frame[column]
        name = _normalised_name(str(column))
        unique = int(series.nunique(dropna=True))
        missing_ratio = float(series.isna().mean())
        unique_ratio = unique / row_count
        score = 0.0
        reasons: list[str] = []
        for priority, hint in enumerate(TARGET_HINTS):
            if name == hint:
                score += 100 - priority
                reasons.append(f"its name exactly matches the common target term '{hint}'")
                break
            if hint in name:
                score += 55 - min(priority, 20)
                reasons.append(f"its name contains the common target term '{hint}'")
                break
        if index == len(frame.columns) - 1:
            score += 15
            reasons.append("it is the last column, a common dataset convention")
        if 2 <= unique <= max(20, int(math.sqrt(row_count))):
            score += 16
            reasons.append("it has a practical number of distinct outcomes")
        if missing_ratio < 0.05:
            score += 6
        elif missing_ratio > 0.4:
            score -= 35
            reasons.append("many values are missing")
        if unique <= 1:
            score -= 100
            reasons.append("it has no useful variation")
        if unique_ratio > 0.95:
            score -= 28
            reasons.append("almost every value is unique")
        if any(name == hint or name.endswith(f"_{hint}") for hint in ID_HINTS):
            score -= 80
            reasons.append("it looks like an identifier")
        rankings.append({"column": str(column), "score": round(score, 2), "reasons": reasons})
    rankings.sort(key=lambda item: item["score"], reverse=True)
    return rankings


def detect_task(series: pd.Series) -> tuple[str, str]:
    clean = series.dropna()
    unique = int(clean.nunique())
    if unique < 2:
        return "unknown", "The selected target needs at least two different values."
    if not pd.api.types.is_numeric_dtype(clean):
        return "classification", "Text or category outcomes indicate a classification task."
    threshold = max(12, min(50, int(math.sqrt(max(1, len(clean))))))
    if unique <= threshold and unique / max(1, len(clean)) <= 0.2:
        return "classification", "The numeric target has a small set of repeated outcomes."
    return "regression", "The numeric target has many distinct values, indicating regression."


def _column_summary(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    result: dict[str, Any] = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "unique": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(clean, errors="coerce").dropna()
        if len(numeric):
            result.update({
                "mean": round(float(numeric.mean()), 4),
                "std": round(float(numeric.std(ddof=1)), 4) if len(numeric) > 1 else 0.0,
                "min": round(float(numeric.min()), 4),
                "median": round(float(numeric.median()), 4),
                "max": round(float(numeric.max()), 4),
            })
    else:
        values = clean.astype(str).value_counts().head(3)
        result["top_values"] = [{"value": str(k), "count": int(v)} for k, v in values.items()]
    return result


def profile_dataset(path: Path, *, original_name: str, dataset_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    size = path.stat().st_size
    limited = size > 120 * 1024 * 1024
    frame = read_dataset(path, nrows=MAX_TRAINING_ROWS if limited else None)
    if frame.empty or not len(frame.columns):
        raise UserFacingError("The uploaded file has no data rows or columns.")
    frame.columns = [str(column).strip() or f"column_{i + 1}" for i, column in enumerate(frame.columns)]
    if len(set(frame.columns)) != len(frame.columns):
        seen: dict[str, int] = {}
        renamed: list[str] = []
        for column in frame.columns:
            seen[column] = seen.get(column, 0) + 1
            renamed.append(column if seen[column] == 1 else f"{column}_{seen[column]}")
        frame.columns = renamed
    rows = estimate_rows(path, frame) if limited else len(frame)
    ranked = rank_target_columns(frame)
    target = ranked[0]["column"] if ranked else str(frame.columns[-1])
    task, task_reason = detect_task(frame[target])
    confidence_gap = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else ranked[0]["score"]
    confidence = "high" if ranked[0]["score"] >= 60 or confidence_gap >= 35 else "medium" if confidence_gap >= 12 else "low"
    duplicate_rows = int(frame.duplicated().sum())
    missing_cells = int(frame.isna().sum().sum())
    estimated_seconds = max(0.2, rows * max(1, len(frame.columns)) / 1_800_000)
    datetime_columns = detect_datetime_columns(frame)
    return {
        "id": dataset_id,
        "filename": original_name,
        "stored_path": str(path),
        "size_bytes": size,
        "size_mb": round(size / 1_000_000, 2),
        "rows": int(rows),
        "profiled_rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "selected_columns": list(frame.columns),
        "target": target,
        "target_confidence": confidence,
        "target_reason": "; ".join(ranked[0]["reasons"][:3]) or "it is the strongest usable candidate",
        "target_rankings": ranked[:8],
        "task": task,
        "task_reason": task_reason,
        "datetime_columns": datetime_columns,
        "date_column": datetime_columns[0] if datetime_columns else None,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / max(1, frame.size) * 100, 2),
        "duplicate_rows": duplicate_rows,
        "column_summaries": [_column_summary(frame[column]) for column in frame.columns],
        "preview": records_for_json(frame),
        "estimated_profile_seconds": round(estimated_seconds, 1),
        "requires_processing_choice": estimated_seconds > 2 or rows > MAX_TRAINING_ROWS,
        "created_at": utc_now(),
        "profile_seconds": round(time.perf_counter() - started, 3),
        "is_cleaned": False,
    }


def model_compatibility(model_id: str, task: str, datetime_columns: list[str]) -> tuple[bool, str]:
    model = MODEL_BY_ID.get(model_id)
    if not model:
        return False, "This model is not available."
    if model_id == "arima":
        if task != "regression":
            return False, "ARIMA needs a continuous numeric target."
        if not datetime_columns:
            return False, "ARIMA also needs a date or time column."
        return True, "ARIMA will use a chronological train/test split."
    if model_id == "kmeans":
        return True, "K-Means explores clusters and does not predict the selected target."
    if task in model["tasks"]:
        return True, f"Suitable for this {task} task."
    return False, f"{model['name']} does not support {task} targets."


def estimate_model_seconds(rows: int, columns: int, model_ids: list[str]) -> float:
    factor = sum(float(MODEL_BY_ID.get(mid, {}).get("factor", 1.0)) for mid in model_ids)
    return round(max(0.2, rows * max(1, columns) * max(0.25, factor) / 1_250_000), 1)


def normalise_processing_mode(mode: str) -> str:
    """Accept old saved values while enforcing the three current resource budgets."""
    legacy = {"sample": "economy", "chunked": "balanced"}
    normalised = legacy.get(str(mode), str(mode))
    if normalised not in PROCESSING_MODES:
        raise UserFacingError("Choose Economy, Balanced, or Full processing.")
    return normalised


def recommend_models(meta: dict[str, Any]) -> dict[str, Any]:
    """Return an explainable beginner pair while leaving manual selection available."""
    task = str(meta.get("task", "unknown"))
    if task == "classification":
        choices = [
            ("logistic_regression", "Provides a fast, explainable probability baseline."),
            ("decision_tree", "Adds readable if/then rules that can capture non-linear patterns."),
        ]
    elif task == "regression" and meta.get("date_column"):
        choices = [
            ("arima", "Uses the selected date column to model chronological patterns."),
            ("linear_regression", "Provides a transparent baseline using the same chronological split."),
        ]
    elif task == "regression":
        choices = [
            ("linear_regression", "Provides a fast and transparent numeric baseline."),
            ("decision_tree", "Adds readable non-linear rules for comparison with the baseline."),
        ]
    else:
        choices = [("kmeans", "Explores groups without requiring a prediction target.")]

    models = []
    for model_id, reason in choices:
        model = MODEL_BY_ID[model_id]
        compatible, _ = model_compatibility(model_id, task, meta.get("datetime_columns", []))
        if compatible:
            models.append({
                "id": model_id,
                "name": model["name"],
                "reason": reason,
                "defaults": dict(model.get("defaults", {})),
            })
    return {
        "task": task,
        "models": models[:2],
        "message": "These starter models were selected for the detected task and target, so changing the resource budget does not change the model pair.",
        "disclaimer": "Resource budgets change how many dataset rows are processed, not which models are selected. Recommendations are guidance, not a performance guarantee; you can replace either model or edit every parameter.",
    }


def processing_preflight(meta: dict[str, Any], model_ids: list[str]) -> dict[str, Any]:
    full_rows = int(meta["rows"])
    columns = len(meta["selected_columns"])
    seconds = estimate_model_seconds(full_rows, columns, model_ids)
    modes = []
    for mode_id, definition in PROCESSING_MODES.items():
        limit = definition["row_limit"]
        rows_used = full_rows if limit is None else min(full_rows, int(limit))
        modes.append({
            "id": mode_id,
            "label": definition["label"],
            "rows": rows_used,
            "estimated_seconds": estimate_model_seconds(rows_used, columns, model_ids),
            "description": definition["description"],
            "tradeoff": definition["tradeoff"],
        })
    recommendation = "economy"
    return {
        "estimated_seconds": seconds,
        "requires_choice": True,
        "full_rows": full_rows,
        "modes": modes,
        "recommendation": recommendation,
        "message": (
            "Choose a resource budget for this run. Row counts and time estimates are transparent guidance, "
            "not guarantees, because hardware, data complexity, and model settings vary."
        ),
    }


def _load_for_training(path: Path, mode: str, seed: int) -> tuple[pd.DataFrame, bool]:
    mode = normalise_processing_mode(mode)
    limit = PROCESSING_MODES[mode]["row_limit"]
    if limit is not None:
        if path.suffix.lower() == ".csv":
            frame = read_dataset(path, nrows=int(limit))
        else:
            frame = read_dataset(path)
        if len(frame) > int(limit):
            frame = frame.sample(int(limit), random_state=seed)
        return frame, True
    return read_dataset(path), False


def _build_preprocessor(frame: pd.DataFrame):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise UserFacingError("The analysis service is temporarily unavailable. Restart DataComparison or contact the administrator.") from exc
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
        ]), categorical))
    if not transformers:
        raise UserFacingError("No usable feature columns remain after selecting the target.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_estimator(model_id: str, task: str, params: dict[str, Any], seed: int):
    try:
        from sklearn.cluster import KMeans
        from sklearn.ensemble import (
            GradientBoostingClassifier, GradientBoostingRegressor,
            RandomForestClassifier, RandomForestRegressor,
        )
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    except ImportError as exc:
        raise UserFacingError("The analysis service is temporarily unavailable. Restart DataComparison or contact the administrator.") from exc

    defaults = dict(MODEL_BY_ID[model_id]["defaults"])
    defaults.update(params or {})
    if defaults.get("criterion") == "auto":
        defaults["criterion"] = "gini" if task == "classification" else "squared_error"
    if defaults.get("loss") == "auto":
        defaults["loss"] = "log_loss" if task == "classification" else "squared_error"
    if defaults.get("max_features") == "auto":
        defaults["max_features"] = None
    if defaults.get("class_weight") == "none":
        defaults["class_weight"] = None
    if model_id == "linear_regression":
        return LinearRegression(**defaults)
    if model_id == "decision_tree":
        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        return cls(random_state=seed, **defaults)
    if model_id == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(random_state=seed, n_jobs=-1, **defaults)
    if model_id == "naive_bayes":
        return GaussianNB(**defaults)
    if model_id == "logistic_regression":
        return LogisticRegression(random_state=seed, **defaults)
    if model_id == "neural_network":
        if isinstance(defaults.get("hidden_layer_sizes"), list):
            defaults["hidden_layer_sizes"] = tuple(defaults["hidden_layer_sizes"])
        cls = MLPClassifier if task == "classification" else MLPRegressor
        return cls(random_state=seed, early_stopping=True, **defaults)
    if model_id == "knn":
        cls = KNeighborsClassifier if task == "classification" else KNeighborsRegressor
        return cls(**defaults)
    if model_id == "gradient_boosting":
        cls = GradientBoostingClassifier if task == "classification" else GradientBoostingRegressor
        return cls(random_state=seed, **defaults)
    if model_id == "kmeans":
        return KMeans(random_state=seed, **defaults)
    raise UserFacingError(f"Model '{model_id}' is not available for standard training.")


def _metric_number(value: float) -> float:
    return round(float(value), 5)


def _run_standard_model(
    model_id: str,
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    task: str,
    train_pct: int,
    seed: int,
    params: dict[str, Any],
    chronological: bool = False,
    date_column: str | None = None,
) -> dict[str, Any]:
    try:
        from sklearn.base import clone
        from sklearn.metrics import (
            accuracy_score, adjusted_rand_score, confusion_matrix, f1_score,
            mean_absolute_error, mean_squared_error, precision_score, r2_score,
            recall_score, silhouette_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise UserFacingError("The analysis service is temporarily unavailable. Restart DataComparison or contact the administrator.") from exc

    used = frame[feature_columns + ([target] if target not in feature_columns else [])].copy()
    if model_id != "kmeans":
        used = used.dropna(subset=[target])
    if len(used) < 10:
        raise UserFacingError("At least 10 usable rows are required for model comparison.")
    if chronological and date_column and date_column in used.columns:
        used[date_column] = pd.to_datetime(used[date_column], errors="coerce")
        used = used.dropna(subset=[date_column]).sort_values(date_column)
        used[date_column] = used[date_column].astype("int64") / 1_000_000_000
    X = used[feature_columns]
    y = used[target] if target in used else None
    preprocessor = _build_preprocessor(X)
    estimator = _make_estimator(model_id, task, params, seed)
    started = time.perf_counter()

    if model_id == "kmeans":
        transformed = preprocessor.fit_transform(X)
        labels = estimator.fit_predict(transformed)
        metrics = {"inertia": _metric_number(estimator.inertia_)}
        if len(set(labels)) > 1 and len(transformed) > len(set(labels)):
            sample_size = min(10_000, len(transformed))
            metrics["silhouette_score"] = _metric_number(
                silhouette_score(transformed, labels, sample_size=sample_size, random_state=seed)
            )
        if y is not None and y.nunique(dropna=True) > 1:
            encoded = pd.factorize(y.astype(str))[0]
            metrics["adjusted_rand_index"] = _metric_number(adjusted_rand_score(encoded, labels))
        counts = pd.Series(labels).value_counts().sort_index()
        return {
            "model_id": model_id, "name": MODEL_BY_ID[model_id]["name"], "task": "clustering",
            "parameters": estimator.get_params(deep=False), "metrics": metrics,
            "training_seconds": round(time.perf_counter() - started, 4),
            "rows_used": len(used), "chart": {"labels": [f"Cluster {i}" for i in counts.index], "values": counts.tolist()},
            "note": "K-Means discovers groups; its scores are not directly comparable with prediction metrics.",
        }

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    if chronological:
        split = max(5, min(len(used) - 3, int(len(used) * train_pct / 100)))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_pct / 100, random_state=seed, stratify=stratify,
        )
    pipeline = Pipeline([("prepare", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    train_pred = pipeline.predict(X_train)
    metrics: dict[str, float]
    chart: dict[str, Any]
    if task == "classification":
        metrics = {
            "accuracy": _metric_number(accuracy_score(y_test, predicted)),
            "precision": _metric_number(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "recall": _metric_number(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "f1_score": _metric_number(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "training_accuracy": _metric_number(accuracy_score(y_train, train_pred)),
        }
        labels = sorted({str(v) for v in y_test} | {str(v) for v in predicted})
        matrix = confusion_matrix(y_test.astype(str), pd.Series(predicted).astype(str), labels=labels)
        chart = {"type": "confusion", "labels": labels, "matrix": matrix.tolist()}
    else:
        mse = mean_squared_error(y_test, predicted)
        metrics = {
            "mae": _metric_number(mean_absolute_error(y_test, predicted)),
            "mse": _metric_number(mse),
            "rmse": _metric_number(math.sqrt(mse)),
            "r2": _metric_number(r2_score(y_test, predicted)),
            "training_r2": _metric_number(r2_score(y_train, train_pred)),
        }
        limit = min(80, len(predicted))
        chart = {
            "type": "actual_predicted",
            "actual": [safe_json_value(v) for v in np.asarray(y_test)[:limit]],
            "predicted": [safe_json_value(v) for v in np.asarray(predicted)[:limit]],
        }
    return {
        "model_id": model_id, "name": MODEL_BY_ID[model_id]["name"], "task": "time_series" if chronological else task,
        "parameters": estimator.get_params(deep=False), "metrics": metrics,
        "training_seconds": round(time.perf_counter() - started, 4),
        "rows_used": len(used), "train_rows": len(X_train), "test_rows": len(X_test),
        "chart": chart,
    }


def _run_arima(
    frame: pd.DataFrame, target: str, date_column: str, train_pct: int, params: dict[str, Any]
) -> dict[str, Any]:
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise UserFacingError(
            "The forecasting service is temporarily unavailable. Restart DataComparison or contact the administrator."
        ) from exc
    used = frame[[date_column, target]].copy()
    used[date_column] = pd.to_datetime(used[date_column], errors="coerce")
    used[target] = pd.to_numeric(used[target], errors="coerce")
    used = used.dropna().sort_values(date_column)
    if len(used) < 30:
        raise UserFacingError("ARIMA requires at least 30 time-ordered numeric observations.")
    split = max(10, min(len(used) - 5, int(len(used) * train_pct / 100)))
    train, test = used[target].iloc[:split], used[target].iloc[split:]
    config = dict(MODEL_BY_ID["arima"]["defaults"])
    config.update(params or {})
    order = (int(config["p"]), int(config["d"]), int(config["q"]))
    started = time.perf_counter()
    fitted = ARIMA(train, order=order).fit()
    predicted = fitted.forecast(steps=len(test))
    mse = mean_squared_error(test, predicted)
    metrics = {
        "mae": _metric_number(mean_absolute_error(test, predicted)),
        "mse": _metric_number(mse), "rmse": _metric_number(math.sqrt(mse)),
        "r2": _metric_number(r2_score(test, predicted)) if len(test) > 1 else 0.0,
    }
    return {
        "model_id": "arima", "name": "ARIMA", "task": "time_series",
        "parameters": {"order": list(order)}, "metrics": metrics,
        "training_seconds": round(time.perf_counter() - started, 4),
        "rows_used": len(used), "train_rows": len(train), "test_rows": len(test),
        "chart": {
            "type": "forecast",
            "labels": [safe_json_value(v) for v in used[date_column].iloc[split:].head(80)],
            "actual": [safe_json_value(v) for v in test.head(80)],
            "predicted": [safe_json_value(v) for v in predicted.head(80)],
        },
    }


def _run_custom_model(
    definition: dict[str, Any],
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    task: str,
    train_pct: int,
    seed: int,
    params: dict[str, Any],
    date_column: str | None = None,
) -> dict[str, Any]:
    """Run a validated local model in a separate, time-limited process.

    This isolates ordinary failures and accidental infinite loops. It is not a
    security boundary for hostile public users, which the UI states clearly.
    """
    try:
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
            mean_squared_error, precision_score, r2_score, recall_score, silhouette_score,
        )
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise UserFacingError("The analysis service is temporarily unavailable. Restart DataComparison or contact the administrator.") from exc
    errors = validate_custom_code(str(definition.get("code", "")))
    if errors:
        raise UserFacingError("The custom model no longer passes validation: " + " ".join(errors))
    declared_task = str(definition.get("task", "classification"))
    if declared_task in {"classification", "regression"} and declared_task != task:
        raise UserFacingError(f"This custom model supports {declared_task}, but the dataset requires {task}.")
    if declared_task == "time_series" and task != "regression":
        raise UserFacingError("A custom time-series model requires a numeric target.")
    if declared_task == "clustering" and not feature_columns:
        raise UserFacingError("A custom clustering model requires at least one input column.")
    metric_task = "regression" if declared_task == "time_series" else declared_task
    if declared_task == "clustering":
        used = frame[feature_columns].dropna(how="all").copy()
        if len(used) < 10:
            raise UserFacingError("At least 10 usable rows are required for a custom clustering model.")
        preprocessor = _build_preprocessor(used)
        X_train_ready = np.asarray(preprocessor.fit_transform(used), dtype=float)
        X_test_ready = X_train_ready
        X_train = used
        X_test = used.iloc[0:0]
        y_train = np.zeros(len(used), dtype=float)
        y_test = np.asarray([], dtype=float)
    else:
        used = frame[feature_columns + [target]].dropna(subset=[target]).copy()
        if declared_task == "time_series":
            if not date_column or date_column not in frame.columns:
                raise UserFacingError("Select a date/time column before running a custom time-series model.")
            order = pd.to_datetime(frame.loc[used.index, date_column], errors="coerce")
            used = used.assign(__time_order=order).dropna(subset=["__time_order"]).sort_values("__time_order").drop(columns="__time_order")
        if len(used) < 10:
            raise UserFacingError("At least 10 usable rows are required for a custom model.")
        X = used[feature_columns]
        y = used[target]
        if declared_task == "time_series":
            split = max(5, min(len(used) - 3, int(len(used) * train_pct / 100)))
            X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
        else:
            stratify = y if metric_task == "classification" and y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_pct / 100, random_state=seed, stratify=stratify,
            )
        preprocessor = _build_preprocessor(X)
        X_train_ready = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
        X_test_ready = np.asarray(preprocessor.transform(X_test), dtype=float)
    worker = Path(__file__).resolve().parent / "custom_model_runner.py"
    if not worker.exists():
        raise UserFacingError("The local custom-model worker is unavailable.")
    with tempfile.TemporaryDirectory(prefix="datacomparison_custom_") as temporary:
        temporary_path = Path(temporary)
        arrays = temporary_path / "arrays.npz"
        request_file = temporary_path / "request.json"
        response_file = temporary_path / "response.json"
        # Unicode arrays avoid pickle while preserving category labels.
        y_train_array = np.asarray(y_train.astype(str) if metric_task == "classification" else pd.to_numeric(y_train), dtype=str if metric_task == "classification" else float)
        np.savez_compressed(arrays, X_train=X_train_ready, X_test=X_test_ready, y_train=y_train_array)
        merged_params = dict(definition.get("defaults", {}))
        merged_params.update(params or {})
        request_file.write_text(json.dumps({
            "code": definition["code"], "params": merged_params, "arrays": str(arrays),
            "response": str(response_file), "task": declared_task,
        }), encoding="utf-8")
        command = [sys.executable, str(worker), str(request_file)]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=45, cwd=str(worker.parent),
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise UserFacingError("The custom model exceeded the 45-second local safety limit.") from exc
        response = load_json(response_file, None)
        if completed.returncode != 0 or not response or not response.get("ok"):
            detail = response.get("error") if response else completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown worker error"
            raise UserFacingError(f"The custom model failed: {detail}")
    predicted_raw = np.asarray(response["predictions"])
    if declared_task == "clustering":
        labels = predicted_raw.astype(str)
        unique, counts = np.unique(labels, return_counts=True)
        metrics = {"clusters": int(len(unique))}
        if len(unique) > 1 and len(unique) < len(labels):
            metrics["silhouette_score"] = _metric_number(silhouette_score(X_train_ready, labels))
        if response.get("inertia") is not None:
            metrics["inertia"] = _metric_number(response["inertia"])
        chart = {"type": "cluster_counts", "labels": unique.tolist(), "counts": counts.tolist()}
    elif metric_task == "classification":
        predicted = predicted_raw.astype(str)
        actual = np.asarray(y_test.astype(str))
        metrics = {
            "accuracy": _metric_number(accuracy_score(actual, predicted)),
            "precision": _metric_number(precision_score(actual, predicted, average="weighted", zero_division=0)),
            "recall": _metric_number(recall_score(actual, predicted, average="weighted", zero_division=0)),
            "f1_score": _metric_number(f1_score(actual, predicted, average="weighted", zero_division=0)),
        }
        labels = sorted(set(actual) | set(predicted))
        chart = {"type": "confusion", "labels": labels, "matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}
    else:
        predicted = predicted_raw.astype(float)
        actual = np.asarray(pd.to_numeric(y_test), dtype=float)
        mse = mean_squared_error(actual, predicted)
        metrics = {
            "mae": _metric_number(mean_absolute_error(actual, predicted)), "mse": _metric_number(mse),
            "rmse": _metric_number(math.sqrt(mse)), "r2": _metric_number(r2_score(actual, predicted)),
        }
        limit = min(80, len(predicted))
        chart = {"type": "actual_predicted", "actual": actual[:limit].tolist(), "predicted": predicted[:limit].tolist()}
    return {
        "model_id": f"custom:{definition['id']}", "name": definition["name"], "task": declared_task,
        "parameters": merged_params, "metrics": metrics,
        "training_seconds": round(float(response.get("training_seconds", 0)), 4),
        "rows_used": len(used), "train_rows": len(X_train), "test_rows": len(X_test), "chart": chart,
        "note": "Executed as trusted local code in a separate time-limited process.",
    }


def compare_models(
    path: Path,
    meta: dict[str, Any],
    model_requests: list[dict[str, Any]],
    *,
    train_pct: int = 70,
    seed: int = 42,
    mode: str = "full",
    progress=None,
) -> dict[str, Any]:
    comparison_started = time.perf_counter()
    if not 1 <= len(model_requests) <= 2:
        raise UserFacingError("Choose one or two models for comparison.")
    if not 1 <= int(train_pct) <= 99:
        raise UserFacingError("Modelling needs at least 1% training data and 1% testing data.")
    mode = normalise_processing_mode(mode)
    progress = progress or (lambda percent, message: None)
    progress(8, "Reading the active dataset")
    frame, sampled = _load_for_training(path, mode, seed)
    sampled = len(frame) < int(meta.get("rows", len(frame)))
    frame.columns = [str(column) for column in frame.columns]
    target = str(meta.get("target"))
    selected = [column for column in meta.get("selected_columns", list(frame.columns)) if column in frame.columns]
    if target not in frame.columns:
        raise UserFacingError("The selected target column is not available in the dataset.")
    if target not in selected:
        selected.append(target)
    feature_columns = [column for column in selected if column != target]
    task, task_reason = detect_task(frame[target])
    date_column = meta.get("date_column") or (meta.get("datetime_columns") or [None])[0]
    progress(18, "Preparing selected columns and checking model compatibility")
    standard_tasks = []
    has_time_series = any(
        request.get("id") == "arima" or (request.get("definition") or {}).get("task") == "time_series"
        for request in model_requests
    )
    for request in model_requests:
        model_id = request.get("id", "")
        if model_id.startswith("custom:"):
            definition = request.get("definition") or {}
            if not definition:
                raise UserFacingError("The custom model definition is unavailable.")
            standard_tasks.append(str(definition.get("task", "unknown")))
            continue
        compatible, reason = model_compatibility(model_id, task, meta.get("datetime_columns", []))
        if not compatible:
            raise UserFacingError(reason)
        if model_id == "kmeans":
            standard_tasks.append("clustering")
        elif model_id == "arima":
            standard_tasks.append("time_series")
        elif has_time_series and task == "regression" and "regression" in MODEL_BY_ID[model_id]["tasks"]:
            standard_tasks.append("time_series")
        else:
            standard_tasks.append(task)
    if len(set(standard_tasks)) > 1:
        raise UserFacingError("These models solve different task types and cannot share a fair comparison chart.")

    results = []
    for index, request in enumerate(model_requests):
        model_id = request.get("id", "")
        progress(25 + index * 35, f"Training {request.get('name') or MODEL_BY_ID.get(model_id, {}).get('name', 'model')}")
        if model_id == "arima":
            if not date_column:
                raise UserFacingError("Select a date/time column before running ARIMA.")
            result = _run_arima(frame, target, date_column, train_pct, request.get("params", {}))
        elif model_id.startswith("custom:"):
            result = _run_custom_model(
                request.get("definition") or {}, frame, target, feature_columns, task,
                train_pct, seed, request.get("params", {}), date_column,
            )
        else:
            result = _run_standard_model(
                model_id, frame, target, feature_columns, task, train_pct, seed, request.get("params", {}),
                chronological=has_time_series, date_column=date_column,
            )
        results.append(result)
    progress(92, "Preparing charts and comparison summary")
    shared_task = results[0]["task"]
    metric_names = sorted({name for result in results for name in result["metrics"]})
    comparison_chart = {
        "labels": metric_names,
        "series": [
            {"name": result["name"], "values": [result["metrics"].get(metric) for metric in metric_names]}
            for result in results
        ],
    }
    training_seconds = round(sum(float(result.get("training_seconds", 0)) for result in results), 4)
    elapsed_seconds = round(time.perf_counter() - comparison_started, 4)
    affordability = {
        "application_fee_myr": 0,
        "paid_external_api_calls": 0,
        "processing_location": "Local computer",
        "enterprise_software_required": False,
        "resource_mode": mode,
        "resource_mode_label": PROCESSING_MODES[mode]["label"],
        "rows_processed": len(frame),
        "rows_available": int(meta["rows"]),
        "training_seconds": training_seconds,
        "elapsed_seconds": elapsed_seconds,
        "note": "The application charged no licence or API fee for this run; ordinary device, electricity, and internet costs are not estimated.",
    }
    return {
        "id": uuid.uuid4().hex,
        "dataset_id": meta["id"], "dataset_name": meta["filename"],
        "target": target, "task": shared_task, "task_reason": task_reason,
        "selected_columns": selected, "feature_columns": feature_columns,
        "train_pct": train_pct, "test_pct": 100 - train_pct, "seed": seed,
        "mode": mode, "sampled": sampled, "rows_available": meta["rows"],
        "rows_processed": len(frame), "models": results, "comparison_chart": comparison_chart,
        "total_training_seconds": training_seconds, "elapsed_seconds": elapsed_seconds,
        "affordability": affordability,
        "transformations": [
            "Rows with a missing target are excluded",
            "Numeric feature gaps use the median and numeric features are standardized",
            "Categorical gaps use the most common value and categories are one-hot encoded",
            "The original uploaded dataset is not overwritten",
        ] if not has_time_series else [
            "Rows are ordered chronologically using the selected date/time column",
            "Rows with an invalid date or missing target are excluded",
            "Regression models and ARIMA use the same earlier-train/later-test boundary",
            "The original uploaded dataset is not overwritten",
        ],
        "created_at": utc_now(),
    }


CONDITION_TOKEN = re.compile(
    r"\s*(?:(?P<number>-?\d+(?:\.\d+)?)|(?P<string>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")|"
    r"(?P<bracket>\[[^\]]+\])|(?P<operator><=|>=|!=|==|=|<|>)|(?P<lparen>\()|(?P<rparen>\))|(?P<word>[A-Za-z_][A-Za-z0-9_]*))"
)


def _condition_tokens(expression: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    position = 0
    while position < len(expression):
        match = CONDITION_TOKEN.match(expression, position)
        if not match:
            raise UserFacingError(f"The condition is invalid near: {expression[position:position + 20]}")
        kind = match.lastgroup or ""
        raw = match.group(kind)
        position = match.end()
        if kind == "number":
            value: Any = float(raw) if "." in raw else int(raw)
        elif kind == "string":
            value = ast.literal_eval(raw)
        elif kind == "bracket":
            value = raw[1:-1]
            kind = "column"
        elif kind == "word" and raw.upper() in {"AND", "OR", "TRUE", "FALSE", "NULL", "NONE"}:
            kind, value = "keyword", raw.upper()
        else:
            value = raw
        tokens.append((kind, value))
    return tokens


def _condition_literal(series: pd.Series, value: Any) -> Any:
    if value is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value) if pd.api.types.is_float_dtype(series) else int(float(value))
        except (TypeError, ValueError) as exc:
            raise UserFacingError(f"'{value}' is not a valid number for column '{series.name}'.") from exc
    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}: return True
        if text in {"false", "0", "no", "n"}: return False
        raise UserFacingError(f"'{value}' is not a valid Boolean for column '{series.name}'.")
    return str(value)


def evaluate_condition(frame: pd.DataFrame, expression: str) -> pd.Series:
    tokens = _condition_tokens(expression)
    position = 0

    def peek() -> tuple[str, Any] | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> tuple[str, Any]:
        nonlocal position
        if position >= len(tokens):
            raise UserFacingError("The condition ended before it was complete.")
        token = tokens[position]
        position += 1
        return token

    def comparison() -> pd.Series:
        nonlocal position
        if peek() and peek()[0] == "lparen":
            take()
            result = or_expression()
            if not peek() or take()[0] != "rparen":
                raise UserFacingError("Close every '(' in the condition with ')'.")
            return result
        left_kind, column = take()
        if left_kind not in {"word", "column"} or str(column) not in frame.columns:
            raise UserFacingError(f"Choose a valid condition column. Use [Column Name] when its name contains spaces.")
        operator_kind, operator = take()
        if operator_kind != "operator":
            raise UserFacingError("Place a comparison operator after the condition column.")
        value_kind, value = take()
        if value_kind == "keyword":
            if value in {"NULL", "NONE"}: value = None
            elif value == "TRUE": value = True
            elif value == "FALSE": value = False
            else: raise UserFacingError("AND and OR must join two complete comparisons.")
        elif value_kind not in {"number", "string", "word"}:
            raise UserFacingError("Use a number, quoted text, Boolean, or NULL after the comparison operator.")
        series = frame[str(column)]
        if value is None:
            if operator in {"=", "=="}: return series.isna()
            if operator == "!=": return series.notna()
            raise UserFacingError("NULL only supports = or != comparisons.")
        value = _condition_literal(series, value)
        if operator in {"=", "=="}: return series == value
        if operator == "!=": return series != value
        try:
            return {"<": series < value, "<=": series <= value, ">": series > value, ">=": series >= value}[operator]
        except TypeError as exc:
            raise UserFacingError(f"Column '{column}' cannot be compared with '{value}'.") from exc

    def and_expression() -> pd.Series:
        result = comparison()
        while peek() == ("keyword", "AND"):
            take()
            result = result & comparison()
        return result

    def or_expression() -> pd.Series:
        result = and_expression()
        while peek() == ("keyword", "OR"):
            take()
            result = result | and_expression()
        return result

    if not tokens:
        return pd.Series(True, index=frame.index)
    result = or_expression()
    if position != len(tokens):
        raise UserFacingError("The condition contains an unexpected extra value or operator.")
    return result.fillna(False).astype(bool)


def apply_cleaning(frame: pd.DataFrame, operation: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    result = frame.copy()
    if operation == "remove_missing":
        columns = [c for c in options.get("columns", []) if c in result.columns]
        before = len(result)
        result = result.dropna(subset=columns or None)
        return result, f"Removed {before - len(result):,} rows containing missing values."
    if operation == "fill_missing":
        columns = [c for c in options.get("columns", []) if c in result.columns] or list(result.columns)
        method = options.get("method", "median")
        for column in columns:
            if not result[column].isna().any():
                continue
            if method == "mean" and pd.api.types.is_numeric_dtype(result[column]):
                value = result[column].mean()
            elif method == "median" and pd.api.types.is_numeric_dtype(result[column]):
                value = result[column].median()
            elif method == "mode" and not result[column].mode(dropna=True).empty:
                value = result[column].mode(dropna=True).iloc[0]
            else:
                value = options.get("value", 0 if pd.api.types.is_numeric_dtype(result[column]) else "Unknown")
            result[column] = result[column].fillna(value)
        return result, f"Filled missing values in {len(columns)} selected column(s)."
    if operation == "remove_duplicates":
        before = len(result)
        result = result.drop_duplicates()
        return result, f"Removed {before - len(result):,} duplicate rows."
    if operation == "replace":
        column = options.get("column")
        if column not in result.columns:
            raise UserFacingError("Choose a valid column for replacement.")
        condition = str(options.get("condition", "")).strip()
        mask = evaluate_condition(result, condition) if condition else pd.Series(True, index=result.index)
        find_value = _condition_literal(result[column], options.get("find"))
        replace_value = _condition_literal(result[column], options.get("replace"))
        matches = mask & result[column].eq(find_value)
        result.loc[matches, column] = replace_value
        suffix = f" where {condition}" if condition else ""
        return result, f"Replaced {int(matches.sum()):,} matching value(s) in '{column}'{suffix}."
    if operation == "remove_outliers":
        columns = [c for c in options.get("columns", []) if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
        before = len(result)
        for column in columns:
            q1, q3 = result[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            result = result[result[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr) | result[column].isna()]
        return result, f"Removed {before - len(result):,} rows outside the IQR range."
    if operation == "normalize":
        columns = [c for c in options.get("columns", []) if c in result.columns and pd.api.types.is_numeric_dtype(result[c])]
        method = options.get("method", "standard")
        for column in columns:
            if method == "minmax":
                span = result[column].max() - result[column].min()
                result[column] = 0.0 if span == 0 else (result[column] - result[column].min()) / span
            else:
                std = result[column].std()
                result[column] = 0.0 if not std else (result[column] - result[column].mean()) / std
        return result, f"Normalized {len(columns)} numeric column(s) using {method}."
    if operation == "drop_columns":
        columns = [c for c in options.get("columns", []) if c in result.columns]
        result = result.drop(columns=columns)
        return result, f"Removed {len(columns)} selected column(s)."
    if operation == "rename_columns":
        mapping = {k: str(v).strip() for k, v in options.get("mapping", {}).items() if k in result.columns and str(v).strip()}
        result = result.rename(columns=mapping)
        return result, f"Renamed {len(mapping)} column(s)."
    if operation == "convert_type":
        column, dtype = options.get("column"), options.get("dtype")
        if column not in result.columns:
            raise UserFacingError("Choose a valid column to convert.")
        before_non_missing = int(result[column].notna().sum())
        if dtype == "integer":
            result[column] = pd.to_numeric(result[column], errors="coerce").round().astype("Int64")
        elif dtype == "decimal":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif dtype == "boolean":
            mapping = {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
            result[column] = result[column].astype("string").str.strip().str.lower().map(mapping).astype("boolean")
        elif dtype == "category":
            result[column] = result[column].astype("category")
        elif dtype == "date":
            result[column] = pd.to_datetime(result[column], errors="coerce").dt.date
        elif dtype == "datetime":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif dtype == "text":
            result[column] = result[column].astype("string")
        else:
            raise UserFacingError("Choose integer, decimal, Boolean, category, date, date and time, or text.")
        failed = max(0, before_non_missing - int(result[column].notna().sum()))
        warning = f" {failed:,} value(s) could not be converted and became missing." if failed else " All non-missing values converted successfully."
        return result, f"Converted '{column}' to {dtype}.{warning}"
    raise UserFacingError("That cleaning operation is not supported.")


def join_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    how: str,
    suffixes: tuple[str, str],
    keep_duplicate_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if how not in {"inner", "left", "right", "outer"}:
        raise UserFacingError("Join type must be inner, left, right, or outer.")
    if left_key not in left.columns or right_key not in right.columns:
        raise UserFacingError("Choose a valid join key for both datasets.")
    merged = left.merge(right, left_on=left_key, right_on=right_key, how=how, suffixes=suffixes)
    keep = set(keep_duplicate_columns or [])
    removed: list[dict[str, str]] = []
    right_suffix = suffixes[1]
    for column in list(merged.columns):
        if not column.endswith(right_suffix) or column in keep:
            continue
        base = column[: -len(right_suffix)]
        left_column = f"{base}{suffixes[0]}" if f"{base}{suffixes[0]}" in merged.columns else base
        if left_column in merged.columns:
            equal = merged[left_column].fillna("__MISSING__").astype(str).equals(
                merged[column].fillna("__MISSING__").astype(str)
            )
            if equal:
                merged = merged.drop(columns=[column])
                removed.append({"column": column, "matches": left_column, "reason": "same name and values"})
    if left_key != right_key and right_key in merged.columns and right_key not in keep:
        equal_keys = merged[left_key].fillna("__MISSING__").astype(str).equals(
            merged[right_key].fillna("__MISSING__").astype(str)
        )
        if equal_keys:
            merged = merged.drop(columns=[right_key])
            removed.append({"column": right_key, "matches": left_key, "reason": "duplicate join key"})
    return merged, removed


ALLOWED_IMPORTS = {"numpy", "pandas", "sklearn", "statsmodels", "math", "statistics"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
    "os", "sys", "subprocess", "socket", "pathlib", "shutil", "requests", "urllib",
}
BANNED_ATTRIBUTES = {"system", "popen", "remove", "unlink", "rmdir", "walk", "listdir", "environ", "exit"}


def validate_custom_code(code: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error on line {exc.lineno}: {exc.msg}"]
    has_builder = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    errors.append(f"Import '{name}' is not allowed in trusted local models.")
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            has_builder = True
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"'{node.id}' is blocked because it can access the computer outside the model task.")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            errors.append(f"Attribute '{node.attr}' is blocked for privacy and file safety.")
    if not has_builder:
        errors.append("Define build_model(params) and return a scikit-learn-compatible estimator.")
    return sorted(set(errors))


def validate_custom_cleaning_code(code: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error on line {exc.lineno}: {exc.msg}"]
    has_cleaner = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    errors.append(f"Import '{name}' is not allowed in trusted local cleaning actions.")
        if isinstance(node, ast.FunctionDef) and node.name == "clean_data":
            has_cleaner = True
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"'{node.id}' is blocked because it can access the computer outside the cleaning task.")
        if isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRIBUTES:
            errors.append(f"Attribute '{node.attr}' is blocked for privacy and file safety.")
    if not has_cleaner:
        errors.append("Define clean_data(df) and return a Pandas DataFrame.")
    return sorted(set(errors))


def custom_model_rules() -> list[str]:
    return [
        "Custom Python models run only on this computer and are marked as trusted local code.",
        "The code must define build_model(params). Predictive models return an estimator with fit and predict methods; clustering models may use fit_predict.",
        "Only NumPy, Pandas, scikit-learn, statsmodels, math, and statistics imports are accepted.",
        "File access, network access, operating-system commands, subprocesses, and dynamic code execution are blocked.",
        "Each parameter must declare a name, type, default value, description, and optional numeric limits.",
        "Custom execution uses a separate process with a time limit; this reduces risk but is not a secure public-hosting sandbox.",
        "Keep private datasets local. Do not paste secrets, passwords, access tokens, or personal data into model code.",
    ]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=safe_json_value), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
