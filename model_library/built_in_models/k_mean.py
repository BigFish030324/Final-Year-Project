
# ----------------------------------------------------
# Prepare Data, Train Model and Evaluate Results
# ----------------------------------------------------

from __future__ import annotations

from typing import Any

import pandas as pd

from error_handler import UserError
from model_library.model_helpers import build_preprocessor


# ----------------------------------------------------
# Main Modelling Function
# ----------------------------------------------------

def k_means(
    frame: pd.DataFrame,
    target_column: str,
    number_of_clusters: int = 3,
    parameters: dict[str, Any] | None = None,
    seed: int = 42,
    selected_input_columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score,
        silhouette_score,
    )

    # K-Means does not use the prediction target for training. Use the
    # selected inputs, or default to non-identifier columns when none exist.
    if selected_input_columns is None:
        feature_columns = [
            column
            for column in frame.columns
            if (
                column != target_column
                and column.lower() not in {"id", "index"}
            )
        ]
    else:
        feature_columns = list(
            dict.fromkeys(selected_input_columns)
        )

        unavailable_columns = [
            column
            for column in feature_columns
            if column not in frame.columns
        ]
        if unavailable_columns:
            raise UserError(
                "One or more selected input columns are unavailable."
            )

        feature_columns = [
            column
            for column in feature_columns
            if column != target_column
        ]

    if not feature_columns:
        raise UserError(
            "K-Means requires at least one input column."
        )

    # Select and prepare the input features
    features = frame[feature_columns].copy()

    # Remove rows where every K-Means input feature is missing
    features = features.dropna(how="all")

    if len(features) < 3:
        raise UserError(
            "K-Means requires at least three usable rows."
        )

    # Silhouette scoring requires at least two clusters.
    if number_of_clusters < 2:
        raise UserError(
            "K-Means requires at least two clusters."
        )

    if number_of_clusters >= len(features):
        raise UserError(
            "The number of K-Means clusters must be smaller than the number of usable rows."
        )

    preprocessor = build_preprocessor(features)
    prepared_features = preprocessor.fit_transform(features)

    # Start with the same K-Means defaults as the test environment.
    model_parameters = {
        "initialization_method": "k-means++",
        "restart_count": 10,
        "maximum_iterations": 300,
        "convergence_tolerance": 0.0001,
    }
    model_parameters.update(parameters or {})

    # Create the K-Means model
    model = KMeans(
        n_clusters=number_of_clusters,
        init=model_parameters["initialization_method"],
        n_init=model_parameters["restart_count"],
        max_iter=model_parameters["maximum_iterations"],
        tol=model_parameters["convergence_tolerance"],
        random_state=seed,
    )

    # Train K-Means and receive the cluster assigned to every row
    cluster_labels = model.fit_predict(
        prepared_features
    )

    discovered_cluster_count = len(
        set(cluster_labels)
    )

    # Identical rows can cause K-Means to discover fewer groups than requested.
    # Silhouette score is unavailable when only one distinct group is found.
    silhouette = None
    if 1 < discovered_cluster_count < len(features):
        silhouette = round(
            float(
                silhouette_score(
                    prepared_features,
                    cluster_labels,
                )
            ),
            5,
        )

    metrics = {
        "inertia": round(
            float(model.inertia_),
            5,
        ),
        "silhouette_score": silhouette,
    }

    # Additional diagnostic:
    # Compare discovered clusters with known target labels
    # The target is not used to train K-Means
    target_for_used_rows = frame.loc[
        features.index,
        target_column,
    ]

    if (
        target_for_used_rows.notna().all()
        and target_for_used_rows.nunique() > 1
    ):
        encoded_target = pd.factorize(
            target_for_used_rows.astype(str)
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

    # Count how many rows were assigned to each cluster
    cluster_counts = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    cluster_count_labels = [
        f"Cluster {cluster_number}"
        for cluster_number in cluster_counts
    ]
    cluster_count_values = [
        int(count)
        for count in cluster_counts.values()
    ]

    # Return a standard result dictionary for Flask and terminal output
    return {
        "model": "K-Means",
        "task": "clustering",
        "target_used_for_training": False,
        "feature_columns": feature_columns,
        "rows_used": len(features),
        "number_of_clusters": number_of_clusters,
        "parameters": model_parameters,
        "metrics": metrics,
        "cluster_counts": {
            f"Cluster {cluster_number}": int(count)
            for cluster_number, count
            in cluster_counts.items()
        },
        "chart": {
            "type": "cluster_sizes",
            "labels": cluster_count_labels,
            "values": cluster_count_values,
        },
        "note": (
            "K-Means discovers groups. Its metrics should not be directly "
            "compared with Decision Tree accuracy."
        ),
    }
