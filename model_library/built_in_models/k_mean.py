
# ----------------------------------------------------
# Prepare, train & evaluate K-Means
# ----------------------------------------------------

from __future__ import annotations

from typing import Any

import pandas as pd

from error_handler import UserError
from model_library.model_helpers import build_preprocessor


# ----------------------------------------------------
# K-Means
# ----------------------------------------------------

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

    # K-Means does not use the prediction target or identifier for training
    feature_columns = [
        column
        for column in frame.columns
        if column not in {target_column, "Id"}
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

    preprocessor = build_preprocessor(features)
    prepared_features = preprocessor.fit_transform(features)

    number_of_clusters = 3

    # Create the K-Means model
    model = KMeans(
        n_clusters=number_of_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=seed,
    )

    # Train K-Means and receive the cluster assigned to every row
    cluster_labels = model.fit_predict(
        prepared_features
    )

    metrics = {
        "inertia": round(
            float(model.inertia_),
            5,
        ),
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

    # Return a standard result dictionary for Flask and terminal output
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
            for cluster_number, count
            in cluster_counts.items()
        },
        "note": (
            "K-Means discovers groups. Its metrics should not be directly "
            "compared with Decision Tree accuracy."
        ),
    }
