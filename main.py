
# ----------------------------------------------------
# Flask Imports and Application Entry Point
# ----------------------------------------------------

from __future__ import annotations
import math
from flask import Flask, render_template, request

# ----------------------------------------------------
# Data Science and Dataset Upload Imports
# ----------------------------------------------------

from data_science import inspect_dataset, run_dataset
from dataset_upload import (
    MAX_UPLOAD_BYTES,
    load_uploaded_dataset,
    save_uploaded_dataset,
)
from error_handler import UserError

# ----------------------------------------------------
# Flask Application Configuration
# ----------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Default Data Model and Parameter Values
DEFAULT_TRAINING_PERCENTAGE = 70
DEFAULT_NUMBER_OF_CLUSTERS = 3
DEFAULT_RANDOM_SEED = 42
DEFAULT_DECISION_TREE_PARAMETERS = {
    "split_quality_criterion": "auto",
    "split_selection_method": "best",
    "maximum_depth": 8,
    "minimum_samples_to_split": 2,
    "minimum_samples_per_leaf": 1,
    "maximum_features_per_split": "auto",
}
DEFAULT_KMEANS_PARAMETERS = {
    "initialization_method": "k-means++",
    "restart_count": 10,
    "maximum_iterations": 300,
    "convergence_tolerance": 0.0001,
}
DEFAULT_SELECTED_DATA_MODELS = [
    "decision_tree",
    "kmeans",
]
ALLOWED_DATA_MODELS = {
    "decision_tree",
    "kmeans",
}

# Comparison - Display Page or Process Uploaded Dataset
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_dataset = None
    result = None
    error = None
    stored_path = None
    new_upload_saved = False
    selected_data_models = DEFAULT_SELECTED_DATA_MODELS.copy()
    selected_training_percentage = DEFAULT_TRAINING_PERCENTAGE
    selected_number_of_clusters = DEFAULT_NUMBER_OF_CLUSTERS
    selected_target = ""
    selected_input_columns = []
    modelling_warnings = []
    decision_tree_parameters = DEFAULT_DECISION_TREE_PARAMETERS.copy()
    kmeans_parameters = DEFAULT_KMEANS_PARAMETERS.copy()

    # Comparison - Inspect Dataset or Render Models
    if request.method == "POST":
        try:
            form_action = request.form.get(
                "form_action",
                "",
            )

            # Comparison - Upload and Inspect Dataset
            if form_action == "inspect_dataset":
                stored_path, original_filename = save_uploaded_dataset(
                    request.files.get("dataset")
                )
                new_upload_saved = True
                uploaded_dataset = inspect_dataset(
                    stored_path,
                    original_filename,
                )
                selected_target = uploaded_dataset["target"]
                selected_input_columns = uploaded_dataset[
                    "default_input_columns"
                ]

                print("\n" + "=" * 70)
                print("DATASET INSPECTION COMPLETED SUCCESSFULLY")
                print("=" * 70)
                print(f"Filename : {uploaded_dataset['filename']}")
                print(f"Rows     : {uploaded_dataset['rows']:,}")
                print(f"Columns  : {uploaded_dataset['columns']:,}\n")

            # Comparison - Validate Settings and Render Models
            elif form_action == "render_models":
                stored_path = load_uploaded_dataset(
                    request.form.get("stored_filename", "").strip()
                )
                original_filename = (
                    request.form.get("original_filename", "").strip()
                    or stored_path.name
                )
                requested_target = request.form.get(
                    "prediction_target",
                    "",
                ).strip()
                uploaded_dataset = inspect_dataset(
                    stored_path,
                    original_filename,
                    selected_target=requested_target,
                )
                selected_target = uploaded_dataset["target"]

                selected_input_columns = list(
                    dict.fromkeys(
                        request.form.getlist("selected_input_columns")
                    )
                )
                selected_input_columns = [
                    column
                    for column in selected_input_columns
                    if column != selected_target
                ]

                if not selected_input_columns:
                    raise UserError(
                        "Choose at least one input column for modelling."
                    )

                unavailable_input_columns = [
                    column
                    for column in selected_input_columns
                    if column not in uploaded_dataset["column_names"]
                ]
                if unavailable_input_columns:
                    raise UserError(
                        "One or more selected input columns are unavailable."
                    )

                selected_data_models = list(
                    dict.fromkeys(
                        request.form.getlist("selected_data_models")
                    )
                )

                if not set(selected_data_models).issubset(ALLOWED_DATA_MODELS):
                    raise UserError(
                        "The selected model is not available."
                    )

                if not selected_data_models:
                    raise UserError(
                        "Choose at least one model in Model Library."
                    )

                try:
                    selected_training_percentage = int(
                        request.form.get(
                            "training_percentage",
                            str(DEFAULT_TRAINING_PERCENTAGE),
                        )
                    )
                except ValueError as exception:
                    raise UserError(
                        "The training percentage must be a whole number."
                    ) from exception

                if (
                    selected_training_percentage < 1
                    or selected_training_percentage > 99
                ):
                    raise UserError(
                        "Choose a training percentage between 1 and 99."
                    )

                if "decision_tree" in selected_data_models:
                    decision_tree_parameters["split_quality_criterion"] = request.form.get(
                        "decision_tree_split_quality_criterion",
                        decision_tree_parameters["split_quality_criterion"],
                    ).strip()
                    decision_tree_parameters["split_selection_method"] = request.form.get(
                        "decision_tree_split_selection_method",
                        decision_tree_parameters["split_selection_method"],
                    ).strip()
                    decision_tree_parameters["maximum_features_per_split"] = request.form.get(
                        "decision_tree_maximum_features_per_split",
                        decision_tree_parameters["maximum_features_per_split"],
                    ).strip()

                    try:
                        decision_tree_parameters["maximum_depth"] = int(
                            request.form.get("decision_tree_maximum_depth", "8")
                        )
                        decision_tree_parameters["minimum_samples_to_split"] = int(
                            request.form.get("decision_tree_minimum_samples_to_split", "2")
                        )
                        decision_tree_parameters["minimum_samples_per_leaf"] = int(
                            request.form.get("decision_tree_minimum_samples_per_leaf", "1")
                        )
                    except ValueError as exception:
                        raise UserError(
                            "Decision Tree number parameters must be whole numbers."
                        ) from exception

                    if decision_tree_parameters["split_quality_criterion"] not in {
                        "auto", "gini", "entropy", "log_loss",
                        "squared_error", "friedman_mse", "absolute_error", "poisson",
                    }:
                        raise UserError("Choose an available Decision Tree criterion.")

                    if decision_tree_parameters["split_selection_method"] not in {"best", "random"}:
                        raise UserError("Choose an available Decision Tree splitter.")

                    if decision_tree_parameters["maximum_features_per_split"] not in {
                        "auto", "sqrt", "log2", "none",
                    }:
                        raise UserError("Choose an available maximum-features option.")

                    if decision_tree_parameters["maximum_depth"] < 1:
                        raise UserError("Decision Tree maximum depth must be at least 1.")

                    if decision_tree_parameters["minimum_samples_to_split"] < 2:
                        raise UserError("Minimum samples to split must be at least 2.")

                    if decision_tree_parameters["minimum_samples_per_leaf"] < 1:
                        raise UserError("Minimum samples per leaf must be at least 1.")

                try:
                    selected_number_of_clusters = int(
                        request.form.get(
                            "number_of_clusters",
                            str(DEFAULT_NUMBER_OF_CLUSTERS),
                        )
                    )
                except ValueError as exception:
                    raise UserError(
                        "The number of clusters must be a whole number."
                    ) from exception

                if "kmeans" in selected_data_models:
                    kmeans_parameters["initialization_method"] = request.form.get(
                        "kmeans_initialization_method",
                        kmeans_parameters["initialization_method"],
                    ).strip()

                    try:
                        kmeans_parameters["restart_count"] = int(
                            request.form.get("kmeans_restart_count", "10")
                        )
                        kmeans_parameters["maximum_iterations"] = int(
                            request.form.get("kmeans_maximum_iterations", "300")
                        )
                        kmeans_parameters["convergence_tolerance"] = float(
                            request.form.get("kmeans_convergence_tolerance", "0.0001")
                        )
                    except ValueError as exception:
                        raise UserError(
                            "K-Means parameters must contain valid numbers."
                        ) from exception

                    if kmeans_parameters["initialization_method"] not in {"k-means++", "random"}:
                        raise UserError("Choose an available K-Means initialization method.")

                    if kmeans_parameters["restart_count"] < 1:
                        raise UserError("K-Means restart count must be at least 1.")

                    if kmeans_parameters["maximum_iterations"] < 1:
                        raise UserError("K-Means maximum iterations must be at least 1.")

                    if kmeans_parameters["convergence_tolerance"] <= 0:
                        raise UserError("K-Means tolerance must be greater than 0.")

                if "decision_tree" in selected_data_models:
                    testing_percentage = 100 - selected_training_percentage

                    if selected_training_percentage < 50:
                        modelling_warnings.append(
                            f"Only {selected_training_percentage}% of the rows were selected for training. "
                            "The Decision Tree may not learn enough patterns or may miss some classes."
                        )

                    if testing_percentage < 10:
                        modelling_warnings.append(
                            f"Only {testing_percentage}% of the rows were selected for testing. "
                            "The reported metrics may be unstable or misleading."
                        )

                if "kmeans" in selected_data_models:
                    maximum_cluster_count = uploaded_dataset["rows"] - 1

                    if (
                        selected_number_of_clusters < 2
                        or selected_number_of_clusters > maximum_cluster_count
                    ):
                        raise UserError(
                            "Choose at least 2 K-Means clusters and fewer clusters than dataset rows."
                        )

                    if selected_number_of_clusters > math.sqrt(uploaded_dataset["rows"]):
                        modelling_warnings.append(
                            "The selected K-Means cluster count is high for this dataset. "
                            "Some clusters may contain very few rows and may not represent meaningful groups."
                        )

                # Comparison - Run Models and Return Results
                result = run_dataset(
                    dataset_path=stored_path,
                    target_column=selected_target,
                    training_percentage=selected_training_percentage,
                    number_of_clusters=selected_number_of_clusters,
                    selected_data_models=selected_data_models,
                    decision_tree_parameters=decision_tree_parameters,
                    kmeans_parameters=kmeans_parameters,
                    seed=DEFAULT_RANDOM_SEED,
                    selected_input_columns=selected_input_columns,
                )

                print("\n" + "=" * 70)
                print("MODELLING AND CHART DATA COMPLETED SUCCESSFULLY")
                print("=" * 70)
                print(f"Filename : {uploaded_dataset['filename']}")
                print(f"Target   : {selected_target}")
                print(
                    "Inputs   : "
                    f"{', '.join(selected_input_columns)}\n"
                )

            else:
                raise UserError(
                    "The submitted comparison action is unavailable."
                )

        except UserError as exception:
            # Dataset Upload - Remove an Unreadable Saved File
            if (
                new_upload_saved
                and stored_path is not None
                and uploaded_dataset is None
            ):
                stored_path.unlink(missing_ok=True)

            error = str(exception)
            print(f"\nRENDER ERROR: {error}\n")

    # Render the Comparison Page
    return render_template(
        "index.html",
        uploaded_dataset=uploaded_dataset,
        modelling_completed=result is not None,
        result=result,
        error=error,
        selected_target=selected_target,
        selected_training_percentage=selected_training_percentage,
        selected_number_of_clusters=selected_number_of_clusters,
        selected_data_models=selected_data_models,
        selected_input_columns=selected_input_columns,
        modelling_warnings=modelling_warnings,
        decision_tree_parameters=decision_tree_parameters,
        kmeans_parameters=kmeans_parameters,
    )


# Dataset Upload - Display File Size Error
@app.errorhandler(413)
def upload_too_large(_error):
    return render_template(
        "index.html",
        uploaded_dataset=None,
        modelling_completed=False,
        result=None,
        error="The uploaded dataset exceeds the 100 MB limit.",
        selected_target="",
        selected_training_percentage=DEFAULT_TRAINING_PERCENTAGE,
        selected_number_of_clusters=DEFAULT_NUMBER_OF_CLUSTERS,
        selected_data_models=DEFAULT_SELECTED_DATA_MODELS,
        selected_input_columns=[],
        modelling_warnings=[],
        decision_tree_parameters=DEFAULT_DECISION_TREE_PARAMETERS,
        kmeans_parameters=DEFAULT_KMEANS_PARAMETERS,
    ), 413

# ----------------------------------------------------
# Run Flask Development Server Locally
# ----------------------------------------------------

# Open at http://127.0.0.1:8000
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        use_reloader=False,
    )
