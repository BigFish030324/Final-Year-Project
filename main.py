
# ----------------------------------------------------
# Browser & combination of all file
# ----------------------------------------------------

from __future__ import annotations
import math
from flask import Flask, render_template, request

# ----------------------------------------------------
# Data Science
# ----------------------------------------------------

from data_science import inspect_dataset, run_dataset
from dataset_upload import (
    MAX_UPLOAD_BYTES,
    save_uploaded_dataset,
)
from error_handler import UserError

# ----------------------------------------------------
# Website
# ----------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Use simple default model settings during this development stage
DEFAULT_TRAIN_PERCENT = 70
DEFAULT_NUMBER_OF_CLUSTERS = 3
DEFAULT_RANDOM_SEED = 42
DEFAULT_DECISION_TREE_PARAMETERS = {
    "criterion": "auto",
    "splitter": "best",
    "max_depth": 8,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "auto",
}
DEFAULT_KMEANS_PARAMETERS = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "tolerance": 0.0001,
}
DEFAULT_SELECTED_MODELS = [
    "decision_tree",
    "kmeans",
]
ALLOWED_MODELS = {
    "decision_tree",
    "kmeans",
}

# Open the upload page with GET and inspect an uploaded dataset with POST
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_dataset = None
    result = None
    error = None
    stored_path = None
    selected_models = DEFAULT_SELECTED_MODELS.copy()
    selected_train_percent = DEFAULT_TRAIN_PERCENT
    selected_cluster_count = DEFAULT_NUMBER_OF_CLUSTERS
    selected_target = ""
    modelling_warnings = []
    decision_tree_parameters = DEFAULT_DECISION_TREE_PARAMETERS.copy()
    kmeans_parameters = DEFAULT_KMEANS_PARAMETERS.copy()

    # Only process a dataset after the upload form is submitted
    if request.method == "POST":
        try:
            selected_models = list(
                dict.fromkeys(
                    request.form.getlist("selected_models")
                )
            )

            if not set(selected_models).issubset(ALLOWED_MODELS):
                raise UserError(
                    "The selected model is not available."
                )

            if not selected_models:
                raise UserError(
                    "Choose at least one model in Model Library."
                )

            try:
                selected_train_percent = int(
                    request.form.get(
                        "train_percent",
                        str(DEFAULT_TRAIN_PERCENT),
                    )
                )
            except ValueError as exception:
                raise UserError(
                    "The training percentage must be a whole number."
                ) from exception

            if selected_train_percent < 1 or selected_train_percent > 99:
                raise UserError(
                    "Choose a training percentage between 1 and 99."
                )

            if "decision_tree" in selected_models:
                decision_tree_parameters["criterion"] = request.form.get(
                    "tree_criterion",
                    decision_tree_parameters["criterion"],
                ).strip()
                decision_tree_parameters["splitter"] = request.form.get(
                    "tree_splitter",
                    decision_tree_parameters["splitter"],
                ).strip()
                decision_tree_parameters["max_features"] = request.form.get(
                    "tree_max_features",
                    decision_tree_parameters["max_features"],
                ).strip()

                try:
                    decision_tree_parameters["max_depth"] = int(
                        request.form.get("tree_max_depth", "8")
                    )
                    decision_tree_parameters["min_samples_split"] = int(
                        request.form.get("tree_min_samples_split", "2")
                    )
                    decision_tree_parameters["min_samples_leaf"] = int(
                        request.form.get("tree_min_samples_leaf", "1")
                    )
                except ValueError as exception:
                    raise UserError(
                        "Decision Tree number parameters must be whole numbers."
                    ) from exception

                if decision_tree_parameters["criterion"] not in {
                    "auto", "gini", "entropy", "log_loss",
                    "squared_error", "friedman_mse", "absolute_error", "poisson",
                }:
                    raise UserError("Choose an available Decision Tree criterion.")

                if decision_tree_parameters["splitter"] not in {"best", "random"}:
                    raise UserError("Choose an available Decision Tree splitter.")

                if decision_tree_parameters["max_features"] not in {
                    "auto", "sqrt", "log2", "none",
                }:
                    raise UserError("Choose an available maximum-features option.")

                if decision_tree_parameters["max_depth"] < 1:
                    raise UserError("Decision Tree maximum depth must be at least 1.")

                if decision_tree_parameters["min_samples_split"] < 2:
                    raise UserError("Minimum samples to split must be at least 2.")

                if decision_tree_parameters["min_samples_leaf"] < 1:
                    raise UserError("Minimum samples per leaf must be at least 1.")

            try:
                selected_cluster_count = int(
                    request.form.get(
                        "number_of_clusters",
                        str(DEFAULT_NUMBER_OF_CLUSTERS),
                    )
                )
            except ValueError as exception:
                raise UserError(
                    "The number of clusters must be a whole number."
                ) from exception

            if "kmeans" in selected_models:
                kmeans_parameters["init"] = request.form.get(
                    "kmeans_init",
                    kmeans_parameters["init"],
                ).strip()

                try:
                    kmeans_parameters["n_init"] = int(
                        request.form.get("kmeans_n_init", "10")
                    )
                    kmeans_parameters["max_iter"] = int(
                        request.form.get("kmeans_max_iter", "300")
                    )
                    kmeans_parameters["tolerance"] = float(
                        request.form.get("kmeans_tolerance", "0.0001")
                    )
                except ValueError as exception:
                    raise UserError(
                        "K-Means parameters must contain valid numbers."
                    ) from exception

                if kmeans_parameters["init"] not in {"k-means++", "random"}:
                    raise UserError("Choose an available K-Means initialization method.")

                if kmeans_parameters["n_init"] < 1:
                    raise UserError("K-Means restart count must be at least 1.")

                if kmeans_parameters["max_iter"] < 1:
                    raise UserError("K-Means maximum iterations must be at least 1.")

                if kmeans_parameters["tolerance"] <= 0:
                    raise UserError("K-Means tolerance must be greater than 0.")

            stored_path, original_filename = save_uploaded_dataset(
                request.files.get("dataset")
            )

            # Read the uploaded dataset so its columns and preview are available.
            uploaded_dataset = inspect_dataset(
                stored_path,
                original_filename,
            )

            # During this stage, the final dataset column is automatically used
            # as the Decision Tree prediction target.
            selected_target = uploaded_dataset["column_names"][-1]

            if "decision_tree" in selected_models:
                testing_percent = 100 - selected_train_percent

                if selected_train_percent < 50:
                    modelling_warnings.append(
                        f"Only {selected_train_percent}% of the rows were selected for training. "
                        "The Decision Tree may not learn enough patterns or may miss some classes."
                    )

                if testing_percent < 10:
                    modelling_warnings.append(
                        f"Only {testing_percent}% of the rows were selected for testing. "
                        "The reported metrics may be unstable or misleading."
                    )

            if "kmeans" in selected_models:
                maximum_cluster_count = uploaded_dataset["rows"] - 1

                if (
                    selected_cluster_count < 2
                    or selected_cluster_count > maximum_cluster_count
                ):
                    raise UserError(
                        "Choose at least 2 K-Means clusters and fewer clusters than dataset rows."
                    )

                if selected_cluster_count > math.sqrt(uploaded_dataset["rows"]):
                    modelling_warnings.append(
                        "The selected K-Means cluster count is high for this dataset. "
                        "Some clusters may contain very few rows and may not represent meaningful groups."
                    )

            # Render Dataset is one complete action: upload, model and return results.
            result = run_dataset(
                dataset_path=stored_path,
                target_column=selected_target,
                train_percent=selected_train_percent,
                number_of_clusters=selected_cluster_count,
                selected_models=selected_models,
                decision_tree_parameters=decision_tree_parameters,
                kmeans_parameters=kmeans_parameters,
                seed=DEFAULT_RANDOM_SEED,
            )

            print("\n" + "=" * 70)
            print("DATASET UPLOADED SUCCESSFULLY")
            print("=" * 70)
            print(f"Filename : {uploaded_dataset['filename']}")
            print(f"Rows     : {uploaded_dataset['rows']:,}")
            print(f"Columns  : {uploaded_dataset['columns']:,}\n")
            print("MODELLING AND CHART DATA COMPLETED SUCCESSFULLY\n")

        except UserError as exception:
            # Remove a saved file only when it could not be inspected.
            if stored_path is not None and uploaded_dataset is None:
                stored_path.unlink(missing_ok=True)

            error = str(exception)
            print(f"\nRENDER ERROR: {error}\n")

    # Render the HTML page (Webpage)
    return render_template(
        "index.html",
        uploaded_dataset=uploaded_dataset,
        modelling_completed=result is not None,
        result=result,
        error=error,
        selected_target=selected_target,
        selected_train_percent=selected_train_percent,
        selected_cluster_count=selected_cluster_count,
        selected_models=selected_models,
        modelling_warnings=modelling_warnings,
        decision_tree_parameters=decision_tree_parameters,
        kmeans_parameters=kmeans_parameters,
    )


# Display an understandable message when the upload exceeds 100 MB
@app.errorhandler(413)
def upload_too_large(_error):
    return render_template(
        "index.html",
        uploaded_dataset=None,
        modelling_completed=False,
        result=None,
        error="The uploaded dataset exceeds the 100 MB limit.",
        selected_target="",
        selected_train_percent=DEFAULT_TRAIN_PERCENT,
        selected_cluster_count=DEFAULT_NUMBER_OF_CLUSTERS,
        selected_models=DEFAULT_SELECTED_MODELS,
        modelling_warnings=[],
        decision_tree_parameters=DEFAULT_DECISION_TREE_PARAMETERS,
        kmeans_parameters=DEFAULT_KMEANS_PARAMETERS,
    ), 413

# ----------------------------------------------------
# Run Website Locally
# ----------------------------------------------------

# Allow user to run the project in webpage 127.0.0.1:8000
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        use_reloader=False,
    )
