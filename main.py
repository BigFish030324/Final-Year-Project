
# ----------------------------------------------------
# Browser & combination of all file
# ----------------------------------------------------

from __future__ import annotations
import math
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# ----------------------------------------------------
# Data Science
# ----------------------------------------------------

from data_science import inspect_dataset, run_dataset
from dataset_upload import (
    MAX_UPLOAD_BYTES,
    load_uploaded_dataset,
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

# Open the upload page with GET and inspect an uploaded dataset with POST
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_dataset = None
    error = None
    stored_path = None

    # Only process a dataset after the upload form is submitted
    if request.method == "POST":
        try:
            stored_path, original_filename = save_uploaded_dataset(
                request.files.get("dataset")
            )

            # Read and inspect the file, but do not run models yet
            uploaded_dataset = inspect_dataset(
                stored_path,
                original_filename,
            )

            print("\n" + "=" * 70)
            print("DATASET UPLOADED SUCCESSFULLY")
            print("=" * 70)
            print(f"Filename : {uploaded_dataset['filename']}")
            print(f"Rows     : {uploaded_dataset['rows']:,}")
            print(f"Columns  : {uploaded_dataset['columns']:,}\n")

        except UserError as exception:
            # Remove a saved file when its contents cannot be read
            if stored_path is not None:
                stored_path.unlink(missing_ok=True)

            error = str(exception)
            print(f"\nUPLOAD ERROR: {error}\n")

    # Render the HTML page (Webpage)
    return render_template(
        "index.html",
        uploaded_dataset=uploaded_dataset,
        modelling_completed=False,
        result=None,
        error=error,
        selected_target="",
        selected_train_percent=DEFAULT_TRAIN_PERCENT,
        selected_cluster_count=DEFAULT_NUMBER_OF_CLUSTERS,
        modelling_warnings=[],
    )


# Run the models only after the user chooses the prediction target
@app.post("/run-models")
def run_models():
    uploaded_dataset = None
    result = None
    error = None
    selected_target = request.form.get(
        "target_column",
        "",
    ).strip()
    train_percent_text = request.form.get(
        "train_percent",
        str(DEFAULT_TRAIN_PERCENT),
    ).strip()
    cluster_count_text = request.form.get(
        "number_of_clusters",
        str(DEFAULT_NUMBER_OF_CLUSTERS),
    ).strip()

    # Keep defaults available when a submitted value cannot be converted.
    selected_train_percent = DEFAULT_TRAIN_PERCENT
    selected_cluster_count = DEFAULT_NUMBER_OF_CLUSTERS
    modelling_warnings = []

    try:
        stored_filename = request.form.get(
            "stored_filename",
            "",
        ).strip()
        # Convert the displayed original name into a safe filename again.
        original_filename = secure_filename(
            request.form.get("original_filename", "")
        ) or stored_filename

        dataset_path = load_uploaded_dataset(
            stored_filename
        )
        uploaded_dataset = inspect_dataset(
            dataset_path,
            original_filename,
        )

        if not selected_target:
            raise UserError(
                "Choose a prediction target before running the models."
            )

        if selected_target not in uploaded_dataset["column_names"]:
            raise UserError(
                "The selected prediction target does not exist in the dataset."
            )

        try:
            selected_train_percent = int(train_percent_text)
        except ValueError as exception:
            raise UserError(
                "The training percentage must be a whole number."
            ) from exception

        if selected_train_percent < 1 or selected_train_percent > 99:
            raise UserError(
                "Choose a training percentage between 1 and 99."
            )

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

        try:
            selected_cluster_count = int(cluster_count_text)
        except ValueError as exception:
            raise UserError(
                "The number of clusters must be a whole number."
            ) from exception

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

        # data_science.py coordinates the two algorithms stored in model_library.
        result = run_dataset(
            dataset_path=dataset_path,
            target_column=selected_target,
            train_percent=selected_train_percent,
            number_of_clusters=selected_cluster_count,
            seed=DEFAULT_RANDOM_SEED,
        )

    except UserError as exception:
        error = str(exception)
        print(f"\nMODELLING ERROR: {error}\n")

    return render_template(
        "index.html",
        uploaded_dataset=uploaded_dataset,
        modelling_completed=result is not None,
        result=result,
        error=error,
        selected_target=selected_target,
        selected_train_percent=selected_train_percent,
        selected_cluster_count=selected_cluster_count,
        modelling_warnings=modelling_warnings,
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
        modelling_warnings=[],
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
