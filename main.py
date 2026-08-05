
# ----------------------------------------------------
# Browser & combination of all file
# ----------------------------------------------------

from __future__ import annotations
from flask import Flask, render_template, request

# ----------------------------------------------------
# Data Science
# ----------------------------------------------------

from data_science import inspect_dataset
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
