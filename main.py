
# ----------------------------------------------------
# Browser & combination of all file
# ----------------------------------------------------

from __future__ import annotations
from flask import Flask, render_template

# ----------------------------------------------------
# Data Science
# ----------------------------------------------------

from data_science import (
    DATASET_PATH,
    run_dataset,
)
from error_handler import UserError

# ----------------------------------------------------
# Website
# ----------------------------------------------------

app = Flask(__name__)

# When user runs or refreshes the home page, run index()
@app.get("/")
def index():
    # Runs the whole data modeling process
    try:
        result = run_dataset()
        error = None
    except UserError as exception:
        result = None
        error = str(exception)
        print(f"\nMODELLING ERROR: {error}\n")

    # Render the HTML page (Webpage)
    return render_template(
        "index.html",
        dataset_path = str(DATASET_PATH),
        modelling_completed = result is not None,
        result=result,
        error = error,
    )

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
