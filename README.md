# DataComparison FYP

This Final Year Project is a Flask website for investigating structured datasets, creating cleaned working copies and comparing machine-learning models on the same train/test split.

## Run locally

The normal Windows command remains:

```powershell
python main.py
```

On macOS or Linux, use `python3 main.py` if the `python` command is unavailable. Then open `http://127.0.0.1:8000`. Python 3.10 or newer is required; Python 3.11 or 3.12 is recommended. `run.py --check` can repair and verify the existing `.venv` when needed.

To prepare and verify the environment without starting the website:

```powershell
python run.py --check
```

If startup fails, read the message printed by `run.py`. Common causes are Python older than 3.10, a missing `venv` module, blocked internet access during the first dependency installation, or a damaged `.venv`. Only `.venv` should be removed when recreating the environment. The ignored `runtime_data` folder contains local accounts and user work after the website is used.

## Main workflow

1. Select the visible train/test split.
2. Upload a local `.csv` or `.xlsx` dataset, or paste a public Kaggle dataset page link. The website downloads public Kaggle data directly without a Kaggle account or extra Kaggle software, then chooses the largest non-empty CSV/XLSX file that passes the 1,000 MB limit.
3. Review the automatic target and task explanation, then override the target or exclude noise columns if needed.
4. Drag or click one or two compatible models. Small jobs run automatically; longer jobs ask whether to use sampled, chunked-profile, or full processing.
5. Follow progress in the terminal-style panel and export the completed result as PDF, PNG, or XLSX.

The six built-in models are Decision Tree and Logistic Regression for classification, Linear Regression and Random Forest for regression, and K-Means and Agglomerative Clustering for clustering.

## Data cleaning

Cleaning never overwrites the original upload. Each action creates a separate working copy that can be downloaded as CSV/XLSX or sent directly to Comparison. Joining supports inner, left, right, and outer joins. Identical duplicate columns are hidden and reported; the user can tick them to bring them back.

## Main project files

- `main.py` starts the Flask development server.
- `website_application.py` controls pages, accounts, uploads, results and API routes.
- `data_science.py` contains dataset profiling, cleaning and modelling functions.
- `custom_model_runner.py` and `custom_cleaning_runner.py` run trusted local custom code separately.
- `templates/` contains the website pages.
- `static/css/website_styles.css` contains the visual design.
- `static/js/website_controller.js` contains the browser interactions.
- `run.py` prepares and checks the virtual environment.

## Accounts and local storage

This version uses local JSON/text files under `runtime_data/`. Passwords are salted and hashed with Werkzeug and are never stored as readable text. The folder is excluded from version control. A future database can replace this storage layer without changing the main data-science services.

Registering a website account is runtime activity, not a source-code upload. It writes to the ignored `runtime_data/` folder and does not create a Git commit or push any private data to GitHub.

## Trusted local custom Python models

Custom models must define `build_model(params)` and return an estimator with `fit` and `predict` methods. Only NumPy, Pandas, scikit-learn, `math`, and `statistics` imports are accepted. Obvious file, network, operating-system, subprocess, and dynamic-code operations are blocked. Execution occurs in a separate process with a 45-second limit.

This reduces accidental risk but is not a secure sandbox for untrusted public users. Keep the feature local, never paste secrets into model code, and use a container-grade isolation service before any public deployment.
