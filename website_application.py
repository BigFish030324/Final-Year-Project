# --------------------
# Website Application
# This file controls the pages, accounts, dataset uploads, saved results and website API routes.
# --------------------
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# --------------------
# Virtual Environment
# Use the project environment when this file is started directly so the required libraries are available.
# --------------------
if __name__ == "__main__":
    environment_directory = Path(__file__).resolve().parent / ".venv"
    bundled_python = environment_directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if bundled_python.exists() and Path(sys.executable).resolve() != bundled_python.resolve():
        raise SystemExit(subprocess.call(
            [str(bundled_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(Path(__file__).resolve().parent),
        ))

import pandas as pd
from flask import (
    Flask, abort, flash, jsonify, redirect, render_template, request,
    send_file, session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from data_science import (
    MAX_TRAINING_ROWS, MODEL_BY_ID, MODEL_CATALOG, PROCESSING_MODES, UserFacingError,
    apply_cleaning, compare_models, custom_model_rules, join_frames, load_json,
    model_compatibility, normalise_processing_mode, processing_preflight, profile_dataset, read_dataset,
    recommend_models,
    records_for_json, save_json, utc_now, validate_custom_cleaning_code, validate_custom_code,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "runtime_data"
ACCOUNT_FILE = DATA_DIR / "accounts.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
HISTORY_DIR = DATA_DIR / "history"
CUSTOM_DIR = DATA_DIR / "custom_models"
CUSTOM_CLEANING_DIR = DATA_DIR / "custom_cleaning"
MAX_UPLOAD_BYTES = 1_000_000_000

for directory in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, HISTORY_DIR, CUSTOM_DIR, CUSTOM_CLEANING_DIR):
    directory.mkdir(parents=True, exist_ok=True)

secret_path = DATA_DIR / ".session_secret"
if not secret_path.exists():
    secret_path.write_text(secrets.token_hex(32), encoding="utf-8")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=secret_path.read_text(encoding="utf-8").strip(),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def current_user() -> str | None:
    username = session.get("username")
    if not username:
        return None
    account = find_account(str(username))
    if not account:
        session.pop("username", None)
        session.pop("active_dataset_id", None)
        return None
    return str(account[1].get("username") or account[0])


def find_account(identifier: str, accounts: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    """Find an account by its stable key, public username, or email address."""
    accounts = accounts if accounts is not None else load_json(ACCOUNT_FILE, {})
    identity = identifier.strip().casefold()
    if not identity:
        return None
    if identity in accounts:
        return identity, accounts[identity]
    for key, account in accounts.items():
        aliases = (account.get("username"), account.get("display_name"), account.get("email"))
        if any(str(alias or "").strip().casefold() == identity for alias in aliases):
            return key, account
    return None


def current_account() -> dict[str, Any]:
    match = find_account(current_user() or "")
    return match[1] if match else {}


def public_username(account: dict[str, Any] | None = None) -> str:
    account = account if account is not None else current_account()
    return str(account.get("display_name") or account.get("username") or current_user() or "User")


def valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value.strip()))


def valid_display_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{1,49}", value.strip()))


def account_identity_taken(
    accounts: dict[str, Any], value: str, *, exclude_key: str | None = None,
) -> bool:
    identity = value.strip().casefold()
    for key, account in accounts.items():
        if exclude_key is not None and key.casefold() == exclude_key.casefold():
            continue
        aliases = (key, account.get("username"), account.get("display_name"), account.get("email"))
        if any(str(alias or "").strip().casefold() == identity for alias in aliases):
            return True
    return False


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user():
            if request.path.startswith("/api/"):
                return json_error("Please sign in to continue.", 401)
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def user_slug() -> str:
    username = current_user()
    if not username:
        raise UserFacingError("Please sign in to continue.")
    return secure_filename(username.lower()) or "user"


def user_data_directory(root: Path) -> Path:
    """Return a user's private directory, creating it on first use."""
    directory = root / user_slug()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def user_upload_dir() -> Path:
    return user_data_directory(UPLOAD_DIR)


def metadata_path(dataset_id: str) -> Path:
    return user_upload_dir() / f"{secure_filename(dataset_id)}.json"


def load_metadata(dataset_id: str) -> dict[str, Any]:
    meta = load_json(metadata_path(dataset_id), None)
    if not meta:
        raise UserFacingError("The selected dataset was not found in your local workspace.")
    path = Path(meta["stored_path"])
    if not path.exists() or path.parent.resolve() != user_upload_dir().resolve():
        raise UserFacingError("The selected dataset file is unavailable.")
    return meta


def active_metadata() -> dict[str, Any]:
    dataset_id = session.get("active_dataset_id")
    if not dataset_id:
        raise UserFacingError("Upload or select a dataset first.")
    return load_metadata(dataset_id)


def save_metadata(meta: dict[str, Any]) -> None:
    save_json(metadata_path(meta["id"]), meta)


def save_derived_dataset(frame: pd.DataFrame, *, name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Persist a generated CSV and its profile using the same workflow everywhere."""
    dataset_id = uuid.uuid4().hex
    stored = user_upload_dir() / f"{dataset_id}.csv"
    frame.to_csv(stored, index=False)
    meta = profile_dataset(stored, original_name=name, dataset_id=dataset_id)
    meta.update(metadata)
    save_metadata(meta)
    return meta


# --------------------
# Kaggle Dataset Import
# Check a public Kaggle dataset link, download it into temporary local storage and keep its largest usable CSV or XLSX file.
# --------------------
def kaggle_dataset_handle(dataset_url: str) -> str:
    value = str(dataset_url or "").strip()
    if not value:
        raise UserFacingError("Paste a Kaggle dataset URL first.")
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or hostname not in {"kaggle.com", "www.kaggle.com"}:
        raise UserFacingError("Use a Kaggle link starting with https://www.kaggle.com/datasets/.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0].lower() != "datasets":
        raise UserFacingError("Use a Kaggle dataset page link, for example kaggle.com/datasets/owner/dataset-name.")
    owner, dataset_name = parts[1], parts[2]
    safe_part = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    if not safe_part.fullmatch(owner) or not safe_part.fullmatch(dataset_name):
        raise UserFacingError("The Kaggle dataset owner or dataset name is not valid.")
    return f"{owner}/{dataset_name}"


def import_kaggle_dataset(dataset_url: str) -> dict[str, Any]:
    handle = kaggle_dataset_handle(dataset_url)
    try:
        import kagglehub
    except ImportError as error:
        raise UserFacingError(
            "Kaggle importing is not installed yet. Run 'python -m pip install -r requirements.txt' and restart the website."
        ) from error

    upload_directory = user_upload_dir()
    with tempfile.TemporaryDirectory(prefix="kaggle_import_", dir=upload_directory) as temporary:
        try:
            downloaded = Path(kagglehub.dataset_download(
                handle,
                output_dir=temporary,
                force_download=True,
            ))
        except Exception as error:
            message = str(error).lower()
            if "401" in message or "403" in message or "permission" in message or "credential" in message:
                raise UserFacingError(
                    "Kaggle did not allow this download. Use a public dataset that does not require account permission."
                ) from error
            raise UserFacingError(
                "The Kaggle dataset could not be downloaded. Check that the public dataset link still exists and try again."
            ) from error

        search_root = downloaded if downloaded.is_dir() else downloaded.parent
        downloaded_files = list(search_root.rglob("*")) if downloaded.is_dir() else [downloaded]
        supported_files = [
            path for path in downloaded_files
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"} and path.stat().st_size > 0
        ]
        if not supported_files:
            raise UserFacingError("This Kaggle dataset has no non-empty CSV or XLSX file to investigate.")

        allowed_files = [path for path in supported_files if path.stat().st_size <= MAX_UPLOAD_BYTES]
        if not allowed_files:
            raise UserFacingError("The CSV or XLSX files in this Kaggle dataset exceed the 1,000 MB limit.")

        first_dataset_error: str | None = None
        for source_path in sorted(allowed_files, key=lambda path: path.stat().st_size, reverse=True):
            dataset_id = uuid.uuid4().hex
            stored_path = upload_directory / f"{dataset_id}{source_path.suffix.lower()}"
            shutil.copy2(source_path, stored_path)
            try:
                metadata = profile_dataset(
                    stored_path,
                    original_name=source_path.name,
                    dataset_id=dataset_id,
                )
            except UserFacingError as error:
                stored_path.unlink(missing_ok=True)
                first_dataset_error = first_dataset_error or str(error)
                continue
            metadata.update({
                "source_type": "kaggle",
                "source_url": f"https://www.kaggle.com/datasets/{handle}",
                "kaggle_handle": handle,
                "kaggle_file": source_path.relative_to(search_root).as_posix(),
            })
            save_metadata(metadata)
            session["active_dataset_id"] = dataset_id
            return metadata

    raise UserFacingError(
        first_dataset_error or "The Kaggle dataset files could not be read as usable tabular data."
    )


def next_numbered_record_name(directory: Path, prefix: str) -> str:
    """Choose the next available display name for a collection of JSON records."""
    pattern = re.compile(rf"{re.escape(prefix)}\s+(\d+)", re.IGNORECASE)
    used_numbers = []
    for path in directory.glob("*.json"):
        record = load_json(path, None)
        match = pattern.fullmatch(str((record or {}).get("name", "")).strip())
        if match:
            used_numbers.append(int(match.group(1)))
    return f"{prefix} {max(used_numbers, default=0) + 1}"


def history_path() -> Path:
    return HISTORY_DIR / f"{user_slug()}.json"


def history_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": result["id"], "dataset_id": result["dataset_id"],
        "dataset_name": result["dataset_name"], "target": result["target"],
        "task": result["task"], "train_pct": result["train_pct"],
        "models": [{"id": m["model_id"], "name": m["name"], "metrics": m["metrics"], "params": m.get("parameters", {})} for m in result["models"]],
        "selected_columns": result.get("selected_columns", []),
        "mode": result.get("mode", "full"),
        "mode_label": result.get("affordability", {}).get("resource_mode_label", str(result.get("mode", "full")).title()),
        "rows_processed": result.get("rows_processed", 0),
        "rows_available": result.get("rows_available", result.get("rows_processed", 0)),
        "total_training_seconds": result.get("total_training_seconds", 0),
        "affordability": result.get("affordability", {}),
        "created_at": result["created_at"],
    }


def add_history(result: dict[str, Any]) -> None:
    entries = load_json(history_path(), [])
    summary = history_summary(result)
    entries.insert(0, summary)
    save_json(history_path(), entries[:100])


def result_path(result_id: str) -> Path:
    return RESULT_DIR / user_slug() / f"{secure_filename(result_id)}.json"


def save_result(result: dict[str, Any]) -> None:
    save_json(result_path(result["id"]), result)


def load_result(result_id: str) -> dict[str, Any]:
    result = load_json(result_path(result_id), None)
    if not result:
        raise UserFacingError("That comparison result is unavailable.")
    return result


def update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        job = JOBS.setdefault(job_id, {})
        job.update(changes)


def run_comparison_job(job_id: str, username: str, dataset_id: str, payload: dict[str, Any]) -> None:
    # Flask's signed session is request-local.  The background worker receives
    # the resolved user and paths explicitly instead of touching session state.
    try:
        upload_dir = UPLOAD_DIR / secure_filename(username.lower())
        meta = load_json(upload_dir / f"{secure_filename(dataset_id)}.json", None)
        if not meta:
            raise UserFacingError("The dataset metadata is unavailable.")
        path = Path(meta["stored_path"])

        def progress(percent: int, message: str):
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["progress"] = percent
                job["logs"].append({"time": utc_now(), "message": message})

        result = compare_models(
            path, meta, payload["models"], train_pct=int(payload.get("train_pct", 70)),
            seed=int(payload.get("seed", 42)), mode=payload.get("mode", "full"), progress=progress,
        )
        result_file = RESULT_DIR / secure_filename(username.lower()) / f"{result['id']}.json"
        save_json(result_file, result)
        entries_file = HISTORY_DIR / f"{secure_filename(username.lower())}.json"
        entries = load_json(entries_file, [])
        entries.insert(0, history_summary(result))
        save_json(entries_file, entries[:100])
        update_job(job_id, status="complete", progress=100, result=result)
        with JOBS_LOCK:
            JOBS[job_id]["logs"].append({"time": utc_now(), "message": "Comparison completed successfully"})
    except Exception as exc:
        message = str(exc) if isinstance(exc, UserFacingError) else "The comparison failed. Check the selected data and model settings."
        update_job(job_id, status="failed", error=message, debug=traceback.format_exc())


@app.errorhandler(413)
def too_large(_error):
    if request.path.startswith("/api/"):
        return json_error("The file exceeds the 1,000 MB upload limit.", 413)
    flash("The file exceeds the 1,000 MB upload limit.", "error")
    return redirect(request.referrer or url_for("comparison"))


@app.errorhandler(UserFacingError)
def handle_user_error(error):
    if request.path.startswith("/api/"):
        return json_error(str(error), 400)
    flash(str(error), "error")
    return redirect(request.referrer or url_for("comparison"))


@app.context_processor
def inject_globals():
    return {
        "current_user": current_user(),
        "current_account_name": public_username(),
        "model_catalog": MODEL_CATALOG,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identity = request.form.get("identity", request.form.get("username", "")).strip()
        password = request.form.get("password", "")
        match = find_account(identity)
        account = match[1] if match else None
        if account and check_password_hash(account["password_hash"], password):
            session.clear()
            session["username"] = account["username"]
            return redirect(request.args.get("next") or url_for("comparison"))
        flash("The email, display name, or password is incorrect.", "error")
    return render_template("auth.html", mode="login")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not display_name or not email or not password:
            flash("Fill in Display Name, Email, and Password before creating your account.", "error")
        elif not valid_display_name(display_name):
            flash("Use 2-50 letters, numbers, spaces, dots, underscores, or hyphens for the display name.", "error")
        elif not valid_email(email):
            flash("Enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Use a password with at least 8 characters.", "error")
        else:
            accounts = load_json(ACCOUNT_FILE, {})
            key = display_name.lower()
            if account_identity_taken(accounts, display_name):
                flash("That display name is already registered.", "error")
            elif account_identity_taken(accounts, email):
                flash("That email address is already registered.", "error")
            else:
                accounts[key] = {
                    "username": display_name, "display_name": display_name, "email": email,
                    "password_hash": generate_password_hash(password), "created_at": utc_now(),
                }
                save_json(ACCOUNT_FILE, accounts)
                session.clear()
                session["username"] = display_name
                return redirect(url_for("comparison"))
    return render_template("auth.html", mode="register")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@app.route("/comparison")
@login_required
def comparison():
    return render_template("comparison.html", active_tab="comparison", page="comparison")


@app.route("/cleaning")
@login_required
def cleaning():
    return render_template("cleaning.html", active_tab="cleaning", page="cleaning")


@app.route("/libraries")
@login_required
def libraries():
    return render_template("libraries.html", active_tab="libraries", page="libraries")


@app.route("/history")
@login_required
def history():
    return render_template("history.html", active_tab="history", page="history")


@app.route("/custom-models")
@login_required
def custom_models():
    return render_template("custom_models.html", active_tab="custom_models", page="custom_models", rules=custom_model_rules())


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", active_tab="settings", page="settings")


@app.get("/api/session")
@login_required
def api_session():
    accounts = load_json(ACCOUNT_FILE, {})
    account = accounts.get(current_user().lower(), {})
    settings_data = load_json(SETTINGS_FILE, {}).get(current_user().lower(), {})
    active = None
    if session.get("active_dataset_id"):
        try:
            active = active_metadata()
        except UserFacingError:
            session.pop("active_dataset_id", None)
    public_account = {key: value for key, value in account.items() if key != "password_hash"}
    return jsonify({"ok": True, "user": public_account, "settings": settings_data, "active_dataset": active})


@app.post("/api/datasets/upload")
@login_required
def upload_dataset():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        raise UserFacingError("Choose a CSV or XLSX dataset to upload.")
    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise UserFacingError("Only CSV and XLSX datasets are accepted; JSON is not supported.")
    dataset_id = uuid.uuid4().hex
    stored = user_upload_dir() / f"{dataset_id}{suffix}"
    uploaded.save(stored)
    if stored.stat().st_size > MAX_UPLOAD_BYTES:
        stored.unlink(missing_ok=True)
        raise UserFacingError("The file exceeds the 1,000 MB upload limit.")
    try:
        meta = profile_dataset(stored, original_name=original_name, dataset_id=dataset_id)
    except Exception:
        stored.unlink(missing_ok=True)
        raise
    save_metadata(meta)
    session["active_dataset_id"] = dataset_id
    return jsonify({"ok": True, "dataset": meta})


@app.post("/api/datasets/kaggle")
@login_required
def upload_kaggle_dataset():
    payload = request.get_json(silent=True) or {}
    metadata = import_kaggle_dataset(str(payload.get("url", "")))
    return jsonify({"ok": True, "dataset": metadata})


@app.get("/api/datasets")
@login_required
def datasets_list():
    items = []
    for file in user_upload_dir().glob("*.json"):
        data = load_json(file, None)
        if data and Path(data.get("stored_path", "")).exists():
            items.append(data)
    items.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return jsonify({"ok": True, "datasets": items, "active_id": session.get("active_dataset_id")})


@app.post("/api/datasets/<dataset_id>/activate")
@login_required
def activate_dataset(dataset_id: str):
    meta = load_metadata(dataset_id)
    session["active_dataset_id"] = meta["id"]
    return jsonify({"ok": True, "dataset": meta})


@app.delete("/api/datasets/<dataset_id>")
@login_required
def delete_dataset(dataset_id: str):
    meta = load_metadata(dataset_id)
    upload_directory = user_upload_dir().resolve()
    stored = Path(meta.get("stored_path", "")).resolve()
    if stored.parent != upload_directory:
        raise UserFacingError("The dataset file is outside this local account and cannot be deleted.")
    stored.unlink(missing_ok=True)
    metadata_path(meta["id"]).unlink(missing_ok=True)
    remaining = []
    for path in user_upload_dir().glob("*.json"):
        item = load_json(path, None)
        if item and Path(item.get("stored_path", "")).exists():
            remaining.append(item)
    remaining.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    next_dataset = remaining[0] if remaining else None
    if session.get("active_dataset_id") == meta["id"]:
        if next_dataset:
            session["active_dataset_id"] = next_dataset["id"]
        else:
            session.pop("active_dataset_id", None)
    return jsonify({"ok": True, "deleted_id": meta["id"], "active_dataset": next_dataset})


@app.delete("/api/datasets")
@login_required
def delete_all_datasets():
    upload_directory = user_upload_dir().resolve()
    metadata_files = list(upload_directory.glob("*.json"))
    dataset_files = [
        path for path in upload_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"}
    ]
    for path in [*dataset_files, *metadata_files]:
        if path.resolve().parent != upload_directory:
            raise UserFacingError("A dataset file is outside this local account and cannot be deleted.")
    for path in [*dataset_files, *metadata_files]:
        path.unlink(missing_ok=True)
    session.pop("active_dataset_id", None)
    return jsonify({"ok": True, "deleted_count": len(metadata_files), "active_dataset": None})


@app.patch("/api/datasets/<dataset_id>")
@login_required
def update_dataset(dataset_id: str):
    meta = load_metadata(dataset_id)
    payload = request.get_json(silent=True) or {}
    path = Path(meta["stored_path"])
    frame = read_dataset(path, nrows=MAX_TRAINING_ROWS)
    if "target" in payload:
        target = str(payload["target"])
        if target not in frame.columns:
            raise UserFacingError("Choose a target column that exists in the dataset.")
        from data_science import detect_task
        task, reason = detect_task(frame[target])
        meta.update({"target": target, "target_confidence": "manual", "target_reason": "selected manually by the user", "task": task, "task_reason": reason})
    if "selected_columns" in payload:
        selected = [str(c) for c in payload["selected_columns"] if str(c) in frame.columns]
        if meta["target"] not in selected:
            selected.append(meta["target"])
        if len(selected) < 2:
            raise UserFacingError("Keep at least one feature column in addition to the target.")
        meta["selected_columns"] = selected
    if "date_column" in payload:
        date_column = payload["date_column"]
        if date_column and date_column not in frame.columns:
            raise UserFacingError("Choose a date/time column that exists in the dataset.")
        meta["date_column"] = date_column
    save_metadata(meta)
    return jsonify({"ok": True, "dataset": meta})


@app.get("/api/models")
@login_required
def models_catalog():
    meta = None
    try:
        meta = active_metadata()
    except UserFacingError:
        pass
    catalog = []
    parameter_help = {
        "fit_intercept": "Whether the line should estimate a constant intercept.",
        "positive": "Restrict fitted coefficients to zero or positive values.",
        "criterion": "Rule used to measure the quality of each tree split.",
        "splitter": "Whether the tree chooses the best or a random valid split.",
        "max_depth": "Maximum number of levels in each decision tree.",
        "min_samples_split": "Minimum rows required before a tree node can split.",
        "min_samples_leaf": "Minimum rows that must remain in each final leaf.",
        "max_features": "Number of input features considered when choosing a tree split.",
        "bootstrap": "Train each forest tree from a resampled copy of the training rows.",
        "n_estimators": "Number of trees or boosting stages built by the model.",
        "C": "Regularisation strength; smaller values create a simpler model.",
        "max_iter": "Maximum number of training iterations.",
        "solver": "Algorithm used to optimise the model during training.",
        "class_weight": "Optionally give under-represented classes more influence.",
        "tol": "Training stops when improvements become smaller than this value.",
        "n_clusters": "Number of groups the clustering model should discover.",
        "n_init": "Number of starting arrangements tried before choosing the best clustering.",
        "init": "Method used to choose the initial cluster centres.",
        "linkage": "Rule used to decide which two groups should be joined next.",
    }
    parameter_options = {
        "splitter": ["best", "random"], "max_features": ["auto", "sqrt", "log2"],
        "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
        "class_weight": ["none", "balanced"],
        "init": ["k-means++", "random"],
        "linkage": ["ward", "complete", "average", "single"],
    }
    parameter_limits = {
        "max_depth": (1, 100), "min_samples_split": (2, 1000), "min_samples_leaf": (1, 1000),
        "n_estimators": (1, 2000), "max_iter": (1, 5000),
        "n_clusters": (2, 100), "n_init": (1, 100),
        "C": (0.000001, 1000000), "tol": (0.000000001, 1),
    }
    for model in MODEL_CATALOG:
        item = dict(model)
        item["parameter_schema"] = []
        for name, value in model.get("defaults", {}).items():
            schema = {
                "name": name, "label": "Splitting Criteria" if name == "criterion" else None,
                "type": "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "text",
                "description": parameter_help.get(name, "Model library default."),
            }
            if name == "criterion" and model["id"] == "decision_tree":
                schema["choices"] = ["auto", "gini", "entropy", "log_loss"]
            elif name == "criterion" and model["id"] == "random_forest":
                schema["choices"] = ["auto", "squared_error", "friedman_mse", "absolute_error", "poisson"]
            elif name in parameter_options:
                schema["choices"] = parameter_options[name]
            if name in parameter_limits:
                schema["min"], schema["max"] = parameter_limits[name]
            item["parameter_schema"].append(schema)
        if meta:
            item["compatible"], item["compatibility_reason"] = model_compatibility(model["id"], meta["task"], meta.get("datetime_columns", []))
        else:
            item["compatible"], item["compatibility_reason"] = None, "Upload a dataset to check compatibility."
        catalog.append(item)
    for file in (CUSTOM_DIR / user_slug()).glob("*.json") if (CUSTOM_DIR / user_slug()).exists() else []:
        custom = load_json(file, None)
        if custom and custom.get("task") != "time_series":
            custom_task = custom.get("task", "classification")
            custom_compatible = None
            custom_reason = "Upload a dataset to check compatibility."
            if meta:
                if custom_task == "clustering":
                    custom_compatible = True
                    custom_reason = "Clustering explores feature groups without using the target for training."
                else:
                    custom_compatible = custom_task == meta.get("task")
                    custom_reason = "Custom model task must match the detected dataset task."
            catalog.append({
                "id": f"custom:{custom['id']}", "name": custom["name"], "family": "Custom Python",
                "tasks": [custom_task], "task_label": custom.get("task_label", ""), "version": "Local", "summary": custom["description"],
                "best_for": "Trusted local experiments written by this user.", "defaults": custom.get("defaults", {}),
                "parameter_schema": custom.get("parameters", []),
                "compatible": custom_compatible, "compatibility_reason": custom_reason,
            })
    return jsonify({"ok": True, "models": catalog})


@app.get("/api/models/recommendations")
@login_required
def model_recommendations():
    return jsonify({"ok": True, **recommend_models(active_metadata())})


@app.post("/api/comparisons/preflight")
@login_required
def comparison_preflight():
    payload = request.get_json(silent=True) or {}
    meta = active_metadata()
    model_ids = [item.get("id", "") for item in payload.get("models", [])]
    if not 1 <= len(model_ids) <= 2:
        raise UserFacingError("Choose one or two model cards first.")
    return jsonify({"ok": True, **processing_preflight(meta, model_ids)})


@app.post("/api/comparisons")
@login_required
def start_comparison():
    payload = request.get_json(silent=True) or {}
    models = payload.get("models", [])
    if not 1 <= len(models) <= 2:
        raise UserFacingError("Choose one or two model cards first.")
    payload["mode"] = normalise_processing_mode(payload.get("mode", "full"))
    custom_directory = CUSTOM_DIR / user_slug()
    for model in models:
        model_id = str(model.get("id", ""))
        if model_id.startswith("custom:"):
            custom_id = secure_filename(model_id.split(":", 1)[1])
            definition = load_json(custom_directory / f"{custom_id}.json", None)
            if not definition:
                raise UserFacingError("The selected custom model could not be found.")
            model["definition"] = definition
    meta = active_metadata()
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id, "status": "running", "progress": 1,
            "logs": [{"time": utc_now(), "message": "Comparison request accepted"}],
        }
    worker = threading.Thread(
        target=run_comparison_job,
        args=(job_id, current_user(), meta["id"], payload), daemon=True,
    )
    worker.start()
    return jsonify({"ok": True, "job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
@login_required
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return json_error("That processing job was not found.", 404)
        public = {key: value for key, value in job.items() if key != "debug"}
    return jsonify({"ok": True, "job": public})


@app.post("/api/cleaning/apply")
@login_required
def cleaning_apply():
    payload = request.get_json(silent=True) or {}
    source = active_metadata()
    frame = read_dataset(Path(source["stored_path"]))
    cleaned, message = apply_cleaning(frame, payload.get("operation", ""), payload.get("options", {}))
    name = f"cleaned_{Path(source['filename']).stem}.csv"
    meta = save_derived_dataset(cleaned, name=name, metadata={
        "is_cleaned": True, "source_dataset_id": source["id"], "cleaning_message": message,
    })
    return jsonify({"ok": True, "dataset": meta, "message": message})


def custom_cleaning_directory() -> Path:
    return user_data_directory(CUSTOM_CLEANING_DIR)


def next_custom_cleaning_name() -> str:
    return next_numbered_record_name(custom_cleaning_directory(), "Custom Cleaning")


def normalise_custom_cleaning(payload: dict[str, Any], action_id: str | None = None) -> dict[str, Any]:
    safe_id = secure_filename(str(action_id or payload.get("id") or uuid.uuid4().hex))
    existing = load_json(custom_cleaning_directory() / f"{safe_id}.json", {})
    name = str(payload.get("name", "")).strip()
    if not name:
        name = str(existing.get("name", "")).strip() or next_custom_cleaning_name()
    code = str(payload.get("code", ""))
    errors = validate_custom_cleaning_code(code)
    if errors:
        raise UserFacingError("The custom cleaning action does not pass validation: " + " ".join(errors))
    return {
        "id": safe_id, "name": name, "description": str(payload.get("description", "")).strip(),
        "code": code, "created_at": existing.get("created_at", utc_now()), "updated_at": utc_now(),
    }


@app.get("/api/custom-cleaning")
@login_required
def list_custom_cleaning():
    actions = [load_json(path, None) for path in custom_cleaning_directory().glob("*.json")]
    return jsonify({"ok": True, "actions": [action for action in actions if action], "suggested_name": next_custom_cleaning_name()})


@app.post("/api/custom-cleaning/validate")
@login_required
def validate_custom_cleaning():
    errors = validate_custom_cleaning_code(str((request.get_json(silent=True) or {}).get("code", "")))
    return jsonify({"ok": not errors, "errors": errors})


@app.post("/api/custom-cleaning")
@login_required
def save_custom_cleaning():
    action = normalise_custom_cleaning(request.get_json(silent=True) or {})
    save_json(custom_cleaning_directory() / f"{action['id']}.json", action)
    return jsonify({"ok": True, "action": action})


@app.put("/api/custom-cleaning/<action_id>")
@login_required
def update_custom_cleaning(action_id: str):
    safe_id = secure_filename(action_id)
    path = custom_cleaning_directory() / f"{safe_id}.json"
    if not safe_id or not path.exists():
        abort(404)
    action = normalise_custom_cleaning(request.get_json(silent=True) or {}, safe_id)
    save_json(path, action)
    return jsonify({"ok": True, "action": action})


@app.delete("/api/custom-cleaning/<action_id>")
@login_required
def delete_custom_cleaning(action_id: str):
    safe_id = secure_filename(action_id)
    path = custom_cleaning_directory() / f"{safe_id}.json"
    if not safe_id or not path.exists():
        abort(404)
    path.unlink()
    return jsonify({"ok": True})


@app.post("/api/custom-cleaning/<action_id>/run")
@login_required
def run_custom_cleaning(action_id: str):
    safe_id = secure_filename(action_id)
    action = load_json(custom_cleaning_directory() / f"{safe_id}.json", None)
    if not safe_id or not action:
        abort(404)
    errors = validate_custom_cleaning_code(str(action.get("code", "")))
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    source = active_metadata()
    frame = read_dataset(Path(source["stored_path"]))
    worker = BASE_DIR / "custom_cleaning_runner.py"
    with tempfile.TemporaryDirectory(prefix="datacomparison_cleaning_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "input.pkl"
        output_path = temporary_path / "output.pkl"
        response_path = temporary_path / "response.json"
        request_path = temporary_path / "request.json"
        frame.to_pickle(input_path)
        request_path.write_text(json.dumps({
            "code": action["code"], "input": str(input_path), "output": str(output_path), "response": str(response_path),
        }), encoding="utf-8")
        try:
            subprocess.run([sys.executable, str(worker), str(request_path)], cwd=str(BASE_DIR), timeout=45, check=False, capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:
            raise UserFacingError("The custom cleaning action exceeded the 45-second local time limit.") from exc
        worker_response = load_json(response_path, None)
        if not worker_response or not worker_response.get("ok") or not output_path.exists():
            raise UserFacingError((worker_response or {}).get("error", "The custom cleaning action could not be completed."))
        cleaned = pd.read_pickle(output_path)
    name = f"custom_cleaned_{Path(source['filename']).stem}.csv"
    message = f"Applied custom cleaning action '{action['name']}'."
    meta = save_derived_dataset(cleaned, name=name, metadata={
        "is_cleaned": True, "source_dataset_id": source["id"], "cleaning_message": message,
        "custom_cleaning_action_id": safe_id,
    })
    return jsonify({"ok": True, "dataset": meta, "message": message})


@app.post("/api/cleaning/join")
@login_required
def cleaning_join():
    payload = request.get_json(silent=True) or {}
    left = active_metadata()
    right = load_metadata(str(payload.get("right_dataset_id", "")))
    left_frame = read_dataset(Path(left["stored_path"]))
    right_frame = read_dataset(Path(right["stored_path"]))
    left_rename = {str(k): str(v).strip() for k, v in payload.get("left_rename", {}).items() if k in left_frame.columns and str(v).strip()}
    right_rename = {str(k): str(v).strip() for k, v in payload.get("right_rename", {}).items() if k in right_frame.columns and str(v).strip()}
    left_key = str(payload.get("left_key", ""))
    right_key = str(payload.get("right_key", ""))
    left_frame = left_frame.rename(columns=left_rename)
    right_frame = right_frame.rename(columns=right_rename)
    left_key = left_rename.get(left_key, left_key)
    right_key = right_rename.get(right_key, right_key)
    suffixes = (str(payload.get("left_suffix", "_left")), str(payload.get("right_suffix", "_right")))
    joined, removed = join_frames(
        left_frame, right_frame, left_key=left_key,
        right_key=right_key, how=str(payload.get("how", "inner")),
        suffixes=suffixes, keep_duplicate_columns=payload.get("keep_duplicate_columns", []),
    )
    mapping = {k: v for k, v in payload.get("rename", {}).items() if k in joined.columns and str(v).strip()}
    joined = joined.rename(columns=mapping)
    name = f"joined_{Path(left['filename']).stem}_{Path(right['filename']).stem}.csv"
    meta = save_derived_dataset(joined, name=name, metadata={
        "is_cleaned": True, "source_dataset_ids": [left["id"], right["id"]],
        "removed_duplicate_columns": removed,
    })
    return jsonify({
        "ok": True, "dataset": meta, "removed_duplicate_columns": removed,
        "message": f"Joined both datasets. {len(removed)} duplicate column(s) were hidden; tick them and run again to bring them back.",
    })


@app.post("/api/datasets/<dataset_id>/send-to-comparison")
@login_required
def send_to_comparison(dataset_id: str):
    meta = load_metadata(dataset_id)
    session["active_dataset_id"] = meta["id"]
    return jsonify({"ok": True, "redirect": url_for("comparison"), "dataset": meta})


@app.get("/api/datasets/<dataset_id>/download.<fmt>")
@login_required
def download_dataset(dataset_id: str, fmt: str):
    meta = load_metadata(dataset_id)
    frame = read_dataset(Path(meta["stored_path"]))
    stem = secure_filename(Path(meta["filename"]).stem)
    if fmt == "csv":
        output = io.BytesIO(frame.to_csv(index=False).encode("utf-8-sig"))
        return send_file(output, as_attachment=True, download_name=f"{stem}.csv", mimetype="text/csv")
    if fmt == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Cleaned Data")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=f"{stem}.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    abort(404)


@app.get("/api/history")
@login_required
def get_history():
    return jsonify({"ok": True, "history": load_json(history_path(), [])})


@app.delete("/api/history")
@login_required
def clear_history():
    save_json(history_path(), [])
    return jsonify({"ok": True})


@app.post("/api/history/<result_id>/restore")
@login_required
def restore_history(result_id: str):
    safe_id = secure_filename(result_id)
    entry = next((item for item in load_json(history_path(), []) if item.get("id") == safe_id), None)
    if not entry:
        raise UserFacingError("That history entry is unavailable.")
    meta = load_metadata(str(entry.get("dataset_id", "")))
    result = load_result(safe_id)
    target = str(entry.get("target", ""))
    if target in meta.get("column_names", []):
        meta["target"] = target
    selected = [name for name in result.get("selected_columns", []) if name in meta.get("column_names", [])]
    if selected:
        meta["selected_columns"] = selected
    save_metadata(meta)
    session["active_dataset_id"] = meta["id"]
    comparison = {
        "train_pct": int(entry.get("train_pct", 70)),
        "mode": result.get("mode", "full"),
        "models": [
            {"id": model.get("model_id"), "name": model.get("name"), "params": model.get("parameters", {})}
            for model in result.get("models", [])
        ],
    }
    return jsonify({"ok": True, "redirect": url_for("comparison"), "comparison": comparison})


@app.get("/api/results/<result_id>")
@login_required
def get_result(result_id: str):
    return jsonify({"ok": True, "result": load_result(result_id)})


def create_result_xlsx(result: dict[str, Any]) -> io.BytesIO:
    output = io.BytesIO()
    receipt = result.get("affordability", {})
    summary = pd.DataFrame([
        {"Field": "Dataset", "Value": result["dataset_name"]},
        {"Field": "Target", "Value": result["target"]},
        {"Field": "Task", "Value": result["task"]},
        {"Field": "Train/Test", "Value": f"{result['train_pct']}% / {result['test_pct']}%"},
        {"Field": "Processing mode", "Value": receipt.get("resource_mode_label", str(result.get("mode", "full")).title())},
        {"Field": "Rows available", "Value": result.get("rows_available", result["rows_processed"])},
        {"Field": "Rows processed", "Value": result["rows_processed"]},
        {"Field": "Total training seconds", "Value": result.get("total_training_seconds", 0)},
    ])
    affordability = pd.DataFrame([
        {"Field": "Application licence/API fee (MYR)", "Value": receipt.get("application_fee_myr", 0)},
        {"Field": "Paid external ML API calls", "Value": receipt.get("paid_external_api_calls", 0)},
        {"Field": "Processing location", "Value": receipt.get("processing_location", "Local computer")},
        {"Field": "Enterprise software required", "Value": "No" if not receipt.get("enterprise_software_required", False) else "Yes"},
        {"Field": "Resource mode", "Value": receipt.get("resource_mode_label", str(result.get("mode", "full")).title())},
        {"Field": "Rows processed / available", "Value": f"{result.get('rows_processed', 0)} / {result.get('rows_available', result.get('rows_processed', 0))}"},
        {"Field": "Measured training seconds", "Value": receipt.get("training_seconds", result.get("total_training_seconds", 0))},
        {"Field": "Receipt note", "Value": receipt.get("note", "No application fee was charged for this local run.")},
    ])
    metrics = []
    for model in result["models"]:
        row = {"Model": model["name"], "Training seconds": model["training_seconds"]}
        row.update(model["metrics"])
        metrics.append(row)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        affordability.to_excel(writer, index=False, sheet_name="Affordability Receipt")
        pd.DataFrame(metrics).to_excel(writer, index=False, sheet_name="Model Metrics")
        pd.DataFrame({"Selected columns": result["selected_columns"]}).to_excel(writer, index=False, sheet_name="Columns")
    output.seek(0)
    return output


@app.get("/api/results/<result_id>/export.<fmt>")
@login_required
def export_result(result_id: str, fmt: str):
    result = load_result(result_id)
    stem = f"comparison_{secure_filename(result['dataset_name'])}_{result_id[:8]}"
    if fmt == "xlsx":
        return send_file(create_result_xlsx(result), as_attachment=True, download_name=f"{stem}.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if fmt == "pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        output = io.BytesIO()
        pdf = canvas.Canvas(output, pagesize=A4)
        width, height = A4
        y = height - 55
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(45, y, "Model Comparison Report")
        y -= 30
        pdf.setFont("Helvetica", 10)
        receipt = result.get("affordability", {})
        for line in [
            f"Dataset: {result['dataset_name']}", f"Target: {result['target']}",
            f"Task: {result['task']}", f"Train/Test: {result['train_pct']}% / {result['test_pct']}%",
            f"Resource mode: {receipt.get('resource_mode_label', str(result.get('mode', 'full')).title())}",
            f"Rows processed / available: {result['rows_processed']:,} / {result.get('rows_available', result['rows_processed']):,}",
            f"Created: {result['created_at']}",
        ]:
            pdf.drawString(45, y, line)
            y -= 16
        y -= 10
        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(45, y, "Affordability and Access Receipt")
        y -= 20
        pdf.setFont("Helvetica", 10)
        for line in [
            f"Application licence/API fee charged: RM {receipt.get('application_fee_myr', 0):.2f}",
            f"Paid external ML API calls: {receipt.get('paid_external_api_calls', 0)}",
            f"Processing location: {receipt.get('processing_location', 'Local computer')}",
            f"Enterprise software required: {'Yes' if receipt.get('enterprise_software_required', False) else 'No'}",
            f"Measured model training time: {receipt.get('training_seconds', result.get('total_training_seconds', 0))} seconds",
        ]:
            pdf.drawString(45, y, line)
            y -= 15
        for model in result["models"]:
            y -= 15
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(45, y, model["name"])
            y -= 20
            pdf.setFont("Helvetica", 10)
            for key, value in model["metrics"].items():
                pdf.drawString(60, y, f"{key.replace('_', ' ').title()}: {value}")
                y -= 15
            pdf.drawString(60, y, f"Training time: {model['training_seconds']} seconds")
            y -= 15
        pdf.save()
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=f"{stem}.pdf", mimetype="application/pdf")
    if fmt == "png":
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB", (1400, 850), "#f6f7fb")
        draw = ImageDraw.Draw(image)
        draw.text((60, 45), "Model Comparison", fill="#111827", font=ImageFont.load_default(size=28))
        draw.text((60, 90), f"{result['dataset_name']} | target: {result['target']} | {result['train_pct']}/{result['test_pct']} split", fill="#475569", font=ImageFont.load_default(size=16))
        colors = ["#2563eb", "#7c3aed"]
        x0 = 80
        metric_names = result["comparison_chart"]["labels"]
        for model_index, series in enumerate(result["comparison_chart"]["series"]):
            draw.text((x0 + model_index * 620, 145), series["name"], fill=colors[model_index], font=ImageFont.load_default(size=22))
            y = 190
            for name, value in zip(metric_names, series["values"]):
                if value is None: continue
                draw.text((x0 + model_index * 620, y), f"{name.replace('_', ' ').title()}: {value}", fill="#111827", font=ImageFont.load_default(size=16))
                y += 38
        output = io.BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=f"{stem}.png", mimetype="image/png")
    abort(404)


def custom_model_directory() -> Path:
    return user_data_directory(CUSTOM_DIR)


def next_custom_model_name() -> str:
    return next_numbered_record_name(custom_model_directory(), "Custom Model")


@app.get("/api/custom-models")
@login_required
def list_custom_models():
    directory = custom_model_directory()
    models = [load_json(path, None) for path in directory.glob("*.json")]
    return jsonify({"ok": True, "models": [m for m in models if m], "rules": custom_model_rules(), "suggested_name": next_custom_model_name()})


def normalise_custom_model(payload: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    code = str(payload.get("code", ""))
    errors = validate_custom_code(code)
    if errors:
        raise UserFacingError("The custom model does not pass validation: " + " ".join(errors))
    task = payload.get("task")
    if task not in {"classification", "regression", "clustering"}:
        raise UserFacingError("Choose classification, regression, or clustering.")
    model_id = secure_filename(str(model_id or payload.get("id") or uuid.uuid4().hex))
    existing = load_json(custom_model_directory() / f"{model_id}.json", {})
    name = str(payload.get("name", "")).strip() or str(existing.get("name", "")).strip() or next_custom_model_name()
    if len(name.split()) > 30:
        raise UserFacingError("Model Name must contain no more than 30 words.")
    parameters = payload.get("parameters", [])
    normalised_parameters = []
    for parameter in parameters:
        if not parameter.get("name") or parameter.get("type") not in {"integer", "number", "boolean", "text", "choice"}:
            raise UserFacingError("Every parameter needs a name and a supported type.")
        item = {
            "name": secure_filename(str(parameter["name"])).replace("-", "_") or "parameter",
            "type": parameter["type"], "description": str(parameter.get("description", "")),
        }
        try:
            if item["type"] == "integer":
                item["default"] = int(parameter.get("default", 0))
                if str(parameter.get("min", "")).strip(): item["min"] = int(parameter["min"])
                if str(parameter.get("max", "")).strip(): item["max"] = int(parameter["max"])
            elif item["type"] == "number":
                item["default"] = float(parameter.get("default", 0))
                if str(parameter.get("min", "")).strip(): item["min"] = float(parameter["min"])
                if str(parameter.get("max", "")).strip(): item["max"] = float(parameter["max"])
            elif item["type"] == "boolean":
                item["default"] = str(parameter.get("default", "false")).lower() in {"true", "1", "yes", "on"}
            else:
                item["default"] = str(parameter.get("default", ""))
        except (TypeError, ValueError) as exc:
            raise UserFacingError(f"Parameter '{item['name']}' has an invalid default or numeric limit.") from exc
        normalised_parameters.append(item)
    parameters = normalised_parameters
    return {
        "id": model_id, "name": name, "description": str(payload.get("description", "")),
        "task": task, "task_label": str(payload.get("task_label", "")).strip(), "parameters": parameters,
        "defaults": {item["name"]: item.get("default") for item in parameters},
        "code": code, "created_at": existing.get("created_at", utc_now()), "updated_at": utc_now(),
        "privacy_notice": "Trusted local code only; do not deploy this runner as a public code-execution service.",
    }


@app.post("/api/custom-models")
@login_required
def save_custom_model():
    payload = request.get_json(silent=True) or {}
    errors = validate_custom_code(str(payload.get("code", "")))
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    model = normalise_custom_model(payload)
    directory = custom_model_directory()
    save_json(directory / f"{model['id']}.json", model)
    return jsonify({"ok": True, "model": model})


@app.put("/api/custom-models/<model_id>")
@login_required
def update_custom_model(model_id: str):
    safe_id = secure_filename(model_id)
    path = custom_model_directory() / f"{safe_id}.json"
    if not safe_id or not path.exists():
        abort(404)
    payload = request.get_json(silent=True) or {}
    errors = validate_custom_code(str(payload.get("code", "")))
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    model = normalise_custom_model(payload, safe_id)
    save_json(path, model)
    return jsonify({"ok": True, "model": model})


@app.delete("/api/custom-models/<model_id>")
@login_required
def delete_custom_model(model_id: str):
    safe_id = secure_filename(model_id)
    path = custom_model_directory() / f"{safe_id}.json"
    if not safe_id or not path.exists():
        abort(404)
    path.unlink()
    return jsonify({"ok": True})


@app.post("/api/custom-models/validate")
@login_required
def validate_custom_model():
    payload = request.get_json(silent=True) or {}
    errors = validate_custom_code(str(payload.get("code", "")))
    return jsonify({"ok": not errors, "errors": errors, "rules": custom_model_rules()})


@app.get("/api/settings")
@login_required
def get_settings():
    defaults = {"theme": "light", "train_pct": 70, "random_seed": 42, "processing": "ask", "tooltips": True, "export": "pdf"}
    defaults.update(load_json(SETTINGS_FILE, {}).get(current_user().lower(), {}))
    defaults["processing"] = {"sample": "economy", "chunked": "balanced"}.get(defaults.get("processing"), defaults.get("processing"))
    if defaults["processing"] not in {"ask", *PROCESSING_MODES}:
        defaults["processing"] = "ask"
    return jsonify({"ok": True, "settings": defaults})


@app.put("/api/settings")
@login_required
def save_settings():
    payload = request.get_json(silent=True) or {}
    processing = {"sample": "economy", "chunked": "balanced"}.get(payload.get("processing"), payload.get("processing", "ask"))
    if processing not in {"ask", *PROCESSING_MODES}:
        raise UserFacingError("Choose Always Ask, Economy, Balanced, or Full processing.")
    settings_data = load_json(SETTINGS_FILE, {})
    settings_data[current_user().lower()] = {
        "theme": payload.get("theme", "light"),
        "train_pct": max(0, min(100, int(payload.get("train_pct", 70)))),
        "random_seed": int(payload.get("random_seed", 42)),
        "processing": processing,
        "tooltips": bool(payload.get("tooltips", True)),
        "export": payload.get("export", "pdf"),
    }
    save_json(SETTINGS_FILE, settings_data)
    return jsonify({"ok": True, "settings": settings_data[current_user().lower()]})


@app.post("/api/account/profile")
@login_required
def update_profile():
    payload = request.get_json(silent=True) or {}
    accounts = load_json(ACCOUNT_FILE, {})
    key = current_user().lower()
    account = accounts[key]
    display_name = str(payload.get("display_name", payload.get("username", ""))).strip()
    if display_name:
        if not valid_display_name(display_name):
            raise UserFacingError("Use 2-50 letters, numbers, spaces, dots, underscores, or hyphens for the display name.")
        if account_identity_taken(accounts, display_name, exclude_key=key):
            raise UserFacingError("That display name is already registered.")
        account["display_name"] = display_name
    current_password = str(payload.get("current_password", ""))
    new_password = str(payload.get("new_password", ""))
    email = str(payload.get("email", account.get("email", ""))).strip().lower()
    email_changed = email != str(account.get("email", "")).strip().lower()
    if email_changed:
        if not valid_email(email):
            raise UserFacingError("Enter a valid email address.")
        if account_identity_taken(accounts, email, exclude_key=key):
            raise UserFacingError("That email address is already registered.")
    if email_changed or new_password:
        if not check_password_hash(account["password_hash"], current_password):
            raise UserFacingError("Enter the current password before changing the email or password.")
    if email_changed:
        account["email"] = email
    if new_password:
        if len(new_password) < 8:
            raise UserFacingError("The new password must contain at least 8 characters.")
        account["password_hash"] = generate_password_hash(new_password)
    accounts[key] = account
    save_json(ACCOUNT_FILE, accounts)
    return jsonify({"ok": True, "user": {k: v for k, v in account.items() if k != "password_hash"}})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="127.0.0.1", port=8000, use_reloader=False)
