from __future__ import annotations

import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from error_handler import UserError


# ----------------------------------------------------
# Dataset Upload Configuration
# ----------------------------------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parent
UPLOAD_DIRECTORY = PROJECT_DIRECTORY / "runtime_data" / "uploads"

# Only allow the two dataset formats supported by data_science.py
ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
}

# Use a smaller limit during the first upload-development stage
MAX_UPLOAD_BYTES = 100_000_000


# ----------------------------------------------------
# Save Uploaded Dataset
# ----------------------------------------------------

def save_uploaded_dataset(
    uploaded_file: FileStorage | None,
) -> tuple[Path, str]:
    # Confirm that the browser submitted a file
    if uploaded_file is None or not uploaded_file.filename:
        raise UserError(
            "Choose a CSV or XLSX dataset first."
        )

    # Remove unsafe path characters from the original filename
    original_filename = secure_filename(
        uploaded_file.filename
    )

    if not original_filename:
        raise UserError(
            "The uploaded dataset needs a valid filename."
        )

    file_extension = Path(
        original_filename
    ).suffix.lower()

    # Reject JSON, TXT, PDF and other unsupported formats before saving
    if file_extension not in ALLOWED_EXTENSIONS:
        raise UserError(
            "Only CSV and XLSX datasets are supported."
        )

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Generate a unique stored filename so two uploads do not overwrite
    # one another when their original filenames are the same
    stored_filename = (
        f"{uuid.uuid4().hex}{file_extension}"
    )
    stored_path = UPLOAD_DIRECTORY / stored_filename

    uploaded_file.save(stored_path)

    # Reject empty uploaded files
    if stored_path.stat().st_size == 0:
        stored_path.unlink(missing_ok=True)
        raise UserError(
            "The uploaded dataset is empty."
        )

    return stored_path, original_filename


# ----------------------------------------------------
# Find a Previously Uploaded Dataset
# ----------------------------------------------------

def load_uploaded_dataset(stored_filename: str) -> Path:
    # The webpage sends only the generated filename back to Flask.
    # Never accept a complete path supplied by the browser.
    safe_filename = secure_filename(stored_filename)

    if not safe_filename or safe_filename != stored_filename:
        raise UserError(
            "The uploaded dataset reference is invalid. Upload the file again."
        )

    dataset_path = (UPLOAD_DIRECTORY / safe_filename).resolve()
    upload_directory = UPLOAD_DIRECTORY.resolve()

    # Confirm that the resolved file remains directly inside uploads.
    if dataset_path.parent != upload_directory:
        raise UserError(
            "The uploaded dataset reference is invalid. Upload the file again."
        )

    if dataset_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UserError(
            "Only CSV and XLSX datasets are supported."
        )

    if not dataset_path.is_file():
        raise UserError(
            "The uploaded dataset is no longer available. Upload it again."
        )

    return dataset_path
