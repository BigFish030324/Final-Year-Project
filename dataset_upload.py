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
