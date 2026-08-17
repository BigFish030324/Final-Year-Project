from __future__ import annotations

# --------------------
# Complete Flask Website
# Import the full application while keeping this file as the simple project starting point.
# --------------------
from website_application import app


# --------------------
# Local Development Server
# Start the FYP website on port 8000 when I run python main.py in the terminal.
# --------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True,
        use_reloader=False,
    )
