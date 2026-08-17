# --------------------
# Custom Cleaning Runner
# Run a saved student cleaning function separately and return a cleaned Pandas dataset.
# --------------------
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    response_path = Path(request["response"])
    try:
        frame = pd.read_pickle(request["input"])
        allowed_builtins = {
            "__import__": __import__, "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "tuple": tuple, "dict": dict, "set": set, "len": len,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "range": range,
            "enumerate": enumerate, "zip": zip, "sorted": sorted,
        }
        namespace = {"__builtins__": allowed_builtins}
        exec(compile(request["code"], "<trusted-local-cleaning>", "exec"), namespace, namespace)
        cleaner = namespace.get("clean_data")
        if not callable(cleaner):
            raise ValueError("clean_data(df) was not defined")
        cleaned = cleaner(frame.copy())
        if not isinstance(cleaned, pd.DataFrame):
            raise ValueError("clean_data(df) must return a Pandas DataFrame")
        cleaned.to_pickle(request["output"])
        payload = {"ok": True, "rows": len(cleaned), "columns": len(cleaned.columns)}
    except Exception as exc:
        payload = {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=4)}
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

