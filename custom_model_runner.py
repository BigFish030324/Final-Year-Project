# --------------------
# Custom Model Runner
# Run a saved student model separately and return its predictions within a time limit.
# --------------------
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> int:
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = Path(request["response"])
    try:
        arrays = np.load(request["arrays"], allow_pickle=False)
        allowed_builtins = {
            "__import__": __import__, "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "tuple": tuple, "dict": dict, "set": set, "len": len,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "range": range,
        }
        namespace = {"__builtins__": allowed_builtins}
        exec(compile(request["code"], "<trusted-local-model>", "exec"), namespace, namespace)
        builder = namespace.get("build_model")
        if not callable(builder):
            raise ValueError("build_model(params) was not defined")
        model = builder(request.get("params", {}))
        started = time.perf_counter()
        if request.get("task") == "clustering":
            if callable(getattr(model, "fit_predict", None)):
                predictions = model.fit_predict(arrays["X_train"])
            elif callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None)):
                model.fit(arrays["X_train"])
                predictions = model.predict(arrays["X_train"])
            else:
                raise ValueError("A clustering model must provide fit_predict, or fit and predict methods")
        else:
            if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
                raise ValueError("build_model must return an estimator with fit and predict methods")
            model.fit(arrays["X_train"], arrays["y_train"])
            predictions = model.predict(arrays["X_test"])
        payload = {
            "ok": True, "predictions": np.asarray(predictions).tolist(),
            "training_seconds": time.perf_counter() - started,
        }
        if hasattr(model, "inertia_"):
            payload["inertia"] = float(model.inertia_)
    except Exception as exc:
        payload = {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=4)}
    response_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

