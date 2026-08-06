"""One logging surface, swappable backend, silent when nothing is installed.

The training code should not know which tracker is in use, and the repo should
still run for someone who installed neither. Every method degrades to a no-op
rather than raising, so `MLOPS_BACKEND=none` is a supported configuration and
not a broken one.

    MLOPS_BACKEND=trackio   local SQLite + a Gradio dashboard
    MLOPS_BACKEND=mlflow    local ./mlruns + the MLflow UI
    MLOPS_BACKEND=none      metrics still print, nothing is stored
"""
from __future__ import annotations

import json
import os
from typing import Any


class Run:
    def __init__(self, backend: str, project: str, name: str, config: dict[str, Any]) -> None:
        self.backend = backend
        self.name = name
        self._impl = None
        if backend == "trackio":
            import trackio

            trackio.init(project=project, name=name, config=_flatten(config))
            self._impl = trackio
        elif backend == "mlflow":
            import mlflow

            mlflow.set_experiment(project)
            mlflow.start_run(run_name=name)
            mlflow.log_params(_flatten(config))
            self._impl = mlflow

    def log(self, step: int, metrics: dict[str, float]) -> None:
        clean = {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
        if self.backend == "trackio":
            self._impl.log(clean, step=step)
        elif self.backend == "mlflow":
            self._impl.log_metrics(clean, step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        """Terminal values for a run. These are what a run table sorts by."""
        if self.backend == "trackio":
            # Logged as a final row so the value lands on the run's chart tail
            # and in its summary column.
            self._impl.log({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        elif self.backend == "mlflow":
            self._impl.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

    def finish(self) -> None:
        if self.backend == "trackio":
            self._impl.finish()
        elif self.backend == "mlflow":
            self._impl.end_run()


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Trackers take flat scalar params; nested provenance dicts must be folded."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value)
        else:
            flat[name] = value
    return flat


def start(project: str, name: str, config: dict[str, Any]) -> Run:
    backend = os.environ.get("MLOPS_BACKEND", "trackio").lower()
    if backend not in ("trackio", "mlflow", "none"):
        raise ValueError(f"unknown MLOPS_BACKEND: {backend}")
    try:
        return Run(backend, project, name, config)
    except ImportError:
        print(f"[track] {backend} not installed; continuing untracked")
        return Run("none", project, name, config)


__all__ = ["Run", "start"]
