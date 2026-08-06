"""Read the arm results and answer one question: is the effect bigger than the noise?

Gate U found that validation noise in this repo exceeded the effects two gates
had reported. Any comparison tool that reports a difference without reporting
the seed-to-seed spread alongside it repeats that mistake with nicer charts.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from statistics import mean

RESULTS = Path("mlops/results")


def load() -> list[dict]:
    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        runs.append({
            "name": config["arm"],
            "route_scale": config["route_scale"],
            "seed": config["seed"],
            "perplexity": payload["summary"]["final_validation_perplexity"],
            "seconds": payload["summary"]["wall_clock_seconds"],
            "commit": (config["provenance"]["git"] or {}).get("commit", "")[:8],
            "dirty": (config["provenance"]["git"] or {}).get("dirty"),
        })
    return runs


def main() -> None:
    runs = [r for r in load() if r["name"].startswith("rs")]
    if not runs:
        print("no arms found; run mlops/run_experiment.py first")
        return

    print(f"{'arm':<16}{'route_scale':>12}{'seed':>7}{'val PPL':>12}{'sec':>8}  {'commit':<10}{'dirty'}")
    for run in sorted(runs, key=lambda r: r["perplexity"]):
        print(f"{run['name']:<16}{run['route_scale']:>12}{run['seed']:>7}"
              f"{run['perplexity']:>12.2f}{run['seconds']:>8.1f}  {run['commit']:<10}{run['dirty']}")

    by_scale: dict[float, list[float]] = {}
    for run in runs:
        by_scale.setdefault(run["route_scale"], []).append(run["perplexity"])

    # Seed spread within one setting is the floor any claimed effect must clear.
    noise = [max(values) - min(values) for values in by_scale.values() if len(values) > 1]
    noise_floor = max(noise) if noise else None

    print()
    for scale, values in sorted(by_scale.items()):
        spread = f", seed spread {max(values) - min(values):.2f}" if len(values) > 1 else ""
        print(f"route_scale {scale:<6} mean PPL {mean(values):>8.2f}  (n={len(values)}{spread})")

    if noise_floor is not None:
        print(f"\nnoise floor (largest within-setting seed spread): {noise_floor:.2f} PPL")
        for a, b in combinations(sorted(by_scale), 2):
            effect = abs(mean(by_scale[a]) - mean(by_scale[b]))
            verdict = "RESOLVED" if effect > noise_floor else "NOT RESOLVED at this budget"
            print(f"  route_scale {a} vs {b}: effect {effect:.2f} PPL vs noise {noise_floor:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
