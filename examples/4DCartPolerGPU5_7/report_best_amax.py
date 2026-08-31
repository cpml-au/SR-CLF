#!/usr/bin/env python3
"""Report cheap and full-exact Artstein a_max on CPU for saved bests.

Examples:
  python report_best_amax.py 83222_best_per_generation.jsonl
  python report_best_amax.py 83222_best_per_generation.jsonl --random 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

# This standalone diagnostic is intentionally CPU-only. These variables are
# set before Evaluate/JAX is imported and do not affect GPU2 training jobs.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SYMCLF_GPU2_ALLOW_CPU"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (PROJECT_ROOT, HERE):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import Evaluate  # noqa: E402
from srcGPU5_7.artstein_gpu5_7 import (  # noqa: E402
    check_artstein_gpu_cheap_many_from_base,
)
from srcGPU5_7.fitness import _compute_exact_result  # noqa: E402
from srcGPU5_7.runtime_exact_candidate import RuntimeExactCandidate  # noqa: E402


def _load(path: Path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "line": line_number,
                    "generation": int(item["generation"]),
                    "expression": str(item["expression"]),
                    "constants": tuple(
                        float(value)
                        for value in item.get("constants", item.get("consts", []))
                    ),
                }
            )
    return records


def _candidate_key(record):
    return record["expression"], record["constants"]


def _a_max(result, gamma1):
    if result is None or result.status != "ok":
        return math.nan
    value = float(result.margin_max) - float(gamma1)
    return value if math.isfinite(value) else math.nan


def _newest_history():
    histories = list(HERE.glob("*_best_per_generation.jsonl"))
    if not histories:
        raise FileNotFoundError(f"No *_best_per_generation.jsonl in {HERE}")
    return max(histories, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history",
        nargs="?",
        type=Path,
        help="Generation-best JSONL; defaults to the newest one in this folder.",
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="N",
        help="Randomly select N generations from the complete history.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=HERE / "best_cheap_exact_amax.csv",
    )
    args = parser.parse_args()

    history = (args.history or _newest_history()).expanduser().resolve()
    records = _load(history)
    if args.random is not None:
        if args.random < 1:
            parser.error("--random must be at least 1")
        records = random.Random(args.seed).sample(
            records, min(args.random, len(records))
        )
    records.sort(key=lambda item: item["generation"])

    # Repeated elites are evaluated once and reported for every generation.
    unique = {}
    for record in records:
        unique.setdefault(_candidate_key(record), record)
    unique_records = list(unique.values())
    base = Evaluate._base

    # Fusion changes batching only, not search points or tolerances. A width of
    # one avoids the former 32-candidate multi-GiB scratch allocation on CPU.
    base.GPU2_CHEAP_FUSION = 1

    cheap_results = {}
    width = max(1, int(getattr(base, "GPU2_CHEAP_FUSION", 32)))
    for start in range(0, len(unique_records), width):
        batch = unique_records[start : start + width]
        candidates = [
            RuntimeExactCandidate(item["expression"], item["constants"])
            for item in batch
        ]
        checked = check_artstein_gpu_cheap_many_from_base(candidates, base)
        for record, result in zip(batch, checked):
            cheap_results[_candidate_key(record)] = result

    exact_results = {}
    for record in unique_records:
        key = _candidate_key(record)
        try:
            exact_results[key] = _compute_exact_result(
                record["expression"], record["constants"], base
            )
        except Exception:
            exact_results[key] = None

    output = args.output_csv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "cheap_a_max", "exact_a_max"],
        )
        writer.writeheader()
        print("generation,cheap_a_max,exact_a_max")
        for record in records:
            key = _candidate_key(record)
            cheap = _a_max(cheap_results.get(key), base.CLF_GAMMA1)
            exact = _a_max(exact_results.get(key), base.CLF_GAMMA1)
            row = {
                "generation": record["generation"],
                "cheap_a_max": cheap,
                "exact_a_max": exact,
            }
            writer.writerow(row)
            print(f"{record['generation']},{cheap:.16g},{exact:.16g}")

    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
