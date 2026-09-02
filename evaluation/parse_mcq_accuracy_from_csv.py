#!/usr/bin/env python3
"""
Parse eval_results_ratio_*_summary.csv files and emit MCQ accuracy blocks
in the same format as parse_mcq_accuracy.py:

length_<ratio>_<dataset>
<T1_accuracy_percent>
<T2_accuracy_percent>
<T3_accuracy_percent>
<T4_accuracy_percent>

Defaults:
  - Reads the four summary CSVs under TS-benchmark_on_LLM:
    eval_results_ratio_60_gpt-4o_summary.csv
    eval_results_ratio_70_gpt-4o_summary.csv
    eval_results_ratio_80_gpt-4o_summary.csv
    eval_results_ratio_90_gpt-4o_summary.csv
  - Writes to mcq_accuracy_summary_from_csv.log in this directory.
"""

from __future__ import annotations

import csv
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple


BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "mcq_accuracy_summary_from_csv.log"
DEFAULT_CSVS = [
    BASE_DIR / "eval_results_ratio_60_gpt-4o_summary.csv",
    BASE_DIR / "eval_results_ratio_70_gpt-4o_summary.csv",
    BASE_DIR / "eval_results_ratio_80_gpt-4o_summary.csv",
    BASE_DIR / "eval_results_ratio_90_gpt-4o_summary.csv",
]

# Only MCQ accuracies for these tasks
TASK_FILTER = {
    "T1": "understanding",
    "T2": "mcq",
    "T3": "pack",
    "T4": "mcq",
}


def _normalize_dataset(raw: str) -> str:
    """Normalize dataset name to lowercase, alnum/underscore only."""
    name = raw.split("|")[0].strip()
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    # Optional manual tweak
    if name == "causalchambers":
        return "causal_chambers"
    return name


def _ratio_from_dataset(raw: str) -> str:
    m = re.search(r"ratio\s*=\s*(\d+)", raw)
    return m.group(1) if m else ""


def _ratio_from_filename(path: pathlib.Path) -> str:
    m = re.search(r"ratio_(\d+)", path.name)
    return m.group(1) if m else ""


def parse_csv(path: pathlib.Path) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """
    Return {(dataset, task): (sum_acc_percent, count)} for MCQ rows only.
    Accuracy Rate in CSV is assumed to be fraction (0-1); convert to percent.
    """
    agg: Dict[Tuple[str, str], List[float | int]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("Task", "").strip()
            subtask = row.get("Subtask", "").strip()
            if task not in TASK_FILTER or TASK_FILTER[task] != subtask:
                continue
            ds_raw = row.get("Dataset", "") or ""
            dataset = _normalize_dataset(ds_raw)
            key = (dataset, task)
            acc_str = row.get("Accuracy Rate") or ""
            try:
                acc = float(acc_str) * 100.0
            except Exception:
                continue
            if key not in agg:
                agg[key] = [0.0, 0]
            agg[key][0] += acc
            agg[key][1] += 1
    return {k: (float(v[0]), int(v[1])) for k, v in agg.items()}


def aggregate_blocks(csv_paths: List[pathlib.Path]) -> List[str]:
    lines: List[str] = []
    for csv_path in csv_paths:
        stats = parse_csv(csv_path)
        ratio = _ratio_from_filename(csv_path)
        if not ratio:
            # Fallback to infer from any dataset field
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ratio = _ratio_from_dataset(row.get("Dataset", "") or "")
                    if ratio:
                        break
        for dataset in sorted({k[0] for k in stats}):
            lines.append(f"length_{ratio}_{dataset}")
            for task in ("T1", "T2", "T3", "T4"):
                s, n = stats.get((dataset, task), (0.0, 0))
                mean = s / n if n else 0.0
                lines.append(f"{mean:.2f}")
    return lines


def main(argv: List[str]) -> int:
    if len(argv := __import__("sys").argv) >= 3:
        out_path = pathlib.Path(argv[1])
        csv_paths = [pathlib.Path(p) for p in argv[2:]]
    else:
        out_path = DEFAULT_OUTPUT
        csv_paths = DEFAULT_CSVS

    lines = aggregate_blocks(csv_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main([]))
