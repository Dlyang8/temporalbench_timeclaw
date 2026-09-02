#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick inspector for the PSML dataset in TS-benchmark.
Loads the JSON file and prints structural statistics for T2/T4 ground_truth blocks,
so we can diagnose why forecast prompts/results might mismatch (e.g., multi-channel dicts).
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


DEFAULT_JSON = "/projects/beei/mweng/TS-benchmark/TS-benchmark/dataset/PSML/results/task_modified.json"


def _sequence_length(value: Any) -> int:
    """Return length if value looks like a list/tuple, otherwise 0."""
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


def _describe_ground_truth(gt: Any) -> Tuple[str, Dict[str, int]]:
    """
    Return (type_label, per_channel_lengths).
    type_label is 'list', 'dict', or actual type name.
    per_channel_lengths maps channel names to lengths (for list → {'__list__': len}).
    """
    if isinstance(gt, list):
        return "list", {"__list__": len(gt)}
    if isinstance(gt, dict):
        lens = {}
        for key, val in gt.items():
            lens[str(key)] = _sequence_length(val)
        return "dict", lens
    if gt is None:
        return "none", {}
    return type(gt).__name__, {}


def inspect_psml(json_path: str, show_samples: int) -> None:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level list of samples.")

    tier_types: Dict[str, Counter] = {"T2": Counter(), "T4": Counter()}
    tier_lengths: Dict[str, Counter] = {"T2": Counter(), "T4": Counter()}
    tier_channels: Dict[str, Counter] = {"T2": Counter(), "T4": Counter()}

    print(f"Loaded {len(data)} samples from {json_path}")
    print("-" * 80)

    for idx, sample in enumerate(data):
        tasks = sample.get("tasks") or {}
        for tier in ("T2", "T4"):
            node = tasks.get(tier)
            if not isinstance(node, dict):
                continue
            gt = node.get("ground_truth")
            gt_type, lens = _describe_ground_truth(gt)
            tier_types[tier][gt_type] += 1
            if lens:
                for channel, length in lens.items():
                    if length:
                        tier_lengths[tier][length] += 1
                tier_channels[tier][len(lens)] += 1

            if show_samples > 0:
                sample_id = sample.get("sample_id") or f"index_{idx}"
                print(f"[Sample {idx:04d}] {sample.get('dataset', 'unknown')} | {sample_id} | Tier {tier}")
                print(f"  ground_truth type: {gt_type}")
                if lens:
                    for channel, length in lens.items():
                        print(f"    - {channel}: len={length}")
                meta = node.get("meta") or sample.get("meta") or {}
                horizon_hint = meta.get("n_horizon") or meta.get("future_len") or meta.get("horizon")
                if horizon_hint:
                    print(f"  meta horizon hint: {horizon_hint}")
                print()
                show_samples -= 1

    print("=" * 80)
    for tier in ("T2", "T4"):
        print(f"Tier {tier}:")
        print(f"  ground_truth types: {dict(tier_types[tier])}")
        print(f"  channel-count distribution: {dict(tier_channels[tier])}")
        print(f"  sequence length distribution: top entries:")
        for length, count in tier_lengths[tier].most_common(10):
            print(f"    len={length}: {count} samples")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Inspect PSML dataset structure.")
    parser.add_argument("--json", default=DEFAULT_JSON, help="Path to PSML task_modified.json")
    parser.add_argument("--show", type=int, default=5, help="Print details for the first N tier instances.")
    args = parser.parse_args()

    inspect_psml(args.json, args.show)


if __name__ == "__main__":
    main()
