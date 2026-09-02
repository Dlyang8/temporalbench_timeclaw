#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selective rerun helper for TS benchmark results.

Goals:
1. Reuse the existing evaluation pipeline in evaluate_llm.py.
2. Rerun only missing / failed tiers instead of rerunning every sample.
3. Fix multivariate MIMIC T2/T4 forecast prompt formatting when needed.

Default behavior:
- Rerun MIMIC T2 for samples whose forecast failed with missing_forecast.
- Merge rerun results into the existing detail JSON.
- Write new detail / summary files with a configurable suffix tag.

Configuration is controlled by environment variables so the script can be run
directly without editing, but the defaults are intentionally conservative.
"""

from __future__ import annotations

import copy
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import evaluate_llm as base
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "unknown")
    raise SystemExit(
        "Failed to import evaluate_llm.py dependencies. "
        f"Missing module: {missing}. "
        "Run this script with the same Python environment you used for evaluate_llm.py."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent


def _env_csv(name: str, default: Iterable[str]) -> List[str]:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


CONFIG = {
    "models": _env_csv("RERUN_MODELS", ["gpt-5.4-mini", "gemini-3.1-flash-lite"]),
    "datasets": _env_csv("RERUN_DATASETS", ["MIMIC"]),
    "tiers": _env_csv("RERUN_TIERS", ["T2"]),
    "error_reasons": set(_env_csv("RERUN_ERROR_REASONS", ["missing_forecast"])),
    "only_failed": _env_bool("RERUN_ONLY_FAILED", True),
    "sample_limit": _env_int("RERUN_SAMPLE_LIMIT", None),
    "dry_run": _env_bool("RERUN_DRY_RUN", False),
    "output_tag": os.environ.get("RERUN_OUTPUT_TAG", "rerun_fix"),
    "force_multivariate_prompt_fix": _env_bool("RERUN_FIX_MULTIVAR_PROMPT", True),
}


def _series_example(length: int) -> str:
    return base._format_forecast_list_example(length)


def _build_multivariate_forecast_json(channel_names: List[str], horizon_steps: int) -> str:
    lines = ['{', '  "forecast": {']
    for idx, name in enumerate(channel_names):
        comma = "," if idx < len(channel_names) - 1 else ""
        lines.append(f'    "{name}": {_series_example(horizon_steps)}{comma}')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _is_multivariate_forecast_task(task_node: Dict[str, Any]) -> bool:
    gt_series = base._series_dict_from_obj(task_node.get("ground_truth"))
    return bool(gt_series and len(gt_series) > 1)


def _channel_names_from_task(task_node: Dict[str, Any]) -> List[str]:
    gt_series = base._series_dict_from_obj(task_node.get("ground_truth")) or {}
    return list(gt_series.keys())


def patched_build_t2_prompts(
    t2: dict,
    input_json_str: str = None,
    sample_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    if not _is_multivariate_forecast_task(t2):
        return _ORIG_BUILD_T2_PROMPTS(t2, input_json_str=input_json_str, sample_meta=sample_meta)

    def _default_t2_background():
        return (
            "Background:\n"
            "Using only the provided multivariate history and aligned future covariates, "
            "forecast the next horizon steps for every target channel.\n"
            "Treat NaN as missing; avoid leakage from future targets."
        )

    raw_base = (t2.get("prompt") or t2.get("task") or "").strip()

    def _looks_like_mcq_block(text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "multiple-choice",
            "multiple choice",
            '"mcq"',
            "future_vs_history",
            "volatility_change",
            "seasonality_shift",
            "question 1",
            "q1)",
            '"mcq":',
        ]
        return any(kw in lowered for kw in keywords)

    if raw_base and not _looks_like_mcq_block(raw_base):
        forecast_base = raw_base
    else:
        forecast_base = _default_t2_background()

    mcq_base = raw_base if raw_base else forecast_base
    input_block = input_json_str if isinstance(input_json_str, str) else base._build_input_preview(t2)
    horizon_steps = base._infer_forecast_horizon(t2, sample_meta=sample_meta, fallback=112)
    horizon_steps = max(int(horizon_steps or 0), 1)
    channel_names = _channel_names_from_task(t2)

    forecast_task = (
        "Task:\n"
        f"- Forecast the next {horizon_steps} steps for each target channel based on the provided history and aligned future covariates.\n"
        f"- Required forecast channels: {', '.join(channel_names)}.\n"
        "- Treat NaN as missing.\n"
        "- Keep predictions finite.\n"
        "- Do not collapse multiple channels into a single unnamed series."
    )
    forecast_out = (
        "Output format (JSON only):\n"
        f"{_build_multivariate_forecast_json(channel_names, horizon_steps)}\n"
        "Constraints:\n"
        f"- Every channel listed above must appear exactly once and have length {horizon_steps}.\n"
        "- Each value must be finite.\n"
        "- Do not include anything outside the JSON object.\n"
        "- Do not use any external knowledge beyond the provided input."
    )
    forecast_prompt = "\n\n".join(
        [part for part in [forecast_base, forecast_task, f"Input (JSON):\n{input_block}", forecast_out] if part]
    ).strip()

    mcq_task = (
        "Task:\n"
        "Based on the same historical data (and aligned future covariates context), "
        "answer the following qualitative questions about the forecast horizon vs history."
    )
    mcq_questions = (
        "Multiple-choice questions:\n"
        "Q1) Median demand level change (forecast horizon vs history)? {Higher, Lower, Similar, Uncertain}\n"
        "Q2) Volatility change (forecast horizon vs history)? {increased, decreased, constant, Uncertain}\n"
        "Q3) Seasonality alignment between history and forecast? {fixed, shifting, no, Uncertain}"
    )
    mcq_out = (
        "Output format (JSON only):\n"
        "{\n"
        '  "mcq": {\n'
        '    "future_vs_history": "<One of { Higher, Lower, Similar, Uncertain }>",\n'
        '    "volatility_change": "<One of { increased, decreased, constant, Uncertain }>",\n'
        '    "seasonality_shift": "<One of { fixed, shifting, no, Uncertain }>"\n'
        "  }\n"
        "}\n"
        "Constraints:\n"
        "- Each answer must be chosen from the listed options exactly.\n"
        "- Do not output forecast values or any other text.\n"
        "- Use only the provided context; do not rely on external knowledge."
    )
    mcq_prompt = "\n\n".join(
        [part for part in [mcq_base, mcq_task, f"Input (JSON):\n{input_block}", mcq_questions, mcq_out] if part]
    ).strip()

    return {"forecast_prompt": forecast_prompt, "mcq_prompt": mcq_prompt}


def patched_build_t4_prompts(
    t4: dict,
    input_json_str: str = None,
    sample_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    if not _is_multivariate_forecast_task(t4):
        return _ORIG_BUILD_T4_PROMPTS(t4, input_json_str=input_json_str, sample_meta=sample_meta)

    horizon_steps = base._infer_forecast_horizon(t4, sample_meta=sample_meta, fallback=112)
    horizon_steps = max(int(horizon_steps or 0), 1)
    background = base.build_t4_background(t4, horizon_steps=horizon_steps)
    input_block = input_json_str if isinstance(input_json_str, str) else base._build_input_preview(t4)
    channel_names = _channel_names_from_task(t4)

    forecast_task = (
        "Task:\n"
        f"- Use the provided history and aligned future covariates to forecast the next {horizon_steps} steps for every target channel.\n"
        f"- Required forecast channels: {', '.join(channel_names)}.\n"
        "- Keep predictions finite.\n"
        "- Do not collapse multiple channels into a single unnamed series."
    )
    forecast_out = (
        "Output format (JSON only):\n"
        f"{_build_multivariate_forecast_json(channel_names, horizon_steps)}\n"
        "Constraints:\n"
        f"- Every channel listed above must appear exactly once and have length {horizon_steps}.\n"
        "- Each value must be finite.\n"
        "- Do not include anything outside the JSON object.\n"
        "- Do not use any external knowledge beyond the provided input."
    )
    forecast_prompt = (
        f"{background}\n\n"
        f"{forecast_task}\n\n"
        f"Input (JSON):\n{input_block}\n\n"
        f"{forecast_out}"
    ).strip()

    mcq_task = (
        "Task:\n"
        "Based on the same historical data and contextual event, analyze the expected qualitative changes between the forecast horizon and history."
    )
    mcq_obj = t4.get("mcq") or {}
    fv_opts = mcq_obj.get("future_vs_history_options") or "{Higher, Lower, Similar, Uncertain}"
    vol_opts = mcq_obj.get("volatility_change_options") or "{increased, decreased, constant, Uncertain}"
    seas_opts = mcq_obj.get("seasonality_shift_options") or "{fixed, shifting, no, Uncertain}"
    mcq_questions = (
        "Multiple-choice questions:\n"
        f"Q1) Median demand level change (forecast horizon vs history)? {fv_opts}\n"
        f"Q2) Volatility change (forecast horizon vs history)? {vol_opts}\n"
        f"Q3) Seasonality alignment between history and forecast? {seas_opts}"
    )
    mcq_out = (
        "Output format (JSON only):\n"
        "{\n"
        '  "mcq": {\n'
        '    "future_vs_history": "<One of { Higher, Lower, Similar, Uncertain }>",\n'
        '    "volatility_change": "<One of { increased, decreased, constant, Uncertain }>",\n'
        '    "seasonality_shift": "<One of { fixed, shifting, no, Uncertain }>"\n'
        "  }\n"
        "}\n"
        "Constraints:\n"
        "- Each answer must be chosen from the listed options exactly.\n"
        "- Do not output forecast values or any other text.\n"
        "- Use only the provided context; do not rely on external knowledge."
    )
    mcq_prompt = (
        f"{background}\n\n"
        f"{mcq_task}\n\n"
        f"Input (JSON):\n{input_block}\n\n"
        f"{mcq_questions}\n\n"
        f"{mcq_out}"
    ).strip()
    return {"forecast_prompt": forecast_prompt, "mcq_prompt": mcq_prompt}


_ORIG_BUILD_T2_PROMPTS = base.build_t2_prompts
_ORIG_BUILD_T4_PROMPTS = base.build_t4_prompts


def patch_prompt_builders() -> None:
    if not CONFIG["force_multivariate_prompt_fix"]:
        return
    base.build_t2_prompts = patched_build_t2_prompts
    base.build_t4_prompts = patched_build_t4_prompts


def load_dataset_sample_map(dataset_name: str) -> Dict[str, Dict[str, Any]]:
    path = base.DATASET_PATHS[dataset_name]
    samples = base.load_samples_for_dataset(path, sample_num=10**9, shuffle=False, seed=base.RANDOM_SEED)
    return {base.sample_id_of(sample): sample for sample in samples}


def discover_detail_file(model_name: str, dataset_name: str) -> Path:
    candidates = sorted(
        SCRIPT_DIR.glob("eval_results_detail_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if row.get("model") == model_name and row.get("dataset") == dataset_name:
                return path
    raise FileNotFoundError(f"Cannot find detail file for model={model_name}, dataset={dataset_name}")


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected detail JSON format: {path}")
    return data


def should_rerun_tier(tier_name: str, tier_result: Dict[str, Any], error_reasons: Set[str], only_failed: bool) -> bool:
    if not isinstance(tier_result, dict):
        return True
    if not only_failed:
        return True

    if tier_name in {"T2", "T4"}:
        forecast_errors = set(tier_result.get("forecast_errors") or [])
        if error_reasons and (forecast_errors & error_reasons):
            return True
        return float(tier_result.get("forecast_exec_success", 0.0) or 0.0) < 1.0

    if tier_name in {"T1", "T3"}:
        if error_reasons:
            mcq_errors = set(tier_result.get("mcq_errors") or [])
            if mcq_errors & error_reasons:
                return True
        exec_val = tier_result.get("exec_success_rate", 1.0 if tier_result.get("ok") else 0.0)
        return float(exec_val or 0.0) < 1.0

    return True


def pick_targets(
    results: List[Dict[str, Any]],
    tiers: List[str],
    error_reasons: Set[str],
    only_failed: bool,
    sample_limit: Optional[int],
) -> List[Tuple[int, str, List[str]]]:
    targets: List[Tuple[int, str, List[str]]] = []
    for idx, row in enumerate(results):
        selected_tiers = []
        tier_map = row.get("tiers") or {}
        for tier in tiers:
            if should_rerun_tier(tier, tier_map.get(tier, {}), error_reasons, only_failed):
                selected_tiers.append(tier)
        if selected_tiers:
            targets.append((idx, row["sample_id"], selected_tiers))
    if sample_limit is not None:
        targets = targets[:sample_limit]
    return targets


def rerun_selected_tiers(sample: Dict[str, Any], model_name: str, tiers: List[str]) -> Dict[str, Any]:
    updated = {}
    for tier in tiers:
        if tier == "T1":
            updated[tier] = base.eval_t1(sample, model_name)
        elif tier == "T2":
            updated[tier] = base.eval_t2(sample, model_name)
        elif tier == "T3":
            updated[tier] = base.eval_t3(sample, model_name)
        elif tier == "T4":
            updated[tier] = base.eval_t4(sample, model_name)
        else:
            updated[tier] = {"ok": False, "error": "unknown_tier"}
    return updated


def write_outputs(results: List[Dict[str, Any]], source_detail_path: Path, output_tag: str) -> Tuple[Path, Path, Path]:
    stem = source_detail_path.stem
    detail_path = source_detail_path.with_name(f"{stem}_{output_tag}.json")
    summary_json_path = detail_path.with_name(detail_path.name.replace("detail_", "summary_"))
    summary_csv_path = summary_json_path.with_suffix(".csv")

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_table = base.summarize_results_multi(results)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)

    cols = [
        "Dataset",
        "Model",
        "Task",
        "Subtask",
        "Exec Success Rate",
        "Accuracy Rate",
        "MAPE",
        "MAE",
        "RMSE",
        "SMAPE",
        "OW_sMAPE",
        "OW_RMSSE",
        "OW_MASE",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in summary_table:
            writer.writerow({k: row.get(k) for k in cols})

    return detail_path, summary_json_path, summary_csv_path


def main() -> None:
    patch_prompt_builders()

    print("Rerun config:")
    print(json.dumps(
        {
            "models": CONFIG["models"],
            "datasets": CONFIG["datasets"],
            "tiers": CONFIG["tiers"],
            "error_reasons": sorted(CONFIG["error_reasons"]),
            "only_failed": CONFIG["only_failed"],
            "sample_limit": CONFIG["sample_limit"],
            "dry_run": CONFIG["dry_run"],
            "output_tag": CONFIG["output_tag"],
            "force_multivariate_prompt_fix": CONFIG["force_multivariate_prompt_fix"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    for dataset_name in CONFIG["datasets"]:
        sample_map = load_dataset_sample_map(dataset_name)
        for model_name in CONFIG["models"]:
            detail_path = discover_detail_file(model_name, dataset_name)
            existing_results = load_existing_results(detail_path)
            scoped_results = [
                row for row in existing_results
                if row.get("model") == model_name and row.get("dataset") == dataset_name
            ]
            targets = pick_targets(
                scoped_results,
                CONFIG["tiers"],
                CONFIG["error_reasons"],
                CONFIG["only_failed"],
                CONFIG["sample_limit"],
            )

            print(f"\n== {model_name} | {dataset_name} ==")
            print(f"detail: {detail_path.name}")
            print(f"target samples: {len(targets)}")
            if not targets:
                continue

            if CONFIG["dry_run"]:
                for _, sample_id, selected_tiers in targets[:10]:
                    print(f"DRY-RUN target: sample_id={sample_id} tiers={selected_tiers}")
                continue

            merged_results = copy.deepcopy(existing_results)
            merged_index = {
                (row.get("model"), row.get("dataset"), row.get("sample_id")): idx
                for idx, row in enumerate(merged_results)
            }

            for run_idx, (_, sample_id, selected_tiers) in enumerate(targets, start=1):
                sample = sample_map.get(sample_id)
                if sample is None:
                    print(f"[Skip] sample not found in dataset source: {sample_id}")
                    continue
                print(f"rerun {run_idx}/{len(targets)} | sample={sample_id} | tiers={selected_tiers}")
                new_tiers = rerun_selected_tiers(sample, model_name, selected_tiers)
                key = (model_name, dataset_name, sample_id)
                if key not in merged_index:
                    print(f"[Skip] result row not found for merge: {key}")
                    continue
                row = merged_results[merged_index[key]]
                row.setdefault("tiers", {}).update(new_tiers)

            out_detail, out_summary_json, out_summary_csv = write_outputs(
                merged_results,
                detail_path,
                CONFIG["output_tag"],
            )
            print("saved:")
            print(f" - {out_detail}")
            print(f" - {out_summary_json}")
            print(f" - {out_summary_csv}")


if __name__ == "__main__":
    main()
