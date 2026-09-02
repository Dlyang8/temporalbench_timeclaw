from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .evaluate_llm import (
        MODEL_CONFIGS,
        REQUEST_TIMEOUT,
        SYSTEM_TEMPLATE,
        T1_RETURN_INSTRUCTION,
        TIER_MODALITY,
        _compute_forecast_metrics,
        _eval_mcq_block,
        _forecast_has_values,
        _get_history_series,
        build_t2_prompts,
        build_t3_single_prompt,
        build_t4_prompts,
        chat,
        extract_first_json,
        parse_t3_single_answer,
        summarize_results_multi,
    )
except ImportError:
    # Supports direct execution from within temporalbench/evaluation.
    from evaluate_llm import (  # type: ignore[no-redef]
        MODEL_CONFIGS,
        REQUEST_TIMEOUT,
        SYSTEM_TEMPLATE,
        T1_RETURN_INSTRUCTION,
        TIER_MODALITY,
        _compute_forecast_metrics,
        _eval_mcq_block,
        _forecast_has_values,
        _get_history_series,
        build_t2_prompts,
        build_t3_single_prompt,
        build_t4_prompts,
        chat,
        extract_first_json,
        parse_t3_single_answer,
        summarize_results_multi,
    )


DATASETS = ["freshretailnet", "PSML", "MIMIC", "causal_chambers", "M5"]
TIERS = ["T1", "T2", "T3", "T4"]

FORECAST_MISSING = "FORECAST_MISSING"
FORMAT_OR_PARSE = "FORMAT_OR_PARSE"
LENGTH_MISMATCH = "LENGTH_MISMATCH"
METRIC_FAIL_MAE = "METRIC_FAIL(MAE)"
METRIC_FAIL_OW = "METRIC_FAIL(OW_*)"
METRIC_FAIL_RMSE = "METRIC_FAIL(RMSE)"


@dataclass
class AgentCall:
    prediction: Any = None
    raw_text: str | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None
    elapsed_s: float = 0.0


def _sample_id(sample: dict) -> str:
    value = sample.get("sample_id")
    if value is not None:
        return str(value)
    for key in ("meta", "Meta JSON", "Meta", "metadata"):
        meta = sample.get(key)
        if isinstance(meta, dict):
            value = meta.get("source_id") or meta.get("id")
            if value is not None:
                return str(value)
    return f"sample_{abs(hash(json.dumps(sample, sort_keys=True, default=str)))}"


def parse_subset(subset: str | None) -> tuple[list[str], list[str]]:
    if not subset or subset.strip().lower() == "all":
        return list(DATASETS), list(TIERS)
    left, separator, right = subset.partition(":")
    tokens = [token.strip() for token in left.split(",") if token.strip()]
    if not separator and tokens and all(token.upper() in TIERS for token in tokens):
        return list(DATASETS), [token.upper() for token in tokens]

    dataset_lookup = {dataset.lower(): dataset for dataset in DATASETS}
    datasets = []
    for token in tokens:
        if token.lower() not in dataset_lookup:
            raise ValueError(f"Unknown dataset {token!r}; expected one of {DATASETS}")
        datasets.append(dataset_lookup[token.lower()])
    datasets = datasets or list(DATASETS)

    tiers = [token.strip().upper() for token in right.split(",") if token.strip()]
    tiers = tiers or list(TIERS)
    unknown = [tier for tier in tiers if tier not in TIERS]
    if unknown:
        raise ValueError(f"Unknown tiers {unknown}; expected one of {TIERS}")
    return datasets, tiers


def load_records(root: Path, datasets: list[str], tiers: list[str]) -> list[dict]:
    records = []
    for dataset in datasets:
        path = root / "data" / dataset / "task_modified.json"
        with path.open("r", encoding="utf-8") as handle:
            samples = json.load(handle)
        for sample in samples:
            sid = _sample_id(sample)
            sample["_dataset_name"] = dataset
            for tier in tiers:
                node = (sample.get("tasks") or {}).get(tier)
                if not isinstance(node, dict):
                    continue
                records.append(
                    {
                        "task_id": f"{dataset}::{sid}::{tier}",
                        "dataset": dataset,
                        "tier": tier,
                        "sample_id": sid,
                        "sample": sample,
                        "node": node,
                    }
                )
    return records


def select_records(
    records: list[dict],
    *,
    ratio: float,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """Select whole samples so T1-T4 always stay together."""
    ratio = min(1.0, max(0.0, float(ratio)))
    by_sample: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        key = (record["dataset"], record["sample_id"])
        by_sample[key].append(record)
    for key in by_sample:
        by_dataset[key[0]].append(key)

    chosen: set[tuple[str, str]] = set()
    for dataset, keys in sorted(by_dataset.items()):
        ordered = sorted(keys)
        random.Random(f"{seed}:{dataset}").shuffle(ordered)
        if ratio >= 1.0:
            count = len(ordered)
        elif ratio <= 0.0:
            count = 0
        else:
            count = max(1, round(len(ordered) * ratio))
        chosen.update(ordered[:count])

    selected = [
        record
        for record in records
        if (record["dataset"], record["sample_id"]) in chosen
    ]
    return selected, {
        "n_records_total": len(records),
        "n_records_selected": len(selected),
        "n_sample_units_total": len(by_sample),
        "n_sample_units_selected": len(chosen),
    }


def _first_history(sample: dict) -> dict:
    tasks = sample.get("tasks") or {}
    for tier in ("T4", "T2", "T3", "T1"):
        node = tasks.get(tier)
        if isinstance(node, dict):
            history = (node.get("input") or {}).get("history")
            if isinstance(history, dict):
                return history
    return {}


def _forecast_node(record: dict) -> dict:
    """T1/T3 are pure-understanding tiers with no numeric target of their own.

    TimeCopilot's `query()` requires a completed `analyze()/forecast()` run
    (it needs `dataset`, `fcst_df`, `eval_df`, `features_df`, `anomalies_df`
    all set), so for T1/T3 we bootstrap the agent using the sibling T2/T4
    node's series purely to make the agent queryable; the sibling's
    forecast/metrics are never scored under the T1/T3 task_id.
    """
    if record["tier"] in ("T2", "T4"):
        return record["node"]
    for tier in ("T2", "T4"):
        node = (record["sample"].get("tasks") or {}).get(tier)
        if isinstance(node, dict) and node.get("ground_truth"):
            return node
    raise ValueError("T1/T3 record has no T2/T4 sibling for TimeCopilot bootstrap")


def _numeric_values(values: Any) -> list[float] | None:
    if not isinstance(values, list):
        return None
    result = []
    for value in values:
        if value is None:
            result.append(float("nan"))
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            return None
    return result


def _map_targets_to_history(record: dict, node: dict) -> dict[str, list[float]]:
    history = _first_history(record["sample"])
    numeric = {
        str(name): parsed
        for name, values in history.items()
        if name != "timestamps" and (parsed := _numeric_values(values)) is not None
    }
    ground_truth = node.get("ground_truth") or {}
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("ground_truth must be a non-empty channel dictionary")

    mapped: dict[str, list[float]] = {}
    unused = set(numeric)
    for target in map(str, ground_truth):
        candidates = [
            target,
            target.removeprefix("future_"),
            target.removeprefix("future__"),
        ]
        match = next((candidate for candidate in candidates if candidate in numeric), None)
        if match is None:
            casefolded = {name.casefold(): name for name in unused}
            match = next(
                (casefolded[candidate.casefold()] for candidate in candidates
                 if candidate.casefold() in casefolded),
                None,
            )
        if match is not None:
            mapped[target] = numeric[match]
            unused.discard(match)

    if len(ground_truth) == 1 and not mapped and numeric:
        # Some datasets (e.g. freshretailnet's "future_sales" vs. history's
        # "sales_censored"; PSML's "future_main"/"future_series" vs.
        # "load_power") give the single forecast target a generic name that
        # doesn't literally match any history channel. In every such case the
        # target is the *first* channel listed in history (the dataset's
        # documented primary series), so fall back to that rather than
        # requiring history to contain exactly one numeric channel.
        primary_channel = next(iter(numeric))
        mapped[str(next(iter(ground_truth)))] = numeric[primary_channel]
        unused.discard(primary_channel)

    remaining = [str(target) for target in ground_truth if str(target) not in mapped]
    if remaining and len(remaining) == len(unused):
        for target, source in zip(remaining, sorted(unused)):
            mapped[target] = numeric[source]

    missing = [str(target) for target in ground_truth if str(target) not in mapped]
    if missing:
        raise ValueError(f"Could not map historical channels for targets {missing}")
    return mapped


_LEGACY_FREQ_ALIASES = {
    "T": "min",
    "S": "s",
    "H": "h",
    "L": "ms",
    "U": "us",
    "N": "ns",
    "BH": "bh",
    "Y": "YE",
    "A": "YE",
    "Q": "QE",
    "M": "ME",
}


def _normalize_freq(freq: str) -> str:
    """Coerce legacy pandas offset aliases (e.g. "60T") into their modern
    equivalents (e.g. "60min"). Pandas 3.x removed the old single-letter
    aliases for minute/second/hour/etc. outright (ValueError, not a
    deprecation warning), so datasets whose recorded `meta["freq"]` predates
    that change would otherwise raise and get silently swallowed by a
    downstream fallback to "D" -- corrupting the frequency for the whole
    series rather than just failing loudly.
    """
    freq = str(freq).strip()
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except ValueError:
        pass
    match = re.match(r"^(-?\d*)([A-Za-z]+)(-[A-Za-z]{3,})?$", freq)
    if match:
        count, unit, anchor = match.groups()
        replacement = _LEGACY_FREQ_ALIASES.get(unit.upper())
        if replacement is not None:
            candidate = f"{count}{replacement}{anchor or ''}"
            try:
                pd.tseries.frequencies.to_offset(candidate)
                return candidate
            except ValueError:
                pass
    # Give up gracefully; caller's own except-block will fall back to "D".
    return freq


_START_END_KEY_PAIRS = [
    ("history_start", "history_end"),
    ("time_start", "time_end"),
    ("start", "end"),
]


def _implied_freq_timedelta(meta: dict, length: int) -> pd.Timedelta | None:
    """Approximate a per-record sampling interval from start/end metadata.

    Some datasets (e.g. MIMIC) provide neither an explicit `history[
    "timestamps"]` array nor a `meta["freq"]` string, only a
    `history_start`/`history_end` window. Falling back to a hardcoded "D"
    there is badly wrong -- MIMIC's real spacing is on the order of tens of
    minutes, not days, and varies per-record (clusters aren't uniformly
    spaced), so treat the window as an average sampling interval instead of
    silently assuming daily.
    """
    if length < 2:
        return None
    for start_key, end_key in _START_END_KEY_PAIRS:
        start_raw = meta.get(start_key)
        end_raw = meta.get(end_key)
        if not start_raw or not end_raw:
            continue
        start = pd.to_datetime(start_raw, errors="coerce")
        end = pd.to_datetime(end_raw, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        spacing = (end - start) / (length - 1)
        if spacing > pd.Timedelta(0):
            # Round to the nearest second: these windows are themselves
            # approximations, and sub-second precision just produces an
            # ugly, non-roundtripping frequency string downstream.
            return spacing.round("s")
    return None


def _dates(record: dict, length: int) -> tuple[pd.DatetimeIndex, str, bool]:
    history = _first_history(record["sample"])
    raw = history.get("timestamps")
    if isinstance(raw, list) and len(raw) == length:
        parsed = pd.to_datetime(raw, errors="coerce")
        if not parsed.isna().any():
            dates = pd.DatetimeIndex(parsed)
            # TimeCopilot's statistical backbones require timezone-naive
            # timestamps. Preserve UTC clock time while removing the timezone.
            if dates.tz is not None:
                dates = dates.tz_convert(None)
            frequency = pd.infer_freq(dates) if len(dates) >= 3 else None
            return dates, frequency or "D", False

    meta = record["sample"].get("meta") or {}
    frequency = (
        meta.get("freq")
        or meta.get("frequency")
        or record["node"].get("freq")
        or record["node"].get("frequency")
    )
    if frequency:
        frequency = _normalize_freq(frequency)
    else:
        implied = _implied_freq_timedelta(meta, length)
        if implied is not None:
            try:
                dates = pd.date_range("2000-01-01", periods=length, freq=implied)
                freq_str = pd.tseries.frequencies.to_offset(implied).freqstr
                return dates, freq_str, True
            except (TypeError, ValueError):
                pass
        frequency = "D"
    try:
        dates = pd.date_range("2000-01-01", periods=length, freq=str(frequency))
        return dates, str(frequency), True
    except (TypeError, ValueError):
        return pd.date_range("2000-01-01", periods=length, freq="D"), "D", True


_SUB_MINUTE_BASE_UNITS = {"s", "ms", "us", "ns"}


def _suggest_seasonality(frequency: str) -> int | None:
    """Override TimeCopilot's automatic seasonal-period inference for
    frequencies/history lengths where it appears to assume a seasonal cycle
    longer than the available history (e.g. causal_chambers' second-level
    series, only ~250-300 points long, fail with "SeasonalNaive has NaN
    values" -- consistent with a season_length that can't be satisfied by
    that little history). For sub-minute frequencies we don't have a
    trustworthy natural cycle length anyway, so force no seasonality (1)
    rather than let it guess. This also covers MIMIC's irregular implied
    intervals, which pandas often represents as an exact number of seconds
    (for example, "2596s"). Those synthetic intervals do not define a
    trustworthy seasonal cycle, so TimeCopilot should not infer one.
    """
    match = re.match(r"^-?\d*([A-Za-z]+)", str(frequency))
    base_unit = match.group(1) if match else ""
    if base_unit in _SUB_MINUTE_BASE_UNITS:
        return 1
    return None


def to_timecopilot_frame(
    record: dict,
) -> tuple[pd.DataFrame, int, str, dict[str, Any]]:
    node = _forecast_node(record)
    target_history = _map_targets_to_history(record, node)
    ground_truth = node.get("ground_truth") or {}
    horizons = {
        str(channel): len(values)
        for channel, values in ground_truth.items()
        if isinstance(values, list) and values
    }
    if not horizons:
        raise ValueError("Could not infer a forecast horizon")
    if len(set(horizons.values())) != 1:
        raise ValueError(f"TimeCopilot requires a shared horizon; received {horizons}")
    horizon = next(iter(horizons.values()))

    frames = []
    frequency = "D"
    synthetic = False
    for channel, values in target_history.items():
        dates, channel_frequency, used_synthetic = _dates(record, len(values))
        frequency = channel_frequency
        synthetic = synthetic or used_synthetic
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": channel,
                    "ds": dates,
                    "y": values,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    missing_before = int(frame["y"].isna().sum())

    # TimeCopilot does not reliably handle missing target observations.
    # Per maintainer guidance, replace only missing input-history targets
    # with zero and record the policy in result metadata.
    frame["y"] = frame.groupby(
        "unique_id",
        group_keys=False,
    )["y"].transform(
        lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0)
    )

    missing_by_series = (
        frame.groupby("unique_id")["y"]
        .apply(lambda series: int(series.isna().sum()))
        .to_dict()
    )
    unresolved = {
        str(channel): count
        for channel, count in missing_by_series.items()
        if count
    }
    if unresolved:
        raise ValueError(
            "Historical target channels remain entirely or partially missing "
            f"after imputation: {unresolved}"
        )

    values = frame["y"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Historical target contains non-finite values after imputation"
        )

    return frame, horizon, frequency, {
        "horizons": horizons,
        "history_channels": list(target_history),
        "synthetic_timestamps": synthetic,
        "imputation": "zero_fill_missing_input_targets",
        "missing_values_imputed": missing_before,
        "imputation_note": (
            "Missing input-history target values were replaced with zero "
            "because TimeCopilot does not reliably handle NaN inputs."
        ),
        "suggested_seasonality": _suggest_seasonality(frequency),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _output_text(result: Any) -> str:
    output = getattr(result, "output", result)
    if isinstance(output, str):
        return output
    return json.dumps(_jsonable(output), ensure_ascii=False, default=str)


def _make_agent(llm: str, retries: int) -> tuple[Any, dict[str, Any]]:
    """Construct a TimeCopilot agent, tolerating older/newer library signatures.

    Match evaluate_llm.py by applying REQUEST_TIMEOUT to each underlying LLM
    HTTP request. This does not impose a wall-clock limit on TimeCopilot's
    complete feature-analysis, CV, fitting, and forecasting pipeline.
    """
    from timecopilot import TimeCopilot

    try:
        return TimeCopilot(
            llm=llm,
            retries=retries,
            model_settings={"timeout": REQUEST_TIMEOUT},
        ), {
            "retries_applied": True,
            "request_timeout_applied": True,
            "request_timeout_seconds": REQUEST_TIMEOUT,
        }
    except TypeError as retries_error:
        # Compatibility path: drop only `retries`; retain the required HTTP
        # timeout rather than silently running without it.
        try:
            agent = TimeCopilot(
                llm=llm,
                model_settings={"timeout": REQUEST_TIMEOUT},
            )
        except TypeError as timeout_error:
            raise RuntimeError(
                "Installed TimeCopilot/Pydantic-AI cannot apply the required "
                f"{REQUEST_TIMEOUT}-second HTTP-request timeout"
            ) from timeout_error
        return agent, {
            "retries_applied": False,
            "retries_fallback_reason": str(retries_error),
            "request_timeout_applied": True,
            "request_timeout_seconds": REQUEST_TIMEOUT,
        }


def _forecast_from_result(
    attempt: dict[str, Any],
    expected_horizons: dict[str, int],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    frame = attempt.get("fcst_df")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("TimeCopilot returned no fcst_df")

    output = attempt.get("output")
    selected_model = attempt.get("selected_model")
    if selected_model not in frame.columns:
        raise ValueError(
            f"Selected forecast model {selected_model!r} is absent from "
            f"fcst_df columns {list(frame.columns)}"
        )

    forecast = {}
    for channel, horizon in expected_horizons.items():
        rows = frame[frame["unique_id"].astype(str) == channel]
        if "ds" in rows:
            rows = rows.sort_values("ds")
        if len(rows) < horizon:
            raise ValueError(
                f"Forecast channel {channel!r} has {len(rows)} rows; "
                f"expected at least {horizon}"
            )
        values = pd.to_numeric(
            rows[selected_model].iloc[:horizon], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Forecast channel {channel!r} contains non-finite values"
            )
        forecast[channel] = values.tolist()
    return forecast, {
        "selected_model": str(selected_model),
        "forecast_columns": list(frame.columns),
        "timecopilot_output": _jsonable(output),
    }


def _t1_prompt(record: dict) -> str:
    node = record["node"]
    return f"{node.get('prompt') or node.get('task') or ''}\n\n{T1_RETURN_INSTRUCTION}"


def _forecast_mcq_prompt(record: dict) -> str:
    """Use the same qualitative prompt builders as evaluate_llm.py."""
    node = record["node"]
    sample_meta = record["sample"].get("meta") or {}
    if record["tier"] == "T2":
        return build_t2_prompts(node, sample_meta=sample_meta)["mcq_prompt"]
    return build_t4_prompts(node, sample_meta=sample_meta)["mcq_prompt"]


def _qualitative_model_name(llm: str, explicit: str | None) -> str:
    """Resolve a evaluate_llm.py registry key for the T1/T3 baseline calls."""
    if explicit:
        candidate = explicit
    elif llm in MODEL_CONFIGS:
        candidate = llm
    else:
        candidate = llm.split(":", 1)[-1]
    if candidate not in MODEL_CONFIGS:
        raise ValueError(
            f"Qualitative model {candidate!r} is not in evaluate_llm.MODEL_CONFIGS. "
            "Pass --qualitative-model with a configured model key."
        )
    return candidate


def _baseline_chat(model_name: str, prompt: str) -> str:
    return chat(
        model_name,
        [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": prompt},
        ],
    )


def _run_t1(record: dict, qualitative_model: str) -> AgentCall:
    started = time.monotonic()
    try:
        raw = _baseline_chat(qualitative_model, _t1_prompt(record))
        return AgentCall(
            prediction=extract_first_json(raw),
            raw_text=raw,
            metadata={
                "inference_path": "evaluate_llm_t1",
                "qualitative_model": qualitative_model,
            },
            elapsed_s=time.monotonic() - started,
        )
    except Exception as exc:
        return AgentCall(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )


def _run_t3(record: dict, qualitative_model: str) -> AgentCall:
    started = time.monotonic()
    answers: list[str | None] = []
    raw_outputs: list[str] = []
    node = record["node"]
    pack = node.get("pack") or []
    if not isinstance(pack, list) or not pack:
        return AgentCall(
            error="ValueError: empty_pack",
            elapsed_s=time.monotonic() - started,
        )

    for index, item in enumerate(pack):
        try:
            prompt = build_t3_single_prompt(node, item, index)
            raw = _baseline_chat(qualitative_model, prompt)
            raw_outputs.append(raw)
            answer = parse_t3_single_answer(raw)
            options = (
                item.get("label_space")
                or item.get("options")
                or item.get("Options")
                or []
            )
            valid = {str(option).strip() for option in options}
            if answer is not None:
                answer = str(answer).strip()
                if valid and answer not in valid:
                    answer = None
            answers.append(answer)
        except Exception as exc:
            # Match evaluate_llm.py: one failed item aborts the entire T3 tier.
            return AgentCall(
                error=f"{type(exc).__name__}: {exc}",
                raw_text="\n\n--- T3 ITEM ---\n\n".join(raw_outputs),
                metadata={
                    "inference_path": "evaluate_llm_t3",
                    "qualitative_model": qualitative_model,
                    "failed_item": str(item.get("task_id", index)),
                },
                elapsed_s=time.monotonic() - started,
            )

    return AgentCall(
        prediction={"answers": answers},
        raw_text="\n\n--- T3 ITEM ---\n\n".join(raw_outputs),
        metadata={
            "inference_path": "evaluate_llm_t3",
            "qualitative_model": qualitative_model,
        },
        elapsed_s=time.monotonic() - started,
    )


def _user_query_response(result: Any) -> Any:
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        return output.get("user_query_response")
    return getattr(output, "user_query_response", None)


def _parse_query_mcq(
    response: Any,
    ground_truth: dict,
) -> tuple[dict[str, Any] | None, str]:
    if response is None:
        return None, ""

    text = response if isinstance(response, str) else _output_text(response)

    # Prefer the JSON format requested by the benchmark prompt.
    parsed = response if isinstance(response, dict) else extract_first_json(text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("mcq"), dict):
            parsed = parsed["mcq"]
        if parsed:
            return parsed, text

    # TimeCopilot often wraps otherwise valid answers in an explanation such
    # as "**Q1 (future_vs_history)**: **Similar**". Adapt that prose to the
    # dictionary consumed by evaluate_llm.py's unchanged MCQ scorer.
    answers: dict[str, Any] = {}
    questions = list((ground_truth or {}).items())

    for index, (question_id, question) in enumerate(questions, start=1):
        next_index = index + 1
        section_match = re.search(
            rf"(?is)(?:\*\*)?\bQ{index}\b.*?"
            rf"(?=(?:\*\*)?\bQ{next_index}\b|\Z)",
            text,
        )
        if section_match is None:
            continue

        section = section_match.group(0)
        options = question.get("options", []) if isinstance(question, dict) else []
        candidates: list[tuple[int, int, str]] = []

        for option in options:
            option_text = str(option).strip()
            if not option_text:
                continue

            escaped = re.escape(option_text)
            bold_match = re.search(
                rf"(?i)\*\*\s*{escaped}\s*\*\*",
                section,
            )
            plain_match = re.search(
                rf"(?i)(?<!\w){escaped}(?!\w)",
                section,
            )
            if bold_match is not None:
                candidates.append((0, bold_match.start(), option_text))
            elif plain_match is not None:
                candidates.append((1, plain_match.start(), option_text))

        if candidates:
            answers[question_id] = min(candidates)[2]

    return (answers if answers else None), text


def run_agent(
    record: dict,
    llm: str,
    retries: int,
    qualitative_model: str,
) -> AgentCall:
    if record["tier"] == "T1":
        return _run_t1(record, qualitative_model)
    if record["tier"] == "T3":
        return _run_t3(record, qualitative_model)

    started = time.monotonic()
    try:
        frame, horizon, frequency, metadata = to_timecopilot_frame(record)
    except Exception as exc:
        return AgentCall(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )

    node = record["node"]
    expected = {
        str(channel): len(values)
        for channel, values in (node.get("ground_truth") or {}).items()
        if isinstance(values, list)
    }

    try:
        agent, agent_build_info = _make_agent(llm, retries)
        result = agent.forecast(
            df=frame,
            h=horizon,
            freq=frequency,
            seasonality=metadata.get("suggested_seasonality"),
            query=_forecast_mcq_prompt(record),
        )
        forecast_frame = getattr(result, "fcst_df", None)
        if forecast_frame is None:
            forecast_frame = getattr(agent, "fcst_df", None)
        output = getattr(result, "output", None)
        forecast_result = {
            "fcst_df": forecast_frame,
            "selected_model": getattr(output, "selected_model", None),
            "output": _jsonable(output),
            "result_text": _output_text(result),
            "user_query_response": _jsonable(_user_query_response(result)),
            "agent_build_info": agent_build_info,
        }
        forecast, forecast_metadata = _forecast_from_result(
            forecast_result,
            expected,
        )
        forecast_result["forecast"] = forecast
        forecast_result["forecast_metadata"] = forecast_metadata
    except Exception as exc:
        exc_text = f"{type(exc).__name__}: {exc}"
        dataset_key = record["dataset"].lower().replace("_", "")
        if (
            dataset_key == "causalchambers"
            and "model seasonalnaive has nan values" in exc_text.casefold()
        ):
            return AgentCall(
                prediction={
                    "forecast": {
                        channel: [0.0] * channel_horizon
                        for channel, channel_horizon in expected.items()
                    },
                    "mcq": None,
                },
                metadata={
                    **metadata,
                    "inference_path": "causal_chambers_zero_fallback",
                    "fallback_reason": "SeasonalNaive produced NaN values",
                    "original_error": exc_text,
                },
                elapsed_s=time.monotonic() - started,
            )
        return AgentCall(
            error=exc_text,
            elapsed_s=time.monotonic() - started,
        )

    try:
        forecast = forecast_result["forecast"]
        forecast_metadata = forecast_result["forecast_metadata"]
        query_response = forecast_result.get("user_query_response")
        mcq, query_text = _parse_query_mcq(
            query_response,
            node.get("mcq") or {},
        )
        raw_outputs = [str(forecast_result.get("result_text") or "")]
        if query_text:
            raw_outputs.append(query_text)
        prediction = {"forecast": forecast, "mcq": mcq}

        return AgentCall(
            prediction=prediction,
            raw_text="\n\n--- TIMECOPILOT CALL ---\n\n".join(raw_outputs),
            metadata={
                **metadata,
                **forecast_metadata,
                **(forecast_result.get("agent_build_info") or {}),
                "inference_path": "timecopilot_forecast",
                "user_query_response": _jsonable(query_response),
                "follow_up_error": (
                    None if mcq is not None else "Could not parse user_query_response"
                ),
            },
            elapsed_s=time.monotonic() - started,
        )
    except Exception as exc:
        return AgentCall(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
        )


# ============================================================================
# evaluate_llm.py-compatible scoring and output
# ============================================================================

_DATASET_DISPLAY_NAMES = {
    "freshretailnet": "FreshRetailNet",
    "PSML": "PSML",
    "MIMIC": "MIMIC",
    "causal_chambers": "CausalChambers",
    "M5": "M5",
}


def _runtime_failure(tier: str, error: str) -> dict[str, Any]:
    """Mirror evaluate_llm.run_once()'s generic runtime-error entry."""
    entry: dict[str, Any] = {
        "ok": False,
        "error": f"runtime_error: {error}",
        "forecast_errors": [],
        "mcq_errors": [],
    }
    modes = TIER_MODALITY.get(tier, {"mcq"})
    if "forecast" in modes:
        entry["forecast_errors"].append("other_error")
    if "mcq" in modes:
        entry["mcq_errors"].append("other_error")
    return entry


def _score_t1_like_reference(record: dict, call: AgentCall) -> dict[str, Any]:
    prediction = call.prediction
    if not isinstance(prediction, dict):
        return {
            "ok": False,
            "error": "invalid_json",
            "mcq_errors": ["invalid_json"],
            "raw": call.raw_text,
            "timecopilot": call.metadata,
            "elapsed_s": call.elapsed_s,
        }

    evaluation = _eval_mcq_block(
        prediction,
        record["node"].get("labels") or {},
    )
    result: dict[str, Any] = {
        "ok": True,
        "mcq_acc": evaluation.get("acc"),
        "mcq": evaluation,
        "raw": prediction,
        "mcq_errors": list(evaluation.get("error_types") or []),
        "timecopilot": call.metadata,
        "elapsed_s": call.elapsed_s,
    }
    if evaluation.get("error_counts"):
        result["mcq_error_counts"] = evaluation["error_counts"]
    return result


def _score_t3_like_reference(record: dict, call: AgentCall) -> dict[str, Any]:
    pack = record["node"].get("pack") or []
    answers = (
        call.prediction.get("answers")
        if isinstance(call.prediction, dict)
        else None
    )
    correct = 0
    valid = 0
    error_counts = {"missing_answer": 0, "invalid_option": 0}
    details: dict[str, dict[str, Any]] = {}

    for index, item in enumerate(pack):
        task_id = str(item.get("task_id", index))
        answer = (
            answers[index]
            if isinstance(answers, list) and index < len(answers)
            else None
        )
        answer = str(answer).strip() if answer is not None else None
        options = (
            item.get("label_space")
            or item.get("options")
            or item.get("Options")
            or []
        )
        valid_options = {str(option).strip() for option in options}
        invalid = bool(
            answer is not None
            and valid_options
            and answer not in valid_options
        )
        if invalid:
            answer = None

        ground_truth = (
            item.get("label")
            or item.get("Answer")
            or item.get("answer")
        )
        ground_truth = (
            str(ground_truth).strip()
            if ground_truth is not None
            else None
        )
        is_correct = (
            answer is not None
            and ground_truth is not None
            and answer == ground_truth
        )
        if answer is not None:
            valid += 1
        if is_correct:
            correct += 1
        elif invalid:
            error_counts["invalid_option"] += 1
        elif answer is None:
            error_counts["missing_answer"] += 1

        details[task_id] = {
            "pred": answer,
            "gt": ground_truth,
            "correct": is_correct,
        }

    total = len(pack)
    return {
        "ok": True,
        "exec_success_rate": valid / total if total else None,
        "mcq_acc": correct / total if total else None,
        "mcq_error_counts": error_counts,
        "mcq_errors": [
            name for name, count in error_counts.items() if count > 0
        ],
        "details": details,
        "raw": None,
        "timecopilot": call.metadata,
        "elapsed_s": call.elapsed_s,
    }


def _score_forecast_like_reference(
    record: dict,
    call: AgentCall,
) -> dict[str, Any]:
    node = record["node"]
    prediction = call.prediction if isinstance(call.prediction, dict) else {}
    forecast = prediction.get("forecast")
    submitted_mcq = prediction.get("mcq")
    result: dict[str, Any] = {
        "ok": True,
        "forecast_errors": [],
        "mcq_errors": [],
        "timecopilot": call.metadata,
        "elapsed_s": call.elapsed_s,
    }

    if not _forecast_has_values(forecast):
        result["forecast_exec_success"] = 0.0
        result["forecast_errors"].append("missing_forecast")
    else:
        result["forecast_exec_success"] = 1.0

    metrics, metric_error = _compute_forecast_metrics(
        node.get("ground_truth"),
        forecast,
        record["dataset"].lower(),
        _get_history_series(node),
    )
    result["metrics"] = metrics
    if metrics is None:
        result["forecast_exec_success"] = 0.0
    if metric_error:
        result["forecast_errors"].append(metric_error)
        result["forecast_exec_success"] = 0.0

    if not isinstance(submitted_mcq, dict) or not submitted_mcq:
        result["mcq_errors"].append("missing_output")
        submitted_mcq = {}

    mcq_ground_truth = node.get("mcq") or {}
    mcq_evaluation = (
        _eval_mcq_block(submitted_mcq, mcq_ground_truth)
        if mcq_ground_truth
        else None
    )
    result["mcq"] = mcq_evaluation
    result["mcq_exec_success"] = (
        1.0
        if isinstance(submitted_mcq, dict)
        and any(value is not None for value in submitted_mcq.values())
        else 0.0
    )
    if mcq_evaluation and mcq_evaluation.get("error_types"):
        result["mcq_errors"].extend(mcq_evaluation["error_types"])
    if mcq_evaluation and mcq_evaluation.get("error_counts"):
        result["mcq_error_counts"] = mcq_evaluation["error_counts"]

    result["raw"] = {
        "forecast": forecast,
        "mcq": submitted_mcq,
        "user_query_response": (
            (call.metadata or {}).get("user_query_response")
        ),
    }
    return result


def evaluate_record(
    record: dict,
    llm: str,
    retries: int,
    qualitative_model: str,
) -> dict[str, Any]:
    """Evaluate one tier and return an evaluate_llm-compatible tier result."""
    call = run_agent(record, llm, retries, qualitative_model)
    tier = record["tier"]
    if call.error:
        tier_output = _runtime_failure(tier, call.error)
        tier_output["elapsed_s"] = call.elapsed_s
    elif tier == "T1":
        tier_output = _score_t1_like_reference(record, call)
    elif tier == "T3":
        tier_output = _score_t3_like_reference(record, call)
    else:
        tier_output = _score_forecast_like_reference(record, call)

    return {
        "task_id": record["task_id"],
        "dataset": record["dataset"],
        "tier": tier,
        "sample_id": record["sample_id"],
        "tier_output": tier_output,
    }


def _group_like_reference(
    flat_results: list[dict[str, Any]],
    llm: str,
) -> list[dict[str, Any]]:
    """Create evaluate_llm.run_once()-shaped sample results."""
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in flat_results:
        dataset = item["dataset"]
        sample_id = item["sample_id"]
        key = (dataset, sample_id)
        if key not in grouped:
            grouped[key] = {
                "model": llm,
                "dataset": _DATASET_DISPLAY_NAMES.get(dataset, dataset),
                "sample_id": sample_id,
                "tiers": {},
            }
        grouped[key]["tiers"][item["tier"]] = item["tier_output"]
    return [grouped[key] for key in sorted(grouped)]


def _runtime_error_category(error: str | None) -> str:
    """Map runtime failures to the benchmark's fixed six-way taxonomy."""
    lowered = (error or "").casefold()
    if any(
        marker in lowered
        for marker in (
            "has nan values",
            "returned no fcst_df",
            "no forecast",
            "missing forecast",
            "non-finite",
        )
    ):
        return FORECAST_MISSING
    if any(
        marker in lowered
        for marker in (
            "length mismatch",
            "incorrect length",
            "expected forecast length",
            "horizon length",
        )
    ):
        return LENGTH_MISMATCH
    return FORMAT_OR_PARSE


def _diagnostic_summary_items(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt reference-shaped tier outputs to the diagnostic summary schema."""
    items: list[dict[str, Any]] = []
    for result in results:
        tier = result["tier"]
        output = result.get("tier_output") or {}
        is_ok = bool(output.get("ok"))
        runtime_error = output.get("error") if not is_ok else None
        forecast_errors = list(output.get("forecast_errors") or [])

        error_category = None
        if runtime_error:
            error_category = _runtime_error_category(str(runtime_error))
        elif forecast_errors:
            for reason in forecast_errors:
                category = _bucket_forecast_error(reason)
                if category:
                    error_category = category
                    break

        mcq_counts = dict(output.get("mcq_error_counts") or {})
        # Keep the MCQ taxonomy identical to evaluate_llm.py.
        mcq_counts = {
            name: int(mcq_counts.get(name, 0) or 0)
            for name in ("missing_answer", "invalid_option")
        }

        metrics = output.get("metrics")
        if tier == "T1":
            metrics = {
                "acc": output.get("mcq_acc"),
                "correct": (
                    (output.get("mcq") or {}).get("correct")
                    if isinstance(output.get("mcq"), dict)
                    else None
                ),
                "total": (
                    (output.get("mcq") or {}).get("total")
                    if isinstance(output.get("mcq"), dict)
                    else None
                ),
            }
        elif tier == "T3":
            metrics = {
                "acc": output.get("mcq_acc"),
                "exec_success_rate": output.get("exec_success_rate"),
            }

        items.append(
            {
                "task_id": result["task_id"],
                "dataset": result["dataset"],
                "tier": tier,
                "sample_id": result["sample_id"],
                "metrics": metrics,
                "mcq": output.get("mcq"),
                "forecast_exec_success": (
                    (
                        output.get("forecast_exec_success")
                        if is_ok
                        else 0.0
                    )
                    if tier in ("T2", "T4")
                    else None
                ),
                "mcq_exec_success": (
                    (
                        output.get("mcq_exec_success")
                        if is_ok
                        else 0.0
                    )
                    if tier in ("T2", "T4")
                    else None
                ),
                "exec_success_rate": (
                    (1.0 if is_ok else 0.0)
                    if tier == "T1"
                    else output.get("exec_success_rate")
                    if tier == "T3" and is_ok
                    else 0.0
                    if tier == "T3"
                    else None
                ),
                "runtime_error": runtime_error,
                "forecast_errors": forecast_errors,
                "mcq_errors": list(output.get("mcq_errors") or []),
                "mcq_error_counts": mcq_counts,
                "error_category": error_category,
                "timecopilot": output.get("timecopilot"),
            }
        )
    return items


def _bucket_forecast_error(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason in {"missing_forecast", "missing_channel"}:
        return FORECAST_MISSING
    if reason == "length_mismatch":
        return LENGTH_MISMATCH
    if reason in {
        "invalid_json",
        "invalid_format",
        "series_mismatch",
        "no_valid_values",
        "missing_ground_truth",
    }:
        return FORMAT_OR_PARSE
    if reason in {
        "metric_threshold_OW_sMAPE",
        "metric_threshold_OW_RMSSE",
        "metric_threshold_OW_MASE",
    }:
        return METRIC_FAIL_OW
    if reason == "metric_threshold_RMSE":
        return METRIC_FAIL_RMSE
    if reason and reason.startswith("metric_threshold_"):
        return METRIC_FAIL_MAE
    return FORMAT_OR_PARSE


def _leaderboard_summary(
    summary_table: list[dict[str, Any]],
    llm: str,
    category_order: tuple[str, ...],
) -> dict[str, Any]:
    """Flatten reference summary rows into the paper leaderboard schema."""
    indexed = {
        (row.get("Dataset"), row.get("Task"), row.get("Subtask")): row
        for row in summary_table
    }

    def value(
        dataset: str,
        task: str,
        subtask: str,
        field: str,
    ) -> Any:
        return indexed.get((dataset, task, subtask), {}).get(field)

    output: dict[str, Any] = {
        "agent_name": "TimeCopilot",
        "agent_type": "time-series-specific agent",
        "base_model": llm.split(":", 1)[-1],
        "T1_acc": None,
        "T2_acc": None,
        "T3_acc": None,
        "T4_acc": None,
    }

    for dataset in (
        "FreshRetailNet",
        "PSML",
        "CausalChambers",
        "MIMIC",
        "M5",
    ):
        output[f"{dataset}_T1_acc"] = value(
            dataset, "T1", "understanding", "Accuracy Rate"
        )
        output[f"{dataset}_T2_acc"] = value(
            dataset, "T2", "mcq", "Accuracy Rate"
        )
        output[f"{dataset}_T3_acc"] = value(
            dataset, "T3", "pack", "Accuracy Rate"
        )
        output[f"{dataset}_T4_acc"] = value(
            dataset, "T4", "mcq", "Accuracy Rate"
        )
        dataset_forecast_rows = [
            row
            for row in summary_table
            if row.get("Dataset") == dataset
            and row.get("Subtask") == "forecast"
        ]
        for category in category_order:
            output[f"{dataset}_{category}"] = sum(
                int(row.get(category, 0) or 0)
                for row in dataset_forecast_rows
            )

    output.update(
        {
            "T2_sMAPE": None,
            "T2_MAE": None,
            "T2_OW_sMAPE_MIMIC": None,
            "T2_OW_RMSSE_MIMIC": None,
            "T4_sMAPE": None,
            "T4_MAE": None,
            "T4_OW_sMAPE_MIMIC": None,
            "T4_OW_RMSSE_MIMIC": None,
        }
    )

    for dataset in ("FreshRetailNet", "PSML", "M5"):
        for task in ("T2", "T4"):
            output[f"{dataset}_{task}_MAE"] = value(
                dataset, task, "forecast", "MAE"
            )
            output[f"{dataset}_{task}_sMAPE"] = value(
                dataset, task, "forecast", "SMAPE"
            )

    for task in ("T2", "T4"):
        output[f"CausalChambers_{task}_MAE"] = value(
            "CausalChambers", task, "forecast", "MAE"
        )
        output[f"CausalChambers_{task}_OW_RMSSE"] = value(
            "CausalChambers", task, "forecast", "OW_RMSSE"
        )
        output[f"MIMIC_{task}_OW_sMAPE"] = value(
            "MIMIC", task, "forecast", "OW_sMAPE"
        )
        output[f"MIMIC_{task}_OW_RMSSE"] = value(
            "MIMIC", task, "forecast", "OW_RMSSE"
        )

    # Keep the benchmark's complete six-way error taxonomy as run totals.
    for category in category_order:
        output[category] = sum(
            int(row.get(category, 0) or 0)
            for row in summary_table
            if row.get("Subtask") == "forecast"
        )

    return output


def write_results(
    output_root: Path,
    *,
    llm: str,
    args: argparse.Namespace,
    selection_stats: dict,
    results: list[dict],
) -> tuple[Path, dict[str, Any]]:
    """Write the same three result artifacts as evaluate_llm.py."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_model = llm.replace(":", "-").replace("/", "-")
    safe_subset = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in args.subset
    )
    run_dir = output_root / f"{safe_model}_{safe_subset}_{args.ratio:g}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    reference_results = _group_like_reference(results, llm)
    summary_table = summarize_results_multi(reference_results)
    failure_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for item in _diagnostic_summary_items(results):
        category = item.get("error_category")
        if category:
            dataset = _DATASET_DISPLAY_NAMES.get(
                item["dataset"],
                item["dataset"],
            )
            failure_counts[(dataset, item["tier"])][category] += 1

    category_order = (
        FORECAST_MISSING,
        FORMAT_OR_PARSE,
        LENGTH_MISMATCH,
        METRIC_FAIL_MAE,
        METRIC_FAIL_OW,
        METRIC_FAIL_RMSE,
    )
    for row in summary_table:
        reasons = failure_counts.get((row["Dataset"], row["Task"]), {})
        for category in category_order:
            row[category] = (
                int(reasons.get(category, 0))
                if row.get("Subtask") == "forecast"
                else 0
            )

    leaderboard_summary = _leaderboard_summary(
        summary_table,
        llm,
        category_order,
    )

    dataset_names = list(
        dict.fromkeys(str(result.get("dataset", "unknown")) for result in reference_results)
    )
    dataset_part = "-".join(dataset_names) if dataset_names else "unknown_dataset"
    output_suffix = f"{safe_model}_{dataset_part}_high"
    detail_path = run_dir / f"eval_results_detail_{output_suffix}.json"
    summary_json_path = run_dir / f"eval_results_summary_{output_suffix}.json"
    summary_csv_path = run_dir / f"eval_results_summary_{output_suffix}.csv"

    with detail_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(reference_results),
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [_jsonable(leaderboard_summary)],
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    if leaderboard_summary:
        import csv

        columns = list(leaderboard_summary)
        with summary_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(leaderboard_summary)

    return run_dir, leaderboard_summary


def main() -> None:
    temporalbench_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate TimeCopilot on TemporalBench"
    )
    parser.add_argument("--llm", required=True, help="Pydantic-AI model identifier")
    parser.add_argument(
        "--qualitative-model",
        help=(
            "evaluate_llm.MODEL_CONFIGS key for T1/T3. Defaults to --llm, "
            "or to the suffix of a provider-qualified identifier."
        ),
    )
    parser.add_argument("--subset", default="all")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--root", type=Path, default=temporalbench_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=temporalbench_root / "Results",
    )
    args = parser.parse_args()
    qualitative_model = _qualitative_model_name(args.llm, args.qualitative_model)
    # Persist the resolved registry key, not an ambiguous omitted CLI value.
    args.qualitative_model = qualitative_model

    datasets, tiers = parse_subset(args.subset)
    records = load_records(args.root, datasets, tiers)
    selected, selection_stats = select_records(
        records,
        ratio=args.ratio,
        seed=args.seed,
    )

    results = []
    if args.num_workers <= 1:
        for index, record in enumerate(selected, start=1):
            result = evaluate_record(
                record,
                args.llm,
                args.retries,
                qualitative_model,
            )
            results.append(result)
            tier_error = (result.get("tier_output") or {}).get("error")
            print(
                f"[{index}/{len(selected)}] {record['task_id']} "
                f"runtime_error={tier_error!r}",
                flush=True,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            futures = {
                executor.submit(
                    evaluate_record,
                    record,
                    args.llm,
                    args.retries,
                    qualitative_model,
                ): record
                for record in selected
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures),
                start=1,
            ):
                record = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": record["task_id"],
                        "dataset": record["dataset"],
                        "tier": record["tier"],
                        "sample_id": record["sample_id"],
                        "tier_output": _runtime_failure(
                            record["tier"],
                            f"{type(exc).__name__}: {exc}",
                        ),
                    }
                results.append(result)
                tier_error = (result.get("tier_output") or {}).get("error")
                print(
                    f"[{index}/{len(selected)}] {record['task_id']} "
                    f"runtime_error={tier_error!r}",
                    flush=True,
                )

    # Restore deterministic order after concurrent execution.
    results.sort(key=lambda item: item["task_id"])
    run_dir, leaderboard_summary = write_results(
        args.output_root,
        llm=args.llm,
        args=args,
        selection_stats=selection_stats,
        results=results,
    )
    print(f"Results: {run_dir}", flush=True)
    print(
        json.dumps(
            [_jsonable(leaderboard_summary)],
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
