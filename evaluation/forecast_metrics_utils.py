#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for computing ALL forecast-task metrics used by the benchmark.

This module is extracted from the logic inside `evaluate_llm.py` so it can be reused
independently. It focuses purely on metric computation and input parsing, and does
not depend on any model or evaluation loop.

Supported metric families:
1) Standard single-series metrics:
   - MAPE, MAE, RMSE, SMAPE
2) MIMIC multi-series metrics (when dataset == "mimic" and multiple channels exist):
   - OW_sMAPE, OW_RMSSE, OW_MASE (weighted across series)

The public entry point is `compute_forecast_metrics(...)`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# =========================
# ===== Default Config =====
# =========================

# For multi-channel series, MAE/RMSE are normalized per channel before averaging.
# Available options: "mean_abs", "std", "range", "none"
DEFAULT_METRIC_NORM_MODE = "mean_abs"

# Numerical safety epsilon for percentage metrics.
DEFAULT_METRIC_NORM_EPS = 1e-8

# MIMIC-only metric configuration.
DEFAULT_OW_WEIGHT_MODE = "length"  # "length" | "scale" | "macro"
DEFAULT_OW_SMAPE_EPS = 1e-8
DEFAULT_MIMIC_SEASONAL_PERIOD = 1

# Thresholds can be used to flag abnormal metric values.
DEFAULT_FORECAST_METRIC_THRESHOLDS = {
    "MAPE": 1e4,       # percentage, i.e. 10000 means 10000%
    "MAE": 5.0,
    "RMSE": 5.0,
    "SMAPE": 200.0,
    "OW_sMAPE": 200.0,
    "OW_RMSSE": 10.0,
    "OW_MASE": 10.0,
}

OW_WEIGHT_ALLOWED = {"length", "scale", "macro"}


# =========================
# ===== Basic Helpers =====
# =========================

def _value_to_float(val: Any) -> Optional[float]:
    """Convert a value to float; return None if conversion fails or is not finite."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        try:
            f = float(str(val))
        except Exception:
            return None
    if not np.isfinite(f):
        return None
    return f


def _to_float_list(x: Any) -> List[float]:
    """
    Convert arbitrary input into a list of floats.
    - Supports dict/list/ndarray/JSON-string.
    - Non-numeric entries become np.nan.
    """
    import json as _json

    if x is None:
        return []

    # If a JSON string, attempt to parse it first.
    if isinstance(x, str):
        try:
            j = _json.loads(x)
            return _to_float_list(j)
        except Exception:
            return []

    # If dict, try common forecast fields or the first list-like value.
    if isinstance(x, dict):
        candidates = [
            "future_sales", "future", "y", "target", "values",
            "data", "ground_truth", "gt"
        ]
        for k in candidates:
            if k in x:
                out = _to_float_list(x[k])
                if out:
                    return out
        for v in x.values():
            out = _to_float_list(v)
            if out:
                return out
        return []

    # If list/tuple/ndarray, coerce each value to float.
    if isinstance(x, (list, tuple, np.ndarray)):
        out: List[float] = []
        for v in x:
            try:
                out.append(float(v) if v is not None else np.nan)
            except Exception:
                try:
                    out.append(float(str(v)))
                except Exception:
                    out.append(np.nan)
        return out

    return []


def _finite_pairwise(gt_arr: Iterable[Any], pred_arr: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two sequences and filter out non-finite values.
    Returns two numpy arrays of the same length (possibly empty).
    """
    gt = np.asarray(gt_arr, dtype=float)
    pr = np.asarray(pred_arr, dtype=float)
    n = min(len(gt), len(pr))
    if n == 0:
        return np.array([]), np.array([])
    gt = gt[:n]
    pr = pr[:n]
    mask = np.isfinite(gt) & np.isfinite(pr)
    return gt[mask], pr[mask]


def _align_and_filter(gt: Iterable[Any], fc: Iterable[Any]) -> Tuple[List[float], List[float]]:
    """Align by length and drop non-finite pairs, returning python lists."""
    gt_arr = np.asarray(gt, dtype=float)
    fc_arr = np.asarray(fc, dtype=float)
    n = min(len(gt_arr), len(fc_arr))
    if n == 0:
        return [], []
    gt_arr, fc_arr = gt_arr[:n], fc_arr[:n]
    mask = np.isfinite(gt_arr) & np.isfinite(fc_arr)
    return gt_arr[mask].tolist(), fc_arr[mask].tolist()


# =========================
# ===== Standard Metrics ===
# =========================

def _channel_scale(gt: np.ndarray, mode: str, eps: float) -> float:
    """
    Compute a per-channel scale to normalize MAE/RMSE.
    - "mean_abs": mean(|gt|)
    - "std": std(gt)
    - "range": max(gt) - min(gt)
    - "none": 1.0
    """
    gt = np.asarray(gt, dtype=float)
    gt_f = gt[np.isfinite(gt)]
    if gt_f.size == 0:
        return 1.0
    if mode == "mean_abs":
        return float(np.mean(np.abs(gt_f)))
    if mode == "std":
        return float(np.std(gt_f))
    if mode == "range":
        return float(np.max(gt_f) - np.min(gt_f))
    return 1.0


def _metrics_per_channel(gt: np.ndarray, pr: np.ndarray, eps: float) -> Dict[str, Optional[float]]:
    """
    Compute raw (non-normalized) metrics for a single channel.
    Returns None values if no valid pairs exist.
    """
    gt, pr = _finite_pairwise(gt, pr)
    if gt.size == 0:
        return {"MAPE": None, "MAE": None, "RMSE": None, "SMAPE": None}

    mae_v = float(np.mean(np.abs(gt - pr)))
    rmse_v = float(np.sqrt(np.mean((gt - pr) ** 2)))
    denom_mape = np.maximum(np.abs(gt), eps)
    mape_v = float(np.mean(np.abs((gt - pr) / denom_mape)))
    denom_smape = np.maximum((np.abs(gt) + np.abs(pr)) / 2.0, eps)
    smape_v = float(np.mean(np.abs(gt - pr) / denom_smape))
    return {"MAPE": mape_v, "MAE": mae_v, "RMSE": rmse_v, "SMAPE": smape_v}


def compute_normed_metrics_aggregate(
    gt_arr: Any,
    pr_arr: Any,
    norm_mode: str = DEFAULT_METRIC_NORM_MODE,
    eps: float = DEFAULT_METRIC_NORM_EPS,
) -> Dict[str, Any]:
    """
    Compute aggregate metrics after per-channel normalization.

    - 1D: return standard metrics (MAE/RMSE are not normalized).
    - 2D: compute per-channel metrics; MAE/RMSE are normalized by a channel scale,
          then averaged across channels; MAPE/SMAPE are averaged directly.

    Return structure:
      {
        "MAPE": float | None,
        "MAE": float | None,   # normalized for multi-channel
        "RMSE": float | None,  # normalized for multi-channel
        "SMAPE": float | None,
        "per_channel": {i: {...}},
        "n_channels": K
      }
    """
    gt = np.asarray(gt_arr, dtype=float)
    pr = np.asarray(pr_arr, dtype=float)
    if gt.shape != pr.shape:
        raise ValueError(f"Forecast/GT shape mismatch: {pr.shape} vs {gt.shape}")

    # Single-channel case.
    if gt.ndim == 1:
        ch = _metrics_per_channel(gt, pr, eps)
        return {
            "MAPE": ch["MAPE"],
            "MAE": ch["MAE"],
            "RMSE": ch["RMSE"],
            "SMAPE": ch["SMAPE"],
            "per_channel": {0: {**ch, "scale": None, "MAE_norm": ch["MAE"], "RMSE_norm": ch["RMSE"]}},
            "n_channels": 1,
        }

    # Multi-channel case.
    K = gt.shape[0]
    per: Dict[int, Dict[str, Any]] = {}
    mape_list: List[float] = []
    smape_list: List[float] = []
    mae_norm_list: List[float] = []
    rmse_norm_list: List[float] = []

    for i in range(K):
        ch = _metrics_per_channel(gt[i], pr[i], eps)
        scale = _channel_scale(gt[i], norm_mode, eps)
        scale = scale if (scale and np.isfinite(scale) and scale > eps) else 1.0

        entry = {**ch, "scale": scale}

        if ch["MAPE"] is not None:
            mape_list.append(ch["MAPE"])
        if ch["SMAPE"] is not None:
            smape_list.append(ch["SMAPE"])

        mae_n = (ch["MAE"] / scale) if (ch["MAE"] is not None) else None
        rmse_n = (ch["RMSE"] / scale) if (ch["RMSE"] is not None) else None
        entry["MAE_norm"] = mae_n
        entry["RMSE_norm"] = rmse_n
        if mae_n is not None:
            mae_norm_list.append(mae_n)
        if rmse_n is not None:
            rmse_norm_list.append(rmse_n)

        per[i] = entry

    def _mean_or_none(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    return {
        "MAPE": _mean_or_none(mape_list),
        "MAE": _mean_or_none(mae_norm_list),
        "RMSE": _mean_or_none(rmse_norm_list),
        "SMAPE": _mean_or_none(smape_list),
        "per_channel": per,
        "n_channels": K,
    }


# =========================
# ===== MIMIC Metrics =====
# =========================

def _series_dict_from_obj(obj: Any) -> Optional[OrderedDict]:
    """Parse a dict-of-series into OrderedDict[str, List[float]]."""
    if not isinstance(obj, dict):
        return None
    series = OrderedDict()
    for key, val in obj.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            seq = []
            for v in val:
                fv = _value_to_float(v)
                seq.append(np.nan if fv is None else fv)
        else:
            seq = _to_float_list(val)
        if seq:
            series[key] = list(seq)
    return series if series else None


def _align_series_dict(gt_series: OrderedDict, pred_series: Optional[OrderedDict]) -> Tuple[bool, OrderedDict]:
    """Check keys and lengths are aligned for each series."""
    if not gt_series or not pred_series:
        return False, OrderedDict()
    aligned = OrderedDict()
    for key, gt_seq in gt_series.items():
        pred_seq = pred_series.get(key)
        if not isinstance(pred_seq, (list, tuple)):
            return False, OrderedDict()
        if len(pred_seq) != len(gt_seq):
            return False, OrderedDict()
        aligned[key] = list(pred_seq)
    return True, aligned


def _seasonal_naive_scale(history_seq: Optional[List[Any]], seasonal_period: int) -> float:
    """Seasonal naive scale used by RMSSE."""
    seq = history_seq if isinstance(history_seq, list) else []
    if not seq or seasonal_period <= 0:
        return 1.0
    diffs = []
    for idx in range(seasonal_period, len(seq)):
        cur = _value_to_float(seq[idx])
        prev = _value_to_float(seq[idx - seasonal_period])
        if cur is None or prev is None:
            continue
        diffs.append((cur - prev) ** 2)
    if not diffs:
        return 1.0
    return sum(diffs) / len(diffs)


def _one_step_naive_scale(history_seq: Optional[List[Any]]) -> float:
    """One-step naive scale used by MASE."""
    seq = history_seq if isinstance(history_seq, list) else []
    if len(seq) < 2:
        return 1.0
    diffs = []
    for idx in range(1, len(seq)):
        cur = _value_to_float(seq[idx])
        prev = _value_to_float(seq[idx - 1])
        if cur is None or prev is None:
            continue
        diffs.append(abs(cur - prev))
    if not diffs:
        return 1.0
    return sum(diffs) / len(diffs)


def _compute_ow_metrics(
    gt_series: OrderedDict,
    pred_series: OrderedDict,
    history_series: Optional[OrderedDict],
    weight_mode: str,
    seasonal_period: int,
    smape_eps: float,
) -> Tuple[Optional[Dict[str, Any]], OrderedDict]:
    """
    Compute MIMIC metrics across multiple series using weighted aggregation.
    Returns (summary_metrics, per_series_details).
    """
    if not gt_series or not pred_series:
        return None, OrderedDict()
    mode = (weight_mode or "length").lower()
    if mode not in OW_WEIGHT_ALLOWED:
        mode = "length"
    if seasonal_period <= 0:
        seasonal_period = 1

    per_series: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    weight_numerators: Dict[str, float] = {}
    valid_keys: List[str] = []
    smape_ratio_map: Dict[str, float] = {}
    history_series = history_series or {}

    for name, gt in gt_series.items():
        pred = pred_series.get(name)
        if pred is None:
            continue
        pairs = []
        for true_val, pred_val in zip(gt, pred):
            yt = _value_to_float(true_val)
            yp = _value_to_float(pred_val)
            if yt is None or yp is None:
                continue
            pairs.append((yt, yp))
        if not pairs:
            per_series[name] = {"sMAPE": float("nan"), "RMSSE": float("nan"), "MASE": float("nan"), "pairs": 0}
            continue

        diffs = [abs(a - b) for (a, b) in pairs]
        smape_vals = [2.0 * d / (abs(a) + abs(b) + smape_eps) for (a, b), d in zip(pairs, diffs)]
        smape_ratio = float(np.mean(smape_vals))
        rmse = math.sqrt(sum((a - b) ** 2 for (a, b) in pairs) / len(pairs))
        mae = float(np.mean(diffs))

        history_seq = history_series.get(name)
        scale_rmsse = _seasonal_naive_scale(history_seq, seasonal_period)
        scale_mase = _one_step_naive_scale(history_seq)
        rmsse = rmse / (scale_rmsse + smape_eps)
        mase = mae / (scale_mase + smape_eps)

        per_series[name] = {"sMAPE": smape_ratio * 100.0, "RMSSE": rmsse, "MASE": mase, "pairs": len(pairs)}
        valid_keys.append(name)
        smape_ratio_map[name] = smape_ratio

        if mode == "length":
            weight_num = len(pairs)
        elif mode == "scale":
            weight_num = sum(abs(a) for (a, _) in pairs)
        else:
            weight_num = 1.0
        weight_numerators[name] = weight_num

    if not valid_keys:
        return None, per_series

    if mode == "macro":
        weights = {k: 1.0 / len(valid_keys) for k in valid_keys}
    else:
        total = sum(weight_numerators.get(k, 0.0) for k in valid_keys)
        if total <= 0:
            weights = {k: 1.0 / len(valid_keys) for k in valid_keys}
        else:
            weights = {k: weight_numerators.get(k, 0.0) / total for k in valid_keys}

    for k in valid_keys:
        per_series[k]["weight"] = weights[k]

    def _weighted_avg(getter):
        acc = 0.0
        wsum = 0.0
        for k in valid_keys:
            val = getter(k)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                continue
            acc += weights[k] * val
            wsum += weights[k]
        return acc / wsum if wsum > 0 else float("nan")

    ow_smape = _weighted_avg(lambda k: smape_ratio_map.get(k))
    ow_rmsse = _weighted_avg(lambda k: per_series[k]["RMSSE"])
    ow_mase = _weighted_avg(lambda k: per_series[k]["MASE"])

    summary = {
        "OW_sMAPE": ow_smape * 100.0 if isinstance(ow_smape, (int, float)) else float("nan"),
        "OW_RMSSE": ow_rmsse,
        "OW_MASE": ow_mase,
        "weight_mode": mode,
    }
    return summary, per_series


# =========================
# ===== Public API =========
# =========================

def check_metric_thresholds(
    metrics: Optional[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Check whether any metric exceeds predefined thresholds.
    Returns a string flag (e.g., "metric_threshold_MAPE") or None.
    """
    if not metrics:
        return None
    thresholds = thresholds or DEFAULT_FORECAST_METRIC_THRESHOLDS
    for key, limit in thresholds.items():
        if limit is None:
            continue
        val = metrics.get(key)
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        # If MAPE is large but SMAPE is still within limit, ignore the MAPE warning.
        if key == "MAPE":
            smape_key = "SMAPE" if "SMAPE" in metrics else ("OW_sMAPE" if "OW_sMAPE" in metrics else None)
            if smape_key:
                smape_val = metrics.get(smape_key)
                smape_limit = thresholds.get(smape_key, float("inf"))
                if isinstance(smape_val, (int, float)) and math.isfinite(smape_val) and abs(smape_val) <= smape_limit:
                    continue
        if abs(val) > limit:
            return f"metric_threshold_{key}"
    return None


def compute_forecast_metrics(
    gt_obj: Any,
    forecast_obj: Any,
    dataset_name: Optional[str] = None,
    history_series: Optional[OrderedDict] = None,
    metric_norm_mode: str = DEFAULT_METRIC_NORM_MODE,
    metric_norm_eps: float = DEFAULT_METRIC_NORM_EPS,
    ow_weight_mode: str = DEFAULT_OW_WEIGHT_MODE,
    ow_smape_eps: float = DEFAULT_OW_SMAPE_EPS,
    mimic_seasonal_period: int = DEFAULT_MIMIC_SEASONAL_PERIOD,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Compute forecast metrics for a single task.

    Behavior:
    - For MIMIC multi-series (dataset_name == "mimic" and gt has >1 channels),
      compute OW_sMAPE / OW_RMSSE / OW_MASE.
    - Otherwise compute standard MAPE/MAE/RMSE/SMAPE.

    Returns:
      (metrics_dict | None, error_flag | None)
    """
    dataset_lower = (dataset_name or "").lower()
    gt_series = _series_dict_from_obj(gt_obj)

    # MIMIC special handling: multi-series with weighted metrics.
    if dataset_lower == "mimic" and gt_series and len(gt_series) > 1:
        pred_series = _series_dict_from_obj(forecast_obj if isinstance(forecast_obj, dict) else {})
        if not pred_series:
            return None, "missing_forecast"
        for key, gt_seq in gt_series.items():
            pred_seq = pred_series.get(key)
            if pred_seq is None:
                return None, "missing_channel"
            if not isinstance(pred_seq, (list, tuple)):
                return None, "invalid_format"
            if len(pred_seq) != len(gt_seq):
                return None, "length_mismatch"
        ok, aligned_pred = _align_series_dict(gt_series, pred_series)
        if not ok:
            return None, "series_mismatch"
        history = history_series or OrderedDict()
        ow_metrics, _ = _compute_ow_metrics(
            gt_series, aligned_pred, history,
            ow_weight_mode, mimic_seasonal_period, ow_smape_eps
        )
        metric_flag = check_metric_thresholds(ow_metrics, thresholds)
        return ow_metrics, metric_flag

    # Standard single-series forecast metrics.
    gt_list = _to_float_list(gt_obj)
    if not gt_list:
        return None, "missing_ground_truth"
    forecast_list = _to_float_list(forecast_obj)
    if not forecast_list:
        return None, "missing_forecast"
    if len(forecast_list) != len(gt_list):
        return None, "length_mismatch"

    gt_clip, fc_clip = _align_and_filter(gt_list, forecast_list)
    if len(gt_clip) == 0:
        return None, "no_valid_values"

    metrics = compute_normed_metrics_aggregate(
        gt_clip, fc_clip, norm_mode=metric_norm_mode, eps=metric_norm_eps
    )
    metric_flag = check_metric_thresholds(metrics, thresholds)
    return metrics, metric_flag


__all__ = [
    "compute_forecast_metrics",
    "compute_normed_metrics_aggregate",
    "check_metric_thresholds",
]

