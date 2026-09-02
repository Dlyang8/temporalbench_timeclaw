#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean benchmark evaluation entrypoint (T1-T4) for sharing with others.

Usage:
  1. Configure datasets / tiers / models in this file.
  2. Export the API keys required by the selected model(s), for example:
       export OPENAI_API_KEY=...
       export ANTHROPIC_API_KEY=...
       export GEMINI_API_KEY=...
  3. Run:
       python evaluate_llm_clean.py

Notes:
  - No API keys are hardcoded in this version.
  - This script keeps the original evaluation logic and output format.
"""

import os
import time
import json
import math
from typing import Any, Dict, List, Optional
import requests
import glob
import random
import numpy as np
from collections import OrderedDict, defaultdict


def _read_env(name: str) -> str:
    return os.environ.get(name, "").strip()

# =========================
# ======== CONFIG =========
# =========================

# 你的数据路径（可为目录/单文件/数组JSON/NDJSON/拼接）
# 可用数据集及其路径（文件/目录/或通配符都行）
# 默认指向本仓库 TemporalBench/data/<dataset>/task_modified.json
# （相对脚本位置解析，无论从哪个目录启动都能找到）。
_BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
DATASET_PATHS = {
    "FreshRetailNet": os.path.join(_BASE, "freshretailnet",  "task_modified.json"),
    "PSML":           os.path.join(_BASE, "PSML",            "task_modified.json"),
    "MIMIC":          os.path.join(_BASE, "MIMIC",           "task_modified.json"),
    "CausalChambers": os.path.join(_BASE, "causal_chambers", "task_modified.json"),
    "M5":             os.path.join(_BASE, "M5",              "task_modified.json"),
}

# 选择要评测哪些数据集（名字要与 DATASET_PATHS 的键一致）
ENABLED_DATASETS = [
    "FreshRetailNet",
    # "PSML",
    # "MIMIC",
    # "CausalChambers",
    # "M5",
]


# 选择要评测的 Tier
ENABLED_TIERS = [
    "T1",
    "T2",
    "T3",
    "T4"
]  

# 选择要评测的模型（从下面 MODEL_CONFIGS 的键中挑）
ENABLED_MODELS = [
    # "gpt-5.4-mini",
    "gemini-3.1-flash-lite",
    # "gpt-4o",
    # "claude",
    # "gemini-2.5-flash-lite",
    # "codestral-latest",
    # "deepseek-chat",
    # "qwen-plus",
    # "grok-4-fast-reasoning",
]


OPENAI_API_KEY = _read_env("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _read_env("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = _read_env("DEEPSEEK_API_KEY")
XAI_OAI_KEY = _read_env("XAI_API_KEY")
QWEN_OAI_KEY = _read_env("QWEN_API_KEY")
MISTRAL_OAI_KEY = _read_env("MISTRAL_API_KEY")
GEMINI_OAI_KEY = _read_env("GEMINI_API_KEY")
# 如需 openai-compatible 端点：
DEEPSEEK_OAI_BASE_URL="https://api.deepseek.com/v1"
XAI_OAI_BASE_URL="https://api.x.ai/v1"
MISTRAL_OAI_BASE_URL="https://api.mistral.ai/v1"
GEMINI_OAI_BASE_URL="https://generativelanguage.googleapis.com/openai/v1"
QWEN_OAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 是否将 T2/T4 的 forecast 与 mcq 拆分成两次独立调用
SPLIT_T2_T4 = True  # True=分开问；False=维持合并问


# 重试与节流
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds base
REQUEST_TIMEOUT = 90  # seconds
# 最小请求间隔（秒）：用于粗略降低请求频率以避免 TPM/429
MIN_REQUEST_INTERVAL = 0.0  # 例如 0.5 或 1.0

# 数值稳定
EPS = 1e-12

# 采样控制
SAMPLE_NUM = 50       # 一次评测多少条样本；设为 1 即评 1 条
SAMPLE_SHUFFLE = False # 是否对候选样本打乱
RANDOM_SEED = 42       # 打乱时的随机种子

# 多通道指标归一化策略：
# - "mean_abs": 用通道的 mean(|gt|) 做尺度（默认）
# - "std":      用标准差 std(gt) 做尺度
# - "range":    用 (max(gt) - min(gt)) 做尺度
# - "none":     不归一化（不建议在多通道上）
METRIC_NORM_MODE = "mean_abs"
METRIC_NORM_EPS = 1e-8  # 防止除零

# MIMIC 专用 forecast 指标配置
OW_WEIGHT_MODE    = os.environ.get("OW_WEIGHT_MODE", "length").lower()
OW_WEIGHT_ALLOWED = {"length", "scale", "macro"}
OW_SMAPE_EPS      = 1e-8
try:
    MIMIC_SEASONAL_PERIOD = int(os.environ.get("MIMIC_SEASONAL_PERIOD", "1") or "1")
except ValueError:
    MIMIC_SEASONAL_PERIOD = 1
if MIMIC_SEASONAL_PERIOD <= 0:
    MIMIC_SEASONAL_PERIOD = 1

# Forecast 指标阈值，当超过该值时视为异常
FORECAST_METRIC_THRESHOLDS = {
    "MAPE": 1e4,       # 百分比（如 10000 => 10000%）
    "MAE": 5.0,
    "RMSE": 5.0,
    "SMAPE": 200.0,
    "OW_sMAPE": 200.0,
    "OW_RMSSE": 10.0,
    "OW_MASE": 10.0,
}

# 每个 Tier 包含的任务类型
TIER_MODALITY = {
    "T1": {"mcq"},
    "T2": {"forecast", "mcq"},
    "T3": {"mcq"},
    "T4": {"forecast", "mcq"},
}


# =========================
# ===== MODEL REGISTRY ====
# =========================
"""
三类 client：
1) openai_native: OpenAI /v1/chat/completions
2) anthropic_native: Anthropic /v1/messages
3) openai_compat: 任意 OpenAI 兼容 Chat Completions API（需 base_url+key）
"""

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # OpenAI
    "gpt-4o": {
        "provider": "openai_native",
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },

    "gpt-5.4-mini": {
        "provider": "openai_native",
        "model": "gpt-5.4-mini",
        "api_key": OPENAI_API_KEY,
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    # Anthropic
    "claude": {
        "provider": "anthropic_native",
        "model": "claude-3-7-sonnet-latest",
        "api_key": ANTHROPIC_API_KEY,
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
    },
    # 以下默认按 OpenAI-compatible 调用（需提供兼容端点）
    "gemini-3.1-flash-lite": {
        "provider": "gemini_native",
        "model": "gemini-3.1-flash-lite-preview",
        "api_key": GEMINI_OAI_KEY,
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com",  # 保持默认即可
    },
    "gemini-2.5-flash-lite": {
        "provider": "gemini_native",
        "model": "gemini-2.5-flash-lite",
        "api_key": GEMINI_OAI_KEY,
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com",  # 保持默认即可
    },
    "codestral-latest": {
        "provider": "openai_compat",
        "model": "codestral-latest",
        "api_key": MISTRAL_OAI_KEY,
        "api_key_env": "MISTRAL_API_KEY",
        "base_url": MISTRAL_OAI_BASE_URL,
    },
    "deepseek-chat": {
        "provider": "openai_compat",
        "model": "deepseek-chat",
        "api_key": DEEPSEEK_API_KEY,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": DEEPSEEK_OAI_BASE_URL,  # e.g. https://api.deepseek.com/v1
    },
    "qwen-plus": {
        "provider": "openai_compat",
        "model": "qwen-plus",
        "api_key": QWEN_OAI_KEY,
        "api_key_env": "QWEN_API_KEY",
        "base_url": QWEN_OAI_BASE_URL,
    },
    "grok-4-fast-reasoning": {
        "provider": "openai_compat",
        "model": "grok-4-fast-reasoning",
        "api_key": XAI_OAI_KEY,
        "api_key_env": "XAI_API_KEY",
        "base_url": XAI_OAI_BASE_URL,  # e.g. https://api.x.ai/v1
    },
    "deepseek-v4-pro": {
        "provider": "openai_compat",
        "model": "deepseek-v4-pro",
        "api_key": DEEPSEEK_API_KEY,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": DEEPSEEK_OAI_BASE_URL
    }
}


# =========================
# ===== JSON UTILITIES ====
# =========================
def _finite_pairwise(gt_arr, pred_arr):
    """对齐并过滤非有限值，返回两个长度一致且全为有限值的数组。"""
    gt = np.asarray(gt_arr, dtype=float)
    pr = np.asarray(pred_arr, dtype=float)
    n = min(len(gt), len(pr))
    if n == 0:
        return np.array([]), np.array([])
    gt = gt[:n]
    pr = pr[:n]
    mask = np.isfinite(gt) & np.isfinite(pr)
    return gt[mask], pr[mask]

def _channel_scale(gt: np.ndarray, mode: str) -> float:
    """计算单通道尺度，用于归一化 MAE/RMSE。"""
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
    return 1.0  # "none"

def _metrics_per_channel(gt: np.ndarray, pr: np.ndarray) -> dict:
    """计算单通道原始指标（未归一化）。"""
    gt, pr = _finite_pairwise(gt, pr)
    if gt.size == 0:
        return {"MAPE": None, "MAE": None, "RMSE": None, "SMAPE": None}

    mae_v  = float(np.mean(np.abs(gt - pr)))
    rmse_v = float(np.sqrt(np.mean((gt - pr) ** 2)))
    # 避免除零
    denom_mape  = np.maximum(np.abs(gt), METRIC_NORM_EPS)
    mape_v  = float(np.mean(np.abs((gt - pr) / denom_mape)))
    denom_smape = np.maximum((np.abs(gt) + np.abs(pr)) / 2.0, METRIC_NORM_EPS)
    smape_v = float(np.mean(np.abs(gt - pr) / denom_smape))

    return {"MAPE": mape_v, "MAE": mae_v, "RMSE": rmse_v, "SMAPE": smape_v}

def compute_normed_metrics_aggregate(gt_arr, pr_arr, norm_mode: str = None) -> dict:
    """
    归一化后再平均的聚合指标：
      - 若是 1D：直接返回单通道指标（MAE/RMSE 不额外归一化）。
      - 若是 2D：对每个通道计算指标；其中 MAE/RMSE 先按通道归一化后再跨通道平均；
                 MAPE/SMAPE（本就无量纲）直接跨通道平均。
    返回:
      {
        "MAPE": 平均MAPE,
        "MAE":  归一化后的平均MAE,
        "RMSE": 归一化后的平均RMSE,
        "SMAPE":平均SMAPE,
        "per_channel": {
            0: {"MAPE":..,"MAE":..,"RMSE":..,"SMAPE":..,"scale":..,"MAE_norm":..,"RMSE_norm":..},
            ...
        },
        "n_channels": K
      }
    """
    if norm_mode is None:
        norm_mode = METRIC_NORM_MODE

    gt = np.array(gt_arr, dtype=float)
    pr = np.array(pr_arr, dtype=float)

    # 形状对齐检查（外层已做过长度裁剪的话，此处只做形状一致性限定）
    if gt.shape != pr.shape:
        raise ValueError(f"Forecast/GT shape mismatch: {pr.shape} vs {gt.shape}")

    # 单通道
    if gt.ndim == 1:
        ch = _metrics_per_channel(gt, pr)
        return {
            "MAPE": ch["MAPE"],
            "MAE": ch["MAE"],
            "RMSE": ch["RMSE"],
            "SMAPE": ch["SMAPE"],
            "per_channel": {0: {**ch, "scale": None, "MAE_norm": ch["MAE"], "RMSE_norm": ch["RMSE"]}},
            "n_channels": 1
        }

    # 多通道：逐通道计算
    K = gt.shape[0]
    per = {}
    mape_list, smape_list = [], []
    mae_norm_list, rmse_norm_list = [], []

    for i in range(K):
        ch = _metrics_per_channel(gt[i], pr[i])
        scale = _channel_scale(gt[i], norm_mode)
        scale = scale if (scale and np.isfinite(scale) and scale > METRIC_NORM_EPS) else 1.0

        # 记录原始指标
        entry = {**ch, "scale": scale}

        # 无量纲的直接纳入平均
        if ch["MAPE"]  is not None: mape_list.append(ch["MAPE"])
        if ch["SMAPE"] is not None: smape_list.append(ch["SMAPE"])

        # 归一化 MAE/RMSE 再纳入平均
        mae_n  = (ch["MAE"]  / scale) if (ch["MAE"]  is not None) else None
        rmse_n = (ch["RMSE"] / scale) if (ch["RMSE"] is not None) else None
        entry["MAE_norm"]  = mae_n
        entry["RMSE_norm"] = rmse_n
        if mae_n  is not None: mae_norm_list.append(mae_n)
        if rmse_n is not None: rmse_norm_list.append(rmse_n)

        per[i] = entry

    def _mean_or_none(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "MAPE":  _mean_or_none(mape_list),
        "MAE":   _mean_or_none(mae_norm_list),   # 注意：这里是归一化后的均值
        "RMSE":  _mean_or_none(rmse_norm_list),  # 注意：这里是归一化后的均值
        "SMAPE": _mean_or_none(smape_list),
        "per_channel": per,
        "n_channels": K
    }


def _value_to_float(val):
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


def _series_dict_from_obj(obj) -> Optional[OrderedDict]:
    if not isinstance(obj, dict):
        return None
    series = OrderedDict()
    for key, val in obj.items():
        seq = None
        if isinstance(val, (list, tuple, np.ndarray)):
            seq = []
            for v in val:
                fv = _value_to_float(v)
                if fv is None:
                    seq.append(np.nan)
                else:
                    seq.append(fv)
        else:
            seq = _to_float_list(val)
        if seq:
            series[key] = list(seq)
    return series if series else None


def _align_series_dict(gt_series: OrderedDict, pred_series: Optional[OrderedDict]):
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


def _flatten_series_pairs(gt_series: OrderedDict, pred_series: OrderedDict):
    g_vals, p_vals = [], []
    for key, gt_seq in gt_series.items():
        pred_seq = pred_series.get(key)
        if pred_seq is None:
            continue
        for g, p in zip(gt_seq, pred_seq):
            gv = _value_to_float(g)
            pv = _value_to_float(p)
            if gv is None or pv is None:
                continue
            g_vals.append(gv)
            p_vals.append(pv)
    if not g_vals:
        return None, None
    return np.asarray(g_vals, dtype=float), np.asarray(p_vals, dtype=float)


def _seasonal_naive_scale(history_seq, seasonal_period: int):
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


def _one_step_naive_scale(history_seq):
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


def _compute_ow_metrics(gt_series: OrderedDict,
                        pred_series: OrderedDict,
                        history_series: Optional[OrderedDict],
                        weight_mode: str,
                        seasonal_period: int):
    if not gt_series or not pred_series:
        return None, OrderedDict()
    mode = (weight_mode or "length").lower()
    if mode not in OW_WEIGHT_ALLOWED:
        mode = "length"
    if seasonal_period <= 0:
        seasonal_period = 1

    per_series = OrderedDict()
    weight_numerators = {}
    valid_keys = []
    smape_ratio_map = {}
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
        smape_vals = [2.0 * d / (abs(a) + abs(b) + OW_SMAPE_EPS) for (a, b), d in zip(pairs, diffs)]
        smape_ratio = float(np.mean(smape_vals))
        rmse = math.sqrt(sum((a - b) ** 2 for (a, b) in pairs) / len(pairs))
        mae = float(np.mean(diffs))
        history_seq = history_series.get(name)
        scale_rmsse = _seasonal_naive_scale(history_seq, seasonal_period)
        scale_mase = _one_step_naive_scale(history_seq)
        rmsse = rmse / (scale_rmsse + OW_SMAPE_EPS)
        mase = mae / (scale_mase + OW_SMAPE_EPS)
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
    ow_mase  = _weighted_avg(lambda k: per_series[k]["MASE"])

    summary = {
        "OW_sMAPE": ow_smape * 100.0 if isinstance(ow_smape, (int, float)) else float("nan"),
        "OW_RMSSE": ow_rmsse,
        "OW_MASE": ow_mase,
        "weight_mode": mode,
    }
    return summary, per_series


def _forecast_has_values(obj) -> bool:
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return True
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return len(obj) > 0
    return False


def _check_metric_thresholds(metrics: Optional[Dict[str, Any]]):
    if not metrics:
        return None
    for key, limit in FORECAST_METRIC_THRESHOLDS.items():
        if limit is None:
            continue
        val = metrics.get(key)
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        if key == "MAPE":
            smape_key = "SMAPE"
            if smape_key not in metrics and "OW_sMAPE" in metrics:
                smape_key = "OW_sMAPE"
            if smape_key in metrics:
                smape_val = metrics.get(smape_key)
                smape_limit = FORECAST_METRIC_THRESHOLDS.get(smape_key, float("inf"))
                if isinstance(smape_val, (int, float)) and math.isfinite(smape_val) and abs(smape_val) <= smape_limit:
                    continue
        if abs(val) > limit:
            return f"metric_threshold_{key}"
    return None


def _dataset_name_lower(sample: Dict[str, Any]) -> str:
    ds = sample.get("_dataset_name") or sample.get("dataset") or ""
    return ds.lower()


def _get_history_series(task_node: Dict[str, Any]) -> Optional[OrderedDict]:
    if not isinstance(task_node, dict):
        return None
    history = task_node.get("input") or {}
    if isinstance(history, dict):
        return _series_dict_from_obj(history.get("history"))
    return None


def _compute_forecast_metrics(gt_obj,
                              forecast_obj,
                              dataset_lower: str,
                              history_series: Optional[OrderedDict]):
    gt_series = _series_dict_from_obj(gt_obj)
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
        history = history_series or {}
        ow_metrics, _ = _compute_ow_metrics(gt_series, aligned_pred, history, OW_WEIGHT_MODE, MIMIC_SEASONAL_PERIOD)
        metric_flag = _check_metric_thresholds(ow_metrics)
        return ow_metrics, metric_flag

    gt_list = _to_float_list(gt_obj)
    if not gt_list:
        return None, "missing_ground_truth"
    forecast_list = _to_float_list(forecast_obj)
    if not forecast_list:
        return None, "missing_forecast"
    
    if len(forecast_list) != len(gt_list):
        # print("len of forecast_seq:", len(forecast_list), "len of gt_seq:", len(gt_list))
        # print("forecast_list:", forecast_list)
        # print("gt_list:", gt_list)
        # exit()
        return None, "length_mismatch"
    gt_clip, fc_clip = _align_and_filter(gt_list, forecast_list)
    if len(gt_clip) == 0:
        return None, "no_valid_values"
    metrics = compute_normed_metrics_aggregate(gt_clip, fc_clip)
    metric_flag = _check_metric_thresholds(metrics)
    return metrics, metric_flag


def extract_first_json(text: str):
    """
    从任意文本中提取第一段有效 JSON 对象（以 '{' 开头的对象）。
    顺序：整体 json.loads -> 从每个 '{' 用 raw_decode -> 字符串感知的栈截取 -> 失败返回 None
    """
    import json as _json
    if not text:
        return None

    s = text.strip()

    # 1) 整体尝试
    try:
        return _json.loads(s)
    except Exception:
        pass

    # 2) 从每个 '{' 起点尝试 raw_decode
    dec = _json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, end = dec.raw_decode(s[i:])
                return obj
            except Exception:
                continue

    # 3) 栈式扫描（识别字符串与转义），截取第一段 {...} 再 loads
    n = len(s)
    i = 0
    while i < n and s[i] != "{":
        i += 1
    if i >= n:
        return None

    start = i
    depth = 0
    in_str = False
    str_char = ""
    escaped = False

    i = start
    while i < n:
        c = s[i]
        if in_str:
            if escaped:
                escaped = False
            else:
                if c == "\\":
                    escaped = True
                elif c == str_char:
                    in_str = False
        else:
            if c == '"' or c == "'":  # 对 JSON 非严格容错
                in_str = True
                str_char = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[start:i+1]
                    try:
                        return _json.loads(chunk)
                    except Exception:
                        # 继续寻找下一段
                        j = i + 1
                        while j < n and s[j] != "{":
                            j += 1
                        if j < n:
                            start = j
                            i = j - 1
                            depth = 0
                            in_str = False
                            str_char = ""
                            escaped = False
                        else:
                            break
        i += 1

    return None


def extract_all_json_objects(text: str):
    """
    从任意文本中依次抽取多个 JSON 对象。优先尝试 json.load；失败则逐段 raw_decode；
    再失败用栈扫描。返回 list[dict]（只收集对象；数组 JSON 会拆成多个对象）。
    """
    import json as _json
    if text is None:
        return []

    # 1) 直接整体 load
    try:
        obj = _json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            return [obj]
        else:
            return []
    except Exception:
        pass

    # 2) 尝试逐个 raw_decode（从每个 '{' 开始）
    res = []
    dec = _json.JSONDecoder()
    s = text
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s[i:])
            if isinstance(obj, dict):
                res.append(obj)
            i += end
        except Exception:
            i += 1
    if res:
        return res

    # 3) 栈式扫描（字符串感知）——提取 {...} 块再 loads
    res = []
    i = 0
    while i < n and s[i] != "{":
        i += 1
    while i < n:
        # 找下一段
        while i < n and s[i] != "{":
            i += 1
        if i >= n:
            break
        start = i
        depth, in_str, str_ch, esc = 0, False, "", False
        while i < n:
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                else:
                    if c == "\\":
                        esc = True
                    elif c == str_ch:
                        in_str = False
            else:
                if c in ('"', "'"):
                    in_str = True
                    str_ch = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = s[start:i+1]
                        try:
                            obj = _json.loads(chunk)
                            if isinstance(obj, dict):
                                res.append(obj)
                        except Exception:
                            pass
                        i += 1
                        break
            i += 1
        # 继续下一段
    return res

def _load_one_file_collect(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        txt = f.read()
    return extract_all_json_objects(txt)

def load_samples_for_dataset(path: str, sample_num: int, shuffle: bool=False, seed: int=42):
    """
    从 path 读取样本：
      - 若 path 是文件：可包含 单对象JSON / 数组JSON / NDJSON / 拼接多对象
      - 若 path 是目录：读取其中所有 *.json
      - 若 path 含通配符：glob 匹配后逐个文件读取
    采样规则：
      - 若 sample_num >= 总数 => 返回全量（若 shuffle=True 则打乱但仍全量）
      - 否则返回前 sample_num（可选打乱）
    """
    import fnmatch
    rng = random.Random(seed)

    files = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        if any(ch in path for ch in ["*", "?", "["]):
            files = sorted(glob.glob(path))
        else:
            files = [path] if os.path.exists(path) else []

    all_samples = []
    for fp in files:
        try:
            all_samples.extend(_load_one_file_collect(fp))
        except Exception:
            continue

    total = len(all_samples)
    if total == 0:
        return []

    if sample_num >= total:
        if shuffle: rng.shuffle(all_samples)
        return all_samples
    if shuffle:
        rng.shuffle(all_samples)
    return all_samples[:sample_num]

def load_all_datasets(dataset_paths: Dict[str, str], enabled: List[str], sample_num: int, shuffle: bool, seed: int):
    """
    返回 [(dataset_name, sample_dict), ...]
    """
    pairs = []
    for ds in enabled:
        path = dataset_paths.get(ds)
        if not path: 
            continue
        samples = load_samples_for_dataset(path, sample_num, shuffle, seed)
        for s in samples:
            # 标记所属数据集
            if isinstance(s, dict):
                s.setdefault("_dataset_name", ds)
            pairs.append((ds, s))
    return pairs


def sample_id_of(s: dict) -> str:
    # 优先 meta.source_id / Meta JSON 的 source_id；其次 tasks.*.id；最后 hash
    for k in ("meta", "Meta JSON", "Meta", "metadata"):
        if isinstance(s.get(k), dict):
            sid = s[k].get("source_id") or s[k].get("id")
            if sid:
                return str(sid)
    t = s.get("tasks") or {}
    for tk in ("T1","T2","T3","T4"):
        if isinstance(t.get(tk), dict):
            sid = t[tk].get("id")
            if sid:
                return str(sid)
    return f"sample_{abs(hash(json.dumps(s, sort_keys=True)))}"


def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default

def _strip_t3_questions_blocks(text: str) -> str:
    """
    强力清理 T3 原始 prompt 中的多题与 answers[] 格式示例：
      1) 删除整段 'Questions:' 列表（直到空行或标题块）。
      2) 删除任意大小写/空白变体的 'Output format (JSON only):' 且包含 answers[] 的整个示例 JSON 块，
         并顺带删除紧跟着与该块相关的约束说明行（直到遇到空行或新标题块）。
    仅保留共享背景/字段说明/单题 Subtask 与 'Output JSON only' 单题格式。
    """
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    i = 0
    n = len(lines)

    def _low(s: str) -> str:
        return s.strip().lower()

    def _is_heading(s_low: str) -> bool:
        # 常见标题起始：Subtask / Output / Constraints / Background / Field(s)
        return (
            s_low.startswith("subtask")
            or s_low.startswith("output")
            or s_low.startswith("constraints")
            or s_low.startswith("background")
            or s_low.startswith("field")
        )

    while i < n:
        ln = lines[i]
        low = _low(ln)

        # -------- 1) 跳过整段 Questions: --------
        if low.startswith("questions:"):
            i += 1
            while i < n:
                nxt_low = _low(lines[i])
                if nxt_low == "" or _is_heading(nxt_low):
                    break
                i += 1
            continue  # 不写入 out

        # -------- 2) 跳过包含 answers[] 的 'Output format (JSON only):' 示例块 --------
        if low.startswith("output format") and "json only" in low:
            # 先判断这个 Output format 块是否与 answers[] 相关（在此行或后续若干行中出现 'answers'）
            j = i
            seen_answers = ("answers" in low)
            # 向前看最多 20 行，防止极端大块
            probe_limit = min(n, i + 20)
            while not seen_answers and j + 1 < probe_limit:
                j += 1
                if "answers" in _low(lines[j]):
                    seen_answers = True
                    break

            if seen_answers:
                # 跳过整个 JSON 示例：用花括号深度匹配（大小写/空白无关）
                i += 1
                brace_depth = 0
                # 如果下一行不是 '{' 也照样尝试匹配，直到配平
                while i < n:
                    t = lines[i]
                    brace_depth += t.count("{") - t.count("}")
                    # 一直走到示例块的 '}' 收尾
                    if "}" in t and brace_depth <= 0:
                        i += 1
                        break
                    i += 1

                # 再跳过紧随其后的、与 answers 示例相关的约束说明（直到空行或新标题）
                while i < n:
                    nxt = lines[i]
                    nxt_low = _low(nxt)
                    if nxt_low == "" or _is_heading(nxt_low):
                        break
                    # 这些是 answers 示例后面常见的说明行，直接跳过
                    i += 1
                continue  # 整块都跳过，不写入 out
            # 若不是 answers 相关，则保留该行（很少见）
            # 注意：这里不提前 i += 1，因为统一在下面默认分支处理

        # 默认保留
        out.append(ln)
        i += 1

    return "\n".join(out).strip()

def build_t3_single_prompt(t3: Dict[str, Any], it: Dict[str, Any], idx: int) -> str:
    """
    生成“单题版”T3 提示：共享背景（已清洗） + 当前子任务 + 严格的输出 JSON（仅一题）。
    只允许：
        { "answer": "<one_label_from_label_space>" }
    """
    # 共享背景（去掉 Questions/answers[] 等块）
    base_raw = (t3.get("prompt") or t3.get("task") or "").strip()
    base = _strip_t3_questions_blocks(base_raw)

    # 当前子任务信息
    tid = it.get("task_id", str(idx))
    name = it.get("name", "")
    q = it.get("question") or it.get("Question") or ""
    label_space = it.get("label_space") or it.get("options") or it.get("Options") or []
    label_space = ", ".join(map(str, label_space)) if isinstance(label_space, (list, tuple)) else str(label_space)

    sub_lines = ["Subtask:"]
    block = [f"- [{tid}]"]
    if name: block.append(f"  Name: {name}")
    if q:    block.append(f"  Question: {q}")
    if label_space: block.append(f"  label_space: {label_space}")
    sub_lines.append("\n".join(block))

    # 单题输出要求
    io_lines = [
        "Output JSON only:",
        '{ "answer": "<one_label_from_label_space>" }',
        "No extra keys, no explanations, no markdown."
    ]

    parts = [p for p in [base, "\n".join(sub_lines), "\n".join(io_lines)] if p.strip()]
    return "\n\n".join(parts).strip()


def parse_t3_single_answer(raw_text: str) -> Optional[str]:
    """
    解析单题回答：
    - 首选 {"answer": "<label>"} 或 {"label": "<label>"} 或 {"prediction": "<label>"}
    - 退化：如果模型直接返回字符串/数组的单元素，也尝试取第一个非空字符串
    返回标准化后的 label（str）或 None
    """
    ans = extract_first_json(raw_text)
    def _norm(x):
        if x is None: return None
        s = str(x).strip()
        return s if s else None

    if isinstance(ans, dict):
        for k in ("answer", "label", "prediction"):
            if k in ans:
                return _norm(ans[k])
        # 容错：有时会放在 {"answers":[...]}
        if isinstance(ans.get("answers"), list) and ans["answers"]:
            return _norm(ans["answers"][0])

    if isinstance(ans, list) and ans:
        return _norm(ans[0])

    if isinstance(ans, str):
        return _norm(ans)

    return None


def build_t3_prompt(t3: Dict[str, Any]) -> str:
    """
    将 T3 的共享 prompt 与每个子任务的关键信息拼接成一个完整提示。
    兼容输出格式：
      优先：{"answers": ["<label_for_Q1>", ...]}  # 与 pack 顺序一致
      备选：{"S4:D": "<label>", "S3:D": "..."}    # 用 task_id 作为键
      备选：{"0":"<label>","1":"<label>", ...}   # 用索引键
    label 必须来自各自 label_space。
    """
    base = t3.get("prompt") or t3.get("task") or ""
    pack = t3.get("pack", [])
    lines = [str(base).rstrip(), "", "Subtasks (answer each with one label from its label_space):"]

    for i, it in enumerate(pack):
        tid = it.get("task_id", str(i))
        name = it.get("name")
        q = it.get("question") or it.get("Question")
        opts = it.get("label_space") or it.get("options") or it.get("Options")
        block = [f"- [{tid}]"]
        if name: block.append(f"  Name: {name}")
        if q:    block.append(f"  Question: {q}")
        if isinstance(opts, (list, tuple)) and len(opts) > 0:
            block.append(f"  label_space: {', '.join(map(str, opts))}")
        lines.append("\n".join(block))

    lines += [
        "",
        "Return ONLY one of the following JSON formats:",
        '1) {"answers": ["<label_for_Q1>", "<label_for_Q2>", ...]}   # order matches the subtasks listed above',
        '2) {"<task_id>": "<label>", ...}                             # keys are task_id (e.g., "S4:D")',
        '3) {"0":"<label>","1":"<label>", ...}                        # keys are zero-based indices',
        "Each label MUST be exactly one string from the corresponding label_space.",
        "No extra text.",
    ]
    return "\n".join(lines).strip()


def _to_float_list(x):
    """把 x 解析成 float 列表；支持 dict/列表/ndarray/JSON字符串；非数值记为 np.nan。"""
    import json as _json
    import numpy as _np

    if x is None:
        return []

    # 字符串：尝试把它当 JSON
    if isinstance(x, str):
        try:
            j = _json.loads(x)
            return _to_float_list(j)
        except Exception:
            # 不是 JSON，就别用了
            return []

    # 字典：优先常见字段，其次尝试第一个 list-like 值
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
        # 没命中就找第一个能变成数字序列的 value
        for v in x.values():
            out = _to_float_list(v)
            if out:
                return out
        return []

    # 列表/元组/ndarray
    if isinstance(x, (list, tuple, np.ndarray)):
        out = []
        for v in x:
            try:
                if v is None:
                    out.append(np.nan)
                else:
                    out.append(float(v))
            except Exception:
                # 常见 'NaN'/'null' 之类
                try:
                    out.append(float(str(v)))
                except Exception:
                    out.append(np.nan)
        return out

    # 其他类型直接放弃
    return []


def _align_and_filter(gt, fc):
    """对齐长度并过滤非有限值；返回 (gt_clip, fc_clip) 两个纯 float 列表。"""
    gt = np.asarray(gt, dtype=float)
    fc = np.asarray(fc, dtype=float)
    n = min(len(gt), len(fc))
    if n == 0:
        return [], []
    gt, fc = gt[:n], fc[:n]
    mask = np.isfinite(gt) & np.isfinite(fc)
    gt = gt[mask]
    fc = fc[mask]
    return gt.tolist(), fc.tolist()



# =========================
# ===== METRICS ===========
# =========================

def _to_np(a):
    return np.asarray(a, dtype=float)

def mae(y_true, y_pred):
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.mean(np.abs(yt - yp)))

def rmse(y_true, y_pred):
    yt, yp = _to_np(y_true), _to_np(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def mape(y_true, y_pred):
    yt, yp = _to_np(y_true), _to_np(y_pred)
    denom = np.maximum(np.abs(yt), EPS)
    return float(np.mean(np.abs((yt - yp) / denom)))

def smape(y_true, y_pred):
    yt, yp = _to_np(y_true), _to_np(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.maximum(denom, EPS)
    return float(np.mean(np.abs(yt - yp) / denom))


# =========================
# ===== MODELS ============
# =========================

class ChatError(Exception):
    pass

# 简易节流：保证相邻请求间隔
_LAST_CALL_TS = 0.0

def _throttle_if_needed():
    global _LAST_CALL_TS
    if MIN_REQUEST_INTERVAL and MIN_REQUEST_INTERVAL > 0:
        now = time.time()
        wait = MIN_REQUEST_INTERVAL - (now - _LAST_CALL_TS)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_TS = time.time()

def chat_gemini_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    """
    Google Gemini 原生 REST (v1beta): POST /v1beta/models/{model}:generateContent?key=API_KEY
    - system 消息合并到 system_instruction
    - user/assistant 分别映射到 user/model
    """
    import requests as _req
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_text = "\n\n".join(sys_parts) if sys_parts else None

    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0},
    }
    if system_text:
        payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_text}]}

    resp = _req.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise ChatError(f"Gemini error {resp.status_code}: {resp.text}")
    data = resp.json()

    # 取第一候选的第一段文本
    candidates = data.get("candidates") or []
    if not candidates:
        raise ChatError(f"Gemini empty response: {data}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = ""
    for p in parts:
        if "text" in p:
            text += p["text"]
    return text or ""


def _retry_loop(func, *args, **kwargs):
    last = None
    for i in range(MAX_RETRIES):
        try:
            _throttle_if_needed()
            return func(*args, **kwargs)
        except Exception as e:
            last = e
            # 如果是 429，尝试读取 "Please try again in Xs" 的建议等待时间
            msg = str(e)
            wait = None
            if "rate_limit" in msg.lower() or "429" in msg:
                import re
                m = re.search(r"try again in ([0-9.]+)s", msg, re.IGNORECASE)
                if m:
                    try:
                        wait = float(m.group(1)) + 0.5
                    except ValueError:
                        wait = None
            if wait is None:
                wait = RETRY_BACKOFF * (2 ** i)
            time.sleep(wait)
    raise ChatError(str(last))

def chat_openai_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise ChatError(f"OpenAI error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def chat_anthropic_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_txt = "\n\n".join(sys_parts) if sys_parts else None
    conv = []
    for m in messages:
        if m["role"] == "system":
            continue
        conv.append({
            "role": m["role"],  # 'user' or 'assistant'
            "content": [{"type": "text", "text": m["content"]}],
        })
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0,
        "messages": conv,
    }
    if system_txt:
        payload["system"] = system_txt
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise ChatError(f"Anthropic error {resp.status_code}: {resp.text}")
    data = resp.json()
    out = ""
    for b in data.get("content", []):
        if b.get("type") == "text":
            out += b.get("text", "")
    return out

def chat_openai_compat(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    if not base_url:
        raise ChatError(f"OpenAI-compatible base_url is empty for model {model}")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        raise ChatError(f"OpenAI-compat error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def chat(model_name: str, messages: List[Dict[str, str]]) -> str:
    conf = MODEL_CONFIGS[model_name]
    provider = conf["provider"]
    # clean 版本优先读配置中已解析的环境变量；保留 api_key_env 仅用于报错提示
    api_key = (conf.get("api_key") or os.environ.get(conf.get("api_key_env", ""), "")).strip()
    if not api_key:
        env_name = conf.get("api_key_env", "API_KEY")
        raise ChatError(f"Missing API key for model {model_name}. Please export {env_name}.")
    base_url = conf.get("base_url", "")


    if provider == "openai_native":
        return _retry_loop(chat_openai_native, conf["model"], api_key, base_url, messages)
    elif provider == "anthropic_native":
        return _retry_loop(chat_anthropic_native, conf["model"], api_key, base_url, messages)
    elif provider == "gemini_native":
        return _retry_loop(chat_gemini_native, conf["model"], api_key, base_url, messages)

    elif provider == "openai_compat":
        return _retry_loop(chat_openai_compat, conf["model"], api_key, base_url, messages)
    else:
        raise ChatError(f"Unknown provider: {provider}")


# =========================
# ===== PROMPTS ===========
# =========================

SYSTEM_TEMPLATE = (
    "You are a rigorous time-series evaluation assistant. "
    "Always return ONLY a valid JSON object with the exact schema requested. "
    "No extra text."
)

T2_T4_RETURN_INSTRUCTION = (
    "Return ONLY a JSON object with keys:\n"
    "{\n"
    '  "forecast": number[],\n'
    '  "mcq": {\n'
    '    "future_vs_history": "Higher|Lower|Similar|Uncertain",\n'
    '    "volatility_change": "increased|decreased|constant|Uncertain",\n'
    '    "seasonality_shift": "fixed|shifting|no|Uncertain"\n'
    "  }\n"
    "}\n"
    "Do not add comments or markdown."
)

T1_RETURN_INSTRUCTION = (
    "Return ONLY a JSON object with keys {trend, volatility, seasonality, outliers}."
)

T3_RETURN_INSTRUCTION = (
    "Return ONLY a JSON object mapping each subtask id (or index) to its answer label.\n"
    'For example: {"S1": "Yes", "S2": "No", ...} or {"0":"A","1":"B",...}.'
)

T2_T4_FORECAST_ONLY = (
    "Return ONLY a JSON object:\n"
    "{\n"
    '  "forecast": number[]\n'
    "}\n"
    "No mcq. No extra text."
)

T2_T4_MCQ_ONLY = (
    "Return ONLY a JSON object with keys:\n"
    "{\n"
    '  "mcq": {\n'
    '    "future_vs_history": "Higher|Lower|Similar|Uncertain",\n'
    '    "volatility_change": "increased|decreased|constant|Uncertain",\n'
    '    "seasonality_shift": "fixed|shifting|no|Uncertain"\n'
    "  }\n"
    "}\n"
    "No forecast. No extra text."
)

# ========= T4 PROMPT BUILDERS (auto) =========

def _minify_series(v, k=8):
    """把数组/列表缩略成前 k 个；非数值/None/NaN 会转成 null，保证 JSON 可用。"""
    import math
    out = []
    if isinstance(v, (list, tuple)):
        for x in v[:k]:
            try:
                fx = float(x) if x is not None else None
                if fx is None or math.isnan(fx) or math.isinf(fx):
                    out.append(None)
                else:
                    out.append(fx)
            except Exception:
                out.append(None)
    return out

def _build_input_preview(t4: dict, max_len_per_field=8) -> str:
    """
    构造一个安全的 Input(JSON) 预览字符串。
    优先使用 t4['input'] / t4['Input']；否则尝试从 t4['history'] / t4['history_preview'] 等推测。
    """
    import json as _json
    # 直接可用的输入块
    for key in ("input", "Input"):
        if isinstance(t4.get(key), (dict, list)):
            try:
                return _json.dumps(t4[key], ensure_ascii=False)
            except Exception:
                pass

    # 兜底：尝试从任务字段里拼一个 mini 预览
    hist = {}
    # 常见历史/协变量字段所在位置
    candidates = [
        ("history", t4.get("history")),
        ("history", t4.get("History")),
        ("history", (t4.get("input") or {}).get("history") if isinstance(t4.get("input"), dict) else None),
    ]
    for name, block in candidates:
        if isinstance(block, dict):
            for k, v in block.items():
                hist[k] = _minify_series(v, max_len_per_field)

    # 如果还是空，给一个最小安全壳
    if not hist:
        hist = {
            "sales_censored": _minify_series([0.0, None, 1.0], max_len_per_field),
        }

    payload = {"history": hist}
    return _json.dumps(payload, ensure_ascii=False)


def _infer_forecast_horizon(task_node: Dict[str, Any],
                            fallback: int = 112,
                            sample_meta: Optional[Dict[str, Any]] = None) -> int:
    """根据 ground_truth / meta 推断预测步数。"""
    def _pick_from_dict(dct: Optional[Dict[str, Any]]):
        if not isinstance(dct, dict):
            return None
        for key in (
            "future_len",
            "future_length",
            "n_horizon",
            "forecast_horizon",
            "horizon_hint",
            "horizon",
        ):
            val = dct.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
        return None

    for candidate in (
        task_node,
        task_node.get("meta") if isinstance(task_node, dict) else None,
        sample_meta,
    ):
        direct = _pick_from_dict(candidate)
        if direct:
            return direct

    gt = task_node.get("ground_truth")
    if isinstance(gt, dict):
        series = _series_dict_from_obj(gt)
        if series:
            first_seq = next(iter(series.values()))
            if isinstance(first_seq, list) and len(first_seq) > 0:
                return len(first_seq)
    else:
        seq = _to_float_list(gt)
        if seq:
            return len(seq)

    return fallback


def _format_forecast_list_example(length: int) -> str:
    length = max(int(length or 0), 1)
    if length == 1:
        return "[v0]"
    if length == 2:
        return "[v0, v1]"
    if length == 3:
        return "[v0, v1, v2]"
    return f"[v0, v1, ..., v{length-1}]"


def build_t4_background(t4: dict, horizon_steps: Optional[int] = None) -> str:
    """
    生成共享背景段：尽量复用样本里的 prompt/task/context/fields 文本；
    若缺失则使用稳健的默认说明。
    """
    # 1) 背景文字
    base = t4.get("prompt") or t4.get("task") or ""
    base = str(base).strip()

    # 2) 尝试发现“Upcoming event/Context”说明（若 base 已含这里就不重复）
    ctx_lines = []
    # 常见存放位置：t4['context'] / t4['event'] / t4['upcoming_event']
    for key in ("context", "event", "upcoming_event"):
        if t4.get(key):
            ctx_lines.append(str(t4[key]).strip())
    context_block = ""
    if ctx_lines:
        context_block = "Upcoming event (context):\n- " + "\n- ".join(ctx_lines)

    # 3) 字段含义
    fields_map = t4.get("fields") or {}
    field_lines = []
    if isinstance(fields_map, dict) and fields_map:
        field_lines = [f'- "{k}": {v}' for k, v in fields_map.items()]
    else:
        # 兜底字段说明
        field_lines = [
            '- "sales_censored": the main target series (history only); NaN means stock-out (demand unobserved).',
            '- "stock_flag": optional binary indicator.',
            '- "discount": price multiplier in [0,1] (smaller = deeper discount).',
            '- "holiday_flag": holiday intensity.',
            '- "precipitation": precipitation level.',
            '- "avg_temperature": average temperature.',
            '- "time_position_in_day": discrete time-of-day slot (0..K-1).',
        ]

    background_parts = []
    # 如果 base 里已写了 Background 段，就直接用；否则给一行默认背景
    if base:
        background_parts.append(str(base))
    else:
        steps = max(int(horizon_steps or 0), 1)
        background_parts.append(
            "Background:\nIn retail-like demand forecasting, the observed series may contain missing values. "
            f"You are given a 480-step hourly history (with aligned auxiliary signals), and aligned future covariates for the next {steps} steps."
        )
    if context_block:
        background_parts.append(context_block)
    background_parts.append("Field meanings:\n" + "\n".join(field_lines))

    return "\n\n".join(background_parts).strip()

def build_t4_prompts(t4: dict,
                     input_json_str: str = None,
                     sample_meta: Optional[Dict[str, Any]] = None) -> dict:
    """
    根据 t4 样本自动构造两个 prompt：
      - forecast_prompt（仅输出 forecast）
      - mcq_prompt（仅输出 mcq）
    可选：传入 input_json_str；不传则自动构造安全的缩略预览。
    返回：
      {"forecast_prompt": "...", "mcq_prompt": "..."}
    """
    horizon_steps = _infer_forecast_horizon(t4, sample_meta=sample_meta, fallback=112)
    horizon_steps = max(int(horizon_steps or 0), 1)
    background = build_t4_background(t4, horizon_steps=horizon_steps)
    input_block = input_json_str if isinstance(input_json_str, str) else _build_input_preview(t4)
    forecast_example = _format_forecast_list_example(horizon_steps)

    # --- Forecast-only ---
    forecast_task = (
        "Task:\n"
        f"- Use the provided history and the aligned future covariates to forecast the next {horizon_steps} steps.\n"
        '- Treat NaN in "sales_censored" as missing.\n'
        "- Capture plausible seasonality/trend and reasonable effects from covariates and the described event.\n"
        "- Keep predictions non-negative and finite."
    )
    forecast_out = (
        "Output format (JSON only):\n"
        "{\n"
        f'  "forecast": {forecast_example}\n'
        "}\n"
        "Constraints:\n"
        f"- Output length must equal {horizon_steps}.\n"
        "- Each value must be finite and non-negative.\n"
        "- Do not include anything outside the JSON object.\n"
        "- Do not use any external knowledge beyond the provided input."
    )
    forecast_prompt = (
        f"{background}\n\n"
        f"{forecast_task}\n\n"
        f"Input (JSON):\n{input_block}\n\n"
        f"{forecast_out}"
    ).strip()

    # --- MCQ-only ---
    mcq_task = (
        "Task:\n"
        "Based on the same historical data and contextual event, analyze the expected qualitative changes between the forecast horizon and history."
    )
    # 题目集合：优先读 t4['mcq'] 的枚举，否则用通用模板
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


def build_t2_prompts(t2: dict,
                     input_json_str: str = None,
                     sample_meta: Optional[Dict[str, Any]] = None) -> dict:
    """
    根据 T2 样本自动构造两个 prompt：
      - forecast_prompt（仅输出 forecast）
      - mcq_prompt（仅输出 mcq）
    复用样本中的 t2['prompt'] / t2['task'] 作为背景说明；没有则给默认背景。
    """
    def _default_t2_background():
        return (
            "Background:\n"
            "Using only the provided history and the aligned future covariates, "
            "forecast the next horizon steps of the main series.\n"
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

    input_block = input_json_str if isinstance(input_json_str, str) else _build_input_preview(t2)

    horizon_steps = _infer_forecast_horizon(t2, sample_meta=sample_meta, fallback=112)
    horizon_steps = max(int(horizon_steps or 0), 1)
    forecast_example = _format_forecast_list_example(horizon_steps)

    # --- Forecast-only ---
    forecast_task = (
        "Task:\n"
        f"- Forecast the next {horizon_steps} steps based on the provided history and aligned future covariates.\n"
        "- Treat NaN as missing.\n"
        "- Keep predictions non-negative and finite."
    )
    forecast_out = (
        "Output format (JSON only):\n"
        "{\n"
        f'  "forecast": {forecast_example}\n'
        "}\n"
        "Constraints:\n"
        f"- Output length must equal {horizon_steps}; values must be finite and non-negative.\n"
        "- Do not include anything outside the JSON object.\n"
        "- Do not use any external knowledge beyond the provided input."
    )
    forecast_prompt_parts = [
        forecast_base,
        forecast_task,
        f"Input (JSON):\n{input_block}",
        forecast_out,
    ]
    forecast_prompt = "\n\n".join([part for part in forecast_prompt_parts if part]).strip()

    # --- MCQ-only ---
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
        "- Answer each MCQ with exactly one option; do not output explanations.\n"
        "- Do not output forecast values or any other text.\n"
        "- Do not use any information beyond the provided input arrays."
    )
    mcq_prompt_parts = [
        mcq_base,
        mcq_task,
        f"Input (JSON):\n{input_block}",
        mcq_questions,
        mcq_out,
    ]
    mcq_prompt = "\n\n".join([part for part in mcq_prompt_parts if part]).strip()
    # print(f"Generated T2 forecast prompt:\n{forecast_prompt}\n")
    # exit()
    return {"forecast_prompt": forecast_prompt, "mcq_prompt": mcq_prompt}


# =========================
# ===== EVALUATION ========
# =========================

def eval_t1(sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    t1 = sample["tasks"]["T1"]
    prompt = t1.get("prompt") or t1.get("task")
    labels = t1["labels"]

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": f"{prompt}\n\n{T1_RETURN_INSTRUCTION}"}
    ]
    raw = chat(model_name, messages)
    ans = extract_first_json(raw)
    if not isinstance(ans, dict):
        return {"ok": False, "error": "invalid_json", "mcq_errors": ["invalid_json"], "raw": raw}

    mcq_eval = _eval_mcq_block(ans, labels)
    _log_mcq_comparison("T1", mcq_eval.get("per_question"))
    result = {
        "ok": True,
        "mcq_acc": mcq_eval.get("acc"),
        "mcq": mcq_eval,
        "raw": ans,
        "mcq_errors": [],
    }
    if mcq_eval.get("error_types"):
        result["mcq_errors"].extend(mcq_eval["error_types"])
    if mcq_eval.get("error_counts"):
        result["mcq_error_counts"] = mcq_eval["error_counts"]
    return result

def _normalize_label(val):
    """将各种形态的标签值规范成小写字符串；支持 dict({'label': ...})、纯字符串、单元素列表等。"""
    if val is None:
        return None
    # 如果是 dict，优先取 'label' / 'Label'
    if isinstance(val, dict):
        for k in ("label", "Label"):
            if k in val:
                return _normalize_label(val[k])
        # 如果 GT 是 {"options":[...]} 但没给 label，就返回 None
        return None
    # 如果是列表/元组，取第一个非空元素
    if isinstance(val, (list, tuple)):
        for x in val:
            s = _normalize_label(x)
            if s:
                return s
        return None
    # 其他：转成字符串，strip + lower
    s = str(val).strip()
    return s.lower() if s else None


def _eval_mcq_block(pred_mcq: Dict[str, Any], gt_mcq: Dict[str, Any]) -> Dict[str, Any]:
    """
    评测 MCQ：支持 gt 为字符串或带 {question, options, label, ...} 的字典。
    比较时对 pred/gt 都做 strip + lower 规范化。
    """
    out = {}
    correct = 0
    total = 0
    error_counts = {"missing_answer": 0, "invalid_option": 0}

    # 有些模型可能把 mcq 放在顶层，这里防御一下
    pred_mcq = pred_mcq or {}

    for k, gt in (gt_mcq or {}).items():
        pred_raw = pred_mcq.get(k, None)

        gt_norm = _normalize_label(gt)
        pred_norm = _normalize_label(pred_raw)
        options = []
        if isinstance(gt, dict):
            options = gt.get("options") or gt.get("label_space") or []
        options_norm = {(_normalize_label(opt) or "") for opt in options if isinstance(opt, (str, dict))}

        category = None
        if pred_raw is None or pred_norm is None:
            category = "missing_answer"
        elif options_norm and pred_norm not in options_norm:
            category = "invalid_option"
            pred_norm = None
        elif gt_norm is not None and pred_norm != gt_norm:
            category = None

        is_ok = (gt_norm is not None and pred_norm is not None and pred_norm == gt_norm)
        if is_ok:
            category = None

        if category and category in error_counts:
            error_counts[category] += 1

        out[k] = {
            "pred": pred_raw,
            "gt": gt,
            "pred_norm": pred_norm,
            "gt_norm": gt_norm,
            "correct": is_ok,
            "error_category": category,
        }
        total += 1
        if is_ok:
            correct += 1

    acc = (correct / total) if total else None
    error_types = [key for key, val in error_counts.items() if val > 0]
    return {
        "per_question": out,
        "acc": acc,
        "error_counts": error_counts,
        "error_types": error_types,
        "total": total,
        "correct": correct,
    }


def _log_mcq_comparison(tier: str, per_question: Optional[Dict[str, Dict[str, Any]]]):
    if not per_question:
        return
    print(f"MCQ per-question comparison ({tier}):")
    for key in per_question:
        detail = per_question[key]
        pred = detail.get("pred")
        gt_disp = detail.get("gt")
        if isinstance(gt_disp, dict):
            gt_disp = gt_disp.get("label") or detail.get("gt_norm")
        elif gt_disp is None:
            gt_disp = detail.get("gt_norm")
        mark = "✓" if detail.get("correct") else "✗"
        print(f"  - {key}: pred={pred} | label={gt_disp}  {mark}")



def eval_t2(sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    T2 评测：支持把 forecast 与 mcq 分开问（SPLIT_T2_T4=True）
    返回字段：
      ok: True/False
      metrics: {MAPE, MAE, RMSE, SMAPE} | None
      mcq: {per_question, acc} | None
      forecast_exec_success: 0/1
      mcq_exec_success: 0/1
      raw: 原始返回（便于排错）
    """
    t2 = sample["tasks"]["T2"]
    prompt_text = t2.get("prompt") or t2.get("task") or ""
    gt_obj = t2.get("ground_truth")
    dataset_lower = _dataset_name_lower(sample)
    history_series = _get_history_series(t2)
    result: Dict[str, Any] = {"ok": True, "forecast_errors": [], "mcq_errors": []}

    if SPLIT_T2_T4:
        # ---------- 使用自动构造的两个独立 Prompt ----------
        prompts = build_t2_prompts(t2, sample_meta=sample.get("meta"))

        # 1) 预测（forecast-only）
        messages_f = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": prompts["forecast_prompt"]},
        ]
        # print("T2 Forecast-only prompt:\n", prompts["forecast_prompt"])
        # exit()
        raw_f = chat(model_name, messages_f)
        # print("T2 Forecast-only raw response:\n", raw_f)
        ans_f = extract_first_json(raw_f) or {}
        # print("T2 Forecast-only extracted JSON:\n", ans_f)
        # exit()
        if not ans_f:
            result["forecast_errors"].append("invalid_json")
        forecast_raw = ans_f.get("forecast")
        if not _forecast_has_values(forecast_raw):
            result["forecast_exec_success"] = 0.0
            result["forecast_errors"].append("missing_forecast")
        else:
            result["forecast_exec_success"] = 1.0
        metrics, metric_err = _compute_forecast_metrics(gt_obj, forecast_raw, dataset_lower, history_series)
        result["metrics"] = metrics
        if metrics is None:
            result["forecast_exec_success"] = 0.0
        if metric_err:
            result["forecast_errors"].append(metric_err)
            result["forecast_exec_success"] = 0.0

        # 2) MCQ（mcq-only）
        messages_m = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": prompts["mcq_prompt"]},
        ]
        raw_m = chat(model_name, messages_m)
        ans_m = extract_first_json(raw_m) or {}
        if not ans_m:
            result["mcq_errors"].append("invalid_json")
        mcq_pred = safe_get(ans_m, "mcq", {})
        if not isinstance(mcq_pred, dict) or not mcq_pred:
            result["mcq_errors"].append("missing_output")
        mcq_gt = t2.get("mcq", {})
        mcq_eval = _eval_mcq_block(mcq_pred, mcq_gt) if mcq_gt else None

        # 执行成功性：返回了 dict 且存在至少一个字段值
        result["mcq_exec_success"] = 1.0 if (isinstance(mcq_pred, dict) and any(v is not None for v in mcq_pred.values())) else 0.0
        result["mcq"] = mcq_eval
        if mcq_eval:
            _log_mcq_comparison("T2", mcq_eval.get("per_question"))
        if mcq_eval and mcq_eval.get("error_types"):
            result["mcq_errors"].extend(mcq_eval["error_types"])
        if mcq_eval and mcq_eval.get("error_counts"):
            result["mcq_error_counts"] = mcq_eval["error_counts"]
        result["raw"] = {"forecast": ans_f, "mcq": ans_m}

    else:
        # ---------- 合并询问（兼容旧逻辑） ----------
        T2_T4_RETURN_INSTRUCTION = (
            "Return ONLY a JSON object with keys:\n"
            "{\n"
            '  "forecast": number[],\n'
            '  "mcq": {\n'
            '    "future_vs_history": "Higher|Lower|Similar|Uncertain",\n'
            '    "volatility_change": "increased|decreased|constant|Uncertain",\n'
            '    "seasonality_shift": "fixed|shifting|no|Uncertain"\n'
            "  }\n"
            "}\n"
            "Do not add comments or markdown."
        )
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": f"{prompt_text}\n\n{T2_T4_RETURN_INSTRUCTION}"},
        ]
        raw = chat(model_name, messages)
        ans = extract_first_json(raw) or {}

        # forecast
        forecast_raw = ans.get("forecast")
        if not _forecast_has_values(forecast_raw):
            result["forecast_exec_success"] = 0.0
            result["forecast_errors"].append("missing_forecast")
        else:
            result["forecast_exec_success"] = 1.0
        metrics, metric_err = _compute_forecast_metrics(gt_obj, forecast_raw, dataset_lower, history_series)
        result["metrics"] = metrics
        if metrics is None:
            result["forecast_exec_success"] = 0.0
        if metric_err:
            result["forecast_errors"].append(metric_err)
            result["forecast_exec_success"] = 0.0

        # mcq
        mcq_gt = t2.get("mcq", {})
        mcq_pred = safe_get(ans, "mcq", {})
        result["mcq"] = _eval_mcq_block(mcq_pred, mcq_gt) if mcq_gt else None
        result["mcq_exec_success"] = 1.0 if (isinstance(mcq_pred, dict) and any(v is not None for v in mcq_pred.values())) else 0.0
        if result["mcq"]:
            _log_mcq_comparison("T2", result["mcq"].get("per_question"))
        if result["mcq"] and result["mcq"].get("error_types"):
            result["mcq_errors"].extend(result["mcq"]["error_types"])
        if result["mcq"] and result["mcq"].get("error_counts"):
            result["mcq_error_counts"] = result["mcq"]["error_counts"]
        if not ans:
            result["mcq_errors"].append("invalid_json")
            result["forecast_errors"].append("invalid_json")
        result["raw"] = ans

    return result


def eval_t3(sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    T3：不合并提问；对 pack 中每个子任务单独构造提示并调用一次模型。
    评测：
      - exec_success_rate = 模型返回了可判定答案（非空且在 label_space 内）的比例
      - mcq_acc = 正确数 / 题目数（pred == gt）
    """
    t3 = sample["tasks"]["T3"]
    pack = t3.get("pack", [])
    if not isinstance(pack, list) or not pack:
        return {"ok": False, "error": "empty_pack"}

    tids: List[str] = []
    gts: List[Optional[str]] = []
    label_spaces: List[List[str]] = []
    preds: List[Optional[str]] = []
    invalid_flags: List[bool] = []
    details: Dict[str, Dict[str, Any]] = {}

    # 先整理 GT 与 label_space
    for i, it in enumerate(pack):
        # print(f"Debug: T3 Pack Item {i}: {it}")
        # exit()
        tid = it.get("task_id", str(i))
        tids.append(tid)
        gt = it.get("label") or it.get("Answer") or it.get("answer")
        gts.append(gt if isinstance(gt, str) else (str(gt).strip() if gt is not None else None))
        ls = it.get("label_space") or it.get("options") or it.get("Options") or []
        ls = [str(x).strip() for x in ls] if isinstance(ls, list) else []
        label_spaces.append(ls)

    # 逐题调用
    for i, it in enumerate(pack):
        single_prompt = build_t3_single_prompt(t3, it, i)
        # print(f"Debug: T3 Single Prompt for task {tids[i]}:\n{single_prompt}\n")
        # exit()
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": single_prompt}
        ]
        raw = chat(model_name, messages)
        pred = parse_t3_single_answer(raw)
        invalid_option = False

        # 规范化 + 限定在 label_space 内（不在就视为无效）
        if pred is not None:
            pred = str(pred).strip()
            if label_spaces[i] and pred not in label_spaces[i]:
                invalid_option = True
                pred = None

        preds.append(pred)
        invalid_flags.append(invalid_option)

    # 汇总评测
    correct = 0
    n_valid = 0
    error_counts = {"missing_answer": 0, "invalid_option": 0}
    for i, tid in enumerate(tids):
        p = preds[i]
        g = gts[i]
        ok = (p is not None and g is not None and p == g)
        if p is not None:
            n_valid += 1
        if ok:
            correct += 1
        else:
            if p is None:
                if invalid_flags[i]:
                    error_counts["invalid_option"] += 1
                else:
                    error_counts["missing_answer"] += 1

        details[tid] = {"pred": p, "gt": g, "correct": ok}

    total = len(tids)
    exec_success_rate = (n_valid / total) if total else None
    mcq_acc = (correct / total) if total else None
    mcq_error_types = [k for k, v in error_counts.items() if v > 0]

    log_map = {
        tid: {
            "pred": info.get("pred"),
            "gt": info.get("gt"),
            "gt_norm": _normalize_label(info.get("gt")),
            "correct": info.get("correct"),
        }
        for tid, info in details.items()
    }
    _log_mcq_comparison("T3", log_map)
    return {
        "ok": True,
        "exec_success_rate": exec_success_rate,
        "mcq_acc": mcq_acc,
        "mcq_error_counts": error_counts,
        "mcq_errors": mcq_error_types,
        "details": details,
        "raw": None  # 单题已分别调用，raw 不再返回聚合体
    }

def eval_t4(sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    T4 评测（支持“预测”和“MCQ”拆分成两次独立询问）。
    依赖：
      - build_t4_prompts(t4)  -> {"forecast_prompt","mcq_prompt"}
      - SYSTEM_TEMPLATE, chat(), extract_first_json()
      - _to_float_list(), _align_and_filter()
      - mape(), mae(), rmse(), smape()
      - _eval_mcq_block()
      - 全局开关：SPLIT_T2_T4 (True=分开问；False=合并问)
    返回：
      {
        ok: True/False,
        metrics: {...} | None,            # 仅当有 gt 且 forecast 可评时
        mcq: { per_question, acc } | None,
        forecast_exec_success: 0/1,
        mcq_exec_success: 0/1,
        raw: {...}                        # 存放模型原始返回（便于排错）
      }
    """
    t4 = sample["tasks"]["T4"]
    prompt_text = t4.get("prompt") or t4.get("task") or ""
    gt_raw = t4.get("ground_truth", None)
    dataset_lower = _dataset_name_lower(sample)
    history_series = _get_history_series(t4)

    result: Dict[str, Any] = {"ok": True, "forecast_errors": [], "mcq_errors": []}

    if SPLIT_T2_T4:
        # ---------- 使用自动构造的两个独立 Prompt ----------
        prompts = build_t4_prompts(t4, sample_meta=sample.get("meta"))

        # 1) 预测（forecast-only）
        messages_f = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": prompts["forecast_prompt"]},
        ]
        raw_f = chat(model_name, messages_f)
        ans_f = extract_first_json(raw_f) or {}
        if not ans_f:
            result["forecast_errors"].append("invalid_json")
        forecast_raw = ans_f.get("forecast")
        if not _forecast_has_values(forecast_raw):
            result["forecast_exec_success"] = 0.0
            result["forecast_errors"].append("missing_forecast")
        else:
            result["forecast_exec_success"] = 1.0
        metrics, metric_err = _compute_forecast_metrics(gt_raw, forecast_raw, dataset_lower, history_series)
        result["metrics"] = metrics
        if metrics is None:
            result["forecast_exec_success"] = 0.0
        if metric_err:
            result["forecast_errors"].append(metric_err)
            result["forecast_exec_success"] = 0.0

        # 2) MCQ（mcq-only）
        messages_m = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": prompts["mcq_prompt"]},
        ]
        raw_m = chat(model_name, messages_m)
        ans_m = extract_first_json(raw_m) or {}
        if not ans_m:
            result["mcq_errors"].append("invalid_json")
        mcq_pred = safe_get(ans_m, "mcq", {})
        if not isinstance(mcq_pred, dict) or not mcq_pred:
            result["mcq_errors"].append("missing_output")
        mcq_gt = t4.get("mcq", {})
        mcq_eval = _eval_mcq_block(mcq_pred, mcq_gt) if mcq_gt else None

        # MCQ 执行成功性：返回了一个 dict 且至少一个字段有值
        result["mcq_exec_success"] = 1.0 if (isinstance(mcq_pred, dict) and any(v is not None for v in mcq_pred.values())) else 0.0
        result["mcq"] = mcq_eval
        if mcq_eval:
            _log_mcq_comparison("T4", mcq_eval.get("per_question"))
        if mcq_eval and mcq_eval.get("error_types"):
            result["mcq_errors"].extend(mcq_eval["error_types"])
        if mcq_eval and mcq_eval.get("error_counts"):
            result["mcq_error_counts"] = mcq_eval["error_counts"]

        # 原始返回保留，便于调试
        result["raw"] = {"forecast": ans_f, "mcq": ans_m}

    else:
        # ---------- 合并询问（兼容旧逻辑） ----------
        T2_T4_RETURN_INSTRUCTION = (
            "Return ONLY a JSON object with keys:\n"
            "{\n"
            '  "forecast": number[],\n'
            '  "mcq": {\n'
            '    "future_vs_history": "Higher|Lower|Similar|Uncertain",\n'
            '    "volatility_change": "increased|decreased|constant|Uncertain",\n'
            '    "seasonality_shift": "fixed|shifting|no|Uncertain"\n'
            "  }\n"
            "}\n"
            "Do not add comments or markdown."
        )
        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": f"{prompt_text}\n\n{T2_T4_RETURN_INSTRUCTION}"},
        ]
        raw = chat(model_name, messages)
        ans = extract_first_json(raw) or {}

        # forecast
        forecast_raw = ans.get("forecast")
        if not _forecast_has_values(forecast_raw):
            result["forecast_exec_success"] = 0.0
            result["forecast_errors"].append("missing_forecast")
        else:
            result["forecast_exec_success"] = 1.0
        metrics, metric_err = _compute_forecast_metrics(gt_raw, forecast_raw, dataset_lower, history_series)
        result["metrics"] = metrics
        if metrics is None:
            result["forecast_exec_success"] = 0.0
        if metric_err:
            result["forecast_errors"].append(metric_err)
            result["forecast_exec_success"] = 0.0

        # mcq
        mcq_gt = t4.get("mcq", {})
        mcq_pred = safe_get(ans, "mcq", {})
        result["mcq"] = _eval_mcq_block(mcq_pred, mcq_gt) if mcq_gt else None
        result["mcq_exec_success"] = 1.0 if (isinstance(mcq_pred, dict) and any(v is not None for v in mcq_pred.values())) else 0.0
        if result["mcq"]:
            _log_mcq_comparison("T4", result["mcq"].get("per_question"))
        if result["mcq"] and result["mcq"].get("error_types"):
            result["mcq_errors"].extend(result["mcq"]["error_types"])
        if result["mcq"] and result["mcq"].get("error_counts"):
            result["mcq_error_counts"] = result["mcq"]["error_counts"]
        if not ans:
            result["forecast_errors"].append("invalid_json")
            result["mcq_errors"].append("invalid_json")
        result["raw"] = ans

    return result

# =========================
# ===== RUNNER ============
# =========================

def run_once(sample: Dict[str, Any], model_name: str, tiers: List[str]) -> Dict[str, Any]:
    dataset_name = sample.get("_dataset_name") or "Unknown"
    out = {"model": model_name, "dataset": dataset_name, "sample_id": sample_id_of(sample), "tiers": {}}
    tasks = sample.get("tasks", sample)
    for t in tiers:
        if t not in tasks:
            out["tiers"][t] = {"ok": False, "error": "tier_not_found"}
            continue
        try:
            if t == "T1":
                out["tiers"][t] = eval_t1(sample, model_name)
            elif t == "T2":
                out["tiers"][t] = eval_t2(sample, model_name)
            elif t == "T3":
                out["tiers"][t] = eval_t3(sample, model_name)
            elif t == "T4":
                out["tiers"][t] = eval_t4(sample, model_name)
            else:
                out["tiers"][t] = {"ok": False, "error": "unknown_tier"}
        except ChatError as ce:
            entry = {"ok": False, "error": f"chat_error: {ce}", "forecast_errors": [], "mcq_errors": []}
            modes = TIER_MODALITY.get(t, {"mcq"})
            if "forecast" in modes:
                entry["forecast_errors"].append("chat_error")
            if "mcq" in modes:
                entry["mcq_errors"].append("chat_error")
            out["tiers"][t] = entry
        except Exception as e:
            entry = {"ok": False, "error": f"runtime_error: {e}", "forecast_errors": [], "mcq_errors": []}
            modes = TIER_MODALITY.get(t, {"mcq"})
            if "forecast" in modes:
                entry["forecast_errors"].append("other_error")
            if "mcq" in modes:
                entry["mcq_errors"].append("other_error")
            out["tiers"][t] = entry
    return out


def summarize_results_multi(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    生成跨数据集的表格化汇总：
      列：Dataset, Model, Task, Subtask, Exec Success Rate, Accuracy Rate, MAPE, MAE, RMSE, SMAPE
    统计：
      - Exec Success Rate:
          T1: ok 且四键齐 => 1，否则 0
          T2: forecast_exec_success / mcq_exec_success 的均值（分别在 "forecast" 与 "mcq" 行中）
          T3: exec_success_rate 字段（逐题单独问的版本已返回）
          T4: 同 T2
      - Accuracy Rate:
          T1: mcq_acc
          T2/T4 forecast: 可置为与 Exec 成功相同或置空（通常不设“准确率”，这里置空）
          T2/T4 mcq: mcq.acc
          T3: mcq_acc
      - 数值指标：仅 T2/T4 的 forecast 行统计 4 个误差的均值
    """
    table = []
    task_sub_map = {"T1":"understanding","T2":"forecast","T3":"pack","T4":"forecast"}
    metric_keys = ("MAPE","MAE","RMSE","SMAPE","OW_sMAPE","OW_RMSSE","OW_MASE")

    # 按 (dataset, model) 分组
    grouped = {}
    for r in results:
        ds = r.get("dataset", "Unknown")
        md = r.get("model", "Unknown")
        grouped.setdefault((ds, md), []).append(r)

    for (dataset, model), rs in grouped.items():
        # 汇总容器
        agg = {}
        def ensure(tkey):
            if tkey not in agg:
                agg[tkey] = {"exec": [], "acc": []}
                for mk in metric_keys:
                    agg[tkey][mk] = []

        for r in rs:
            for t, out in (r.get("tiers") or {}).items():
                ensure(t)
                if not isinstance(out, dict) or not out.get("ok"):
                    # 执行失败计 0（仅对 exec 成功率）
                    if t=="T1":
                        agg[t]["exec"].append(0.0)
                    elif t in ("T2","T4"):
                        mslot = agg.setdefault(f"{t}_mcq", {"exec": [], "acc": []})
                        for mk in metric_keys:
                            mslot.setdefault(mk, [])
                        agg[t]["exec"].append(0.0)
                        mslot["exec"].append(0.0)
                    elif t=="T3":
                        agg[t]["exec"].append(0.0)
                    continue

                    # 正常成功路径：
                if t=="T1":
                    # 执行成功性：答案齐全（你也可更精细）
                    agg[t]["exec"].append(1.0)
                    # 准确率
                    if out.get("mcq_acc") is not None:
                        agg[t]["acc"].append(out["mcq_acc"])

                elif t in ("T2","T4"):
                    # forecast 部分
                    f_exec = out.get("forecast_exec_success")
                    if f_exec is not None:
                        agg[t]["exec"].append(float(f_exec))
                    met = out.get("metrics")
                    if met and f_exec:
                        for k in metric_keys:
                            if met.get(k) is not None:
                                agg[t][k].append(float(met[k]))
                    # mcq 部分单列
                    mkey = f"{t}_mcq"
                    if mkey not in agg:
                        agg[mkey] = {"exec": [], "acc": []}
                        for mk in metric_keys:
                            agg[mkey][mk] = []
                    mcq = out.get("mcq")
                    if mcq and isinstance(mcq, dict):
                        if mcq.get("acc") is not None:
                            agg[mkey]["acc"].append(float(mcq["acc"]))
                    m_exec = out.get("mcq_exec_success")
                    if m_exec is not None:
                        agg[mkey]["exec"].append(float(m_exec))

                elif t=="T3":
                    if out.get("exec_success_rate") is not None:
                        agg[t]["exec"].append(float(out["exec_success_rate"]))
                    if out.get("mcq_acc") is not None:
                        agg[t]["acc"].append(float(out["mcq_acc"]))

        # 写表：T1/T3 一行；T2/T4 两行（forecast 与 mcq）
        for t in ("T1","T2","T3","T4"):
            if t in agg:
                row = {
                    "Dataset": dataset,
                    "Model": model,
                    "Task": t,
                    "Subtask": task_sub_map.get(t, t),
                    "Exec Success Rate": float(np.mean(agg[t]["exec"])) if agg[t]["exec"] else None,
                    "Accuracy Rate": float(np.mean(agg[t]["acc"])) if agg[t]["acc"] else None,
                }
                for k in metric_keys:
                    row[k] = float(np.mean(agg[t][k])) if agg[t][k] else None
                table.append(row)
            # T2/T4 的 mcq 子行
            mkey = f"{t}_mcq"
            if mkey in agg:
                row = {
                    "Dataset": dataset,
                    "Model": model,
                    "Task": t,
                    "Subtask": "mcq",
                    "Exec Success Rate": float(np.mean(agg[mkey]["exec"])) if agg[mkey]["exec"] else None,
                    "Accuracy Rate": float(np.mean(agg[mkey]["acc"])) if agg[mkey]["acc"] else None,
                }
                for k in metric_keys:
                    row[k] = None
                table.append(row)

    return table


def summarize_error_stats(results: List[Dict[str, Any]]):
    def _nested_dict():
        return defaultdict(lambda: defaultdict(int))

    forecast_stats = defaultdict(_nested_dict)
    mcq_stats = defaultdict(_nested_dict)

    for r in results:
        dataset = r.get("dataset", "Unknown")
        for tier, out in (r.get("tiers") or {}).items():
            if not isinstance(out, dict):
                continue
            tier_forecast = forecast_stats[dataset][tier]
            for reason in out.get("forecast_errors", []) or []:
                tier_forecast[reason] += 1
                tier_forecast["__total__"] += 1

            tier_mcq = mcq_stats[dataset][tier]
            for reason in out.get("mcq_errors", []) or []:
                tier_mcq[reason] += 1
                tier_mcq["__total__"] += 1

            counts = out.get("mcq_error_counts")
            if isinstance(counts, dict):
                for reason, count in counts.items():
                    if reason == "__total__":
                        continue
                    if isinstance(count, (int, float)) and count > 0:
                        tier_mcq[reason] += int(count)
                        tier_mcq["__total__"] += int(count)
    return forecast_stats, mcq_stats


def print_error_breakdown(title: str, stats: Dict[str, Dict[str, Dict[str, int]]]):
    print(f"\n===== {title} Error Breakdown by Dataset & Tier =====")
    any_data = False
    for dataset in sorted(stats.keys()):
        ds_has_data = False
        for tier in sorted(stats[dataset].keys()):
            total = stats[dataset][tier].get("__total__", 0)
            if total <= 0:
                continue
            if not ds_has_data:
                print(f"-- Dataset: {dataset} --")
                ds_has_data = True
                any_data = True
            print(f"{tier}: total {total}")
            for reason, count in sorted(stats[dataset][tier].items()):
                if reason == "__total__":
                    continue
                pct = (count / total) * 100.0 if total else 0.0
                print(f"  - {reason}: {count} ({pct:.2f}%)")
        if ds_has_data:
            print()
    if not any_data:
        print("No errors recorded.")

def build_output_suffix(model_name: str, dataset_names: List[str]) -> str:
    dataset_part = "-".join(dataset_names) if dataset_names else "unknown_dataset"
    return f"{model_name}_{dataset_part}_high"

def main():
    pairs = load_all_datasets(DATASET_PATHS, ENABLED_DATASETS, SAMPLE_NUM, SAMPLE_SHUFFLE, RANDOM_SEED)
    if not pairs:
        raise RuntimeError(f"No samples found from selected datasets: {ENABLED_DATASETS}")

    all_results: List[Dict[str, Any]] = []
    for m in ENABLED_MODELS:
        if m not in MODEL_CONFIGS:
            print(f"[Skip] Model {m} not in MODEL_CONFIGS.")
            continue
        for idx, (ds_name, sample) in enumerate(pairs):
            sid = sample_id_of(sample)
            print(f"== Running {m} | {ds_name} | tiers {ENABLED_TIERS} | sample {idx+1}/{len(pairs)}: {sid} ==")
            r = run_once(sample, m, ENABLED_TIERS)
            all_results.append(r)

    output_suffix = build_output_suffix(ENABLED_MODELS[0], ENABLED_DATASETS)
    detail_path = f"eval_results_detail_{output_suffix}.json"
    summary_json_path = f"eval_results_summary_{output_suffix}.json"
    summary_csv_path = f"eval_results_summary_{output_suffix}.csv"

    # 保存：详细结果
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 保存：表格化汇总（含 Dataset/Model）
    summary_table = summarize_results_multi(all_results)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, ensure_ascii=False, indent=2)

    # 另存 CSV
    if summary_table:
        import csv
        cols = ["Dataset","Model","Task","Subtask","Exec Success Rate","Accuracy Rate",
                "MAPE","MAE","RMSE","SMAPE","OW_sMAPE","OW_RMSSE","OW_MASE"]
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in summary_table:
                writer.writerow({k: row.get(k) for k in cols})

    forecast_error_stats, mcq_error_stats = summarize_error_stats(all_results)
    print_error_breakdown("Forecast", forecast_error_stats)
    print_error_breakdown("MCQ", mcq_error_stats)

    print(f"\nSaved:\n - {detail_path}\n - {summary_json_path}\n - {summary_csv_path}")

if __name__ == "__main__":
    main()
