#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长度扰动实验版：基于 evaluate_llm.py，针对不同保留比例（60/70/80/90%）
对 prompt 中的 Input(JSON) 里时间序列做“截掉开头、保留末尾”的截断，然后运行原有 LLM 评测。

特点：
- 不修改原始数据文件，只在内存中处理。
- 按比例生成新的数据集标识（dataset|ratio=80%），便于汇总区分。
- 输出文件名带上比例后缀。
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import evaluate_llm as eb

# ============== 实验参数（直接改这里） ==============
LENGTH_RATIOS: List[float] = [
    # 0.6, 
    # 0.7, 
    # 0.8, 
    0.9
    ]
ENABLED_DATASETS = eb.ENABLED_DATASETS[:]  # 如需定制，可改成 ["FreshRetailNet", ...]
ENABLED_TIERS = eb.ENABLED_TIERS[:]        # T1/T2/T3/T4
ENABLED_MODELS = ["gpt-4o"]      # 模型列表
SAMPLE_NUM = 50               # 采样条数
SAMPLE_SHUFFLE = eb.SAMPLE_SHUFFLE
RANDOM_SEED = eb.RANDOM_SEED

# 只认为“长度 >= MIN_SERIES_LEN 且元素为数值/None”的 list 是时间序列
MIN_SERIES_LEN = 12


# ============== 长度扰动工具 ==============
def _find_json_bounds(text: str, start_search: int = 0) -> Tuple[int, int]:
    start = text.find("{", start_search)
    if start == -1:
        return -1, -1
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, idx
    return -1, -1


def _shrink_series_in_obj(obj: Any, ratio: float) -> Tuple[Any, Dict[str, Any]]:
    """递归截断时间序列 list（保留末尾、丢掉开头）。"""
    changes: List[Tuple[int, int]] = []
    length_map: Dict[int, int] = {}

    def helper(node: Any) -> Any:
        if isinstance(node, list):
            if len(node) >= MIN_SERIES_LEN and all(isinstance(x, (int, float)) or x is None for x in node):
                orig_len = len(node)
                new_len = length_map.setdefault(orig_len, max(1, int(math.floor(orig_len * ratio))))
                if new_len < orig_len:
                    changes.append((orig_len, new_len))
                    return node[-new_len:]
                return node
            return [helper(x) for x in node]
        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = helper(v)
        return node

    updated = helper(obj)
    return updated, {"series_touched": len(changes), "length_changes": changes}


def _perturb_prompt(prompt: str, ratio: float) -> Tuple[str, Dict[str, Any]]:
    """找到 prompt 中的第一个 JSON 对象，截断其中的时间序列后返回新 prompt 与统计信息。"""
    if not isinstance(prompt, str):
        return prompt, {}
    input_pos = prompt.lower().find("input")
    start, end = _find_json_bounds(prompt, start_search=max(0, input_pos))
    if start == -1 or end == -1:
        start, end = _find_json_bounds(prompt, start_search=0)
    if start == -1 or end == -1:
        return prompt, {}
    raw_json = prompt[start : end + 1]
    try:
        parsed = json.loads(raw_json)
    except Exception:
        return prompt, {}

    updated, stats = _shrink_series_in_obj(parsed, ratio)
    if stats.get("series_touched", 0) <= 0:
        return prompt, stats
    new_json = json.dumps(updated, ensure_ascii=False)
    new_prompt = prompt[:start] + new_json + prompt[end + 1 :]
    return new_prompt, stats


def _perturb_sample(sample: Dict[str, Any], ratio: float, length_counter: Dict[Tuple[int, int], int]) -> Dict[str, Any]:
    """深拷贝样本并对各 tier prompt 做长度扰动（仅修改 prompt 字段）。"""
    s = copy.deepcopy(sample)
    tasks = s.get("tasks")
    if not isinstance(tasks, dict):
        return s

    for tier, node in tasks.items():
        if not isinstance(node, dict):
            continue
        prompt = node.get("prompt")
        new_prompt, stats = _perturb_prompt(prompt, ratio)
        if new_prompt != prompt:
            node["prompt"] = new_prompt
        for orig, new in stats.get("length_changes", []):
            length_counter[(orig, new)] += 1
    return s


def _print_length_summary(length_counter: Dict[Tuple[int, int], int], ratio: float) -> None:
    if not length_counter:
        print(f"[长度扰动] 比例 {ratio:.0%}：未检测到可截断的时间序列。")
        return
    parts = []
    for (orig, new), cnt in sorted(length_counter.items(), key=lambda x: (x[0][0], x[0][1])):
        rel = (new / orig) if orig else 0.0
        suffix = f" x{cnt}" if cnt > 1 else ""
        parts.append(f"{orig}->{new} ({rel:.1%}){suffix}")
    print(f"[长度扰动] 比例 {ratio:.0%}：长度变化汇总: " + "; ".join(parts))


# ============== 主流程 ==============
def main():
    all_results: List[Dict[str, Any]] = []

    for ratio in LENGTH_RATIOS:
        print(f"\n===================== 长度比例 {ratio:.0%} =====================")
        pairs = eb.load_all_datasets(eb.DATASET_PATHS, ENABLED_DATASETS, SAMPLE_NUM, SAMPLE_SHUFFLE, RANDOM_SEED)
        if not pairs:
            raise RuntimeError(f"未加载到样本，数据集列表：{ENABLED_DATASETS}")

        length_counter: Dict[Tuple[int, int], int] = defaultdict(int)
        # 构造带比例的新样本（深拷贝）
        perturbed_pairs = []
        for ds_name, sample in pairs:
            new_sample = _perturb_sample(sample, ratio, length_counter)
            new_sample["_dataset_name"] = f"{ds_name}|ratio={int(ratio*100)}%"
            perturbed_pairs.append((new_sample["_dataset_name"], new_sample))

        _print_length_summary(length_counter, ratio)

        ratio_results: List[Dict[str, Any]] = []
        for m in ENABLED_MODELS:
            if m not in eb.MODEL_CONFIGS:
                print(f"[Skip] 模型 {m} 不在 MODEL_CONFIGS 中。")
                continue
            for idx, (ds_name, sample) in enumerate(perturbed_pairs):
                sid = eb.sample_id_of(sample)
                print(f"== Running {m} | {ds_name} | tiers {ENABLED_TIERS} | sample {idx+1}/{len(perturbed_pairs)}: {sid} ==")
                r = eb.run_once(sample, m, ENABLED_TIERS)
                ratio_results.append(r)

        # 保存当前比例的结果
        if ratio_results:
            base = f"eval_results_ratio_{int(ratio*100)}_{ENABLED_MODELS[0]}"
            with open(f"{base}_detail.json", "w", encoding="utf-8") as f:
                json.dump(ratio_results, f, ensure_ascii=False, indent=2)
            summary_table = eb.summarize_results_multi(ratio_results)
            with open(f"{base}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_table, f, ensure_ascii=False, indent=2)
            if summary_table:
                import csv
                cols = ["Dataset","Model","Task","Subtask","Exec Success Rate","Accuracy Rate",
                        "MAPE","MAE","RMSE","SMAPE","OW_sMAPE","OW_RMSSE","OW_MASE"]
                with open(f"{base}_summary.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=cols)
                    writer.writeheader()
                    for row in summary_table:
                        writer.writerow({k: row.get(k) for k in cols})

            forecast_error_stats, mcq_error_stats = eb.summarize_error_stats(ratio_results)
            eb.print_error_breakdown("Forecast", forecast_error_stats)
            eb.print_error_breakdown("MCQ", mcq_error_stats)

        all_results.extend(ratio_results)

    if not all_results:
        print("\n⚠️ 未收集到任何结果。")
        return

    print(f"\n✅ 全部比例完成，累计样本数：{len(all_results)}。")


if __name__ == "__main__":
    main()
