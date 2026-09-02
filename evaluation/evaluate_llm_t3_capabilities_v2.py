#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 evaluator (single prompt per sample) with capability-wise stats.
Prompts pack questions together to reduce token cost vs per-question calls.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import evaluate_llm as eb

# 统一规范化
def _norm(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


# ===== 手动配置 =====
MODEL = "qwen-plus"  # 可选: "gpt-4o", "claude-3-7-sonnet-latest", "gemini-2.5-flash-lite", "codestral-latest", "deepseek-chat", "qwen-plus", "grok-4-fast-reasoning"
SAMPLE_NUM = 50     # 每个数据集最多采样多少条（>=总数则全量）
SAMPLE_SEED = 42
SLEEP_BETWEEN_REQ = 1.0  # 每个样本请求后 sleep，减轻速率限制
ENABLED_DATASETS = [
    "FreshRetailNet",
    "PSML",
    "MIMIC",
    "CausalChambers",
]


# ===== 统计工具 =====
def _update_capability_stats(cap_stats, dataset_name, pack: List[dict], answers: Dict[str, Any]):
    ds_bucket = cap_stats[dataset_name]

    def _resolve_pred(item, idx):
        keys = [
            item.get("task_id"),
            item.get("name"),
            item.get("sub_id"),
            str(idx),
            f"{idx}",
            f"Q{idx+1}",
        ]
        for k in keys:
            if k is None:
                continue
            if k in answers:
                return _norm(answers[k])
        return None

    for idx, item in enumerate(pack):
        caps = item.get("capabilities") or []
        if not isinstance(caps, list):
            continue
        pred = _resolve_pred(item, idx)
        label = _norm(item.get("label"))
        options = item.get("label_space") or item.get("options") or []
        opts_norm = {_norm(o) for o in options} if isinstance(options, list) else set()
        opts_norm = {o for o in opts_norm if o}
        valid = False
        if pred:
            valid = (pred in opts_norm) if opts_norm else True
        correct = bool(valid and label and pred == label)
        for cap in caps:
            b = ds_bucket[str(cap)]
            b["total"] += 1
            if valid:
                b["valid"] += 1
            if isinstance(label, str):
                b["label_total"] += 1
                if correct:
                    b["correct"] += 1


def _print_capability_report(cap_stats):
    print("\n===== Capability Stats =====")
    agg = defaultdict(lambda: {"total": 0, "valid": 0, "label_total": 0, "correct": 0})
    for ds, caps in cap_stats.items():
        print(f"\n📂 数据集: {ds}")
        if not caps:
            print("  (无记录)")
            continue
        print("  capability | total | success_rate | accuracy")
        for cap, b in sorted(caps.items()):
            total = b["total"]
            succ = b["valid"] / total if total else 0.0
            acc = b["correct"] / b["label_total"] if b["label_total"] else 0.0
            print(f"  {cap:10s} | {total:5d} | {succ:12.2%} | {acc:8.2%}")
            agg[cap]["total"] += b["total"]
            agg[cap]["valid"] += b["valid"]
            agg[cap]["label_total"] += b["label_total"]
            agg[cap]["correct"] += b["correct"]
    print("\n📊 汇总 (所有数据集)")
    if not agg:
        print("  (无记录)")
    else:
        print("  capability | total | success_rate | accuracy")
        for cap, b in sorted(agg.items()):
            total = b["total"]
            succ = b["valid"] / total if total else 0.0
            acc = b["correct"] / b["label_total"] if b["label_total"] else 0.0
            print(f"  {cap:10s} | {total:5d} | {succ:12.2%} | {acc:8.2%}")


def main():
    random.seed(SAMPLE_SEED)
    tier_success = {"ok": 0, "tot": 0}
    tier_accuracy = []
    capability_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "valid": 0, "label_total": 0, "correct": 0}))

    for ds in ENABLED_DATASETS:
        path = eb.DATASET_PATHS.get(ds)
        if not path:
            print(f"[WARN] dataset path not found for key: {ds}")
            continue
        samples = eb.load_samples_for_dataset(path, SAMPLE_NUM, shuffle=False, seed=SAMPLE_SEED)
        print(f"\n📂 Dataset: {ds} ({path}) | model={MODEL} | sample_num={len(samples)}")
        if not samples:
            print("⚠️ No samples loaded; skip.")
            continue

        for sample in samples:
            sid = eb.sample_id_of(sample)
            t3 = sample.get("tasks", {}).get("T3")
            if not isinstance(t3, dict):
                continue
            pack = t3.get("pack") or []
            if not isinstance(pack, list) or not pack:
                continue
            prompt = eb.build_t3_prompt(t3)
            messages = [
                {"role": "system", "content": eb.SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ]
            try:
                raw = eb.chat(MODEL, messages)
            except Exception as e:
                print(f"❌ chat error on sample {sid}: {e}")
                tier_success["tot"] += len(pack)
                _update_capability_stats(capability_stats, ds, pack, {})
                time.sleep(SLEEP_BETWEEN_REQ)
                continue
            finally:
                time.sleep(SLEEP_BETWEEN_REQ)

            ans = eb.extract_first_json(raw) or {}
            answers_raw = ans.get("answers")

            # 统一答案为 dict
            if isinstance(answers_raw, list):
                answers = {item.get("task_id") or item.get("name") or f"Q{i+1}": val for i, (item, val) in enumerate(zip(pack, answers_raw))}
            elif isinstance(answers_raw, dict):
                answers = answers_raw
            else:
                answers = {}

            def _pick_answer(item, idx):
                keys = [
                    item.get("task_id"),
                    item.get("name"),
                    item.get("sub_id"),
                    str(idx),
                    f"{idx}",
                    f"Q{idx+1}",
                ]
                for k in keys:
                    if k is None:
                        continue
                    if k in answers:
                        return answers[k]
                return None

            valid_cnt = 0
            correct_cnt = 0
            label_cnt = 0
            answers_norm = {}
            for idx, item in enumerate(pack):
                tid = item.get("task_id") or item.get("name") or item.get("sub_id")
                pred_raw = _pick_answer(item, idx)
                pred = _norm(pred_raw)
                answers_norm[tid] = pred
                opts = item.get("label_space") or item.get("options") or []
                if isinstance(opts, list):
                    opts_norm = {_norm(o) for o in opts if _norm(o)}
                else:
                    opts_norm = set()
                label = _norm(item.get("label"))
                valid = False
                if pred:
                    if opts_norm:
                        valid = pred in opts_norm
                    else:
                        valid = True
                if valid:
                    valid_cnt += 1
                if label:
                    label_cnt += 1
                    if pred == label:
                        correct_cnt += 1

            tier_success["ok"] += valid_cnt
            tier_success["tot"] += len(pack)
            if label_cnt:
                tier_accuracy.append(correct_cnt / label_cnt)
            print(f"[RESULT] {sid}: success {valid_cnt}/{len(pack)}, acc {(correct_cnt/label_cnt if label_cnt else 0):.2%}")

            _update_capability_stats(capability_stats, ds, pack, answers_norm)

    print("\n===== Overall T3 Success =====")
    ok, tot = tier_success["ok"], tier_success["tot"]
    rate = ok / tot if tot else 0.0
    print(f"T3 success: {ok}/{tot} ({rate:.2%})")
    if tier_accuracy:
        avg_acc = sum(tier_accuracy) / len(tier_accuracy)
        print(f"Avg T3 accuracy: {avg_acc:.2%} over {len(tier_accuracy)} sample(s)")
    else:
        print("No accuracy recorded (no labels found).")

    _print_capability_report(capability_stats)


if __name__ == "__main__":
    main()
