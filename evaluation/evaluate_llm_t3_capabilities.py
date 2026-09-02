#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3-only evaluator with capability-wise stats (per dataset + global).
Uses evaluate_llm.py utilities (chat, prompt builders, dataset loader).
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any, Dict, List

import evaluate_llm as eb

# ===== 手动配置 =====
MODEL = "gemini-2.5-flash-lite"  # 可选: "gpt-4o", "claude-3-7-sonnet-latest", "gemini-2.5-flash-lite", "codestral-latest", "deepseek-chat", "qwen-plus", "grok-4-fast-reasoning"
SAMPLE_NUM = 50     # 每个数据集最多采样多少条（>=总数则全量）
SAMPLE_SEED = 42
SLEEP_BETWEEN_REQ = 1.0  # 每次调用后 sleep，减轻速率限制
ENABLED_DATASETS = [
    "FreshRetailNet",
    "PSML",
    "MIMIC",
    "CausalChambers",
]


# ===== 统计工具 =====
def _update_capability_stats(cap_stats, dataset_name, entry, pred, valid):
    caps = entry.get("capabilities") or []
    if not isinstance(caps, list):
        return
    label = entry.get("label")
    correct = bool(valid and isinstance(label, str) and pred == label)
    for cap in caps:
        bucket = cap_stats[dataset_name][str(cap)]
        bucket["total"] += 1
        if valid:
            bucket["valid"] += 1
        if isinstance(label, str):
            bucket["label_total"] += 1
            if correct:
                bucket["correct"] += 1


def _print_capability_report(cap_stats):
    print("\n===== Capability Stats =====")
    agg = defaultdict(lambda: {"total": 0, "valid": 0, "label_total": 0, "correct": 0})
    for ds, caps in cap_stats.items():
        print(f"\n📂 数据集: {ds}")
        if not caps:
            print("  (无记录)")
            continue
        print("  capability | total | success_rate | accuracy")
        for cap, bucket in sorted(caps.items()):
            total = bucket["total"]
            succ = bucket["valid"] / total if total else 0.0
            acc = bucket["correct"] / bucket["label_total"] if bucket["label_total"] else 0.0
            print(f"  {cap:10s} | {total:5d} | {succ:12.2%} | {acc:8.2%}")
            agg[cap]["total"] += bucket["total"]
            agg[cap]["valid"] += bucket["valid"]
            agg[cap]["label_total"] += bucket["label_total"]
            agg[cap]["correct"] += bucket["correct"]
    print("\n📊 汇总 (所有数据集)")
    if not agg:
        print("  (无记录)")
    else:
        print("  capability | total | success_rate | accuracy")
        for cap, bucket in sorted(agg.items()):
            total = bucket["total"]
            succ = bucket["valid"] / total if total else 0.0
            acc = bucket["correct"] / bucket["label_total"] if bucket["label_total"] else 0.0
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
            print(f"\n--- Sample {sid} ---")
            preds = []
            labels = []
            label_spaces = []

            for idx, item in enumerate(pack):
                prompt = eb.build_t3_single_prompt(t3, item, idx)
                messages = [
                    {"role": "system", "content": eb.SYSTEM_TEMPLATE},
                    {"role": "user", "content": prompt},
                ]
                try:
                    raw = eb.chat(MODEL, messages)
                except Exception as e:
                    print(f"❌ chat error on {sid}-{item.get('task_id')}: {e}")
                    raw = None
                finally:
                    time.sleep(SLEEP_BETWEEN_REQ)
                pred = eb.parse_t3_single_answer(raw) if raw is not None else None
                labels.append(item.get("label"))
                ls = item.get("label_space") or item.get("options") or []
                if isinstance(ls, list):
                    ls = [str(x) for x in ls]
                else:
                    ls = []
                label_spaces.append(ls)

                valid = False
                if pred is not None:
                    if ls:
                        valid = pred in ls
                        if not valid:
                            pred = None
                    else:
                        valid = True
                preds.append(pred)

                print(f"   [{item.get('task_id')}] pred={pred} | label={item.get('label')} | options={ls}")
                _update_capability_stats(capability_stats, ds, item, pred, valid)
                tier_success["tot"] += 1
                if valid:
                    tier_success["ok"] += 1

            # accuracy per sample
            correct = 0
            total_label = 0
            for p, g in zip(preds, labels):
                if isinstance(g, str):
                    total_label += 1
                    if p == g:
                        correct += 1
            if total_label:
                tier_accuracy.append(correct / total_label)
                print(f"📊 sample acc: {correct}/{total_label} ({correct/total_label:.2%})")

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
