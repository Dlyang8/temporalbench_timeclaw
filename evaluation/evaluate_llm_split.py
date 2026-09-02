#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 evaluator (single prompt per sample; pack merged) with basic success/accuracy stats.
Self-contained: does not import helpers from evaluate_llm.py.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import requests

# =========================
# ====== 配置区域 =========
# =========================
DATASET_PATHS = {
    "FreshRetailNet": "/projects/beei/mweng/TS-benchmark/TS-benchmark/dataset/freshretailnet/results/task_modified.json",
    "PSML": "/projects/beei/mweng/TS-benchmark/TS-benchmark/dataset/PSML/results",
    "MIMIC": "/projects/beei/mweng/TS-benchmark/TS-benchmark/dataset/MIMIC/results",
    "CausalChambers": "/projects/beei/mweng/TS-benchmark/TS-benchmark/dataset/causal_chambers/results",
}

ENABLED_DATASETS = [
    "FreshRetailNet",
    "PSML",
    "MIMIC",
    "CausalChambers",
]

MODEL = "qwen-plus"  # 可选: "gpt-4o", "claude-3-7-sonnet-latest", "gemini-2.5-flash-lite", "codestral-latest", "deepseek-chat", "qwen-plus", "grok-4-fast-reasoning"
SAMPLE_NUM = 5
SAMPLE_SEED = 42
SLEEP_BETWEEN_REQ = 1.0
REQUEST_TIMEOUT = 90

# 模型配置
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "provider": "openai_native",
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY") or "sk-your-openai-key",
        "base_url": "https://api.openai.com/v1",
    },
    "claude-3-7-sonnet-latest": {
        "provider": "anthropic_native",
        "model": "claude-3-7-sonnet-latest",
        "api_key": os.environ.get("ANTHROPIC_API_KEY") or "sk-your-anthropic-key",
        "base_url": "https://api.anthropic.com",
    },
    "gemini-2.5-flash-lite": {
        "provider": "gemini_native",
        "model": "gemini-2.5-flash-lite",
        "api_key": os.environ.get("GOOGLE_API_KEY") or "YOUR_GEMINI_KEY",
        "base_url": "https://generativelanguage.googleapis.com",
    },
    "codestral-latest": {
        "provider": "openai_compat",
        "model": "codestral-latest",
        "api_key": os.environ.get("MISTRAL_API_KEY") or "sk-your-mistral-key",
        "base_url": "https://api.mistral.ai/v1",
    },
    "deepseek-chat": {
        "provider": "openai_compat",
        "model": "deepseek-chat",
        "api_key": os.environ.get("DEEPSEEK_API_KEY") or "sk-your-deepseek-key",
        "base_url": "https://api.deepseek.com/v1",
    },
    "qwen-plus": {
        "provider": "openai_compat",
        "model": "qwen-plus",
        "api_key": os.environ.get("DASHSCOPE_API_KEY") or "sk-your-dashscope-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "grok-4-fast-reasoning": {
        "provider": "openai_compat",
        "model": "grok-4-fast-reasoning",
        "api_key": os.environ.get("GROK_API_KEY") or "sk-your-grok-key",
        "base_url": "https://api.x.ai/v1",
    },
}

SYSTEM_TEMPLATE = (
    "You are a precise time-series MCQ analyst. "
    "Answer each subtask with EXACTLY one option from its label_space. "
    "Return ONLY JSON, no markdown or explanations."
)

# =========================
# ====== 基础工具 =========
# =========================
def _norm(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def extract_first_json(text: str) -> Dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def load_samples_for_dataset(path: str, sample_num: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    files: List[str] = []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        if any(ch in path for ch in ["*", "?", "["]):
            files = sorted(glob.glob(path))
        elif os.path.exists(path):
            files = [path]
    samples: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                txt = f.read()
            objs = re.findall(r"\{[\s\S]*?\}", txt)
            for chunk in objs:
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        samples.append(obj)
                except Exception:
                    continue
        except Exception:
            continue
    if sample_num < len(samples):
        samples = rng.sample(samples, sample_num)
    return samples


def sample_id_of(s: Dict[str, Any]) -> str:
    for k in ("meta", "Meta", "Meta JSON", "metadata"):
        if isinstance(s.get(k), dict):
            sid = s[k].get("source_id") or s[k].get("id")
            if sid:
                return str(sid)
    tasks = s.get("tasks", {})
    if isinstance(tasks, dict):
        for tk in ("T1", "T2", "T3", "T4"):
            node = tasks.get(tk)
            if isinstance(node, dict) and node.get("id"):
                return str(node["id"])
    return f"sample_{abs(hash(json.dumps(s, sort_keys=True)))}"


# =========================
# ====== 模型调用 =========
# =========================
class ChatError(Exception):
    pass


def chat_openai_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise ChatError(f"OpenAI error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat_anthropic_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": api_key, "content-type": "application/json", "anthropic-version": "2023-06-01"}
    payload = {
        "model": model,
        "max_tokens": 1500,
        "system": "\n\n".join([m["content"] for m in messages if m["role"] == "system"]),
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise ChatError(f"Anthropic error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["content"][0]["text"]
    except Exception:
        return ""


def chat_gemini_native(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    sys_msgs = "\n\n".join([m["content"] for m in messages if m["role"] == "system"])
    user_msgs = "\n\n".join([m["content"] for m in messages if m["role"] == "user"])
    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": sys_msgs}]},
        "contents": [{"role": "user", "parts": [{"text": user_msgs}]}],
        "generation_config": {"temperature": 0},
    }
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise ChatError(f"Gemini error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def chat_openai_compat(model: str, api_key: str, base_url: str, messages: List[Dict[str, str]]) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise ChatError(f"OAI-compat error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def chat(model_name: str, messages: List[Dict[str, str]]) -> str:
    cfg = MODEL_CONFIGS.get(model_name)
    if not cfg:
        raise ChatError(f"Unknown model {model_name}")
    provider = cfg.get("provider")
    api_key = cfg.get("api_key")
    base_url = cfg.get("base_url")
    model = cfg.get("model")
    if provider == "openai_native":
        return chat_openai_native(model, api_key, base_url, messages)
    if provider == "anthropic_native":
        return chat_anthropic_native(model, api_key, base_url, messages)
    if provider == "gemini_native":
        return chat_gemini_native(model, api_key, base_url, messages)
    return chat_openai_compat(model, api_key, base_url, messages)


# =========================
# ====== T3 相关 ==========
# =========================
def build_t3_prompt(t3: Dict[str, Any]) -> str:
    base = t3.get("prompt") or t3.get("task") or ""
    pack = t3.get("pack") or []
    lines = [str(base).strip(), "", "Subtasks (answer each with one label from its label_space):"]
    for i, it in enumerate(pack):
        tid = it.get("task_id") or it.get("name") or it.get("sub_id") or f"Q{i+1}"
        name = it.get("name") or ""
        q = it.get("question") or it.get("Question") or ""
        opts = it.get("label_space") or it.get("options") or it.get("Options") or []
        if isinstance(opts, (list, tuple)):
            opts_str = ", ".join(map(str, opts))
        else:
            opts_str = str(opts)
        block = [f"- [{tid}]"]
        if name:
            block.append(f"  Name: {name}")
        if q:
            block.append(f"  Question: {q}")
        if opts_str:
            block.append(f"  label_space: {opts_str}")
        lines.append("\n".join(block))
    lines += [
        "",
        "Return ONLY JSON:",
        '1) {"answers": ["<label_for_Q1>", "<label_for_Q2>", ...]}  # order matches the subtasks listed above',
        '2) {"answers": {"<task_id>":"<label>", ...}}               # task_id keyed',
        "Each label MUST be exactly one string from its label_space. No extra text.",
    ]
    return "\n".join(lines)


# =========================
# ====== 主流程 ===========
# =========================
def main():
    random.seed(SAMPLE_SEED)
    tier_success = {"ok": 0, "tot": 0}
    tier_accuracy: List[float] = []

    for ds in ENABLED_DATASETS:
        path = DATASET_PATHS.get(ds)
        if not path:
            print(f"[WARN] dataset path not found: {ds}")
            continue
        samples = load_samples_for_dataset(path, SAMPLE_NUM, seed=SAMPLE_SEED)
        print(f"\n📂 Dataset: {ds} ({path}) | model={MODEL} | sample_num={len(samples)}")
        if not samples:
            print("⚠️ No samples loaded; skip.")
            continue

        for sample in samples:
            sid = sample_id_of(sample)
            t3 = sample.get("tasks", {}).get("T3") if isinstance(sample.get("tasks"), dict) else None
            if not isinstance(t3, dict):
                continue
            pack = t3.get("pack") or []
            if not isinstance(pack, list) or not pack:
                continue

            prompt = build_t3_prompt(t3)
            messages = [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt},
            ]
            try:
                raw = chat(MODEL, messages)
            except Exception as e:
                print(f"❌ chat error on sample {sid}: {e}")
                tier_success["tot"] += len(pack)
                time.sleep(SLEEP_BETWEEN_REQ)
                continue
            finally:
                time.sleep(SLEEP_BETWEEN_REQ)

            parsed = extract_first_json(raw) or {}
            answers_raw = parsed.get("answers")
            answers: Dict[str, Any] = {}
            if isinstance(answers_raw, list):
                for i, (item, val) in enumerate(zip(pack, answers_raw)):
                    tid = item.get("task_id") or item.get("name") or item.get("sub_id") or f"Q{i+1}"
                    answers[tid] = val
            elif isinstance(answers_raw, dict):
                answers = answers_raw

            valid_cnt = 0
            correct_cnt = 0
            label_cnt = 0
            for i, item in enumerate(pack):
                tid_keys = [
                    item.get("task_id"),
                    item.get("name"),
                    item.get("sub_id"),
                    str(i),
                    f"{i}",
                    f"Q{i+1}",
                ]
                pred = None
                for k in tid_keys:
                    if k is None:
                        continue
                    if k in answers:
                        pred = _norm(answers[k])
                        break
                opts = item.get("label_space") or item.get("options") or []
                opts_norm = {_norm(o) for o in opts} if isinstance(opts, list) else set()
                opts_norm = {o for o in opts_norm if o}
                label = _norm(item.get("label"))
                valid = False
                if pred:
                    valid = (pred in opts_norm) if opts_norm else True
                if valid:
                    valid_cnt += 1
                if label:
                    label_cnt += 1
                    if pred == label:
                        correct_cnt += 1

            tier_success["ok"] += valid_cnt
            tier_success["tot"] += len(pack)
            acc_val = (correct_cnt / label_cnt) if label_cnt else 0.0
            if label_cnt:
                tier_accuracy.append(acc_val)
            print(f"[RESULT] {sid}: success {valid_cnt}/{len(pack)}, acc {acc_val:.2%}")

    print("\n===== Overall T3 Success =====")
    ok, tot = tier_success["ok"], tier_success["tot"]
    rate = ok / tot if tot else 0.0
    print(f"T3 success: {ok}/{tot} ({rate:.2%})")
    if tier_accuracy:
        avg_acc = sum(tier_accuracy) / len(tier_accuracy)
        print(f"Avg T3 accuracy: {avg_acc:.2%} over {len(tier_accuracy)} sample(s)")
    else:
        print("No accuracy recorded (no labels found).")


if __name__ == "__main__":
    main()
