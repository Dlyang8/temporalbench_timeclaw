#!/usr/bin/env python3
"""
Count token lengths of the *final user prompts* exactly as constructed in
evaluate_llm.py (per tier), using the same dataset paths and prompt builders.
We only count the user content (system/tool messages不计)。

T2/T4：仅统计 mcq 提示（forecast 提示忽略）。
T3：按 pack 中每个子任务的单题 prompt 计数后求平均。
输出格式与先前类似：
  model_<name>_<dataset>
  <T1_avg_tokens>
  <T2_avg_tokens>   # mcq-only
  <T3_avg_tokens>
  <T4_avg_tokens>   # mcq-only
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

# 若缺 numpy，注册简易 stub 以便导入 evaluate_llm（只为取 prompt 构造函数）
try:
    import numpy  # type: ignore
except ImportError:
    class _NpStub(types.SimpleNamespace):
        ndarray = object
        def __getattr__(self, name):
            if name in ("asarray", "array"):
                return lambda x, *a, **k: x
            if name in ("isfinite",):
                import math
                return lambda x: math.isfinite(x) if isinstance(x, (int, float)) else False
            return None
    sys.modules["numpy"] = _NpStub()

# 直接引用 evaluate_llm.py 配置/构建函数
from evaluate_llm import DATASET_PATHS, MODEL_CONFIGS, T1_RETURN_INSTRUCTION, build_t2_prompts, build_t3_single_prompt, build_t4_prompts

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "prompt_token_stats_final.log"
# 只跑 gpt-4o（与生产配置一致，如需更多可扩展此列表）
ENABLED_MODELS = ["gpt-4o"]


def _sanitize_name(name: str) -> str:
    import re
    n = (name or "").lower()
    n = re.sub(r"[^a-z0-9]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n


SANITIZED_DATASET_MAP = { _sanitize_name(k): k for k in DATASET_PATHS.keys() }
# 常见别名补充
SANITIZED_DATASET_MAP.update({
    "causal_chambers": "CausalChambers",
    "freshretailnet": "FreshRetailNet",
    "psml": "PSML",
    "mimic": "MIMIC",
})


def load_tokenizer(model_name: str) -> Callable[[str], int]:
    model_id = MODEL_CONFIGS.get(model_name, {}).get("model", model_name)
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_id)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        return lambda text: len(enc.encode(text))
    except Exception:
        print("[WARN] tiktoken not available; using whitespace token count (rough).")
        return lambda text: len(text.split())


def _collect_files(path_str: str) -> List[Path]:
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("task_modified*.json"))
        if not files:
            files = sorted(p.glob("*.json"))
        return files
    return sorted(Path().glob(path_str))


def iter_user_prompts(sample: dict) -> Iterable[Tuple[str, str]]:
    """Yield (tier, user_prompt_text) for this sample, using evaluate_llm builders."""
    ds_raw = sample.get("dataset") or sample.get("meta", {}).get("dataset") or "unknown"
    ds_key = SANITIZED_DATASET_MAP.get(_sanitize_name(ds_raw), ds_raw)
    tasks = sample.get("tasks") or {}

    # T1
    if "T1" in tasks:
        t1 = tasks["T1"]
        prompt = t1.get("prompt") or t1.get("task")
        if prompt:
            yield ("T1", f"{prompt}\n\n{T1_RETURN_INSTRUCTION}")

    # T2 (mcq-only)
    if "T2" in tasks:
        t2 = tasks["T2"]
        try:
            prompts = build_t2_prompts(t2, sample_meta=sample.get("meta"))
            mcq_prompt = prompts.get("mcq_prompt")
            if mcq_prompt:
                yield ("T2", mcq_prompt)
        except Exception as exc:
            print(f"[WARN] build_t2_prompts failed for dataset={ds_key}: {exc}")

    # T3 (per subtask)
    if "T3" in tasks:
        t3 = tasks["T3"]
        pack = t3.get("pack") or []
        for i, it in enumerate(pack):
            try:
                sp = build_t3_single_prompt(t3, it, i)
                if sp:
                    yield ("T3", sp)
            except Exception as exc:
                print(f"[WARN] build_t3_single_prompt failed idx={i} ds={ds_key}: {exc}")

    # T4 (mcq-only)
    if "T4" in tasks:
        t4 = tasks["T4"]
        try:
            prompts = build_t4_prompts(t4, sample_meta=sample.get("meta"))
            mcq_prompt = prompts.get("mcq_prompt")
            if mcq_prompt:
                yield ("T4", mcq_prompt)
        except Exception as exc:
            print(f"[WARN] build_t4_prompts failed for dataset={ds_key}: {exc}")


def aggregate(file_paths: List[Path], count_fn: Callable[[str], int]) -> Dict[Tuple[str, str], List[int]]:
    stats: Dict[Tuple[str, str], List[int]] = {}
    for fp in file_paths:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read {fp}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            for tier, prompt in iter_user_prompts(rec):
                ds_raw = rec.get("dataset") or rec.get("meta", {}).get("dataset") or "unknown"
                ds_key = SANITIZED_DATASET_MAP.get(_sanitize_name(ds_raw), ds_raw)
                key = (ds_key, tier)
                stats.setdefault(key, []).append(count_fn(prompt))
    return stats


def main() -> int:
    lines: List[str] = []
    for model_name in ENABLED_MODELS:
        count_fn = load_tokenizer(model_name)
        for ds_name, path_str in DATASET_PATHS.items():
            files = _collect_files(path_str)
            if not files:
                print(f"[WARN] No files for dataset {ds_name} at {path_str}")
                continue
            stats = aggregate(files, count_fn)
            lines.append(f"model_{model_name}_{ds_name}")
            for tier in ("T1", "T2", "T3", "T4"):
                arr = stats.get((ds_name, tier), [])
                mean = sum(arr) / len(arr) if arr else 0.0
                lines.append(f"{mean:.2f}")

    DEFAULT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {DEFAULT_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
