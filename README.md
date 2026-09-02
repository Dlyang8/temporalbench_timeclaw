# TemporalBench

一个面向**大语言模型（LLM）时间序列理解与预测能力**的基准。每条样本提供一段历史时间序列（及对齐的辅助变量/事件上下文），配套 **T1–T4 四类任务**，覆盖"理解 vs 预测"和"无情境 vs 有情境"四个象限。

本目录是从分散的多个工作区**整理后的精简、可分享版本**，包含：5 个子数据集的成品 benchmark 数据、任务构造脚本、干净的 LLM 评测代码、任务设计文档，以及所有阈值/错误判定口径的说明。

---

## 1. 任务设计（T1–T4）

|                | 理解任务 (understanding) | 预测任务 (prediction) |
| -------------- | --- | --- |
| **无情境** Non-contextual | **T1** | **T2** |
| **有情境** Contextual | **T3** | **T4** |

- **T1** —— 无情境理解：对历史序列回答趋势/波动/季节性/离群的选择题（MCQ）。
- **T2** —— 无情境预测：给定历史预测未来；含预测序列 + 关于变化的 MCQ。
- **T3** —— 有情境理解：给定背景/字段说明，对历史序列做更复杂的情境推理（一个样本一"包" pack 的多道 MCQ，覆盖 C1–C6 能力）。
- **T4** —— 有情境预测：在历史 + 未来协变量 + 事件上下文下预测未来；含预测序列 + MCQ。

完整的任务族（S1–S6 / EVT…）、每个子任务的问法、选项、自动标注口径见 **[docs/TASK_DESIGN.md](docs/TASK_DESIGN.md)**。

---

## 2. 子数据集（5 个）

| 子数据集 | 领域 | 样本数 | 事件来源 |
| --- | --- | --- | --- |
| `freshretailnet` | 生鲜零售需求（含缺货删失） | 44 | 注入事件（storm / promotion / holiday） |
| `PSML` | 电网负荷（多尺度能源） | 50 | 注入事件（heatwave / cold_snap / storm / eclipse…） |
| `MIMIC` | ICU 生命体征（HR/Temp/RR/SpO₂/SBP/DBP） | 47 | 真实事件（用药 / 操作 / 转入转出） |
| `causal_chambers` | 物理实验台（风机负载） | 50 | 真实实验操作事件 |
| **`M5`** *(第五个，新并入)* | Walmart 零售销量 | 50 | 注入事件（holiday / price_drop / price_rise / snap_start） |

每个子数据集就是一个文件：`data/<dataset>/task_modified.json`，结构统一：

```jsonc
[
  {
    "dataset":   "M5",
    "sample_id": "...",
    "meta":      { "history_len": 112, "future_len": 28, ... },
    "tasks":     { "T1": {...}, "T2": {...}, "T3": {...}, "T4": {...} }
  },
  ...
]
```

> `task_modified.json` 是各数据集的**最终成品**（旧的 `task_high_score.json` 是其早期子集，例如 causal 42 → 50 条；本仓库统一采用 `task_modified.json`）。

---

## 3. 目录结构

```
TemporalBench/
├── README.md                  ← 本文件
├── data/                      ← 成品 benchmark 数据（5 个子数据集，直接可用）
│   ├── freshretailnet/task_modified.json
│   ├── PSML/task_modified.json
│   ├── MIMIC/task_modified.json
│   ├── causal_chambers/task_modified.json
│   └── M5/task_modified.json
├── construction/              ← 任务构造脚本（前四数据集；展示 benchmark 如何生成）
│   ├── <dataset>/build_task*.py, task_t1..t4.py, task_common.py,
│   │            event_injection.py, modify_data.py, read_data.py,
│   │            preview_sample.py, raw_dataset/(部分)
│   └── ...
├── evaluation/                ← 干净的 LLM 评测代码（无任何硬编码密钥）
│   ├── evaluate_llm.py            主入口（T1–T4 全流程）
│   ├── forecast_metrics_utils.py  预测指标工具
│   ├── evaluate_llm_split.py / _length_study.py / _t3_capabilities*.py  专项实验
│   ├── count_prompt_tokens_final.py / parse_mcq_accuracy_from_csv.py / inspect_psml_dataset.py
│   ├── rerun_missing_eval.py      补跑缺失样本
│   └── run_benchmark.sbatch       Slurm 提交脚本
└── docs/
    ├── TASK_DESIGN.md         ← 任务/子任务/能力(C1–C6)完整设计文档（含示例与配图）
    ├── THRESHOLDS.md          ← 所有阈值与“错误/异常”判定口径（重点阅读）
    └── imgs/                  ← 文档配图
```

---

## 4. 如何评测 LLM

评测脚本无 argparse，**配置写在 `evaluation/evaluate_llm.py` 顶部**：

1. 选数据集 / 任务 / 模型：编辑 `ENABLED_DATASETS`、`ENABLED_TIERS`、`ENABLED_MODELS`。
   - `DATASET_PATHS` 已默认指向本仓库 `data/<dataset>/task_modified.json`（相对脚本位置解析，含 `M5`）。
2. 导出所选模型的 API key（脚本只从环境变量读取，**不含任何硬编码密钥**）：
   ```bash
   export OPENAI_API_KEY=...      # gpt-4o / gpt-5.4-mini
   export ANTHROPIC_API_KEY=...   # claude
   export GEMINI_API_KEY=...      # gemini-*
   export DEEPSEEK_API_KEY=...    # deepseek-chat
   # 还支持 XAI_API_KEY / QWEN_API_KEY / MISTRAL_API_KEY
   ```
3. 运行：
   ```bash
   cd evaluation
   python evaluate_llm.py
   # 或 Slurm:  sbatch run_benchmark.sbatch
   ```

输出（写在运行目录）：
- `eval_results_detail_<MODEL>_<DATASET>_*.json` —— 每条样本的细节
- `eval_results_summary_<MODEL>_<DATASET>_*.json` / `.csv` —— 按 数据集×任务×子任务 聚合

**评分口径**：MCQ → 准确率；预测 → MAPE/MAE/RMSE/SMAPE（MIMIC 多通道用 OW_sMAPE/OW_RMSSE/OW_MASE）。
预测误差超过上限会被判为异常 —— 详见 **[docs/THRESHOLDS.md](docs/THRESHOLDS.md)**。

---

## 5. 阈值与"错误"判定（务必阅读）

benchmark 有两套独有的数值口径，全部整理在 **[docs/THRESHOLDS.md](docs/THRESHOLDS.md)**：

1. **评测端**：预测误差超过 `FORECAST_METRIC_THRESHOLDS`（如 SMAPE>200%、MAE>5.0、MAPE>10000%）即判为异常；外加长度不匹配、缺通道、选项非法等结构性错误。
2. **标注端**：构造 ground-truth 时把连续量切成离散选项的 bar/band（如 T1 趋势 slope、波动 MAD 比值；T3 的 ±8%/±4% 效应带）。M5 的 T1 阈值经网格搜索单独选定，文档中有专门小节。

---

## 6. 整理说明 / 已知取舍

- **数据来源**：`data/` 的前四个 `task_modified.json` 来自 `TS-benchmark/new_version/`；`M5` 采用 `TS-benchmark-plus/m5/` 的官方**公开 50 条**切分 `variants/public_private_split/task_modified_full_v3_refined_75_public_50.json`（即 full 上下文、v3、refined_75 的 public_50；是 100 条全集的子集，agent_eval 默认就用它）。所有文件经 md5 校验复制，原始工作区保持不动。
- **M5 仅以成品数据并入（50 条公开集）**：未复制 M5 的构造脚本 / 其它变体（ctx_v*/full_v2/private_25）/ agent_eval 等中间产物；它们仍在原 `TS-benchmark-plus/m5/`。若日后需要 100 条全集或 private_25，可从该工作区取。
- **评测代码取干净版**：采用原 `evaluate_llm_clean.py`（重命名为 `evaluate_llm.py`，便于其余脚本 `from evaluate_llm import ...`）。原 `evaluate_llm.py`、`count_prompt_tokens.py` 含**硬编码 API 密钥，已刻意排除**；本目录已扫描确认无任何密钥。
- **`construction/` 排除了 `results/`**（约 750MB 的实验输出/中间产物）与 `__pycache__`、`.DS_Store`，保留构造脚本与 `raw_dataset`。
- **PSML / MIMIC 无独立 `raw_dataset/`**：其源数据原先位于被排除的 `results/` 下或需从外部下载（PhysioNet / Zenodo）；构造脚本 `read_data.py`/`modify_data.py` 说明了来源。`task_modified.json` 自包含，不依赖原始 CSV 即可评测。
