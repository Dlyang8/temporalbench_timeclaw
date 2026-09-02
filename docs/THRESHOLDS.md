# TemporalBench — 阈值与"错误"判定 (Thresholds & Error Definitions)

本文件集中整理 TemporalBench 中所有**基于数值阈值的判定逻辑**，分为两类：

1. **评测端阈值（Evaluation-side）** —— 预测结果"超出某个范围就判为错误/异常"的逻辑，定义在 LLM 评测脚本里。
2. **标注端阈值（Label-construction-side）** —— 构造 ground-truth 标签时，用来把连续量切成离散选项（up/down/flat、Yes/No/Uncertain 等）的"bar / band"，定义在各数据集的任务构造脚本里。

> 第 1 类决定"模型答得对不对/预测是否被判废"，第 2 类决定"正确答案本身是怎么算出来的"。两者都属于 benchmark 独有的判定口径，必须连同数据一起记录，否则结果不可复现。

---

## 1. 评测端阈值：预测超范围 → 判为异常

文件：[`evaluation/evaluate_llm.py`](../evaluation/evaluate_llm.py)

### 1.1 预测指标上限 `FORECAST_METRIC_THRESHOLDS`（约 line 132–141）

适用于 T2 / T4 的**预测（forecast）子任务**。当某条预测算出的误差指标超过下表上限时，这条预测被标记为 `metric_threshold_<KEY>`（视为异常/不计入有效预测）：

| 指标 | 上限值 | 含义 | 适用 |
| --- | --- | --- | --- |
| `MAPE` | `1e4` (=10000%) | 平均绝对百分比误差 | 单序列 |
| `MAE` | `5.0` | 平均绝对误差（归一化后） | 单序列 |
| `RMSE` | `5.0` | 均方根误差（归一化后） | 单序列 |
| `SMAPE` | `200.0` | 对称 MAPE（%） | 单序列 |
| `OW_sMAPE` | `200.0` | MIMIC 多通道 owsMAPE（%） | MIMIC |
| `OW_RMSSE` | `10.0` | MIMIC 多通道相对标准化误差 | MIMIC |
| `OW_MASE` | `10.0` | MIMIC 多通道平均绝对标准化误差 | MIMIC |

### 1.2 判定逻辑 `_check_metric_thresholds()`（约 line 568–588）

```python
for key, limit in FORECAST_METRIC_THRESHOLDS.items():
    ...
    if key == "MAPE":
        # 特例：若 MAPE 很大但 SMAPE 在其上限内，则忽略 MAPE 超限
        smape_key = "SMAPE" if "SMAPE" in metrics else "OW_sMAPE"
        if smape_val <= smape_limit:
            continue
    if abs(val) > limit:
        return f"metric_threshold_{key}"   # ← “超出范围 = 错误”
```

要点：
- 判废条件是 **`abs(metric) > limit`**。
- **MAPE 特例**：MAPE 对接近 0 的真值极不稳定，因此若 `SMAPE`（或 MIMIC 的 `OW_sMAPE`）仍在各自上限内，则不因 MAPE 超限而判废。
- 这些上限可调：直接改 `FORECAST_METRIC_THRESHOLDS` 字典即可。

### 1.3 结构性错误（非阈值，但也算"判错"）

预测在进入指标计算前会先做对齐/合法性检查，失败则直接返回错误标签：

| 错误标签 | 触发条件 | 位置 |
| --- | --- | --- |
| `length_mismatch` | 预测长度 ≠ ground-truth 长度 | line ~621 / ~642 |
| `no_valid_values` | 过滤非有限值后无可用配对 | line ~645 |
| `missing_channel` | MIMIC 多通道里缺某个通道 | line ~617 |
| `invalid_format` | 通道预测不是 list/tuple | — |
| `series_mismatch` | 多通道序列对不齐 | line ~624 |

### 1.4 MCQ 子任务的"错误"分类（T1/T2/T3/T4 的选择题部分）

来源：`_eval_mcq_block()`（约 line 1830–1849）与 T3 pack 评分（约 line 2090–2104）。

| 类别 | 含义 |
| --- | --- |
| `missing_answer` | 模型没有给出可解析的答案 |
| `invalid_option` | 答案不在该题的合法选项集合内 |
| 判对条件 | 归一化（小写、去空格）后 `pred == gt` |

### 1.5 数值稳定常量

`EPS = 1e-12`、`METRIC_NORM_EPS = 1e-8`、`OW_SMAPE_EPS = 1e-8`、`METRIC_NORM_MODE = "mean_abs"`（多通道指标按 `mean(|gt|)` 归一化，可选 `std`/`range`/`none`）。

---

## 2. 标注端阈值：ground-truth 标签是怎么算出来的

这些 bar / band 定义在 **各数据集的任务构造脚本**里，决定连续统计量映射到哪个离散选项。位置：`construction/<dataset>/task_t1.py`、`task_t3.py`、`task_common.py`。

> 说明：T3 的逐子任务 bar 在 `docs/TASK_DESIGN.md` 的 "Auto-label summary" 列里有逐条文字描述（如 ≥+8% → Yes、±4% → Uncertain）。本节给出代码里的**模块级常量**，作为权威数值来源。

### 2.1 T1（趋势 / 波动 / 季节 / 离群）标签 bar

| 数据集 | Trend（slope/rel）| Volatility（MAD 比值）| 其他 | 文件:行 |
| --- | --- | --- | --- | --- |
| causal_chambers | rel ≥ +0.05 → up；≤ −0.05 → down；\|rel\| ≤ 0.02 → flat | ≥1.3 inc / ≤0.8 dec / [0.95,1.05] const | level_shift ≥ 0.05 | task_t1.py:198–239 |
| PSML | 归一化 slope thr = 0.01 | ≥1.25 inc / ≤0.80 dec | seasonality amp_bar = 0.10×scale | task_t1.py:251–263 |
| MIMIC | slope ≥ 0.01 up / ≤ −0.01 down | ≥1.3 inc / ≤0.75 dec | amp < 0.05×base → 无季节 | task_t1.py:271–281 |
| freshretailnet | thr = 0.01×scale | ≥1.25 inc / ≤0.8 dec | — | task_t1.py:130–135 |

### 2.2 T3（情境理解）效应 bar / 不确定带

| 数据集 | 关键常量 | 文件:行 |
| --- | --- | --- |
| PSML | `EFFECT_BAR = 0.08`（~8% 视为显著）、`UNCERT_BAND = 0.04`（±4% 视为无明显变化）、相关性 `rho ≥ 0.35`、`SCORE_BAR = 0.10` | task_t3.py:34–35, 281, 1256 |
| MIMIC | `REL_EFFECT_BAR = 0.05`、`UNCERT_BAR = 0.02`、`TREND_SLOPE_BAR = 0.3`、`PEAK_THRESHOLD_FACTOR = 0.9`、`FEVER_THRESHOLD = 37.2`（°C） | task_t3.py:24–32 |
| freshretailnet | `T3_EFFECT_YES = 0.12`、`T3_UNCERT_MARGIN = 0.05`、`T3_SCORE_BAR = 0.10`，以及随样本量自适应的 `_bar_by_n(base=0.05, floor=0.03)` | task_t3.py:82–92, 236 |
| causal_chambers | 各子任务用自适应 `bar`/`unc_band`（无模块级常量，逻辑见 `docs/TASK_DESIGN.md` 中 causal 表） | task_t3.py |

> 自适应 bar `_bar_by_n(n, base, floor)`：样本越少要求越严，避免小样本上凑出"显著"结论。

---

## 3. 第五个子数据集 M5 的标注阈值（新增）

M5 是新并入的第五个子数据集，T1 标签阈值是**经网格搜索后选定**的（最大化标签分布均衡），与前四个数据集的固定 bar 口径略有不同。权威来源：`m5_final_summary.json` 的 `recommended_t1_thresholds`（M5 原始工作区，未随精简版打包）。

| 维度 | 选定参数 |
| --- | --- |
| trend | `smooth_win = 21`，`up_bar = 0.002`，`const_bar = 0.001` |
| volatility | `inc_bar = 1.15`，`dec_bar = 0.85`，`const_low = 0.9`，`const_high = 1.1` |
| seasonality | `amp_none = 0.08`，`fixed_cos = 0.9`，`shift_cos = 0.8` |
| outliers | `spike_bar = 5.0`，`shift_bar = 0.1`，`stable_spike = 4.5`，`stable_shift = 0.08` |

> 注意：M5 以官方**公开 50 条**切分（`full_v3_refined_75_public_50`）并入 `data/M5/task_modified.json`（已是构造好的成品，是 100 条全集的子集）。构造脚本、阈值网格搜索、其它变体（ctx_*/v2/private_25）保留在原 `TS-benchmark-plus/m5/` 工作区，未复制进精简版。评测端阈值（第 1 节）对 M5 与其他四个数据集**完全一致**。

---

## 4. 新旧 benchmark 阈值差异速查

| 方面 | 旧版（task_high_score.json，前四数据集 fixed bar） | 新版（task_modified.json + M5） |
| --- | --- | --- |
| T1 trend/vol bar | 各数据集硬编码（见 §2.1） | 前四数据集不变；M5 用网格搜索选定值（§3） |
| 评测预测判废 | `FORECAST_METRIC_THRESHOLDS`（§1.1），口径相同 | 相同（M5 复用同一套上限） |
| 样本量 | causal 42 / 等 | causal 50 / PSML 50 / MIMIC 47 / fresh 44 / **M5 50（公开集）** |

调阈值时只需改：评测端 → `evaluation/evaluate_llm.py` 的 `FORECAST_METRIC_THRESHOLDS`；标注端 → 对应 `construction/<dataset>/task_t*.py` 顶部常量后重新构造。
