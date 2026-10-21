# Optimizer Forge

《面向机器学习的最优化方法》作业 10-12：在 AI 辅助下设计一个新的梯度类方法，并用实验验证。

- **方法**：AIPG（自适应区间预条件梯度法）——逐坐标用梯度差分估计曲率区间，步长取最小最大意义下的最优解。
- **工程**：Optimizer Forge — 提出候选 → 静态闸 → 理论闸 → 实验闸 → 记录，全流程可复现。
- **报告**：`report/report.pdf`

完整设计见 [SPEC.md](SPEC.md)。

## 环境

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`（可选 `matplotlib` 用于出图，`sympy` 用于符号校验）。

## 快速开始

```bash
# 冒烟：单问题、单种子、少量迭代
python -m forge.bench --smoke

# 全量实验（约 20–30 分钟，20 核）
python -m forge.bench --full --seeds 7

# 重放 AI 设计搜索（离线，无需 API key）
python -m forge.search --replay design_log/rounds.jsonl

# 重建全部图表与报告
bash scripts/make_all.sh
```

## 目录

| 路径 | 内容 |
|---|---|
| `forge/` | 问题集、方法、闸门、benchmark、搜索、绘图 |
| `skills/` | 供 Coding Agent 复用的 skill（设计 / 验证 / 评测 / 报告） |
| `design_log/` | 每轮候选、否决理由、token 记账 |
| `results/` | 原始日志、汇总表、图 |
| `report/` | LaTeX 源码与 PDF |

## 复现约定

- 随机性全部由 `numpy.random.SeedSequence` 固定。
- 每个数字都能追溯到 `results/raw/*.jsonl`。
- `--proposer recorded` 为默认，离线可重放；`--proposer llm` 需要 `OPENAI_API_KEY`（OpenAI 兼容端点）。
