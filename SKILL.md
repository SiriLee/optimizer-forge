---
name: optimizer-forge
description: 在"梯度类方法"的设计空间里提出、校验并评测新方法（AIPG 及其变体），产出可复现的实验表、图与中文报告；用于课程作业式的算法设计-验证-报告一体化任务。
---

# Optimizer Forge

把"设计一个新梯度类方法"当作可验证的搜索问题：候选是数据（DSL），闸门负责否决，实验负责排序，报告由结果自动生成。

## 何时使用

- 需要在一个明确定义的设计空间里做算法变体搜索，并要求每个结论可追溯到原始日志。
- 需要"理论条件"与"实验结果"互相约束，而不是先写故事再补数据。
- 需要把过程（提议、否决、预算）本身作为交付物的一部分。

## 流水线

```
Brief ──► Propose ──► 静态闸 ──► 理论闸 ──► 实验闸 ──► Report
        (LLM/网格)   DSL 合法性   R1–R6 不变量  冒烟→全量    表/图/PDF
```

## 命令

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m forge.bench --suite full --seeds 3 --tag final_s3
.venv/bin/python -m forge.search --proposer recorded --rounds 12
.venv/bin/python -m forge.plots --raw results/raw/final_s3.jsonl
.venv/bin/python report/build.py
```

一次跑完：`bash scripts/make_all.sh`。

## 交付标准

- `results/raw/*.jsonl` 里每个 run 都能定位到问题、方法、种子与预算。
- 报告中每个数字都能在 `results/tables/*.csv` 或 `design_log/rounds.jsonl` 找到。
- 任何被报告的方法版本都必须先通过 `forge.verify` 的 R1–R6；未通过的候选只允许出现在“被否决”表里。
- 结论必须写明失效边界；不允许只报胜出的问题。

## 组成

| 组件 | 职责 | 文件 |
|---|---|---|
| 设计 | 在 DSL 里提候选、解释理由 | `skills/method-design/SKILL.md` |
| 校验 | 静态/理论闸，给出否决理由 | `skills/theory-verify/SKILL.md` |
| 评测 | 公平计数器、协议、失败统计 | `skills/benchmark/SKILL.md` |
| 报告 | 从结果生成表、图、PDF | `skills/report/SKILL.md` |
