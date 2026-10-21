---
name: benchmark
description: 按公平协议评测梯度类方法：统一计数器、统一种子与停止判据、失败也计入；用于"跑实验、出主表"这一步。
---

# 评测协议

## 公平性五条

1. **主指标是梯度调用次数**，不是迭代数（避免"每步更便宜"的耍赖）。
2. 所有方法共享同一初值、同一光滑常数估计 $\hat L$、同一停止判据。
3. 未收敛不剔除，记入失败率；不挑最好种子。
4. 每个 (问题, 方法) 至少 3 个种子，报中位数与四分位。
5. 线搜索的额外函数求值单独计数，不混进梯度计数。

## 命令

```bash
.venv/bin/python -m forge.bench --smoke                       # 30 秒冒烟
.venv/bin/python -m forge.bench --suite full --seeds 7 --tag final_s7
.venv/bin/python -m forge.bench --suite P1 --methods gd,aipg --seeds 2
```

## 输出

- `results/raw/<tag>.jsonl`：每个 run 一行，含 trace（降采样）、gap、gnorm、计数、耗时、fallback。
- `results/tables/<tag>_summary.csv`：按 (问题, 方法) 聚合的中位数、IQR、成功率。

## 检查清单

- 失败率 > 0 的问题，报告里必须点名，并说明是"方法不行"还是"预算不够"。
- 若某方法在预算内全部失败，主表用"—"表示，不要用预算值冒充成绩。
- 结构量（条件数、耦合度）随结果一起记录，便于事后解释而不是事后找补。
