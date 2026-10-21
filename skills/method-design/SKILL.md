---
name: method-design
description: 在 AIPG 的 7 维 DSL 中提出候选梯度类方法，说明动机与预期收益，并按搜索历史做增量改进；用于"提出下一个候选"这一步。
---

# 方法设计

候选必须是**数据**（JSON），不是代码补丁。这样它才能被静态闸检查、被日志记录、被离线重放。

## 输入

- 当前排行榜（`design_log/leaderboard.csv`）与被否决候选的原因（`design_log/rounds.jsonl`）。
- 失败诊断：在哪些问题上慢、慢多少、fallback 次数、证书是否被违反。

## 七个维度

| 维度 | 取值 | 典型动机 |
|---|---|---|
| precond.type | `diag` / `scalar` | 判断增益是否来自“逐坐标” |
| precond.window | 1–50 | 短窗追局部曲率，长窗更稳 |
| precond.ema | 0–1 | 1.0 = 完全信任最新 secant（易过度自信） |
| precond.adaptive | true/false | false = 固定全局区间，作为对照组 |
| step.rule | `minimax` / `invu` | `2/(u+ℓ)` vs `1/u` |
| accel.type | `none` / `chebyshev` | 动量在耦合/非凸问题上买稳健性 |
| safeguard.type | `armijo` / `none` | 无保护只允许配 `invu`+`none`（静态闸会拒绝其它组合） |

## 输出格式

```json
{"proposer": "llm:deepseek", "tokens": {"prompt": 800, "completion": 220},
 "notes": "假设：长窗口降低区间噪声，预期在耦合问题上更稳",
 "candidate": {"precond": {"window": 20, "ema": 0.2}, "step": {"rule": "minimax"},
               "accel": {"type": "chebyshev"}, "safeguard": {"type": "armijo"}}}
```

追加到 `design_log/proposals.jsonl`，每行一个。

## 规矩

- 一次只改一到两个维度，并在 `notes` 写清"改什么、为什么、预期在哪类问题上变好或变差"。
- 不要提出无法被否决的候选（例如同时改 5 个维度、或没有明确假设的随机组合）。
- 提议被闸门拒绝是正常结果；拒绝原因必须留在日志里，不要悄悄删掉。
