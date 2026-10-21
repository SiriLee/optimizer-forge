# Optimizer Forge · AIPG

《面向机器学习的最优化方法》课程作业 10–12 的交付物：在 AI 辅助下设计一个新的梯度类方法，并用实验验证。

- **方法**：AIPG（Adaptive Interval-Preconditioned Gradient）——逐坐标用梯度差分估计曲率**区间**，步长取"最坏情况收缩因子最小"的闭式解 $2/(u+\ell)$，叠加 Chebyshev 三项递推与 Armijo 保护。
- **工程**：Optimizer Forge——把"提出候选 → 静态闸 → 理论闸 → 实验闸 → 记录"做成可离线重放的流水线。
- **报告**：[`report/report.pdf`](report/report.pdf)（13 页，中文）。

## 结果一览

达到 $f-f^\star\le10^{-10}$ 所需梯度调用次数（3 个种子的中位数，"—" 表示在预算内未收敛（凸问题 40000 次迭代））：

| 问题 | 结构 | AIPG | NAG | Adam | L-BFGS |
|---|---|---|---|---|---|
| P1 对角二次（κ=10⁴） | 跨度 10⁴ | **44** | 1165 | 490 | 628 |
| P2 旋转二次（κ=10³） | 跨度 2.1 | 1084 | **369** | — | 209 |
| P3 i.i.d. 最小二乘 | 跨度 9826 | **111** | 1232 | 4006 | 475 |
| P3 Toeplitz 最小二乘 | 跨度 10⁴ | **5993** | — | 7027 | 8883 |
| P4 逻辑回归 | 跨度 900 | **76** | 339 | 766 | 103 |
| P5 Rosenbrock | 非凸 | 2578 | 10254 | 5127 | **44** |

一句话结论：**AIPG 的收益来自坐标尺度差异（对角跨度 900–10⁴ 时快 4–26 倍，跨度 2.1 时退化）；耦合度只决定同一族问题内部打几折（按加速比看约 1.5 倍）。逐点收缩证书在实际配置下不成立，这一点我们在报告里作为负结果报告。**

## 快速开始

```bash
# 报告管线需要系统库 pango / cairo（如未安装：sudo apt install libpango-1.0-0 libcairo2）
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python tests/test_smoke.py                       # 冒烟测试
.venv/bin/python -m forge.bench --suite full --seeds 3     # 主实验（约 80 秒）
.venv/bin/python -m forge.search --proposer recorded       # 设计循环（离线重放）
.venv/bin/python -m forge.plots --raw results/raw/final_s3.jsonl
.venv/bin/python report/build.py                           # 生成 report/report.pdf
```

一次跑完：`bash scripts/make_all.sh`。

## 目录

| 路径 | 内容 |
|---|---|
| `forge/` | 问题集、方法（AIPG + 8 个基线）、DSL、闸门、benchmark、搜索、绘图 |
| `skills/` | 供 Agent 复用的 skill：方法设计 / 理论校验 / 评测 / 报告 |
| `design_log/` | 12 轮候选、否决理由、token 记账、胜者 |
| `results/` | 原始日志、汇总表、图、证书数据、oracle 实验 |
| `report/` | 报告源文件（Markdown）与生成的 PDF |
| `tests/` | 4 个回归测试 |

## 设计空间

梯度类方法被参数化为 7 个维度：预条件结构、曲率估计、区间规则、步长、加速、保护、噪声处理。候选是纯数据（DSL），因此可以先于运行被检查，也可以被离线重放：

```json
{"precond": {"type": "diag", "window": 20, "ema": 0.2, "adaptive": true},
 "step":    {"rule": "minimax"},
 "accel":   {"type": "chebyshev", "beta_cap": 0.9},
 "safeguard": {"type": "armijo", "c": 1e-4, "halvings": 25}}
```

## 理论承诺

| 编号 | 内容 | 状态 |
|---|---|---|
| 命题 1 | $\alpha^\star=2/(u+\ell)$ 是区间上的最小最大最优步长 | 已证 |
| 定理 1 | 对角二次下的逐坐标收缩 $(\kappa-1)/(\kappa+1)$，加速时 $(\sqrt\kappa-1)/(\sqrt\kappa+1)$ | 已证 |
| 定理 2 | 一般凸 + 保护：单调下降与 $O(1/\sqrt K)$ 间隙界 | 已证 |
| 定理 3 | 强凸：线性收敛（常数很保守） | 已证 |
| 负结果 | 实用配置（区间平滑 + 动量）下逐点收缩证书失效，只剩单调性 | 已报告 |
| 边界 | 尺度跨度接近 1 时退化为 GD；旋转耦合下精确对角预条件会发散 | 用 oracle 实验证实，未证 |

## 复现约定

- 随机性由 `numpy.random.default_rng(seed)` 固定。
- 报告里每个数字都能追溯到 `results/raw/final_s3.jsonl`、`results/tables/final_s3_summary.csv` 或 `design_log/rounds.jsonl`。
- `--proposer recorded` 不需要任何 API key。
