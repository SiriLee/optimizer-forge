---
name: report
description: 从 results/ 与 design_log/ 自动生成中文报告（表、图、PDF），保证每个数字可追溯；用于"写报告、出交付物"这一步。
---

# 报告生成

报告是**结果的下游**，不是独立写作。所有表格与图由脚本注入。

## 命令

```bash
.venv/bin/python -m forge.plots --raw results/raw/final_s3.jsonl   # 图
.venv/bin/python report/build.py                                    # PDF
```

## 结构（固定）

1. 引言：作业要求 → 问题重述 → 三点贡献
2. 设计空间：教材框架、7 维分解、结构量（耦合度）
3. 方法：伪代码、三条性质、与既有方法关系表
4. 理论：命题与定理，明确写出假设与失效条件
5. 实验：协议、主表、结构分析、消融、证书核对、失败案例
6. AI 设计循环：流水线、DSL、轮次表、token 预算、被否决候选
7. 讨论与局限
8. 结论

## 注入占位符

`report/report.md` 里用 `{{TABLE:main}}`、`{{TABLE:ablation}}`、`{{TABLE:structure}}`、`{{TABLE:rounds}}`、`{{FIG:f1_main_bars}}` 等，`report/build.py` 负责替换为 CSV/JSON 生成的内容与 PNG 图。

## 校验

- 生成后随机抽 3 个数字，回 CSV/JSONL 核对。
- 检查没有"未收敛"被写成具体次数。
- 检查失败案例与局限章节存在且点名了具体问题。
