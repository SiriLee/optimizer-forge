#!/usr/bin/env bash
# Build the submission package: report PDF + source archive + email draft.
# usage: bash scripts/make_submission.sh <学号> <姓名>
set -euo pipefail
cd "$(dirname "$0")/.."

SID="${1:?usage: make_submission.sh <学号> <姓名>}"
NAME="${2:?usage: make_submission.sh <学号> <姓名>}"
OUT="dist/作业10-12-${SID}-${NAME}"
mkdir -p "$OUT"

cp report/report.pdf "$OUT/作业10-12-${SID}-${NAME}-报告.pdf"
git archive --format=zip --prefix=optimizer-forge/ -o "$OUT/optimizer-forge-源码.zip" HEAD

cat > "$OUT/邮件正文.txt" <<TXT
主题：作业10-12-${SID}-${NAME}-最优化方法

老师您好：

附件为《面向机器学习的最优化方法》作业 10-12 的提交材料。

一、作业内容
针对梯度类方法，设计了一个新的梯度类方法 AIPG（自适应区间预条件梯度法），
并用实验验证其有效性。交付形式为 harness 工程 + skill set + 报告。

二、方法一句话
逐坐标用梯度差分估计曲率区间 [ℓ_i, u_i]，步长取最坏情况收缩因子最小的闭式解
α_i = 2/(u_i+ℓ_i)，叠加 Chebyshev 三项递推与 Armijo 保护。

三、主要结果
在 6 个问题、9 个基线、3 个随机种子上测试（指标为达到 f−f* ≤ 1e-10 的梯度调用次数）：
  · 对角二次（κ=1e4）：44 次，NAG 为 1165 次（约 1/26）
  · i.i.d. 最小二乘：111 次，NAG 为 1232 次（约 1/11）
  · 逻辑回归：76 次，NAG 为 339 次（约 1/4.5）
  · 旋转二次：1084 次，慢于 NAG 的 369 次（失效边界，报告中已如实分析）
  · 报告还给出一个可测的结构量（非对角耦合度）来解释"何时有效、何时失效"。

四、交付物
  1. 报告 PDF（12 页，中文）
  2. 源码压缩包（含 harness 工程、skill set、实验脚本与结果）
     · 一键复现：bash scripts/make_all.sh
     · 不需要任何 API key（设计循环支持离线重放）

五、仓库
https://github.com/SiriLee/optimizer-forge

如有需要补充的材料，请随时联系。

此致
敬礼

${NAME}
${SID}
TXT

echo "submission → $OUT"
ls -lh "$OUT"
