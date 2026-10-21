# 设计日志（自动生成）

- 提出候选：12；通过闸门：7；最终候选：round 3

| 轮次 | 候选 | 静态闸 | 理论闸 | 冒烟分数 | 全量分数 | 决策 |
|---|---|---|---|---|---|---|
| 1 | `diag/w5/ema0.5 + minimax + chebyshev(beta<=0.9) + armijo` | pass | True | 37.11 | 1.61 | reject |
| 2 | `diag/w1/ema1.0 + minimax + chebyshev(beta<=0.9) + armijo` | pass | False | nan | nan | reject |
| 3 | `diag/w20/ema0.2 + minimax + chebyshev(beta<=0.9) + armijo` | pass | True | 69.73 | 1.39 | promote |
| 4 | `scalar/w5/ema0.5 + minimax + chebyshev(beta<=0.9) + armijo` | pass | True | 21933.99 | nan | reject |
| 5 | `diag/w5/ema0.5/fixed + minimax + chebyshev(beta<=0.9) + armijo` | pass | True | 3811.19 | nan | reject |
| 6 | `diag/w5/ema0.5 + invu + none + none` | pass | False | nan | nan | reject |
| 7 | `diag/w5/ema0.5 + minimax + chebyshev(beta<=0.9) + none` | R3: 无 safeguard 时不允许使用动量加速（无法保证下降引理前提）;  | False | nan | nan | reject |
| 8 | `diag/w5/ema0.5 + minimax + chebyshev(beta<=0.99) + armijo` | pass | True | 36.47 | 1.60 | reject |
| 9 | `diag/w50/ema0.1 + minimax + chebyshev(beta<=0.9) + armijo` | pass | True | 122.68 | nan | reject |
| 10 | `diag/w5/ema1.5 + minimax + chebyshev(beta<=0.9) + armijo` | precond.ema=1.5 超出范围 [0.0, 1.0] | False | nan | nan | reject |
| 11 | `diag/w5/ema0.5 + invu + chebyshev(beta<=0.9) + armijo` | pass | True | 42.43 | 1.89 | reject |
| 12 | `diag/w5/ema0.5 + minimax + nesterov + armijo` | accel.type='nesterov' 不在允许集合 ['chebyshev | False | nan | nan | reject |
