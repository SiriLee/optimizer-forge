"""Benchmark runner: one loop, fair counters, resumable JSONL logs.

Usage:
    python -m forge.bench --smoke
    python -m forge.bench --suite full --seeds 7
    python -m forge.bench --suite P1 --methods gd,aipg --seeds 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .methods import BASELINES, Evaluator, build, run_lbfgs
from .problems import Problem, build_suite

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"

DEFAULTS = dict(eps_gap=1e-10, eps_grad=1e-8)


def _max_iters(prob: Problem) -> int:
    if prob.kind == "nonconvex":
        return 20000
    return 40000


def run_one(prob: Problem, method: str, cfg: dict | None = None, seed: int = 0,
            max_iters: int | None = None, budget_grad: int = 40000,
            eps_gap: float = DEFAULTS["eps_gap"],
            eps_grad: float = DEFAULTS["eps_grad"]) -> dict:
    ev = Evaluator(prob)
    t0 = time.perf_counter()
    trace: list[tuple[int, float, float]] = []

    if method == "lbfgs":
        out = run_lbfgs(prob, ev, max_iters=max_iters or 20000,
                        eps_gap=eps_gap, eps_grad=eps_grad)
        gap = out["f"] - prob.fstar
        converged = gap <= eps_gap * (1 + abs(prob.fstar)) or out["gnorm"] <= eps_grad
        return dict(
            problem=prob.name, kind=prob.kind, method="lbfgs", seed=seed,
            coupling=prob.meta.get("coupling"),
            converged=bool(converged), grad_calls=ev.grad_calls, f_calls=ev.f_calls,
            iters=out["iters"], gap=float(gap), gnorm=float(out["gnorm"]),
            seconds=time.perf_counter() - t0, rho_theory=None, rho_emp=None,
            fallbacks=0, trace=out["trace"][:: max(1, len(out["trace"]) // 200)],
        )

    m = build(method, prob, ev, cfg)
    trace.append((m.gap(), m.gnorm(), ev.grad_calls))
    iters_max = max_iters or _max_iters(prob)
    rho_emp_max, rho_emp_last = None, None
    rec_next = 1
    xs = prob.xstar

    while True:
        gap = m.gap()
        if gap <= eps_gap * (1 + abs(prob.fstar)) or m.gnorm() <= eps_grad:
            break
        if m.iters >= iters_max or ev.grad_calls >= budget_grad:
            break
        x_before = m.x
        m.step()
        if prob.kind == "quadratic-diag" and xs is not None:
            den = np.abs(x_before - xs)
            num = np.abs(m.x - xs)
            ok = den > 1e-12 * np.maximum(1.0, np.abs(xs))
            if ok.any():
                r = float(np.max(num[ok] / den[ok]))
                if np.isfinite(r) and r < 1e6:
                    rho_emp_max = r if rho_emp_max is None else max(rho_emp_max, r)
                    rho_emp_last = r
        if m.iters >= rec_next:
            trace.append((m.gap(), m.gnorm(), ev.grad_calls))
            rec_next = max(m.iters + 1, int(math.ceil(m.iters * 1.5)))

    trace.append((m.gap(), m.gnorm(), ev.grad_calls))
    gap = m.gap()
    converged = gap <= eps_gap * (1 + abs(prob.fstar)) or m.gnorm() <= eps_grad
    diag = m.diagnostics() if hasattr(m, "diagnostics") else {}
    return dict(
        problem=prob.name, kind=prob.kind, method=method, seed=seed,
        coupling=prob.meta.get("coupling"),
        converged=bool(converged), grad_calls=ev.grad_calls, f_calls=ev.f_calls,
        iters=m.iters, gap=float(gap), gnorm=m.gnorm(),
        seconds=time.perf_counter() - t0,
        rho_theory=diag.get("rho_theory_max"), rho_emp=rho_emp_max,
        fallbacks=diag.get("fallbacks", 0),
        trace=trace,
    )


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate per (problem, method): median grad calls, IQR, success rate."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["problem"], r["method"]), []).append(r)
    out = []
    for (prob, meth), rs in sorted(groups.items()):
        calls = np.array([r["grad_calls"] for r in rs], dtype=float)
        secs = np.array([r["seconds"] for r in rs], dtype=float)
        out.append(dict(
            problem=prob, method=meth, runs=len(rs),
            grad_calls_median=float(np.median(calls)),
            grad_calls_q25=float(np.percentile(calls, 25)),
            grad_calls_q75=float(np.percentile(calls, 75)),
            success_rate=float(np.mean([r["converged"] for r in rs])),
            seconds_median=float(np.median(secs)),
            gap_median=float(np.median([r["gap"] for r in rs])),
        ))
    return out


def print_table(summary: list[dict]) -> None:
    problems = sorted({s["problem"] for s in summary})
    order = ["gd", "gd_armijo", "heavy_ball", "nag", "bb1", "bb2", "adam",
             "aipg", "aipg_plain", "aipg_scalar", "aipg_fixed", "lbfgs"]
    methods = [m for m in order if any(s["method"] == m for s in summary)]
    cell = {(s["problem"], s["method"]): s for s in summary}
    print(f"\n{'':22s}" + "".join(f"{m:>12s}" for m in methods))
    for p in problems:
        row = f"{p:22s}"
        for m in methods:
            s = cell.get((p, m))
            row += f"{'-':>12s}" if s is None else f"{s['grad_calls_median']:>12.0f}"
        print(row)
    print("\n(梯度调用次数的中位数；'-' = 未运行)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None,
                    choices=["smoke", "full", "structure", "P1", "P2", "P3i", "P3c", "P4i", "P5"])
    ap.add_argument("--smoke", action="store_true", help="alias: --suite smoke --seeds 1 --iters 2000")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", default=None, help="comma separated, default all")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--budget-grad", type=int, default=45000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    suite = args.suite or ("smoke" if args.smoke else "full")
    seeds = 1 if args.smoke else args.seeds
    methods = (args.methods.split(",") if args.methods
               else (["gd", "aipg"] if args.smoke else BASELINES))
    tag = args.tag or f"{suite}_s{seeds}"
    RAW.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RAW / f"{tag}.jsonl"

    rows: list[dict] = []
    n_runs = 0
    for seed in range(seeds):
        for prob in build_suite(suite, seed=seed):
            for meth in methods:
                r = run_one(prob, meth, seed=seed, max_iters=args.iters,
                            budget_grad=args.budget_grad)
                rows.append(r)
                n_runs += 1
                print(f"[{n_runs:3d}] {prob.name:22s} {meth:10s} seed={seed} "
                      f"calls={r['grad_calls']:6d} gap={r['gap']:.2e} "
                      f"conv={int(r['converged'])} t={r['seconds']:.2f}s", flush=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    csv_path = TABLES / f"{tag}_summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        cols = list(summary[0].keys())
        fh.write(",".join(cols) + "\n")
        for s in summary:
            fh.write(",".join(str(s[c]) for c in cols) + "\n")

    print_table(summary)
    print(f"\nraw   → {out_path}\ntable → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
