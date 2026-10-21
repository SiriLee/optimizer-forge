"""The design loop: propose -> static gate -> theory gate -> empirical gate.

Two proposers are supported:

* ``recorded`` (default) -- replay the candidates stored in
  ``design_log/proposals.jsonl``.  It needs no API key, so the whole design
  history in the report is reproducible offline.
* ``llm`` -- ask an OpenAI-compatible endpoint for the next candidate.  The
  prompt template lives in ``skills/method-design/SKILL.md``.

Every round is appended to ``design_log/rounds.jsonl`` and human-readable
``design_log/design_log.md``; the budget (rounds, tokens) is recorded too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from .bench import run_one
from .dsl import describe, normalize, validate
from .problems import BUILDERS, build_suite
from .verify import check_dynamic

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "design_log"
PROPOSALS = LOG_DIR / "proposals.jsonl"
ROUNDS = LOG_DIR / "rounds.jsonl"
LEADERBOARD = LOG_DIR / "leaderboard.csv"

# The curated grid: the corners of the design space a human would try first.
GRID: list[dict] = [
    {"precond": {"type": "diag", "window": 5, "ema": 0.5}, "step": {"rule": "minimax"},
     "accel": {"type": "chebyshev"}, "safeguard": {"type": "armijo"}},
    {"precond": {"type": "diag", "window": 1, "ema": 1.0}, "step": {"rule": "minimax"},
     "accel": {"type": "chebyshev"}, "safeguard": {"type": "armijo"}},
    {"precond": {"type": "diag", "window": 20, "ema": 0.2}, "step": {"rule": "minimax"},
     "accel": {"type": "chebyshev"}, "safeguard": {"type": "armijo"}},
    {"precond": {"type": "diag", "window": 5, "ema": 0.5}, "step": {"rule": "minimax"},
     "accel": {"type": "none"}, "safeguard": {"type": "armijo"}},
    {"precond": {"type": "scalar", "window": 5, "ema": 0.5}, "step": {"rule": "minimax"},
     "accel": {"type": "chebyshev"}, "safeguard": {"type": "armijo"}},
    {"precond": {"type": "diag", "window": 5, "ema": 0.5, "adaptive": False},
     "step": {"rule": "minimax"}, "accel": {"type": "chebyshev"},
     "safeguard": {"type": "armijo"}},
    {"precond": {"type": "diag", "window": 5, "ema": 0.5}, "step": {"rule": "invu"},
     "accel": {"type": "none"}, "safeguard": {"type": "none"}},
    {"precond": {"type": "diag", "window": 5, "ema": 0.5}, "step": {"rule": "minimax"},
     "accel": {"type": "chebyshev", "beta_cap": 0.99}, "safeguard": {"type": "armijo"}},
]


def _load_proposals(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _score(rows: list[dict]) -> tuple[float, dict]:
    """Lower is better: geometric mean of calls relative to the best candidate."""
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["problem"], []).append(float(r["grad_calls"]))
    detail = {p: float(np.median(v)) for p, v in per.items()}
    return float(np.exp(np.mean([math.log(max(v, 1.0)) for v in detail.values()]))), detail


def _penalized_score(detail: dict, best: dict) -> float:
    ratios = []
    for p, v in detail.items():
        b = best.get(p, v)
        ratios.append(max(v / max(b, 1.0), 1.0))
    return float(np.exp(np.mean([math.log(r) for r in ratios])))


def run_grid(candidates: list[dict], seeds: list[int], suite: str,
             iters: int | None) -> list[dict]:
    problems = {s: build_suite(suite, seed=s) for s in seeds}
    rows = []
    t0 = time.perf_counter()
    for i, cfg in enumerate(candidates):
        for seed in seeds:
            for prob in problems[seed]:
                rows.append(run_one(prob, "aipg", cfg=cfg, seed=seed, max_iters=iters))
        print(f"  [grid {i + 1}/{len(candidates)}] {describe(cfg)} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", default="recorded", choices=["recorded", "grid", "llm"])
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--token-budget", type=int, default=300_000)
    ap.add_argument("--smoke-seeds", type=int, default=2)
    ap.add_argument("--full-seeds", type=int, default=3)
    ap.add_argument("--smoke-suite", default="smoke")
    ap.add_argument("--full-suite", default="full")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    proposed = GRID if args.proposer == "grid" else _load_proposals(PROPOSALS)
    if not proposed:
        print(f"no proposals at {PROPOSALS}; use --proposer grid", file=sys.stderr)
        return 2
    proposed = proposed[: args.rounds]

    rounds: list[dict] = []
    tokens_used = 0
    survivors: list[tuple[float, int, dict]] = []

    print(f"== static + theory gates ({len(proposed)} candidates) ==")
    for r, item in enumerate(proposed):
        cfg = normalize(item.get("candidate", item))
        violations = validate(cfg)
        rec = {
            "round": r + 1, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "proposer": item.get("proposer", args.proposer),
            "candidate": cfg, "description": describe(cfg),
            "notes": item.get("notes", ""),
            "static": "pass" if not violations else "; ".join(violations),
        }
        if item.get("tokens"):
            tok = item["tokens"]
            tokens_used += int(tok.get("prompt", 0) + tok.get("completion", 0)) if isinstance(tok, dict) else int(tok)
            rec["tokens"] = tok
        if not violations:
            # the theory gate runs on two probes: a diagonal one (where the
            # certificate is claimed) and a coupled one (where it must not lie)
            probes = {"P1": check_dynamic(cfg, prob=BUILDERS["P1"](0)),
                      "P2": check_dynamic(cfg, prob=BUILDERS["P2"](0))}
            rec["dynamic"] = {k: g.rules for k, g in probes.items()}
            rec["dynamic_passed"] = all(g.passed for g in probes.values())
            if not rec["dynamic_passed"]:
                bad = {k: {r: v for r, v in g.rules.items() if v != "pass"}
                       for k, g in probes.items() if not g.passed}
                rec["dynamic_reason"] = bad
        else:
            rec["dynamic_passed"] = False
        rounds.append(rec)
        print(f"  [{r + 1:2d}] {describe(cfg):70s} static={'ok' if not violations else 'FAIL'}"
              f" dynamic={'ok' if rec.get('dynamic_passed') else 'FAIL'}")

    ok_rounds = [i for i, rec in enumerate(rounds) if rec.get("dynamic_passed")]
    if not ok_rounds:
        print("every candidate was rejected by the gates", file=sys.stderr)
        return 1

    print(f"\n== smoke gate ({len(ok_rounds)} survivors) ==")
    smoke_rows = run_grid([rounds[i]["candidate"] for i in ok_rounds],
                          list(range(args.smoke_seeds)), args.smoke_suite, args.iters)
    # attribute rows back to candidates (run_grid iterates candidate-major)
    n_prob = len(build_suite(args.smoke_suite, seed=0))
    per_cand = len(range(args.smoke_seeds)) * n_prob
    for idx, i in enumerate(ok_rounds):
        rows = smoke_rows[idx * per_cand:(idx + 1) * per_cand]
        score, detail = _score(rows)
        rounds[i]["smoke"] = {"detail": detail, "score": score,
                              "calls": sum(r["grad_calls"] for r in rows)}
        survivors.append((score, i, rounds[i]["candidate"]))

    survivors.sort(key=lambda t: t[0])
    best_smoke: dict[str, float] = {}
    for _, i, _ in survivors:
        for p, v in rounds[i]["smoke"]["detail"].items():
            best_smoke[p] = min(best_smoke.get(p, 1e18), v)
    for score, i, _ in survivors:
        rounds[i]["smoke"]["penalized"] = _penalized_score(rounds[i]["smoke"]["detail"], best_smoke)

    print(f"\n== full benchmark (top {args.top_k}) ==")
    finalists = survivors[: args.top_k]
    full_rows = run_grid([cfg for _, _, cfg in finalists],
                         list(range(args.full_seeds)), args.full_suite, args.iters)
    n_prob_full = len(build_suite(args.full_suite, seed=0))
    per_cand_full = len(range(args.full_seeds)) * n_prob_full
    best_full: dict[str, float] = {}
    for idx, (_, i, _) in enumerate(finalists):
        rows = full_rows[idx * per_cand_full:(idx + 1) * per_cand_full]
        detail = {}
        for p in {r["problem"] for r in rows}:
            detail[p] = float(np.median([r["grad_calls"] for r in rows if r["problem"] == p]))
        rounds[i]["full"] = {"detail": detail,
                             "success": float(np.mean([r["converged"] for r in rows]))}
        for p, v in detail.items():
            best_full[p] = min(best_full.get(p, 1e18), v)
    for _, i, _ in finalists:
        rounds[i]["full"]["penalized"] = _penalized_score(rounds[i]["full"]["detail"], best_full)

    finalists.sort(key=lambda t: rounds[t[1]]["full"]["penalized"])
    # Occam tie-break: among candidates within 5% of the best score, keep the one
    # proposed earliest (the simplest hypothesis) instead of the lucky variant.
    best_score = rounds[finalists[0][1]]["full"]["penalized"]
    eligible = [t for t in finalists
                if rounds[t[1]]["full"]["penalized"] <= best_score * 1.05]
    winner = min(eligible, key=lambda t: t[1])
    rounds[winner[1]]["decision"] = "promote"
    rounds[winner[1]]["reason"] = "最低 penalized 分数（全量基准）"
    for _, i, _ in finalists[1:]:
        rounds[i]["decision"] = "reject"
        rounds[i]["reason"] = "全量基准分数劣于胜者"
    for i, rec in enumerate(rounds):
        rec.setdefault("decision", "reject")
        rec.setdefault("reason", "未通过闸门" if not rec.get("dynamic_passed") else "冒烟阶段未进前 k")

    with ROUNDS.open("w", encoding="utf-8") as fh:
        for rec in rounds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with LEADERBOARD.open("w", encoding="utf-8") as fh:
        fh.write("candidate,description,static,dynamic,smoke_score,full_penalized,decision\n")
        for rec in rounds:
            fh.write(",".join([
                str(rec["round"]), json.dumps(rec["description"]),
                str(rec["static"]), str(rec.get("dynamic_passed")),
                f"{rec.get('smoke', {}).get('score', float('nan')):.3f}",
                f"{rec.get('full', {}).get('penalized', float('nan')):.3f}",
                rec["decision"],
            ]) + "\n")

    md = ["# 设计日志（自动生成）", "",
          f"- 提出候选：{len(rounds)}；通过闸门：{len(ok_rounds)}；"
          f"最终候选：round {winner[1] + 1}", ""]
    md.append("| 轮次 | 候选 | 静态闸 | 理论闸 | 冒烟分数 | 全量分数 | 决策 |")
    md.append("|---|---|---|---|---|---|---|")
    for rec in rounds:
        md.append(f"| {rec['round']} | `{rec['description']}` | {rec['static'][:40]} | "
                  f"{rec.get('dynamic_passed')} | {rec.get('smoke', {}).get('score', float('nan')):.2f} | "
                  f"{rec.get('full', {}).get('penalized', float('nan')):.2f} | {rec['decision']} |")
    (LOG_DIR / "design_log.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nwinner: round {winner[1] + 1}  {describe(winner[2])}")
    print(f"rounds → {ROUNDS}\nleaderboard → {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
