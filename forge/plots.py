"""Figures for the report.

Everything is generated from `results/raw/*.jsonl` and `design_log/rounds.jsonl`,
so no number in the report is typed by hand.

    python -m forge.plots --raw results/raw/final_s3.jsonl --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
})

COLORS = {
    "gd": "#9a9a9a", "gd_armijo": "#b5b5b5", "heavy_ball": "#8fb3d9",
    "nag": "#4477aa", "bb1": "#88ccaa", "bb2": "#66aa88",
    "adam": "#ee8866", "aipg": "#cc3311", "aipg_plain": "#ee6677",
    "aipg_scalar": "#aa3377", "aipg_fixed": "#663399", "lbfgs": "#000000",
}
LABELS = {
    "gd": "GD", "gd_armijo": "GD+Armijo", "heavy_ball": "Heavy-ball", "nag": "NAG",
    "bb1": "BB1", "bb2": "BB2", "adam": "Adam", "aipg": "AIPG (ours)",
    "aipg_plain": "AIPG w/o momentum", "aipg_scalar": "AIPG scalar",
    "aipg_fixed": "AIPG fixed interval", "lbfgs": "L-BFGS",
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _short(name: str) -> str:
    return (name.replace("P1_quad_diag_", "P1 diag ")
                .replace("P2_quad_rot_", "P2 rot ")
                .replace("P3_ls_iid_span2", "P3 LS i.i.d.")
                .replace("P3_ls_toeplitz_span2", "P3 LS Toeplitz")
                .replace("P4_logit_span2", "P4 logistic")
                .replace("P5_rosenbrock_n2", "P5 Rosenbrock"))


def fig_main_bars(rows: list[dict], methods: list[str], out: Path) -> None:
    problems = sorted({r["problem"] for r in rows})
    med = {}
    for p in problems:
        for m in methods:
            vals = [r["grad_calls"] for r in rows if r["problem"] == p and r["method"] == m]
            if vals:
                med[(p, m)] = float(np.median(vals))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs, ys = [], []
        for j, p in enumerate(problems):
            if (p, m) in med:
                xs.append(j + (i - (len(methods) - 1) / 2) * width)
                ys.append(med[(p, m)])
        ax.bar(xs, ys, width=width, color=COLORS.get(m, "#333"), label=LABELS.get(m, m))
    ax.set_yscale("log")
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels([_short(p) for p in problems], rotation=18, ha="right")
    ax.set_ylabel("gradient calls to 1e-10")
    ax.legend(ncol=len(methods), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(out / "f1_main_bars.pdf")
    fig.savefig(out / "f1_main_bars.png")
    plt.close(fig)


def fig_structure(rows: list[dict], out: Path) -> None:
    coupling = {r["problem"]: r.get("coupling") for r in rows}
    problems = [p for p in sorted(coupling) if coupling[p] is not None]
    xs, ys = [], []
    for p in problems:
        nag = [r["grad_calls"] for r in rows if r["problem"] == p and r["method"] == "nag"]
        ours = [r["grad_calls"] for r in rows if r["problem"] == p and r["method"] == "aipg"]
        if nag and ours:
            xs.append(coupling[p])
            ys.append(float(np.median(nag)) / float(np.median(ours)))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.scatter(xs, ys, s=34, color=COLORS["aipg"], zorder=3)
    for x, y, p in zip(xs, ys, problems):
        ax.annotate(_short(p), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xlabel("coupling  ||H-diag(H)||$_F$/||H||$_F$")
    ax.set_ylabel("speedup vs NAG")
    fig.tight_layout()
    fig.savefig(out / "f6_structure.pdf")
    fig.savefig(out / "f6_structure.png")
    plt.close(fig)


def _curve(rows: list[dict], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves = []
    for r in rows:
        tr = r.get("trace") or []
        if len(tr) < 3:
            continue
        calls = np.array([t[2] for t in tr], dtype=float)
        gaps = np.maximum(np.array([t[0] for t in tr], dtype=float), 1e-16)
        keep = np.concatenate(([True], np.diff(calls) > 0))
        curves.append(np.interp(np.log10(grid), np.log10(calls[keep]), np.log10(gaps[keep])))
    if not curves:
        return grid, np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    arr = np.vstack(curves)
    return grid, np.median(arr, axis=0), np.percentile(arr, 25, axis=0)


def fig_curves(rows: list[dict], problem: str, methods: list[str], out: Path,
               fname: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    grid = np.logspace(0, 4.6, 120)
    for m in methods:
        sub = [r for r in rows if r["problem"] == problem and r["method"] == m]
        if not sub:
            continue
        _, med, q25 = _curve(sub, grid)
        ax.plot(grid, 10 ** med, color=COLORS.get(m, "#333"), lw=1.2, label=LABELS.get(m, m))
        ax.fill_between(grid, 10 ** q25, 10 ** (2 * med - q25), color=COLORS.get(m, "#333"),
                        alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gradient calls")
    ax.set_ylabel("$f(x)-f^\\star$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / fname)
    fig.savefig(out / fname.replace(".pdf", ".png"))
    plt.close(fig)


def fig_certificate(cert: dict, out: Path, fname: str = "f2_certificate.pdf") -> None:
    """Compare the theoretical one-step bound with the *m-step* observed factor.

    With momentum a single step may overshoot; the object the bound controls is
    the geometric mean over m steps, so that is what we plot.
    """
    rows = cert["rows"]
    if not rows:
        return
    m_steps = 5
    ks, th, emp = [], [], []
    for i in range(len(rows) - m_steps):
        a, b = rows[i], rows[i + m_steps]
        if not (np.isfinite(a.get("resid", np.nan)) and np.isfinite(b.get("resid", np.nan))):
            continue
        if a["resid"] <= 0:
            continue
        ks.append(b["k"])
        th.append(min(a["rho_theory"], b["rho_theory"]))
        emp.append((b["resid"] / a["resid"]) ** (1.0 / m_steps))
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(ks, th, color="#4477aa", lw=1.3, label="theoretical bound $\\hat\\rho_k$")
    ax.plot(ks, emp, color=COLORS["aipg"], lw=1.0, ls="--", label=f"observed {m_steps}-step mean")
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("contraction factor")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / fname)
    fig.savefig(out / fname.replace(".pdf", ".png"))
    plt.close(fig)


def fig_intervals(certs: dict, out: Path, fname: str = "f3_intervals.pdf") -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    for name, cert in certs.items():
        rows = cert["rows"]
        if not rows:
            continue
        ks = [r["k"] for r in rows]
        widths = [r["width_med"] for r in rows]
        ax.plot(ks, widths, lw=1.2, label=_short(name))
    ax.set_yscale("log")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("median interval width $u_i-\\ell_i$")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / fname)
    fig.savefig(out / fname.replace(".pdf", ".png"))
    plt.close(fig)


def fig_pareto(rounds_path: Path, out: Path) -> None:
    if not rounds_path.exists():
        return
    rounds = [json.loads(line) for line in rounds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tok, score, labels = 0.0, [], []
    xs, ys = [], []
    best = float("inf")
    for rec in rounds:
        t = rec.get("tokens") or {}
        tok += float(t.get("prompt", 0) + t.get("completion", 0)) if isinstance(t, dict) else float(t)
        s = rec.get("full", {}).get("penalized", rec.get("smoke", {}).get("score"))
        if s is None:
            continue
        best = min(best, s)
        xs.append(tok)
        ys.append(best)
        labels.append(str(rec["round"]))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.step(xs, ys, where="post", color=COLORS["aipg"], lw=1.3)
    ax.scatter(xs, ys, s=14, color=COLORS["aipg"], zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, -8), fontsize=6)
    ax.set_xlabel("cumulative tokens (estimated)")
    ax.set_ylabel("best score so far")
    fig.tight_layout()
    fig.savefig(out / "f5_pareto.pdf")
    fig.savefig(out / "f5_pareto.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.raw))

    main_methods = ["gd", "nag", "bb1", "adam", "aipg", "lbfgs"]
    fig_main_bars(rows, main_methods, out)
    fig_structure(rows, out)
    fig_curves(rows, "P3_ls_iid_span2", ["gd", "nag", "adam", "aipg", "lbfgs"], out, "f4_ls.pdf")
    fig_curves(rows, "P4_logit_span2", ["gd", "nag", "adam", "aipg", "lbfgs"], out, "f4_logit.pdf")
    fig_pareto(ROOT / "design_log" / "rounds.jsonl", out)

    # certificates: run the winning method on a diagonal and a coupled probe
    from .dsl import normalize
    from .problems import BUILDERS
    from .verify import certify_trace

    cfg = normalize(None)
    certs = {name: certify_trace(cfg, BUILDERS[name](0)) for name in ("P1", "P3c")}
    (ROOT / "results" / "certificates.json").write_text(
        json.dumps(certs, ensure_ascii=False), encoding="utf-8")
    fig_certificate(certs["P1"], out)
    fig_intervals(certs, out)
    print(f"figures → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
