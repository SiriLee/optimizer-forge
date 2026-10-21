"""Build report/report.pdf from report/report.md.

Pipeline: markdown -> HTML (+ injected tables/figures) -> display equations
rendered to SVG with matplotlib mathtext -> WeasyPrint PDF.

Nothing is typed twice: tables come from results/tables/*.csv and
design_log/rounds.jsonl, figures from results/figures/*.png.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import markdown
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
EQ = REPORT / "eq"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})


# --------------------------------------------------------------------------- #
# equations
# --------------------------------------------------------------------------- #
_MACROS = [
    (r"\\left", ""), (r"\\right", ""), (r"\\bigl", ""), (r"\\bigr", ""),
    (r"\\Bigl", ""), (r"\\Bigr", ""),
    (r"\\le(?![a-zA-Z])", r"\\leq"), (r"\\ge(?![a-zA-Z])", r"\\geq"),
    (r"\\star", r"\\ast"), (r"\\!", ""), (r"\\;", r"\\ "),
]


def _sanitize(tex: str) -> str:
    tex = (tex.replace("&gt;", ">").replace("&lt;", "<")
              .replace("&amp;", "&").replace("&quot;", '"'))
    for pattern, repl in _MACROS:
        tex = re.sub(pattern, repl, tex)
    return tex


def render_eq(tex: str, fontsize: float = 10.5) -> str:
    EQ.mkdir(parents=True, exist_ok=True)
    tex = _sanitize(tex)
    key = hashlib.sha1(f"{tex}{fontsize}".encode()).hexdigest()[:16]
    path = EQ / f"{key}.svg"
    if not path.exists():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=fontsize)
        fig.savefig(path, bbox_inches="tight", transparent=True, pad_inches=0.02)
        plt.close(fig)
    return f"report/eq/{key}.svg"



_MAP = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "mu": "μ", "lambda": "λ", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "varphi": "φ", "kappa": "κ", "ell": "ℓ", "eta": "η",
    "theta": "θ", "xi": "ξ", "pi": "π", "omega": "ω", "Delta": "Δ",
    "nabla": "∇", "partial": "∂", "infty": "∞", "star": "⋆", "ast": "⋆",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠",
    "approx": "≈", "sim": "∼", "propto": "∝", "in": "∈", "to": "→",
    "top": "ᵀ", "odot": "⊙", "sum": "Σ", "prod": "Π", "times": "×",
    "cdot": "·", "ldots": "…", "dots": "…", "partial": "∂",
    "lesssim": "≲", "gtrsim": "≳", "ll": "≪", "gg": "≫",
    "Rightarrow": "⇒", "Leftrightarrow": "⇔", "longrightarrow": "→",
    "|": "‖", "{": "{", "}": "}", "max": "max", "min": "min",
    "mathrm": "", "mathcal": "", "quad": "\u2003", "qquad": "\u2003\u2003",
    ",": "\u2009", ";": "\u2005", "!": "", " ": " ",
}


def _group(s: str, i: int) -> tuple[str, int]:
    """Read a {...} group starting at i; return (content, next index)."""
    if i >= len(s) or s[i] != "{":
        return s[i], i + 1
    depth, j = 1, i + 1
    while j < len(s) and depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return s[i + 1:j - 1], j


def _token(s: str, i: int) -> tuple[str, int]:
    """Read the next math token: a {group}, a \\macro, or one character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        j = i + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        return s[i:j], (j if j > i + 1 else i + 2)
    return s[i], i + 1


def inline_math(tex: str) -> str:
    """Typeset an inline formula as HTML/Unicode (small text stays crisp)."""
    tex = (tex.replace("&gt;", ">").replace("&lt;", "<")
              .replace("&amp;", "&").replace("&quot;", '"'))
    out, i = [], 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            j = i + 1
            while j < len(tex) and tex[j].isalpha():
                j += 1
            name = tex[i + 1:j] if j > i + 1 else tex[i + 1:i + 2]
            if name == "frac":
                a, k = _group(tex, j)
                b, k = _group(tex, k)
                out.append(f"({inline_math(a)})/({inline_math(b)})")
                i = k
                continue
            if name == "sqrt":
                base, k = _token(tex, j)
                body = inline_math(base)
                out.append("√" + (body if len(base) <= 1 else f"({body})"))
                i = k
                continue
            if name == "hat":
                base, k = _token(tex, j)
                out.append(inline_math(base) + "\u0302")
                i = k
                continue
            out.append(_MAP.get(name, name))
            i = j if j > i + 1 else i + 2
            continue
        if c == "{":
            body, k = _group(tex, i)
            out.append(inline_math(body))
            i = k
            continue
        if c in "^_":
            body, k = _token(tex, i + 1)
            tag = "sup" if c == "^" else "sub"
            out.append(f"<{tag}>{inline_math(body)}</{tag}>")
            i = k
            continue
        if c == "<":
            out.append("&lt;")
        elif c == ">":
            out.append("&gt;")
        elif c == "&":
            out.append("&amp;")
        else:
            out.append("−" if c == "-" else c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# tables from data
# --------------------------------------------------------------------------- #
LABEL = {
    "gd": "GD", "gd_armijo": "GD+Armijo", "heavy_ball": "Heavy-ball", "nag": "NAG",
    "bb1": "BB1", "bb2": "BB2", "adam": "Adam", "aipg": "**AIPG**",
    "aipg_plain": "w/o momentum", "aipg_scalar": "scalar precond.",
    "aipg_fixed": "fixed interval", "lbfgs": "L-BFGS",
}
SHORT = {
    "P1_quad_diag_k10000": "P1 对角二次",
    "P2_quad_rot_k1000": "P2 旋转二次",
    "P3_ls_iid_span2": "P3 i.i.d. 最小二乘",
    "P3_ls_toeplitz_span2": "P3 Toeplitz 最小二乘",
    "P4_logit_span2": "P4 逻辑回归",
    "P5_rosenbrock_n2": "P5 Rosenbrock",
}


def _fmt(v: str, success: str | None = None) -> str:
    try:
        x = float(v)
    except ValueError:
        return v
    if success is not None and float(success) < 0.5:
        return "—"
    return f"{x:.0f}"


def table_main() -> str:
    path = TAB / "final_s3_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    methods = ["gd", "heavy_ball", "nag", "bb1", "adam", "aipg", "lbfgs"]
    cell = {(r["problem"], r["method"]): r for r in rows}
    problems = [p for p in SHORT if any(r["problem"] == p for r in rows)]
    out = ["| 问题 | " + " | ".join(LABEL[m] for m in methods) + " |",
           "|---|" + "---|" * len(methods)]
    for p in problems:
        line = [SHORT[p]]
        for m in methods:
            r = cell.get((p, m))
            line.append(_fmt(r["grad_calls_median"], r.get("success_rate")) if r else "—")
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out)


def table_ablation() -> str:
    path = TAB / "final_s3_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    methods = ["aipg", "aipg_plain", "aipg_scalar", "aipg_fixed"]
    cell = {(r["problem"], r["method"]): r for r in rows}
    problems = [p for p in SHORT if any(r["problem"] == p for r in rows)]
    names = {"aipg": "完整 AIPG", "aipg_plain": "去动量",
             "aipg_scalar": "全局标量", "aipg_fixed": "固定区间"}
    out = ["| 问题 | " + " | ".join(names[m] for m in methods) + " |",
           "|---|" + "---|" * len(methods)]
    for p in problems:
        out.append("| " + SHORT[p] + " | " + " | ".join(
            _fmt(cell[(p, m)]["grad_calls_median"], cell[(p, m)].get("success_rate"))
            if (p, m) in cell else "—" for m in methods) + " |")
    return "\n".join(out)


def table_structure() -> str:
    path = TAB / "final_s3_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    import sys
    sys.path.insert(0, str(ROOT))
    from forge.problems import BUILDERS

    meta, spread = {}, {}
    for key in ("P1", "P2", "P3i", "P3c", "P4i", "P5"):
        prob = BUILDERS[key](0)
        meta[prob.name] = prob.meta
        if prob.H0 is not None:
            d = np.diag(prob.H0)
            spread[prob.name] = float(d.max() / max(d.min(), 1e-300))

    med, ok = {}, {}
    for r in rows:
        med[(r["problem"], r["method"])] = float(r["grad_calls_median"])
        ok[(r["problem"], r["method"])] = float(r.get("success_rate", 1.0)) >= 0.5

    out = ["| 问题 | 条件数 κ | 对角跨度 | 耦合度 | AIPG | NAG | 加速比 |",
           "|---|---|---|---|---|---|---|"]
    for p, name in SHORT.items():
        if p not in meta:
            continue
        kappa = meta[p].get("kappa")
        coup = meta[p].get("coupling")
        a, n = med.get((p, "aipg")), med.get((p, "nag"))
        if a and n and ok.get((p, "aipg")) and ok.get((p, "nag")):
            sp = f"{n / a:.1f}×"
        elif a and ok.get((p, "aipg")):
            sp = f"≥{40001 / a:.1f}×（NAG 未收敛）"
        else:
            sp = "—"
        ks = f"{kappa:.1e}" if kappa is not None else "—"
        sd = f"{spread[p]:.0f}" if p in spread else "—"
        cs = f"{coup:.2f}" if coup is not None else "—"
        aa = f"{a:.0f}" if a and ok.get((p, "aipg")) else "未收敛"
        nn = f"{n:.0f}" if n and ok.get((p, "nag")) else "未收敛"
        out.append(f"| {name} | {ks} | {sd} | {cs} | {aa} | {nn} | {sp} |")
    return "\n".join(out)


def table_rounds() -> str:
    path = ROOT / "design_log" / "rounds.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = ["| 轮 | 候选（预条件/步长/加速/保护） | 静态闸 | 理论闸 | 全量分数 | 决策 |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        g1 = "通过" if r["static"] == "pass" else "拒绝"
        g2 = "通过" if r.get("dynamic_passed") else "拒绝"
        full = r.get("full", {}).get("penalized")
        out.append(f"| {r['round']} | `{r['description']}` | {g1} | {g2} | "
                   f"{full:.3f} | {r['decision']} |" if full is not None else
                   f"| {r['round']} | `{r['description']}` | {g1} | {g2} | — | {r['decision']} |")
    return "\n".join(out)


def fig(name: str, width: str = "100%") -> str:
    return f'<img src="results/figures/{name}.png" style="width:{width}">'


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
CSS = """
@page { size: A4; margin: 2.1cm 2.0cm 2.0cm 2.0cm;
        @bottom-center { content: counter(page); font-size: 8.5pt; color: #666; } }
body { font-family: "Noto Serif SC", "Noto Serif HK", "Source Han Serif SC", serif;
       font-size: 10.5pt; line-height: 1.72; color: #111; text-align: left; }
.imath { font-family: "STIXGeneral", "Noto Serif CJK SC", serif; white-space: nowrap; }
.imath { font-family: "STIXGeneral", "Noto Serif SC", serif; white-space: nowrap; }
sup, sub { font-size: 72%; line-height: 0; position: relative; vertical-align: baseline; }
sup { top: -0.42em; }
sub { bottom: -0.22em; }
h1 { font-family: "Noto Sans SC", "Noto Sans HK", sans-serif; font-size: 17pt;
     font-weight: 600; margin: 0 0 .25em 0; letter-spacing: .01em; }
h2 { font-family: "Noto Sans SC", "Noto Sans HK", sans-serif; font-size: 12.4pt;
     font-weight: 600; margin: 1.2em 0 .4em 0;
     border-bottom: .6pt solid #ccc; padding-bottom: .12em; }
h3 { font-family: "Noto Sans SC", "Noto Sans HK", sans-serif; font-size: 11pt;
     font-weight: 600; margin: .95em 0 .3em 0; }
p { margin: .45em 0; }
.sub { color: #555; font-size: 9.4pt; margin-bottom: 1.2em; }
.abstract { background: #f7f7f7; border-left: 2.5pt solid #bbb; padding: .5em .8em; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: .6em 0 1.05em 0;
        page-break-inside: avoid; }
h2, h3 { page-break-after: avoid; }
th, td { border: .4pt solid #bbb; padding: 3pt 4pt; text-align: center; }
th { background: #eee; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
.eq { text-align: center; margin: .55em 0; }
.eq img { vertical-align: middle; max-width: 94%; }
.imath { vertical-align: -0.18em; }
figure { margin: .9em 0 1.2em 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: #444; margin-top: .45em; line-height: 1.5; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: 8.8pt;
       background: #f2f2f2; padding: 0 .2em; }
pre { background: #f5f5f5; padding: .5em .7em; font-size: 8.4pt; white-space: pre-wrap;
      border-left: 2pt solid #ccc; }
ul, ol { margin: .3em 0 .3em 1.2em; padding: 0; }
li { margin: .15em 0; }
.small { font-size: 8.8pt; color: #444; }
.nobreak { page-break-inside: avoid; }
"""


def build(author: str = "", out: Path | None = None) -> Path:
    src = (REPORT / "report.md").read_text(encoding="utf-8")
    src = src.replace("{{AUTHOR}}", f" · {author}" if author else "")

    src = src.replace("{{TABLE:main}}", table_main())
    src = src.replace("{{TABLE:ablation}}", table_ablation())
    src = src.replace("{{TABLE:structure}}", table_structure())
    src = src.replace("{{TABLE:rounds}}", table_rounds())
    for name in ("f1_main_bars", "f2_certificate", "f3_intervals", "f4_ls",
                 "f4_logit", "f5_pareto", "f6_structure"):
        src = src.replace("{{FIG:" + name + "}}", fig(name))

    html = markdown.markdown(src, extensions=["tables", "fenced_code", "toc"])

    def sub_display(m):
        return f'<div class="eq"><img src="{render_eq(m.group(1).strip())}"></div>'

    html = re.sub(r"<p>\s*\$\$(.+?)\$\$\s*</p>", sub_display, html, flags=re.S)

    def sub_inline(m):
        return f'<span class="imath">{inline_math(m.group(1).strip())}</span>'

    html = re.sub(r"(?<!\$)\$([^$<>]+?)\$(?!\$)", sub_inline, html)

    doc = (f'<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
           f"<style>{CSS}</style></head><body>{html}</body></html>")
    (REPORT / "report.html").write_text(doc, encoding="utf-8")
    target = Path(out) if out else REPORT / "report.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=doc, base_url=str(ROOT)).write_pdf(target)
    return target


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="", help="追加到封面的署名（提交版用，仓库版留空）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pdf = build(author=args.author, out=args.out)
    print(f"pdf → {pdf}")
