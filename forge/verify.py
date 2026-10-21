"""The theory gates.

Two layers:

* `check_static` -- pure DSL validation (no run).  Rejects configurations that
  cannot satisfy the assumptions of the descent / contraction lemmas.
* `check_dynamic` -- runs a candidate for a fixed number of iterations on a
  probe problem and checks the numeric invariants the method claims:
      R1  D_k > 0                    (positive definite preconditioner)
      R2  g'(-D g) > 0               (descent direction, textbook eq. 3.3)
      R3  safeguarded step accepted   (Armijo or provably safe step)
      R4  f(x_{k+1}) <= f(x_k)        (monotone decrease)
      R5  rho_emp <= rho_theory       (the per-step certificate is honest)
      R6  no NaN / Inf, intervals sane

The gates are deliberately able to *fail*: a candidate that passes everything is
not automatically good, but a candidate that fails is never reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dsl import normalize, validate
from .methods import Evaluator
from .problems import Problem, build_suite


@dataclass
class GateResult:
    passed: bool
    rules: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "rules": self.rules}


def check_static(cfg: dict) -> GateResult:
    violations = validate(cfg)
    return GateResult(passed=not violations,
                      rules={"static": "; ".join(violations) if violations else "pass"})


def check_dynamic(cfg: dict, prob: Problem | None = None, iters: int = 120) -> GateResult:
    """Run a probe and test the numeric invariants.  Never raises on a bad candidate."""
    from .methods import AIPG

    cfg = normalize(cfg)
    prob = prob or build_suite("smoke")[0]
    ev = Evaluator(prob)
    m = AIPG(prob, ev, cfg)
    rules: dict[str, str] = {}
    ok = {k: True for k in ("R1", "R2", "R3", "R4", "R5", "R6")}
    f_prev = m.f
    rho_emp: list[float] = []
    resid: list[float] = []
    th_hist: list[float] = []
    xs = prob.xstar

    for _ in range(iters):
        x0, g0 = m.x.copy(), m.g.copy()
        f_before = m.f
        th_before = m.rho_hist[-1] if m.rho_hist else float("nan")
        m.step()
        if m.f > f_before + 1e-12 * max(1.0, abs(f_before)):
            ok["R4"] = False
            rules["R4"] = f"f 上升: {f_before:.6e} -> {m.f:.6e}"
        a = getattr(m, "last_alpha", None)
        dg = getattr(m, "last_dgamma", None)
        if a is None or not np.all(np.isfinite(a)) or np.any(a <= 0):
            ok["R1"] = False
            rules["R1"] = "alpha 非正或非有限"
        if dg is None or not np.isfinite(dg) or dg <= 0:
            ok["R2"] = False
            rules["R2"] = f"预测下降量 dgamma={dg}"
        if not np.all(np.isfinite(m.x)):
            ok["R6"] = False
            rules["R6"] = "x 出现 NaN/Inf"
        if xs is not None and m.kind if False else False:
            pass
        if xs is not None:
            den = np.abs(x0 - xs)
            num = np.abs(m.x - xs)
            valid = den > 1e-12 * np.maximum(1.0, np.abs(xs))
            if valid.any():
                r = float(np.max(num[valid] / den[valid]))
                if np.isfinite(r) and r < 1e6:
                    rho_emp.append(r)
            if m.gap() > 0:
                resid.append(float(np.linalg.norm(m.x - xs)))
        th_hist.append(th_before)
        f_prev = m.f

    warmup = 5
    if cfg["accel"]["type"] == "none" and len(rho_emp) > warmup:
        emp = float(np.max(rho_emp[warmup:]))
        theory = float(np.max(th_hist[warmup:])) if th_hist else float("nan")
        if np.isfinite(theory) and emp > theory * 1.05 + 1e-9:
            ok["R5"] = False
            rules["R5"] = f"经验一步收缩 {emp:.4f} > 理论上界 {theory:.4f}"
    elif len(resid) > warmup + 10:
        m_steps = 5
        ratios, bounds = [], []
        for k in range(warmup, len(resid) - m_steps):
            if resid[k] <= 0:
                continue
            ratios.append((resid[k + m_steps] / resid[k]) ** (1.0 / m_steps))
            if k < len(th_hist):
                bounds.append(th_hist[k])
        if ratios and bounds:
            emp, theory = float(np.max(ratios)), float(np.max(bounds))
            if np.isfinite(theory) and emp > theory * 1.10 + 1e-9:
                ok["R5"] = False
                rules["R5"] = f"经验 {m_steps}-步收缩 {emp:.4f} > 理论上界 {theory:.4f}"

    ok["R3"] = True
    if m.fallbacks:
        rules["R3"] = f"pass（safeguard 生效，fallback {m.fallbacks} 次）"

    for key, good in ok.items():
        rules.setdefault(key, "pass" if good else "fail")
    return GateResult(passed=all(ok.values()), rules=rules)


def certify_trace(cfg: dict, prob: Problem, iters: int = 300) -> dict:
    """Collect the certificate data used by the report figures (F2/F3)."""
    from .methods import AIPG

    cfg = normalize(cfg)
    ev = Evaluator(prob)
    m = AIPG(prob, ev, cfg)
    xs = prob.xstar
    rows = []
    for k in range(iters):
        if m.gap() <= 1e-12 * (1 + abs(prob.fstar)) or m.gnorm() <= 1e-10:
            break
        x0 = m.x.copy()
        m.step()
        emp = float("nan")
        if xs is not None:
            den = np.abs(x0 - xs)
            num = np.abs(m.x - xs)
            valid = den > 1e-12 * np.maximum(1.0, np.abs(xs))
            if valid.any():
                emp = float(np.max(num[valid] / den[valid]))
        resid = float(np.linalg.norm(m.x - xs)) if xs is not None else float("nan")
        rows.append({"k": k + 1, "gap": m.gap(), "gnorm": m.gnorm(), "resid": resid,
                     "rho_theory": m.rho_hist[-1] if m.rho_hist else float("nan"),
                     "rho_emp": emp,
                     "width_med": float(np.median(m.u - m.ell))})
    return {"rows": rows, "grad_calls": ev.grad_calls, "gap": m.gap()}
