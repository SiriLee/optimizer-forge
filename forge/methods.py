"""Gradient-type methods: baselines and AIPG.

A method is an iterator with a tiny protocol:
    m = build(cfg, prob, ev)
    while not done: m.step()
    m.f, m.g          # current objective and gradient
    m.extra          # method-specific diagnostics (certificates, fallback counts)

All function/gradient evaluations go through `Evaluator` so that every method is
charged the same way.  Nothing here knows about the runner, plotting or reports.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .problems import Problem


class Evaluator:
    """Counts function and gradient evaluations for fair comparisons."""

    def __init__(self, prob: Problem):
        self.prob = prob
        self.f_calls = 0
        self.grad_calls = 0

    def f(self, x: np.ndarray) -> float:
        self.f_calls += 1
        return float(self.prob.f(x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.grad_calls += 1
        return self.prob.grad(x)


class Method:
    name = "base"

    def __init__(self, prob: Problem, ev: Evaluator, cfg: Optional[dict] = None):
        self.prob = prob
        self.ev = ev
        self.cfg = cfg or {}
        self.x = np.array(prob.x0, dtype=float)
        self.f = ev.f(self.x)
        self.g = ev.grad(self.x)
        self.n = self.x.size
        self.iters = 0
        self.extra: dict = {}

    # -- helpers ----------------------------------------------------------- #
    def _set(self, x: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.f = self.ev.f(self.x)
        self.g = self.ev.grad(self.x)

    def step(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def gap(self) -> float:
        return self.f - self.prob.fstar

    def gnorm(self) -> float:
        return float(np.linalg.norm(self.g))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
class GD(Method):
    name = "gd"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        self.alpha = float(self.cfg.get("alpha", 1.0 / prob.L))

    def step(self) -> None:
        self.x = self.x - self.alpha * self.g
        self.f = self.ev.f(self.x)
        self.g = self.ev.grad(self.x)
        self.iters += 1


class GDArmijo(Method):
    name = "gd_armijo"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        self.alpha0 = float(self.cfg.get("alpha0", 1.0 / prob.L))
        self.c = float(self.cfg.get("c", 1e-4))
        self.beta = float(self.cfg.get("beta", 0.5))

    def step(self) -> None:
        a = self.alpha0
        g2 = float(self.g @ self.g)
        for _ in range(40):
            x_new = self.x - a * self.g
            f_new = self.ev.f(x_new)
            if f_new <= self.f - self.c * a * g2:
                break
            a *= self.beta
        self.x, self.f = x_new, f_new
        self.g = self.ev.grad(self.x)
        self.iters += 1


class HeavyBall(Method):
    name = "heavy_ball"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        L = prob.L
        mu = max(prob.mu, 1e-6 * L)
        sL, smu = np.sqrt(L), np.sqrt(mu)
        self.alpha = 4.0 / (sL + smu) ** 2
        self.beta = ((sL - smu) / (sL + smu)) ** 2
        self.x_prev = self.x.copy()

    def step(self) -> None:
        x_new = self.x - self.alpha * self.g + self.beta * (self.x - self.x_prev)
        self.x_prev = self.x
        self.x = x_new
        self.f = self.ev.f(self.x)
        self.g = self.ev.grad(self.x)
        self.iters += 1


class NAG(Method):
    name = "nag"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        L = prob.L
        mu = max(prob.mu, 1e-6 * L)
        self.alpha = 1.0 / L
        self.beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
        self.y = self.x.copy()

    def step(self) -> None:
        x_new = self.y - self.alpha * self.g
        y_new = x_new + self.beta * (x_new - self.x)
        self.x, self.y = x_new, y_new
        self.f = self.ev.f(self.x)
        self.g = self.ev.grad(self.x)
        self.iters += 1


class BB(Method):
    """Barzilai-Borwein with a monotone safeguard (non-monotone BB is unsafe)."""

    name = "bb"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        self.variant = int(self.cfg.get("variant", 1))
        self.alpha = 1.0 / prob.L
        self.x_prev, self.g_prev = self.x.copy(), self.g.copy()

    def step(self) -> None:
        s = self.x - self.x_prev
        y = self.g - self.g_prev
        sy = float(s @ y)
        if sy > 1e-12:
            if self.variant == 1:
                a = float(s @ s) / sy
            else:
                a = sy / float(y @ y)
            a = min(max(a, 1e-8 / max(self.prob.L, 1e-12), ), 1e8 / max(self.prob.L, 1e-12))
        else:
            a = 1.0 / self.prob.L
        g2 = float(self.g @ self.g)
        for _ in range(20):
            x_new = self.x - a * self.g
            f_new = self.ev.f(x_new)
            if f_new <= self.f - 1e-4 * a * g2:
                break
            a *= 0.5
        self.x_prev, self.g_prev = self.x.copy(), self.g.copy()
        self.x, self.f = x_new, f_new
        self.g = self.ev.grad(self.x)
        self.iters += 1


class Adam(Method):
    name = "adam"

    def __init__(self, prob, ev, cfg=None):
        super().__init__(prob, ev, cfg)
        self.lr = float(self.cfg.get("lr", 0.01))
        self.b1 = float(self.cfg.get("beta1", 0.9))
        self.b2 = float(self.cfg.get("beta2", 0.999))
        self.eps = float(self.cfg.get("eps", 1e-8))
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)

    def step(self) -> None:
        self.m = self.b1 * self.m + (1 - self.b1) * self.g
        self.v = self.b2 * self.v + (1 - self.b2) * self.g ** 2
        t = self.iters + 1
        mhat = self.m / (1 - self.b1 ** t)
        vhat = self.v / (1 - self.b2 ** t)
        self.x = self.x - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        self.f = self.ev.f(self.x)
        self.g = self.ev.grad(self.x)
        self.iters += 1


# --------------------------------------------------------------------------- #
# AIPG
# --------------------------------------------------------------------------- #
DEFAULT_AIPG = {
    "name": "aipg",
    "precond": {"type": "diag", "source": "secant", "window": 5, "ema": 0.5},
    "step": {"rule": "minimax"},
    "accel": {"type": "none"},
    "safeguard": {"type": "armijo", "c": 1e-4, "halvings": 25},
}


class AIPG(Method):
    """Adaptive interval-preconditioned gradient.

    Per coordinate i we maintain [ell_i, u_i] with a secant-based estimate of the
    local curvature h_i, then take alpha_i = 2/(u_i + ell_i): the step that
    minimises the worst-case contraction |1 - alpha h| over h in [ell_i, u_i].
    """

    name = "aipg"

    def __init__(self, prob, ev, cfg=None):
        cfg = {**(cfg or {})}
        merged = {}
        for section, values in DEFAULT_AIPG.items():
            if isinstance(values, dict):
                merged[section] = {**values, **cfg.get(section, {})}
            else:
                merged[section] = cfg.get(section, values)
        for section, values in cfg.items():
            merged.setdefault(section, values)
        cfg = merged
        super().__init__(prob, ev, cfg)
        L = prob.L
        self.mu0 = max(1e-8, float(prob.mu) if prob.mu > 0 else 1e-6 * L)
        self.ell = np.full(self.n, self.mu0)
        self.u = np.full(self.n, L)
        self.window = int(cfg["precond"].get("window", 5))
        self.ema = float(cfg["precond"].get("ema", 0.5))
        self.scalar = cfg["precond"].get("type", "diag") == "scalar"
        self.adaptive = bool(cfg["precond"].get("adaptive", True))
        self.rule = cfg["step"].get("rule", "minimax")
        self.safeguard = cfg["safeguard"].get("type", "armijo")
        self.c = float(cfg["safeguard"].get("c", 1e-4))
        self.halvings = int(cfg["safeguard"].get("halvings", 25))
        self.accel = cfg["accel"].get("type", "none")
        self.beta_cap = float(cfg["accel"].get("beta_cap", 0.9))
        self.x_prev = self.x.copy()
        self.secants: list[np.ndarray] = []
        self.fallbacks = 0
        self.rho_theory = np.zeros(self.window * 4 + 1)  # reused ring buffer
        self.rho_hist: list[float] = []
        self._last_dx = np.zeros(self.n)

    # -- step size ---------------------------------------------------------- #
    def _alpha(self) -> np.ndarray:
        if self.rule == "minimax":
            return 2.0 / (self.u + self.ell)
        if self.rule == "invu":
            return 1.0 / self.u
        raise ValueError(f"unknown step rule {self.rule!r}")

    # -- one iteration ------------------------------------------------------ #
    def step(self) -> None:
        alpha = self._alpha()
        if self.scalar:
            alpha = np.full(self.n, float(np.mean(alpha)))

        if self.accel == "chebyshev":
            sq_u, sq_l = np.sqrt(self.u), np.sqrt(self.ell)
            beta = ((sq_u - sq_l) / (sq_u + sq_l)) ** 2
            beta = np.minimum(beta, self.beta_cap)
            d = beta * (self.x - self.x_prev) - alpha * self.g
        else:
            d = -alpha * self.g
        dgamma = float(-self.g @ d)          # predicted decrease (linear model)
        if not np.isfinite(dgamma) or dgamma <= 0:
            d = -alpha * self.g
            dgamma = float(np.sum(alpha * self.g * self.g))

        t, ok, x_new, f_new = 1.0, False, None, None
        if self.safeguard == "armijo":
            for _ in range(self.halvings):
                x_try = self.x + t * d
                f_try = self.ev.f(x_try)
                if np.isfinite(f_try) and f_try <= self.f - self.c * t * dgamma:
                    x_new, f_new, ok = x_try, f_try, True
                    break
                t *= 0.5
            if not ok:
                # provably descending fallback: step 1/L along -g
                x_new = self.x - (1.0 / self.prob.L) * self.g
                f_new = self.ev.f(x_new)
                self.fallbacks += 1
                self.extra["last_fallback"] = True
                if not np.isfinite(f_new) or f_new > self.f:
                    x_new, f_new = self.x.copy(), self.f
        else:
            x_new = self.x + d
            f_new = self.ev.f(x_new)

        self.last_alpha = alpha
        self.last_dgamma = dgamma
        g_new = self.ev.grad(x_new)
        self._update_intervals(x_new - self.x, g_new - self.g)
        self.x_prev = self.x.copy()
        self.x, self.f, self.g = x_new, f_new, g_new
        self.iters += 1

    # -- interval bookkeeping ----------------------------------------------- #
    def _update_intervals(self, dx: np.ndarray, dg: np.ndarray) -> None:
        if not self.adaptive:
            self._record_rho()
            return
        scale = np.maximum(1.0, np.abs(self.x))
        mask = np.abs(dx) > 1e-12 * scale
        h = np.full(self.n, np.nan)
        h[mask] = dg[mask] / dx[mask]
        good = np.isfinite(h) & (h > 0)
        if good.any():
            self.secants.append(np.where(good, h, np.nan))
            self.secants = self.secants[-self.window:]
            S = np.vstack(self.secants)
            fin = np.isfinite(S)
            if not fin.any():
                self.secants.pop()
                return
            lo = np.where(fin, S, np.inf).min(axis=0)
            hi = np.where(fin, S, -np.inf).max(axis=0)
            lo = np.where(np.isfinite(lo), lo, self.ell)
            hi = np.where(np.isfinite(hi), hi, self.u)
            lo = np.where(np.isfinite(lo), lo, self.ell)
            hi = np.where(np.isfinite(hi), hi, self.u)
            hi = np.maximum(hi, lo)
            self.ell = (1 - self.ema) * self.ell + self.ema * np.maximum(lo, self.mu0)
            self.u = (1 - self.ema) * self.u + self.ema * np.minimum(np.maximum(hi, self.ell), self.prob.L)
            self.ell = np.minimum(self.ell, self.u)
        self._record_rho()
        self._last_dx = dx

    # -- certificate -------------------------------------------------------- #
    def _record_rho(self) -> None:
        if self.accel == "chebyshev":
            sq_u, sq_l = np.sqrt(self.u), np.sqrt(self.ell)
            rho = float(np.max((sq_u - sq_l) / (sq_u + sq_l)))
        else:
            rho = float(np.max((self.u - self.ell) / (self.u + self.ell)))
        self.rho_hist.append(rho)

    # -- diagnostics -------------------------------------------------------- #
    def diagnostics(self) -> dict:
        if not self.rho_hist:
            return {}
        return {
            "fallbacks": self.fallbacks,
            "rho_theory_max": float(np.max(self.rho_hist)),
            "rho_theory_last": float(self.rho_hist[-1]),
            "interval_width_med": float(np.median(self.u - self.ell)),
        }


# --------------------------------------------------------------------------- #
# L-BFGS reference (scipy), charged per evaluation
# --------------------------------------------------------------------------- #
def run_lbfgs(prob: Problem, ev: Evaluator, max_iters: int = 20000,
              eps_gap: float = 1e-10, eps_grad: float = 1e-8) -> dict:
    from scipy.optimize import minimize

    trace: list[tuple[int, float, float]] = []
    state = {"stop": False, "x": np.asarray(prob.x0, dtype=float).copy()}

    def fun(x):
        state["x"] = np.asarray(x, dtype=float).copy()
        fx = ev.f(x)
        gx = ev.grad(x)
        gap = fx - prob.fstar
        trace.append((ev.grad_calls, fx, float(np.linalg.norm(gx))))
        if gap <= eps_gap * (1 + abs(prob.fstar)) or np.linalg.norm(gx) <= eps_grad:
            state["stop"] = True
        return fx, gx

    def cb(xk):
        state["x"] = np.asarray(xk, dtype=float).copy()
        if state["stop"]:
            raise StopIteration

    try:
        minimize(fun, prob.x0.copy(), jac=True, method="L-BFGS-B",
                 callback=cb, options=dict(maxiter=max_iters, gtol=0.0, ftol=0.0))
    except StopIteration:
        pass
    x = state["x"]
    g_final = prob.grad(x)
    f_final = prob.f(x)
    return {
        "x": x, "f": float(f_final), "gnorm": float(np.linalg.norm(g_final)),
        "iters": len(trace), "trace": trace,
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
REGISTRY = {
    "gd": GD,
    "gd_armijo": GDArmijo,
    "heavy_ball": HeavyBall,
    "nag": NAG,
    "bb1": None,  # filled below
    "bb2": None,
    "adam": Adam,
    "aipg": AIPG,
    "aipg_plain": AIPG,
    "aipg_cheb": AIPG,
    "aipg_scalar": AIPG,
    "aipg_fixed": AIPG,
}

BASELINES = ["gd", "gd_armijo", "heavy_ball", "nag", "bb1", "bb2", "adam",
             "aipg", "aipg_plain", "aipg_scalar", "aipg_fixed", "lbfgs"]


_WINNER_PRECOND = {"type": "diag", "source": "secant", "window": 20, "ema": 0.2,
                   "adaptive": True}

PRESETS: dict[str, dict] = {
    # the method as reported: interval preconditioner + Chebyshev acceleration
    "aipg": {"precond": dict(_WINNER_PRECOND), "accel": {"type": "chebyshev"}},
    # one-factor-at-a-time ablations, all inheriting the winner's preconditioner
    "aipg_plain": {"precond": dict(_WINNER_PRECOND), "accel": {"type": "none"}},
    "aipg_scalar": {"precond": {**_WINNER_PRECOND, "type": "scalar"},
                    "accel": {"type": "chebyshev"}},
    "aipg_fixed": {"precond": {**_WINNER_PRECOND, "adaptive": False},
                   "accel": {"type": "chebyshev"}},
    "aipg_cheb": {"precond": dict(_WINNER_PRECOND), "accel": {"type": "chebyshev"}},
}


def build(name: str, prob: Problem, ev: Evaluator, cfg: Optional[dict] = None) -> Method:
    if name in PRESETS:
        merged = {**PRESETS[name], **(cfg or {})}
        return AIPG(prob, ev, merged)
    if name == "bb1":
        return BB(prob, ev, {"variant": 1})
    if name == "bb2":
        return BB(prob, ev, {"variant": 2})
    if name == "lbfgs":
        raise ValueError("lbfgs is run through run_lbfgs(), not build()")
    cls = REGISTRY[name]
    if cls is None:
        raise ValueError(f"unknown method {name!r}")
    return cls(prob, ev, cfg)
