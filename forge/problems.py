"""Problem suite for the AIPG experiments.

Design principle: keep everything a *problem* property, not a *method* property.
Every problem exposes the same surface:

    f, grad, x0, fstar, L (smoothness), mu (strong convexity lower bound), H0

`H0` is the Hessian at x0 (exact for quadratics and least squares, the
Gauss-Newton matrix for logistic).  It is used only for *diagnostics* -- the
coupling index in `meta` and the oracle experiments in the report -- never inside
the optimisation loop, so it cannot leak information into a method.

The suite deliberately spans the structural axis the report is about:

    coupling(H) := ||H - diag(H)||_F / ||H||_F

P1 and P3i are (near) diagonal -- AIPG's home; P2 and P3c are strongly coupled --
the documented failure boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Problem:
    name: str
    kind: str
    n: int
    f: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    fstar: float
    L: float
    mu: float = 0.0
    xstar: Optional[np.ndarray] = None
    H0: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


def coupling_index(H: np.ndarray) -> float:
    """Relative mass of the off-diagonal part, in Frobenius norm."""
    H = np.asarray(H, dtype=float)
    d = np.diag(np.diag(H))
    return float(np.linalg.norm(H - d) / max(np.linalg.norm(H), 1e-300))


# --------------------------------------------------------------------------- #
# P1: ill-conditioned diagonal quadratic  (coupling = 0)
# --------------------------------------------------------------------------- #
def quad_diag(n: int = 200, kappa: float = 1e4, seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    h = np.logspace(0.0, np.log10(kappa), n)
    b = rng.standard_normal(n)
    xstar = b / h
    fstar = float(-0.5 * b @ xstar)

    def f(x):
        return float(0.5 * np.sum(h * x * x) - b @ x)

    def grad(x):
        return h * x - b

    return Problem(
        name=f"P1_quad_diag_k{int(kappa)}", kind="quadratic-diag", n=n,
        f=f, grad=grad, x0=np.zeros(n), fstar=fstar,
        L=float(h.max()), mu=float(h.min()), xstar=xstar,
        H0=np.diag(h), meta={"kappa": float(h.max() / h.min()), "coupling": 0.0},
    )


# --------------------------------------------------------------------------- #
# P2: rotated ill-conditioned quadratic  (coupling high: diagonal cannot help)
# --------------------------------------------------------------------------- #
def quad_rot(n: int = 200, kappa: float = 1e3, seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    h = np.logspace(0.0, np.log10(kappa), n)
    H = (Q * h) @ Q.T
    H = 0.5 * (H + H.T)
    b = rng.standard_normal(n)
    xstar = np.linalg.solve(H, b)
    fstar = float(-0.5 * b @ xstar)

    def f(x):
        return float(0.5 * x @ (H @ x) - b @ x)

    def grad(x):
        return H @ x - b

    return Problem(
        name=f"P2_quad_rot_k{int(kappa)}", kind="quadratic-rot", n=n,
        f=f, grad=grad, x0=np.zeros(n), fstar=fstar,
        L=float(h.max()), mu=float(h.min()), xstar=xstar, H0=H,
        meta={"kappa": float(h.max() / h.min()), "coupling": coupling_index(H)},
    )


# --------------------------------------------------------------------------- #
# P3: least squares, i.i.d. design (low coupling) vs Toeplitz design (high)
# --------------------------------------------------------------------------- #
def least_squares(design: str = "iid", m: int = 300, d: int = 100,
                  scale_span: float = 2.0, rho: float = 0.8,
                  noise: float = 1e-2, seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    if design == "iid":
        A = rng.standard_normal((m, d)) / np.sqrt(m)
    elif design == "toeplitz":
        idx = np.arange(m)[:, None] - np.arange(d)[None, :]
        A = rho ** np.abs(idx)
        A /= np.linalg.norm(A, axis=0, keepdims=True) / np.sqrt(d)
    else:
        raise ValueError(design)
    if scale_span > 0:
        A = A * np.logspace(0.0, -scale_span, d)[None, :]

    w = rng.standard_normal(d)
    b = A @ w + noise * rng.standard_normal(m)
    xstar, *_ = np.linalg.lstsq(A, b, rcond=None)
    fstar = float(0.5 * np.mean((A @ xstar - b) ** 2))
    H = A.T @ A / m
    L = float(np.linalg.norm(A, 2) ** 2 / m)
    mu = float(np.linalg.eigvalsh(H)[0])

    def f(x):
        r = A @ x - b
        return float(0.5 * np.mean(r * r))

    def grad(x):
        return A.T @ (A @ x - b) / m

    return Problem(
        name=f"P3_ls_{design}_span{scale_span:g}", kind="least-squares", n=d,
        f=f, grad=grad, x0=np.zeros(d), fstar=fstar,
        L=max(L, 1e-12), mu=max(mu, 1e-12), xstar=xstar, H0=H,
        meta={"m": m, "d": d, "design": design, "scale_span": scale_span,
              "kappa": float(max(L, 1e-12) / max(mu, 1e-12)),
              "coupling": coupling_index(H)},
    )


# --------------------------------------------------------------------------- #
# P4: L2 logistic regression, i.i.d. design (low coupling)
# --------------------------------------------------------------------------- #
def logistic(m: int = 2000, d: int = 100, lam: float = 1e-3,
             feature_span: float = 2.0, overlap: float = 0.5,
             seed: int = 0) -> Problem:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, d)) / np.sqrt(m) * np.sqrt(d)
    A = A * np.logspace(0.0, -feature_span, d)[None, :]
    A = A * np.sqrt(4.0 * m) / np.linalg.norm(A, 2)   # => ||A||^2/(4m) = 1
    w = rng.standard_normal(d)
    margin = A @ w
    y = np.sign(margin + overlap * np.std(margin) * rng.standard_normal(m))
    y[y == 0] = 1.0

    def f(x):
        z = -y * (A @ x)
        return float(np.mean(np.logaddexp(0.0, z)) + 0.5 * lam * x @ x)

    def grad(x):
        z = -y * (A @ x)
        sig = 1.0 / (1.0 + np.exp(-z))
        return A.T @ (-y * sig) / m + lam * x

    L = float(np.linalg.norm(A, 2) ** 2 / (4 * m) + lam)
    xstar = _refine_minimiser(f, grad, np.zeros(d))
    D0 = 0.25 * np.ones(m)  # worst-case sigmoid weight
    H0 = A.T @ (D0[:, None] * A) / m + lam * np.eye(d)

    return Problem(
        name=f"P4_logit_span{feature_span:g}", kind="logistic", n=d,
        f=f, grad=grad, x0=np.zeros(d), fstar=float(f(xstar)),
        L=L, mu=lam, xstar=xstar, H0=H0,
        meta={"m": m, "d": d, "lam": lam, "feature_span": feature_span,
              "kappa": float(L / lam), "coupling": coupling_index(H0)},
    )


# --------------------------------------------------------------------------- #
# P5: Rosenbrock (non-convex; no coupling index, no convexity)
# --------------------------------------------------------------------------- #
def rosenbrock(n: int = 2, seed: int = 0) -> Problem:
    def f(x):
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(x):
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(n, -1.2)
    x0[1::2] = 1.0
    L = _estimate_local_lipschitz(grad, x0, n, seed)
    return Problem(
        name=f"P5_rosenbrock_n{n}", kind="nonconvex", n=n,
        f=f, grad=grad, x0=x0, fstar=0.0, L=L, mu=0.0, xstar=np.ones(n),
        meta={"coupling": None},
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _refine_minimiser(f, grad, x0, gtol: float = 1e-12):
    from scipy.optimize import minimize

    res = minimize(lambda x: (f(x), grad(x)), x0, jac=True, method="L-BFGS-B",
                   options=dict(gtol=gtol, ftol=1e-16, maxiter=20000))
    x = res.x
    for _ in range(30):  # Newton polish: fstar must be exact for a 1e-10 target
        g = grad(x)
        if np.linalg.norm(g, np.inf) < 1e-13:
            break
        eps = 1e-6
        H = np.empty((x.size, x.size))
        for i in range(x.size):
            e = np.zeros(x.size)
            e[i] = eps
            H[i] = (grad(x + e) - g) / eps
        try:
            x = x - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return x


def _estimate_local_lipschitz(grad, x0, n, seed, samples: int = 300) -> float:
    """Deterministic empirical estimate, shared by every method."""
    rng = np.random.default_rng(10_000 + seed)
    scale = max(1.0, float(np.linalg.norm(x0)) / np.sqrt(n))
    pts = x0 + rng.uniform(-1.0, 1.0, size=(samples, n)) * scale
    ratios = []
    for i in range(samples - 1):
        d = pts[i + 1] - pts[i]
        nd = np.linalg.norm(d)
        if nd > 1e-12:
            ratios.append(np.linalg.norm(grad(pts[i + 1]) - grad(pts[i])) / nd)
    return max(float(np.percentile(ratios, 95.0)) * 1.5, 1.0)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
BUILDERS = {
    "P1": lambda seed: quad_diag(),
    "P2": lambda seed: quad_rot(),
    "P3i": lambda seed: least_squares(design="iid", scale_span=2.0, seed=seed),
    "P3c": lambda seed: least_squares(design="toeplitz", scale_span=2.0, seed=seed),
    "P4i": lambda seed: logistic(feature_span=2.0, seed=seed),
    "P5": lambda seed: rosenbrock(n=2),
}

SUITES = {
    "smoke": ["P1", "P3i"],
    "full": ["P1", "P2", "P3i", "P3c", "P4i", "P5"],
    "structure": ["P1", "P3i", "P3c", "P2"],
}


def build_suite(which: str = "smoke", seed: int = 0) -> list[Problem]:
    keys = SUITES.get(which, [which])
    return [BUILDERS[k](seed) for k in keys]
