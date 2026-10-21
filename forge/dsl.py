"""The method DSL: a candidate method is *data*, not code.

A candidate is a small dict that fully determines an AIPG instance.  Keeping it
data is what allows the static gate (and a human) to check a proposal before it
runs, and what makes the search log readable and replayable.

Every key has a default; `validate()` returns a list of human-readable
violations (empty list = legal).
"""

from __future__ import annotations

import copy
from typing import Any

from .methods import AIPG, Evaluator
from .problems import Problem

DEFAULT: dict[str, Any] = {
    "name": "aipg",
    "precond": {"type": "diag", "source": "secant", "window": 5, "ema": 0.5, "adaptive": True},
    "step": {"rule": "minimax"},
    "accel": {"type": "chebyshev", "beta_cap": 0.9},
    "safeguard": {"type": "armijo", "c": 1e-4, "halvings": 25},
}

_ALLOWED = {
    ("precond", "type"): {"diag", "scalar"},
    ("precond", "source"): {"secant"},
    ("step", "rule"): {"minimax", "invu"},
    ("accel", "type"): {"none", "chebyshev"},
    ("safeguard", "type"): {"none", "armijo"},
}

_RANGES = {
    ("precond", "window"): (1, 50),
    ("precond", "ema"): (0.0, 1.0),
    ("accel", "beta_cap"): (0.0, 0.999),
    ("safeguard", "c"): (0.0, 0.5),
    ("safeguard", "halvings"): (0, 60),
}


def normalize(cfg: dict | None) -> dict:
    out = copy.deepcopy(DEFAULT)
    for section, values in (cfg or {}).items():
        if section == "name":
            out["name"] = values
        elif isinstance(values, dict):
            out.setdefault(section, {}).update(values)
        else:
            out[section] = values
    return out


def validate(cfg: dict) -> list[str]:
    """Static gate.  Returns violations; an empty list means the candidate is legal."""
    cfg = normalize(cfg)
    bad: list[str] = []
    for (section, key), allowed in _ALLOWED.items():
        value = cfg.get(section, {}).get(key)
        if value not in allowed:
            bad.append(f"{section}.{key}={value!r} 不在允许集合 {sorted(allowed)} 内")
    for (section, key), (lo, hi) in _RANGES.items():
        value = cfg.get(section, {}).get(key)
        if value is None:
            bad.append(f"缺少 {section}.{key}")
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            bad.append(f"{section}.{key}={value!r} 不是数值")
            continue
        if not (lo <= value <= hi):
            bad.append(f"{section}.{key}={value} 超出范围 [{lo}, {hi}]")

    # R3 (static part): without a safeguard the step must be provably safe, i.e.
    # no momentum and a diagonal model that reduces to D <= 1/L.
    if cfg["safeguard"]["type"] == "none" and cfg["accel"]["type"] != "none":
        bad.append("R3: 无 safeguard 时不允许使用动量加速（无法保证下降引理前提）")
    if cfg["safeguard"]["type"] == "none" and cfg["step"]["rule"] != "invu":
        bad.append("R3: 无 safeguard 时步长规则必须是 1/u（保证 alpha <= 1/L）")
    return bad


def to_method(cfg: dict, prob: Problem, ev: Evaluator) -> AIPG:
    cfg = normalize(cfg)
    return AIPG(prob, ev, cfg)


def describe(cfg: dict) -> str:
    cfg = normalize(cfg)
    pr, ac = cfg["precond"], cfg["accel"]
    pre = f"{pr['type']}/w{pr['window']}/ema{pr['ema']}"
    if not pr.get("adaptive", True):
        pre += "/fixed"
    acc = ac["type"]
    if acc == "chebyshev":
        acc += f"(beta<={ac['beta_cap']})"
    return f"{pre} + {cfg['step']['rule']} + {acc} + {cfg['safeguard']['type']}"
