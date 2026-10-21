"""Minimal regression tests: run with `python tests/test_smoke.py` or pytest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.bench import run_one
from forge.dsl import normalize, validate
from forge.problems import BUILDERS
from forge.verify import check_dynamic


def test_dsl_validation():
    assert validate(normalize(None)) == []
    assert validate(normalize({"accel": {"type": "nesterov"}}))
    assert validate(normalize({"precond": {"ema": 1.5}}))
    # no safeguard + momentum must be refused by the static gate
    assert validate(normalize({"safeguard": {"type": "none"}}))


def test_aipg_converges_on_diagonal_quadratic():
    prob = BUILDERS["P1"](0)
    r = run_one(prob, "aipg", seed=0)
    assert r["converged"], r
    assert r["grad_calls"] < 200, r["grad_calls"]
    assert r["gap"] < 1e-8


def test_runner_fields_and_monotonicity():
    prob = BUILDERS["P3i"](0)
    r = run_one(prob, "aipg", seed=0)
    for key in ("problem", "method", "seed", "grad_calls", "gap", "trace", "converged"):
        assert key in r
    gaps = [row[0] for row in r["trace"]]
    assert gaps[-1] <= gaps[0], gaps


def test_gate_accepts_default_and_rejects_unsafe():
    assert check_dynamic(normalize(None)).passed
    unsafe = normalize({"step": {"rule": "invu"}, "accel": {"type": "none"},
                        "safeguard": {"type": "none"}})
    res = check_dynamic(unsafe, prob=BUILDERS["P2"](0), iters=80)
    assert not res.passed, res.rules


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
