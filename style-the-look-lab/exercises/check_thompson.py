"""Checks for the Thompson sampling exercise.

    python exercises/check_thompson.py              # your file: exercises/thompson_exercise.py
    python exercises/check_thompson.py --solution   # the reference solution
"""
import importlib
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "solutions"))
mod = importlib.import_module("thompson_solution" if "--solution" in sys.argv else "thompson_exercise")
BetaBernoulliTS, LinearTS = mod.BetaBernoulliTS, mod.LinearTS


def t_beta_posterior_mean():
    ts = BetaBernoulliTS(2, seed=0)
    for r in (1, 1, 1, 0):
        ts.update(0, r)
    for r in (0, 0, 0, 0):
        ts.update(1, r)
    assert np.allclose(ts.posterior_mean(), [4 / 6, 1 / 6]), ts.posterior_mean()


def t_beta_select_valid_and_seeded():
    a, b = BetaBernoulliTS(4, seed=3), BetaBernoulliTS(4, seed=3)
    sa, sb = [a.select() for _ in range(50)], [b.select() for _ in range(50)]
    assert sa == sb, "same seed must give the same sequence"
    assert all(isinstance(s, int) and 0 <= s < 4 for s in sa), "select must return an int in [0, n_arms)"
    assert len(set(sa)) > 1, "with a flat prior it should explore several arms"


def t_beta_learns():
    p = np.array([0.1, 0.5, 0.9])
    env, ts, picks = np.random.default_rng(5), BetaBernoulliTS(3, seed=1), []
    for _ in range(3000):
        a = ts.select()
        ts.update(a, float(env.random() < p[a]))
        picks.append(a)
    frac = np.mean(np.array(picks[-500:]) == 2)
    assert frac > 0.85, f"best arm played in only {frac:.0%} of the last 500 rounds"


def t_linear_posterior_mean_is_ridge():
    rng = np.random.default_rng(0)
    n, d, lam = 200, 4, 2.0
    ts = LinearTS(2, d, lam=lam, v=1.0, seed=0)
    X, r, arms = rng.normal(size=(n, d)), rng.normal(size=n), rng.integers(0, 2, n)
    for x, y, a in zip(X, r, arms):
        ts.update(int(a), x, float(y))
    for a in (0, 1):
        Xa, ya = X[arms == a], r[arms == a]
        ridge = np.linalg.solve(lam * np.eye(d) + Xa.T @ Xa, Xa.T @ ya)
        assert np.allclose(ts.posterior_mean(a), ridge), f"arm {a}: posterior mean != ridge solution"


def t_linear_v0_is_greedy():
    rng = np.random.default_rng(1)
    ts = LinearTS(3, 3, v=0.0, seed=0)
    for _ in range(100):
        x = rng.normal(size=3)
        ts.update(int(rng.integers(0, 3)), x, float(rng.normal()))
    for _ in range(50):
        x = rng.normal(size=3)
        expected = int(np.argmax([x @ ts.posterior_mean(a) for a in range(3)]))
        assert ts.select(x) == expected, "with v=0 select must be argmax of x . posterior_mean"


def t_linear_explores_then_concentrates():
    x = np.array([1.0, 0.0, 0.0])
    fresh = LinearTS(3, 3, v=1.0, seed=0)
    assert len({fresh.select(x) for _ in range(200)}) == 3, "with no data every arm should get tried"
    ts = LinearTS(3, 3, v=1.0, seed=0)
    for _ in range(500):  # arm 1 is clearly best for this context
        for a, y in ((0, 0.0), (1, 1.0), (2, 0.0)):
            ts.update(a, x, y)
    frac = np.mean([ts.select(x) == 1 for _ in range(200)])
    assert frac > 0.95, f"after lots of evidence arm 1 chosen only {frac:.0%} of the time"


def t_linear_learns():
    rng = np.random.default_rng(7)
    K, d = 3, 3
    theta = rng.normal(size=(K, d))
    ts, ok = LinearTS(K, d, v=1.0, seed=2), []
    for _ in range(3000):
        x = rng.normal(size=d)
        a = ts.select(x)
        ts.update(a, x, float(x @ theta[a] + 0.1 * rng.normal()))
        ok.append(a == int(np.argmax(theta @ x)))
    frac = np.mean(ok[-500:])
    assert frac > 0.7, f"oracle arm chosen in only {frac:.0%} of the last 500 rounds"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
failed = 0
for t in TESTS:
    try:
        t()
        print(f"PASS  {t.__name__}")
    except NotImplementedError:
        failed += 1
        print(f"TODO  {t.__name__}  (not implemented yet)")
    except AssertionError as e:
        failed += 1
        print(f"FAIL  {t.__name__}: {e}")
    except Exception:
        failed += 1
        print(f"ERROR {t.__name__}:\n{traceback.format_exc()}")
print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
sys.exit(1 if failed else 0)
