"""Exercise: build Thompson sampling from scratch (numpy only).

Fill in the two classes, then run:   python exercises/check_thompson.py
(Check the reference solution with: python exercises/check_thompson.py --solution)

Part A - BetaBernoulliTS (no context).
  Each arm a has an unknown click rate theta_a with a Beta(alpha_a, beta_a) posterior.
  select(): draw one sample per arm from its posterior and play the arm with the largest draw.
  update(arm, reward): reward is 0 or 1; add it to that arm's posterior counts.
  posterior_mean(): alpha / (alpha + beta) for every arm.

Part B - LinearTS (contextual, one linear model per arm).
  Per arm keep A = lam * I + sum x x^T   and   b = sum reward * x.
  The posterior over the arm's weights is  N(A^-1 b, v^2 * A^-1).
  select(x): for every arm draw weights theta_a from its posterior, score = x . theta_a, play the argmax.
  update(arm, x, reward): update only the arm that was played.
  posterior_mean(arm): A^-1 b  (this is exactly the ridge regression solution).
"""
import numpy as np


class BetaBernoulliTS:
    def __init__(self, n_arms, alpha0=1.0, beta0=1.0, seed=None):
        # TODO: store posterior parameters for every arm and a numpy Generator seeded with `seed`
        raise NotImplementedError("TODO")

    def select(self) -> int:
        # TODO
        raise NotImplementedError("TODO")

    def update(self, arm: int, reward: float) -> None:
        # TODO
        raise NotImplementedError("TODO")

    def posterior_mean(self) -> np.ndarray:
        # TODO
        raise NotImplementedError("TODO")


class LinearTS:
    def __init__(self, n_arms, dim, lam=1.0, v=1.0, seed=None):
        # TODO: per-arm A (lam * identity) and b (zeros); keep v and a seeded Generator
        raise NotImplementedError("TODO")

    def posterior_mean(self, arm: int) -> np.ndarray:
        # TODO
        raise NotImplementedError("TODO")

    def select(self, x: np.ndarray) -> int:
        # TODO: if v == 0 the draw should simply be the posterior mean (greedy)
        raise NotImplementedError("TODO")

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        # TODO
        raise NotImplementedError("TODO")
