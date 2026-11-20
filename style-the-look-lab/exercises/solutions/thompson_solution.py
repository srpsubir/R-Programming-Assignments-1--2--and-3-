"""Reference solution for exercises/thompson_exercise.py. Try the exercise first."""
import numpy as np


class BetaBernoulliTS:
    def __init__(self, n_arms, alpha0=1.0, beta0=1.0, seed=None):
        self.alpha = np.full(n_arms, alpha0, dtype=float)
        self.beta = np.full(n_arms, beta0, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def posterior_mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


class LinearTS:
    def __init__(self, n_arms, dim, lam=1.0, v=1.0, seed=None):
        self.A = np.stack([lam * np.eye(dim) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self.v = v
        self.rng = np.random.default_rng(seed)

    def posterior_mean(self, arm: int) -> np.ndarray:
        return np.linalg.solve(self.A[arm], self.b[arm])

    def select(self, x: np.ndarray) -> int:
        scores = np.empty(len(self.A))
        for a in range(len(self.A)):
            mu = self.posterior_mean(a)
            if self.v > 0:
                cov = self.v ** 2 * np.linalg.inv(self.A[a])
                cov = (cov + cov.T) / 2.0
                theta = self.rng.multivariate_normal(mu, cov)
            else:
                theta = mu
            scores[a] = x @ theta
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
