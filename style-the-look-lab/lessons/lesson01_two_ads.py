"""Lessons 1-2 demo: two ads, regret, why simple strategies fail, and Thompson sampling.

Run:  python lessons/lesson01_two_ads.py      (writes results/lesson_beta_posteriors.png)
Everything is vectorised over RUNS independent repetitions of the same experiment.
"""
import os

import numpy as np

P = np.array([0.04, 0.06])  # true click rates of ad A and ad B (unknown to the algorithm)
T, RUNS = 5000, 2000
GAP = P.max() - P  # expected clicks lost each time an ad is shown


def simulate(policy, seed):
    rng = np.random.default_rng(seed)
    n = np.zeros((RUNS, 2))  # impressions per ad
    s = np.zeros((RUNS, 2))  # clicks per ad
    regret = np.zeros(RUNS)
    worse_late = np.zeros(RUNS)  # times the worse ad was shown in the last 1000 rounds
    idx = np.arange(RUNS)
    for t in range(T):
        a = policy(t, n, s, rng)
        click = rng.random(RUNS) < P[a]
        n[idx, a] += 1
        s[idx, a] += click
        regret += GAP[a]
        if t >= T - 1000:
            worse_late += (a == 0)
    return regret, worse_late / 1000


def pick_best_observed(n, s, rng):
    rate = s / np.maximum(n, 1) + rng.random(n.shape) * 1e-9  # random tie-break
    return rate.argmax(1)


def random_policy(t, n, s, rng):
    return rng.integers(0, 2, RUNS)


def greedy(t, n, s, rng):  # show each ad once, then always the best observed rate
    return np.full(RUNS, t) if t < 2 else pick_best_observed(n, s, rng)


def explore_then_commit(t, n, s, rng):  # 500 rounds of 50/50, then the best observed rate
    return np.full(RUNS, t % 2) if t < 500 else pick_best_observed(n, s, rng)


def eps_greedy(t, n, s, rng):
    a = np.full(RUNS, t) if t < 2 else pick_best_observed(n, s, rng)
    explore = rng.random(RUNS) < 0.1
    return np.where(explore, rng.integers(0, 2, RUNS), a)


def thompson(t, n, s, rng):
    draws = rng.beta(1 + s, 1 + n - s)  # one plausible click rate per ad, from its Beta posterior
    return draws.argmax(1)


if __name__ == "__main__":
    print(f"Two ads, true click rates {P[0]:.0%} and {P[1]:.0%}; {T} impressions; {RUNS} repeated experiments.")
    print("Regret = expected clicks lost vs always showing the better ad (0.02 per impression of the worse ad).\n")
    print(f"{'strategy':<22} {'mean regret':>11} {'sd':>6} {'runs where worse ad got most of the last 1000':>48}")
    for i, (name, pol) in enumerate([
        ("random 50/50", random_policy), ("greedy", greedy), ("explore-then-commit", explore_then_commit),
        ("epsilon-greedy (10%)", eps_greedy), ("Thompson sampling", thompson)]):
        reg, worse = simulate(pol, seed=i)
        print(f"{name:<22} {reg.mean():11.1f} {reg.std():6.1f} {np.mean(worse > 0.5):>47.1%}")

    # --- Lesson 2: what a Beta belief looks like and how often Thompson sampling would play ad B
    from scipy.stats import beta as beta_dist
    rng = np.random.default_rng(0)
    stages = [("after 60 impressions", (2, 20), (6, 40)),
              ("after 600 impressions", (20, 200), (60, 400)),
              ("after 6,000 impressions", (200, 2000), (600, 4000))]
    print("\nSame observed rates (ad A 10%, ad B 15%), more and more evidence:")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
    x = np.linspace(0, 0.35, 600)
    for ax, (title, (sa, na), (sb, nb)) in zip(axes, stages):
        a_post, b_post = (1 + sa, 1 + na - sa), (1 + sb, 1 + nb - sb)
        pb = np.mean(rng.beta(*b_post, 200000) > rng.beta(*a_post, 200000))
        print(f"  {title:<24} A: {sa}/{na} clicks -> Beta{a_post};  B: {sb}/{nb} -> Beta{b_post};  "
              f"chance a draw favours B = {pb:.0%}")
        ax.plot(x, beta_dist.pdf(x, *a_post), label=f"ad A ({sa}/{na})")
        ax.plot(x, beta_dist.pdf(x, *b_post), label=f"ad B ({sb}/{nb})")
        ax.set_title(f"{title}\nP(B drawn above A) = {pb:.0%}", fontsize=10)
        ax.set_xlabel("possible true click rate")
        ax.legend(fontsize=8)
        ax.set_yticks([])
    axes[0].set_ylabel("belief (density)")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/lesson_beta_posteriors.png", dpi=130)
    print("\nplot: results/lesson_beta_posteriors.png")
