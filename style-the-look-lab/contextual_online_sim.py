"""Online contextual-bandit simulation with multiple seeds and uncertainty bands.

obp's own run_bandit_simulation is a REPLAY of logged data (a policy learns only on rounds where it
happens to pick the logged action). This script instead lets every policy interact live with a
synthetic environment whose true click probabilities are known, so regret is exact.

  * environment : obp SyntheticBanditDataset (binary reward = Bernoulli(expected_reward)),
                  a fresh environment per seed
  * policies    : obp EpsilonGreedy / BernoulliTS (no context), LinEpsilonGreedy, LinUCB, LinTS,
                  LogisticTS, plus our from-scratch LinearTS (reference solution of the exercise)
  * output      : mean +/- sd of cumulative regret across seeds, last-1000-round regret slope,
                  and a plot with +/-1 sd bands  (results/contextual_regret.png)

Run:  python contextual_online_sim.py [--seeds 8] [--rounds 5000]
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from obp.dataset import SyntheticBanditDataset, logistic_reward_function
from obp.policy import BernoulliTS, EpsilonGreedy, LinEpsilonGreedy, LinTS, LinUCB, LogisticTS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exercises", "solutions"))
from thompson_solution import LinearTS as ScratchLinearTS  # noqa: E402

K, D = 10, 5


def make_policies(seed):
    return {
        "EpsGreedy (no ctx)": ("obp_free", EpsilonGreedy(n_actions=K, epsilon=0.1, random_state=seed)),
        "BernoulliTS (no ctx)": ("obp_free", BernoulliTS(n_actions=K, random_state=seed)),
        "LinEpsGreedy": ("obp_ctx", LinEpsilonGreedy(dim=D, n_actions=K, epsilon=0.1, random_state=seed)),
        "LinUCB": ("obp_ctx", LinUCB(dim=D, n_actions=K, epsilon=0.1, random_state=seed)),
        "LinTS (obp)": ("obp_ctx", LinTS(dim=D, n_actions=K, random_state=seed)),
        "LinTS (scratch)": ("scratch", ScratchLinearTS(K, D, lam=1.0, v=1.0, seed=seed)),
        "LogisticTS": ("obp_ctx", LogisticTS(dim=D, n_actions=K, random_state=seed)),
    }


def run_online(kind, policy, p, ctx, rng):
    n = len(ctx)
    regret = np.empty(n)
    for i in range(n):
        if kind == "obp_free":
            a = int(policy.select_action()[0])
            r = float(rng.random() < p[i, a])
            policy.update_params(action=a, reward=r)
        elif kind == "obp_ctx":
            c = ctx[i].reshape(1, D)
            a = int(policy.select_action(c)[0])
            r = float(rng.random() < p[i, a])
            policy.update_params(action=a, reward=r, context=c)
        else:  # scratch
            a = policy.select(ctx[i])
            r = float(rng.random() < p[i, a])
            policy.update(a, ctx[i], r)
        regret[i] = p[i].max() - p[i, a]
    return np.cumsum(regret)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    n = args.rounds

    curves, random_regret = {}, []
    t0 = time.time()
    for seed in range(args.seeds):
        ds = SyntheticBanditDataset(
            n_actions=K, dim_context=D, reward_type="binary",
            reward_function=logistic_reward_function, random_state=seed,
        )
        bf = ds.obtain_batch_bandit_feedback(n_rounds=n)
        p, ctx = bf["expected_reward"], bf["context"]
        random_regret.append(np.cumsum(p.max(1) - p.mean(1)))
        for name, (kind, pol) in make_policies(seed).items():
            curves.setdefault(name, []).append(run_online(kind, pol, p, ctx, np.random.default_rng(1000 + seed)))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    mark = min(1000, n)
    print(f"\n{args.seeds} seeds x {n} rounds, K={K}, dim={D}. Cumulative regret, mean +/- sd across seeds (lower is better)")
    print(f"{'policy':<22} {'@' + str(mark):>14} {'@' + str(n):>16} {'regret/round, last 1000':>25}")
    rr = np.array(random_regret)
    print(f"{'always random':<22} {rr[:, mark - 1].mean():8.1f} +/-{rr[:, mark - 1].std():5.1f} "
          f"{rr[:, -1].mean():9.1f} +/-{rr[:, -1].std():5.1f} {np.mean((rr[:, -1] - rr[:, -1001]) / 1000):>20.4f}")
    for name, cs in sorted(curves.items(), key=lambda kv: np.mean([c[-1] for c in kv[1]])):
        c = np.array(cs)
        slope = np.mean((c[:, -1] - c[:, -1001]) / 1000)
        print(f"{name:<22} {c[:, mark - 1].mean():8.1f} +/-{c[:, mark - 1].std():5.1f} "
              f"{c[:, -1].mean():9.1f} +/-{c[:, -1].std():5.1f} {slope:>20.4f}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = np.arange(1, n + 1)
        for name, cs in curves.items():
            c = np.array(cs)
            m, s = c.mean(0), c.std(0)
            ax.plot(x, m, label=name, lw=1.6)
            ax.fill_between(x, m - s, m + s, alpha=0.12)
        ax.plot(x, rr.mean(0), "k--", lw=1, label="always random")
        ax.set_xlabel("round")
        ax.set_ylabel("cumulative regret (expected clicks lost)")
        ax.set_title(f"Online contextual bandits, {args.seeds} seeds, bands = +/-1 sd")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("results/contextual_regret.png", dpi=130)
        print("\nplot: results/contextual_regret.png")


if __name__ == "__main__":
    main()
