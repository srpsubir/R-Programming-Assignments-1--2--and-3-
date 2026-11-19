"""From-scratch bandits and OPE on a synthetic environment where the true values are known.

Part 1: cumulative regret of epsilon-greedy, UCB1 and Thompson sampling (Beta-Bernoulli).
Part 2: how IPW / SNIPS error shrinks with log size when the logging policy is uniform
        and click rates are low (similar to the Open Bandit sample).

Run:  python bandit_sim.py
"""
import numpy as np

K = 10
CTR = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015, 0.020])
BEST = CTR.max()


def run_policy(name, T, rng):
    n = np.zeros(K)  # pulls
    s = np.zeros(K)  # successes
    regret = 0.0
    for t in range(1, T + 1):
        if name == "eps-greedy":
            a = rng.integers(K) if rng.random() < 0.1 else int(np.argmax(s / np.maximum(n, 1)))
        elif name == "ucb1":
            if t <= K:
                a = t - 1
            else:
                a = int(np.argmax(s / n + np.sqrt(2 * np.log(t) / n)))
        elif name == "thompson":
            a = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        else:
            raise ValueError(name)
        r = rng.random() < CTR[a]
        n[a] += 1
        s[a] += r
        regret += BEST - CTR[a]
    return regret


def part1(T=20000, seeds=20):
    print(f"Part 1: cumulative regret (expected clicks lost), T={T}, {seeds} seeds, best CTR={BEST}")
    for name in ["eps-greedy", "ucb1", "thompson"]:
        r = np.array([run_policy(name, T, np.random.default_rng(100 + i)) for i in range(seeds)])
        print(f"  {name:<11} mean={r.mean():7.2f}  sd={r.std():6.2f}")
    print(f"  (always-random would lose about {T * (BEST - CTR.mean()):.1f})")


def ucb_regret(c, T, rng):
    n, s, regret = np.zeros(K), np.zeros(K), 0.0
    for t in range(1, T + 1):
        a = t - 1 if t <= K else int(np.argmax(s / n + c * np.sqrt(2 * np.log(t) / n)))
        r = rng.random() < CTR[a]
        n[a] += 1
        s[a] += r
        regret += BEST - CTR[a]
    return regret


def part1b(T=20000, seeds=20):
    # UCB1's textbook bonus (c=1) assumes rewards on a scale where gaps are large.
    # With CTRs of 0.2-2% the bonus dwarfs the gaps, so UCB1 behaves almost like random.
    # NOTE: c is tuned on the same environment and seeds used to report it, so small c is flattered.
    print("\nPart 1b: UCB1 exploration-bonus scale c (c=1 is textbook)")
    for c in [1.0, 0.3, 0.1, 0.05, 0.02]:
        r = np.array([ucb_regret(c, T, np.random.default_rng(100 + i)) for i in range(seeds)])
        print(f"  c={c:<5} mean={r.mean():7.2f}  sd={r.std():6.2f}")


def part2(reps=500):
    # target policy: softmax over true CTR (a stand-in for a learned policy); logging: uniform
    pi = np.exp(CTR / 0.003)
    pi /= pi.sum()
    true_value = float(pi @ CTR)
    base = CTR.mean()
    print(f"\nPart 2: OPE error vs log size. true target value={true_value:.5f} "
          f"({true_value / base:.2f} x uniform), uniform logging CTR={base:.5f}")
    print(f"  {'n logged':>9} {'exp. clicks':>12} {'IPW mean abs err':>18} {'SNIPS mean abs err':>20}")
    rng = np.random.default_rng(7)
    for n in [1000, 5000, 10000, 50000, 200000]:
        e_ipw, e_snips = [], []
        for _ in range(reps):
            a = rng.integers(0, K, n)
            r = rng.random(n) < CTR[a]
            w = pi[a] / (1.0 / K)
            e_ipw.append(abs(np.mean(w * r) - true_value))
            e_snips.append(abs(np.sum(w * r) / np.sum(w) - true_value))
        print(f"  {n:9d} {n * base:12.1f} {np.mean(e_ipw) / true_value:17.1%} {np.mean(e_snips) / true_value:19.1%}")
    print("  errors are relative to the true value")


def part3(reps=300):
    """DM vs DR when the true value is known, under three reward models.

    good    : per-arm CTR, cross-fitted (2 folds), mild shrinkage
    shrunk  : per-arm CTR shrunk hard toward the global CTR (prior of 300 pseudo-impressions)
    nothing : constant = global CTR (a model that knows nothing about arms)
    """
    pi = np.exp(CTR / 0.003)
    pi /= pi.sum()
    true_value = float(pi @ CTR)
    print(f"\nPart 3: DM vs DR, uniform logging, true value={true_value:.5f}. "
          "Cells: relative bias / relative mean abs error")
    models = {"good": 5.0, "shrunk": 300.0, "nothing": None}
    header = f"  {'n':>7} {'IPW':>13}" + "".join(f" {'DM-' + m:>13} {'DR-' + m:>13}" for m in models)
    print(header)
    rng = np.random.default_rng(11)
    for n in [2000, 10000, 50000]:
        res = {k: [] for k in ["IPW"] + [f"{e}-{m}" for m in models for e in ("DM", "DR")]}
        for _ in range(reps):
            a = rng.integers(0, K, n)
            r = (rng.random(n) < CTR[a]).astype(float)
            w = pi[a] * K
            res["IPW"].append(np.mean(w * r))
            fold = rng.integers(0, 2, n)
            for m, prior in models.items():
                dm_i = np.empty(n)
                resid = np.empty(n)
                for f in (0, 1):
                    tr, te = fold != f, fold == f
                    g = r[tr].mean()
                    if prior is None:
                        q = np.full(K, g)
                    else:
                        cnt = np.bincount(a[tr], minlength=K)
                        suc = np.bincount(a[tr], weights=r[tr], minlength=K)
                        q = (suc + prior * g) / (cnt + prior)
                    dm_i[te] = pi @ q
                    resid[te] = r[te] - q[a[te]]
                res[f"DM-{m}"].append(dm_i.mean())
                res[f"DR-{m}"].append((dm_i + w * resid).mean())
        def cell(v):
            v = np.array(v) - true_value
            return f"{v.mean() / true_value:+6.1%}/{np.abs(v).mean() / true_value:5.1%}"
        row = f"  {n:7d} {cell(res['IPW']):>13}"
        for m in models:
            row += f" {cell(res['DM-' + m]):>13} {cell(res['DR-' + m]):>13}"
        print(row)


if __name__ == "__main__":
    part1()
    part1b()
    part2()
    part3()
