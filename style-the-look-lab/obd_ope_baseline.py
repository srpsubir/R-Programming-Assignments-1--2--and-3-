"""Off-policy evaluation (OPE) baseline on the Open Bandit Dataset sample shipped with obp.

What this does
  1. Loads the logs of the *random* behaviour policy (uniform over 80 items, 3 slots).
  2. Replicates the production Bernoulli Thompson sampling (BTS) policy as the *target* policy.
  3. Estimates the target policy's value with obp's IPW / SNIPS and with our own numpy IPW,
     and checks the two implementations agree.
  4. Adds bootstrap 95% intervals, because the sample has very few clicks.

Run:  python obd_ope_baseline.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from obp.dataset import OpenBanditDataset
from obp.ope import InverseProbabilityWeighting as IPW
from obp.ope import OffPolicyEvaluation
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPS
from obp.policy import BernoulliTS

SEED = 12345
N_BOOT = 2000


def iw_values(bf, action_dist):
    """Importance weights pi_target(a|x) / pi_logging(a|x) for the logged (action, position)."""
    idx = np.arange(bf["n_rounds"])
    return action_dist[idx, bf["action"], bf["position"]] / bf["pscore"]


def ipw_and_snips(reward, w):
    return np.mean(w * reward), np.sum(w * reward) / np.sum(w)


def main():
    ds = OpenBanditDataset(behavior_policy="random", campaign="all")
    bf = ds.obtain_batch_bandit_feedback()
    n, reward = bf["n_rounds"], bf["reward"]
    base = reward.mean()
    print(f"rounds={n}  clicks={int(reward.sum())}  random-policy CTR={base:.5f}")

    policy = BernoulliTS(
        n_actions=ds.n_actions,
        len_list=ds.len_list,
        is_zozotown_prior=True,  # replicate the ZOZOTOWN production prior
        campaign="all",
        random_state=SEED,
    )
    action_dist = policy.compute_batch_action_dist(n_sim=100000, n_rounds=n)

    # --- obp's estimators
    ope = OffPolicyEvaluation(bandit_feedback=bf, ope_estimators=[IPW(), SNIPS()])
    obp_est = ope.estimate_policy_values(action_dist=action_dist)

    # --- our own numpy implementation; must match obp
    w = iw_values(bf, action_dist)
    my_ipw, my_snips = ipw_and_snips(reward, w)
    assert np.isclose(my_ipw, obp_est["ipw"]), (my_ipw, obp_est["ipw"])
    assert np.isclose(my_snips, obp_est["snipw"]), (my_snips, obp_est["snipw"])
    print("own IPW / SNIPS match obp: OK")

    # --- bootstrap over logged rounds
    rng = np.random.default_rng(SEED)
    boots = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots[b] = ipw_and_snips(reward[i], w[i])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

    print(f"\n{'estimator':<8} {'value':>9} {'rel. to random':>15}   95% bootstrap CI (rel. to random)")
    for name, v, l, h in [("IPW", my_ipw, lo[0], hi[0]), ("SNIPS", my_snips, lo[1], hi[1])]:
        print(f"{name:<8} {v:9.6f} {v / base:15.3f}   [{l / base:.3f}, {h / base:.3f}]")

    # --- reference: observed CTR of the BTS logging policy in the sample (NOT ground truth)
    bf_bts = OpenBanditDataset(behavior_policy="bts", campaign="all").obtain_batch_bandit_feedback()
    r_bts = bf_bts["reward"]
    ref = np.array([r_bts[rng.integers(0, len(r_bts), len(r_bts))].mean() for _ in range(N_BOOT)])
    print(
        f"\nreference: BTS logged CTR={r_bts.mean():.5f} ({int(r_bts.sum())} clicks) -> "
        f"{r_bts.mean() / base:.3f} x random; bootstrap CI "
        f"[{np.percentile(ref, 2.5) / base:.3f}, {np.percentile(ref, 97.5) / base:.3f}]"
    )
    print("note: the sample has ~40 clicks per file, so these intervals are wide by construction.")


if __name__ == "__main__":
    main()
