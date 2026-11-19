"""Direct method (DM), doubly robust (DR) and self-normalized DR on the Open Bandit sample.

Extends obd_ope_baseline.py. Same setup: logs from the random policy, target = replicated
Bernoulli Thompson sampling policy.

  * Reward model q(x, a): one logistic regression per slot on [user context, item features],
    cross-fitted (2 folds) so q is never evaluated on rows it was trained on.
  * DM, DR and SNDR are implemented in numpy and asserted equal to obp's when both are fed the
    same reward estimates.
  * Bootstrap refits the reward model each time. Folds are assigned per ORIGINAL round, so
    duplicated rows in a bootstrap sample stay in the same fold (no train/test leakage).
  * A constant reward model (global CTR) is included as a sanity check: DM with a model that
    knows nothing about the target policy must return exactly the logging policy's CTR.

Run:  python obd_dr_dm.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from obp.dataset import OpenBanditDataset
from obp.ope import DirectMethod as OBP_DM
from obp.ope import DoublyRobust as OBP_DR
from obp.ope import OffPolicyEvaluation
from obp.ope import SelfNormalizedDoublyRobust as OBP_SNDR
from obp.policy import BernoulliTS
from sklearn.linear_model import LogisticRegression

SEED = 12345
N_BOOT = 300
N_FOLDS = 2
C_REG = 0.1


def fit_predict_q(context, action_ctx, action, reward, position, fold, n_actions, len_list):
    """Cross-fitted q_hat[i, a] for the slot logged in round i (shape n_rounds x n_actions)."""
    n = len(reward)
    q = np.zeros((n, n_actions))
    for f in range(N_FOLDS):
        test = fold == f
        for pos in range(len_list):
            tr = (~test) & (position == pos)
            te = test & (position == pos)
            if te.sum() == 0:
                continue
            X_tr = np.hstack([context[tr], action_ctx[action[tr]]])
            y_tr = reward[tr]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                q[te, :] = y_tr.mean() if len(y_tr) else 0.0
                continue
            clf = LogisticRegression(C=C_REG, max_iter=500).fit(X_tr, y_tr)
            ctx_te = context[te]
            # predict for every action: tile contexts x actions
            X_te = np.hstack(
                [np.repeat(ctx_te, n_actions, axis=0), np.tile(action_ctx, (len(ctx_te), 1))]
            )
            q[te, :] = clf.predict_proba(X_te)[:, 1].reshape(len(ctx_te), n_actions)
    return q


def estimators(reward, action, position, pscore, action_dist, q):
    """DM, DR, SNDR from per-round q (n x K at the logged slot). action_dist is n x K x L."""
    n = len(reward)
    idx = np.arange(n)
    pi = action_dist[idx, :, position]  # n x K at the logged slot
    iw = pi[idx, action] / pscore
    dm_i = (pi * q).sum(axis=1)
    resid = reward - q[idx, action]
    dm = dm_i.mean()
    dr = (dm_i + iw * resid).mean()
    sndr = (dm_i + iw * resid / iw.mean()).mean()
    return dm, dr, sndr


def to_3d(q, position, len_list):
    """Place q at the logged slot so obp's estimators (which index [round, :, position]) can use it."""
    n, K = q.shape
    out = np.zeros((n, K, len_list))
    out[np.arange(n), :, position] = q
    return out


def main():
    ds = OpenBanditDataset(behavior_policy="random", campaign="all")
    bf = ds.obtain_batch_bandit_feedback()
    n, K, L = bf["n_rounds"], ds.n_actions, ds.len_list
    ctx, act_ctx = bf["context"], bf["action_context"]
    a, pos, r, ps = bf["action"], bf["position"], bf["reward"], bf["pscore"]
    base = r.mean()
    print(f"rounds={n} clicks={int(r.sum())} random CTR={base:.5f}")

    policy = BernoulliTS(n_actions=K, len_list=L, is_zozotown_prior=True, campaign="all", random_state=SEED)
    action_dist = policy.compute_batch_action_dist(n_sim=100000, n_rounds=n)

    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, N_FOLDS, n)
    q = fit_predict_q(ctx, act_ctx, a, r, pos, fold, K, L)
    q_const = np.full((n, K), base)

    # --- own vs obp, same q
    dm, dr, sndr = estimators(r, a, pos, ps, action_dist, q)
    q3 = to_3d(q, pos, L)
    ope = OffPolicyEvaluation(bandit_feedback=bf, ope_estimators=[OBP_DM(), OBP_DR(), OBP_SNDR()])
    o = ope.estimate_policy_values(action_dist=action_dist, estimated_rewards_by_reg_model=q3)
    for name, mine, theirs in [("dm", dm, o["dm"]), ("dr", dr, o["dr"]), ("sndr", sndr, o["sndr"])]:
        assert np.isclose(mine, theirs, rtol=1e-6), (name, mine, theirs)
    print("own DM / DR / SNDR match obp given the same reward model: OK")

    dm_c, dr_c, sndr_c = estimators(r, a, pos, ps, action_dist, q_const)
    assert np.isclose(dm_c, base), (dm_c, base)
    print("DM with a constant reward model equals the logging CTR: OK")

    # --- IPW / SNIPS for the same table
    iw = action_dist[np.arange(n), a, pos] / ps
    ipw, snips = np.mean(iw * r), np.sum(iw * r) / np.sum(iw)

    # --- bootstrap (refit q each time; folds follow original rounds)
    names = ["IPW", "SNIPS", "DM (logreg)", "DR (logreg)", "SNDR (logreg)", "DM (constant)", "DR (constant)"]
    boots = np.empty((N_BOOT, len(names)))
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        iwb = iw[i]
        qb = fit_predict_q(ctx[i], act_ctx, a[i], r[i], pos[i], fold[i], K, L)
        d, g, s = estimators(r[i], a[i], pos[i], ps[i], action_dist[i], qb)
        dc, gc, _ = estimators(r[i], a[i], pos[i], ps[i], action_dist[i], np.full((n, K), r[i].mean()))
        boots[b] = [np.mean(iwb * r[i]), np.sum(iwb * r[i]) / np.sum(iwb), d, g, s, dc, gc]
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    point = [ipw, snips, dm, dr, sndr, dm_c, dr_c]

    print(f"\n{'estimator':<15} {'rel. to random':>15}   95% bootstrap CI (rel. to random)")
    for nm, v, l, h in zip(names, point, lo, hi):
        print(f"{nm:<15} {v / base:15.3f}   [{l / base:.3f}, {h / base:.3f}]")

    r_bts = OpenBanditDataset(behavior_policy="bts", campaign="all").obtain_batch_bandit_feedback()["reward"]
    print(f"\nreference: BTS logged CTR = {r_bts.mean() / base:.3f} x random ({int(r_bts.sum())} clicks; not ground truth)")
    print(f"reward model spread: q ranges {q.min():.5f}..{q.max():.5f} (random CTR {base:.5f}); "
          f"{int(r.sum())} clicks is very little to learn from")


if __name__ == "__main__":
    main()
