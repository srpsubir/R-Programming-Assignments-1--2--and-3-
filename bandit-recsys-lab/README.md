# bandit-recsys-lab

Hands-on refresher for fashion-e-commerce personalization: contextual bandits, off-policy evaluation (OPE) and an item recommender. Status below is what actually ran in this folder.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python obd_ope_baseline.py
python bandit_sim.py
```

Pins matter: `obp` 0.4.1 fails on pandas 2+ (`DataFrame.drop(col, 1)`), and pandas 1.5.x has no wheels for Python 3.13.

## 1. Open Bandit sample: `obd_ope_baseline.py`

Uses the small sample bundled inside the `obp` package (10,000 rows per policy and campaign; 80 items, 3 slots, 89 columns). The full dataset lives on a host that was blocked in the session where this was built, so it has **not** been used.

Results from the run (target = replicated Bernoulli Thompson sampling, logs = random policy):

| estimator | value relative to random | 95% bootstrap CI |
|---|---|---|
| IPW | 1.198 | [0.396, 2.487] |
| SNIPS | 1.257 | [0.418, 2.611] |
| reference: BTS logged CTR (42 clicks) | 1.105 | [0.789, 1.448] |

- The IPW value of 1.198 matches the figure printed in the OBP README, and our numpy IPW/SNIPS match obp's (asserted in the script).
- The random-policy log has only 38 clicks, so the intervals are very wide. The sample cannot rank these estimators.
- The "reference" row is the observed CTR of the BTS logging policy in the sample. It is not a ground-truth policy value for the replicated policy.

## 2. From-scratch bandits and OPE: `bandit_sim.py`

Synthetic 10-arm environment, CTRs 0.2%-2%, so the true values are known.

- Regret over T=20,000, 20 seeds: Thompson 72.3, epsilon-greedy (0.1) 111.5, UCB1 218.0, random about 230.
- UCB1 is near random because its textbook bonus is far larger than the CTR gaps. Shrinking the bonus scale to 0.05 gives 55.1 (sd 39.7). That scale was tuned on the same environment and seeds, so it flatters UCB1.
- OPE error vs log size (uniform logging, softmax target): about 47% relative error at ~8 expected clicks, 15% at ~85 clicks, 3% at ~1,700 clicks. IPW and SNIPS were nearly identical here.

## Not done yet

- H&M Personalized Fashion Recommendations: not downloaded (Kaggle host was blocked; needs an API credential and rule acceptance on the Kaggle account).
- Delayed returns (about 30 days after sale) are not modelled anywhere.
- Slate-level estimators and doubly robust / direct-method comparisons are not run.
