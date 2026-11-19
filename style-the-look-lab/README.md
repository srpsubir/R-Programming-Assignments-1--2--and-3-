# style-the-look-lab

Hands-on refresher for fashion-e-commerce personalization: contextual bandits, off-policy evaluation (OPE) and, later, a Style the Look item recommender. Status below is what actually ran in this folder.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python obd_ope_baseline.py
python obd_dr_dm.py
python bandit_sim.py
```

Pins matter: `obp` 0.4.1 fails on pandas 2+ (`DataFrame.drop(col, 1)`), and pandas 1.5.x has no wheels for Python 3.13.

## 1. Open Bandit sample: IPW and SNIPS (`obd_ope_baseline.py`)

Uses the small sample bundled inside the `obp` package (10,000 rows per policy and campaign; 80 items, 3 slots, 89 columns). The full dataset (listed by ZOZO as 26M rows) could not be downloaded in the session where this was built, so it has **not** been used.

Target = replicated Bernoulli Thompson sampling, logs = random policy. IPW gives 1.198 x the random policy, the same value the OBP README prints. The numpy IPW/SNIPS match obp's (asserted).

## 2. Open Bandit sample: DM, DR, SNDR (`obd_dr_dm.py`)

Reward model: per-slot logistic regression on user context plus item features, cross-fitted in 2 folds. Bootstrap (300 resamples) refits the model each time, with folds tied to the original round.

| estimator | relative to random | 95% bootstrap CI |
|---|---|---|
| IPW | 1.198 | [0.390, 2.462] |
| SNIPS | 1.257 | [0.410, 2.581] |
| DM (logreg) | 0.931 | [0.565, 1.214] |
| DR (logreg) | 1.239 | [0.428, 2.502] |
| SNDR (logreg) | 1.254 | [0.406, 2.590] |
| DM (constant model) | 1.000 | [0.684, 1.330] |
| DR (constant model) | 1.245 | [0.442, 2.508] |
| reference: BTS logged CTR (42 clicks) | 1.105 | not ground truth |

- Checks that pass: own DM/DR/SNDR equal obp's given the same reward estimates; DM with a constant model equals the logging CTR exactly.
- DM (0.931) sits below IPW, DR and SNIPS (1.198 to 1.257). All intervals overlap.
- The random-policy log has 38 clicks, so the reward model has very little to learn from and the sample cannot rank these estimators. IPW's interval differs slightly from `obd_ope_baseline.py` (300 vs 2000 resamples, different random draws).

## 3. From-scratch bandits and OPE on known truth (`bandit_sim.py`)

Synthetic 10-arm environment, CTRs 0.2%-2%, so true values are known.

- **Regret** over T=20,000, 20 seeds: Thompson 72.3, epsilon-greedy (0.1) 111.5, UCB1 218.0, random about 230. UCB1 is near random because its textbook bonus dwarfs the CTR gaps; scaling the bonus to 0.05 gives 55.1 (sd 39.7), tuned on the same seeds so it flatters UCB1.
- **OPE error vs log size** (IPW): about 47% relative error at ~8 expected clicks, 15% at ~85, 3% at ~1,700.
- **DM vs DR** (300 repetitions per row; relative bias / relative mean abs error):

| n | IPW | DM good | DR good | DM shrunk | DR shrunk | DM nothing | DR nothing |
|---|---|---|---|---|---|---|---|
| 2,000 | +7.1% / 32.1% | +3.6% / 30.3% | +6.8% / 31.7% | -37.4% / 37.9% | +6.9% / 31.7% | -52.0% / 52.0% | +6.9% / 31.8% |
| 10,000 | -0.6% / 14.6% | -1.2% / 14.4% | -0.7% / 14.6% | -20.5% / 20.9% | -0.7% / 14.6% | -53.4% / 53.4% | -0.7% / 14.5% |
| 50,000 | +0.1% / 6.0% | 0.0% / 5.9% | +0.2% / 5.9% | -5.6% / 7.2% | +0.2% / 5.9% | -53.1% / 53.1% | +0.1% / 5.9% |

  DM with a no-information model stays about 52-53% too low at every n. DM with a heavily shrunk model is biased but the bias shrinks as n grows. DR stayed close to unbiased under all three reward models. In this setting DR's error was about the same as IPW's, so no variance reduction showed up. Simulation noise was not quantified.

## Not done yet

- H&M Personalized Fashion Recommendations: not downloaded (Kaggle and Hugging Face are blocked from the build session; Kaggle also requires accepting the competition rules). Schema, row counts and images were confirmed through Apify, not loaded.
- Style the Look item recommender: no code yet.
- Delayed returns (about 30 days after sale) are not modelled anywhere.
