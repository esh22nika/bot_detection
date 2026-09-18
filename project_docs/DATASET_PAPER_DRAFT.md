# BotDriftBench: An Empirically Calibrated Synthetic Longitudinal Benchmark for Drift-Aware Social Bot Detection

**Authors:** Eshan Amballa

---

## Abstract

Evaluating social bot detectors under temporal behavioral drift requires
longitudinal benchmark data with known drift ground truth — a resource that
existing static datasets do not provide. We present **BotDriftBench v0.4**, an
empirically calibrated synthetic longitudinal benchmark designed for
drift-aware bot detection research. The benchmark contains 1,000 accounts
(500 human, 500 bot) observed across 20 temporal windows, producing 20,000
observations with 27 behavioral and profile features. Feature distributions
are fitted from the TwiBot-22 reliable subset (294,478 human, 38,986 bot
users) and cross-referenced against Cresci-2017. Five controlled drift
scenarios inject gradual behavioral shifts into distinct feature subsets at
known onset points, enabling systematic evaluation of drift detection,
attribution, and adaptation methods. Version 0.4 reduces measurement noise
(4% relative / 1.5% absolute) to better reflect realistic monthly behavioral
aggregates, strengthens archetype feature modifiers for improved class
separation, adds a `difficulty_tier` column (easy/realistic/hard) for
tier-stratified evaluation, and provides a pre-registered 3-way account-
disjoint evaluation protocol (600 train / 200 validation / 200 test accounts,
split seed 99). SHA-256:
`4b20d673eb3fc994f652bcd9ea5042f90f50d9d03372c181b112f85143bd8f21`.
We document the calibration methodology, validate that injected drift
produces measurable distributional shifts while control accounts remain stable,
and report baseline evaluation results demonstrating the benchmark's utility.
BotDriftBench v0.4 is released under CC BY 4.0 with full generation code and
configuration for reproducibility.

**Keywords:** benchmark dataset, social bot detection, concept drift,
synthetic data generation, TwiBot-22, temporal evaluation, longitudinal
benchmark, account-disjoint split

---

## 1. Introduction

Social bot detection has advanced significantly over the past decade, driven
by increasingly sophisticated benchmark datasets [1, 2, 3]. However, the
dominant evaluation paradigm treats bot detection as a static classification
task: models are trained and tested on data collected at a single point in
time, with no mechanism to study how performance changes as bot behavior
evolves.

This static paradigm has a fundamental limitation. Real-world bot operators
adapt their strategies over time [4, 5], causing the feature distributions
that classifiers rely on to shift — a phenomenon known as behavioral drift
or concept drift [6]. A detector that performs well on a static benchmark
may degrade silently when deployed on evolving data, and there is no
standardized way to evaluate its robustness to such changes.

Three specific gaps motivate the creation of BotDriftBench:

1. **No longitudinal benchmarks with drift ground truth.** Existing bot
   detection datasets [1, 2, 3] provide cross-sectional snapshots. While
   some longitudinal studies have been conducted [4], no publicly available
   benchmark provides temporal trajectories with known drift onset, affected
   features, and drift magnitude.

2. **No controlled drift scenarios.** Without ground truth about when and
   how drift occurs, it is impossible to systematically evaluate drift
   detection methods. Observed behavioral changes in real data confound
   genuine bot adaptation with platform changes, data collection artifacts,
   and natural user behavior evolution.

3. **No standardized evaluation protocol.** Different studies use different
   train/test splits, features, and temporal granularities, making
   cross-study comparison difficult.

BotDriftBench addresses these gaps by providing a synthetic benchmark with
*controlled* drift — the drift onset, affected features, and magnitude are
specified by the benchmark designer and known to the evaluator. This enables
rigorous, reproducible evaluation of drift detection accuracy, attribution
precision, and adaptation effectiveness.

The key design decision is to generate *synthetic* accounts rather than
collecting longitudinal data from live platforms. This choice is driven by
practical and ethical considerations: longitudinal bot data with verified
labels is extremely difficult to collect at scale, platform API access is
increasingly restricted, and redistributing raw user data raises privacy
concerns. By fitting feature distributions from existing labeled datasets
(TwiBot-22 [1], Cresci-2017 [3]) and generating synthetic accounts from
those distributions, BotDriftBench provides a controlled testbed that
captures empirically observed feature patterns while enabling reproducible
evaluation under known conditions.

**Contributions:**

- A synthetic longitudinal benchmark with 27 calibrated features, 20
  temporal windows, and 5 controlled drift scenarios plus a control group.
- A documented calibration methodology using empirical distributions from
  TwiBot-22 and cross-dataset validation against Cresci-2017.
- Validation showing that injected drift produces measurable distributional
  shifts while control accounts remain stable.
- Full release of generation code, configuration, and the dataset under
  CC BY 4.0 for reproducibility.

---

## 2. Related Work

### 2.1 Bot Detection Datasets

The bot detection research community has developed several benchmark
datasets over the past decade. The Cresci-2017 dataset [3] introduced the
notion of social spambot generations and remains widely used. TwiBot-20 [7]
and TwiBot-22 [1] provided larger-scale benchmarks with graph structure.
The MGTAB dataset [8] included multi-granularity labels. These datasets
have enabled significant advances in bot detection but share a common
limitation: they provide static snapshots of account behavior at a single
time period.

### 2.2 Temporal and Longitudinal Studies

Hays et al. [4] conducted a longitudinal study demonstrating that bot
behavior shifts substantially over multi-year periods. Cresci [9] surveyed
a decade of bot detection and highlighted the arms race between detectors
and bot operators. Rauchfleisch and Kaiser [10] showed that commercial bot
detection tools disagree substantially on the same accounts, partly due to
temporal instability. However, no publicly available benchmark dataset
provides temporal trajectories with labeled drift events.

### 2.3 Synthetic Data for Evaluation

Synthetic data generation has been used in other machine learning domains
to create controlled evaluation environments [11]. In anomaly detection,
synthetic benchmarks with injected anomalies are standard practice [12].
For bot detection specifically, synthetic approaches have been limited
to augmentation for class balancing rather than longitudinal evaluation.

### 2.4 Drift Benchmarks in Machine Learning

The concept drift literature has produced synthetic benchmarks such as the
SEA generator [13], rotating hyperplane [14], and STAGGER concepts [15].
These are domain-agnostic and do not capture the specific feature semantics
of social bot detection. BotDriftBench fills this gap by providing a
domain-specific drift benchmark calibrated from real bot detection data.

---

## 3. Feature Audit and Selection

### 3.1 Source Datasets

We derive calibration parameters from two established bot detection datasets:

**TwiBot-22** [1]: A large-scale Twitter bot detection benchmark with
1,000,000 users. We use the reliable subset (users with high-confidence
labels): 294,478 human and 38,986 bot users. Profile features are extracted
from user.json; tweet features are aggregated from approximately 88 million
tweets. Licensed under CC BY-NC-ND 4.0.

**Cresci-2017** [3]: A multi-generation social spambot dataset collected
from Twitter in 2017. Used for cross-dataset validation, not for parameter
fitting, due to era and collection methodology differences.

### 3.2 Feature Selection Criteria

We select 27 features that meet all of the following criteria:

1. **Computable from public data.** The feature must be derivable from
   publicly available profile metadata or tweet content, without requiring
   network topology or platform-internal data.

2. **Available in TwiBot-22.** The feature must be computable from the
   TwiBot-22 user.json and tweet files to enable empirical calibration.

3. **Label-independent.** The feature must not require knowledge of account
   labels for computation. This excludes network-topology features such as
   bot-to-bot edge counts.

4. **Semantically meaningful.** The feature must capture a behavioral or
   profile characteristic relevant to bot detection.

### 3.3 Features Excluded

Six features present in earlier benchmark versions (v0.2) were removed
for v0.3:

| Feature | Reason for Exclusion |
|---|---|
| friends_count | Redundant with following_count (same data in TwiBot-22) |
| favourites_count | Unreliable; Twitter deprecated public favourite counts |
| default_profile | Insufficient variance in TwiBot-22 reliable subset |
| verified | Twitter verification policy changed fundamentally in 2023 |
| follower_friend_ratio | Derived from included features; adds no information |
| retweet_ratio | Requires native retweet metadata unavailable in aggregated form |

Two features were added in v0.3:

| Feature | Justification |
|---|---|
| name_length | Distinguishes auto-generated vs. human-chosen display names |
| observed_tpd | Tweets-per-day from available tweet data; distinguishes from statuses_count which includes all-time total |

### 3.4 Feature Categories

The final 27 features are organized into two categories:

**Profile features (11):** account_age_days, statuses_count,
followers_count, following_count, listed_count, default_profile_image,
has_url, description_length, screen_name_length, name_length,
reputation_score.

**Behavioral features (16):** observed_tpd, posting_regularity,
night_post_ratio, weekend_ratio, burst_score, avg_tweet_length,
text_entropy, vocabulary_richness, emoji_ratio, caps_ratio,
content_similarity, reply_ratio, rt_prefix_ratio, mention_count_avg,
hashtag_count_avg, url_share_ratio.

Note: rt_prefix_ratio captures the frequency of "RT @" text prefixes as a
textual proxy for retweet-like behavior. It is not a ground-truth
measurement of the platform's native retweet mechanism.

### 3.5 Cross-Dataset Comparison

We computed feature distributions from both TwiBot-22 and Cresci-2017 and
compared them using the KS statistic. Notable divergences include:

| Feature | KS Statistic | Interpretation |
|---|---|---|
| posting_regularity | 0.996 | Extreme divergence; collection era differences |
| emoji_ratio | 0.841 | Emoji usage changed between 2017 and 2020 |
| content_similarity | 0.810 | Different tweet volumes affect self-similarity |

These divergences reflect era differences, platform changes, and collection
methodology differences between the two datasets — not purely behavioral
drift. We use TwiBot-22 as the primary calibration source because it is
more recent and larger. Cresci-2017 is used for qualitative validation
only, not for parameter fitting.

---

## 4. Calibration Methodology

### 4.1 Distribution Fitting

For each feature, we fit a parametric distribution to the TwiBot-22 data
separately for human and bot classes:

- **Bounded features** (ratios in [0, 1]): Beta distribution,
  Beta(α, β), fitted via maximum likelihood.
- **Count features** (non-negative, unbounded): Lognormal distribution,
  LogNormal(μ, σ), fitted via maximum likelihood.
- **Binary features** (0/1): Bernoulli distribution, Bernoulli(p),
  fitted as the sample proportion.

The fitted parameters are stored in a configuration file
(`v03_fitted_distributions.json`) and used by the generation script to
sample synthetic account features.

**Scope of calibration.** "Empirically calibrated" refers to marginal
distribution fitting from TwiBot-22, not to calibration of temporal dynamics
or joint feature distributions. Each feature's distribution is fitted
independently; inter-feature correlations are not modeled. The TwiBot-22
provenance chain is: raw files (`user.json`, `label.csv`, tweet files) →
fitting script (`outputs/v03_fit_distributions.py`) → calibration parameters
(`outputs/v03_fitted_distributions.json`, n_human_reliable=294,478,
n_bot_reliable=38,986) → generator (`generate_dataset_v03.py`).

### 4.2 Trajectory Generation

Each synthetic account is generated as a 20-window temporal trajectory
using the following procedure:

1. **Class assignment.** 500 accounts are assigned label "human" and 500
   "bot." Bot accounts are partitioned into drift scenario groups
   (83-84 bots each) and a no_drift control group (84 bots).

2. **Initial state (window 0).** Features are sampled from the fitted
   distributions for the account's class.

3. **Temporal evolution.** For windows 1–19, features evolve using a
   first-order Markov carry-forward with noise:
   x_j^{(t)} = x_j^{(t-1)} + ε, where ε ~ N(0, σ_j^2) and σ_j is a
   small noise scale proportional to the feature's empirical standard
   deviation. Features are clipped to their valid ranges after each step.

4. **Drift injection.** For bot accounts in drift scenarios, targeted
   features are additionally shifted during windows τ_start through τ_end:
   x_j^{(t)} = x_j^{(t)} + m · δ_j · α(t), where m is the drift
   magnitude, δ_j is the feature-specific drift direction, and α(t) is
   a ramp function increasing linearly from 0 at τ_start to 1 at τ_end.

5. **Constraint enforcement.** Bounded features are clipped to [0, 1],
   count features to [0, ∞), and binary features are re-binarized.

### 4.3 Drift Scenarios

Five drift scenarios target distinct feature groups:

| Scenario | Drifted Features | Bot Accounts |
|---|---|---|
| activity_drift | observed_tpd, posting_regularity, night_post_ratio, burst_score, weekend_ratio | 84 |
| linguistic_drift | avg_tweet_length, text_entropy, vocabulary_richness, emoji_ratio, caps_ratio | 83 |
| coordination_drift | rt_prefix_ratio, mention_count_avg, hashtag_count_avg, content_similarity, reply_ratio | 83 |
| network_drift | followers_count, following_count, listed_count, reputation_score, name_length (profile-level proxies of network status, not graph-topology features) | 83 |
| combined_drift | observed_tpd, posting_regularity, text_entropy, vocabulary_richness, content_similarity, reply_ratio, rt_prefix_ratio | 83 |

The no_drift control group (84 bots) experiences temporal noise but no
injected drift.

Human accounts (500) have scenario "none" and experience no injected drift.
This is a simplification: in reality, legitimate user behavior also evolves
over time. This limitation means BotDriftBench evaluates drift detection
under the assumption that only bot behavior changes.

### 4.4 Configuration Parameters (v0.4)

| Parameter | v0.3 Value | **v0.4 Value** |
|---|---|---|
| Drift onset (τ_start) | Window 5 | Window 5 |
| Drift end (τ_end) | Window 15 | Window 15 |
| Drift magnitude | 1.0 | 1.0 |
| Random seed (generation) | 42 | 42 |
| Measurement noise | 12% rel / 6% abs | **4% rel / 1.5% abs** |
| Natural evolution noise | 2.0× std | **1.0× std** |
| Archetype modifiers | moderate | **strengthened** |
| Difficulty tiers | — | **easy / realistic / hard** |
| Split seed | — | **99 (3-way 600/200/200)** |

The v0.4 measurement noise reduction (from 12%/6% to 4%/1.5%) reflects
realistic monthly behavioral aggregates where measurement precision is higher
than daily aggregates assumed in v0.3. Natural evolution noise reduction
(from 2.0× to 1.0×) makes individual account trajectories more stable,
improving baseline separability. Strengthened archetype modifiers produce
clearer class discrimination for medium-signal features.

Only magnitude 1.0 is included in v0.4. The benchmark is designed to
support magnitudes [0.25, 0.50, 1.00, 2.00], but only 1.0 has been
generated and validated.

---

## 5. Benchmark Validation

### 5.1 Structural Validation

| Check | Result |
|---|---|
| Unique accounts | 1,000 (500 human + 500 bot) |
| Temporal windows | 20 (0-19) |
| Account-window rows | 20,000 |
| Features present | 27/27 |
| NaN values | 0 |
| Inf values | 0 |
| Duplicate rows | 0 |
| Label balance | 10,000 human / 10,000 bot |
| Temporal continuity | Every account has all 20 windows |
| Label consistency | Each account has exactly 1 label |

### 5.2 Feature Range Validation

- All bounded features (ratios) are in [0, 1]
- All count features are non-negative
- No impossible values (e.g., negative follower counts)

### 5.3 Control Group Stability

To verify that the no_drift control group does not exhibit spurious drift,
we computed per-feature KS statistics between the control group's pre-drift
(windows 0-4) and during-drift (windows 5-15) observations. No feature
exceeded KS = 0.3, confirming that temporal noise alone does not trigger
drift-like distributional shifts.

### 5.4 Drift Effect Verification

For each drift scenario, we verified that the declared drifted features
show measurable distributional shift after drift onset. All targeted
features achieved KS > 0.04 between pre-drift and during-drift periods.

Non-targeted features in drift scenarios show minimal shift, confirming
that drift injection is localized to the intended feature subsets.

### 5.5 Drift Onset Correctness

No `drift_active` flag is set before window τ_start = 5 for any account
in any scenario, confirming that drift injection respects the specified
onset time.

### 5.6 Leakage Audit

The benchmark design was audited for potential label leakage:

- Account IDs are not correlated with labels
- Drift scenario assignment is not encoded in feature values
- Window indices do not leak temporal position information into features
- **Account-disjoint split is required.** A row-level random split leaks
  account information: since each account has 20 temporally correlated
  observations, a random split places some windows of the same account in
  both train and test, inflating F1 from ~0.81 (honest) to ~0.96 (leaked).
  The recommended evaluation protocol (Section 9.4) specifies account-level
  splitting to prevent this.

---

## 6. Benchmark Utility Demonstration

To demonstrate that BotDriftBench **v0.4** produces meaningful experimental
results, we report baseline experiments using an XGBoost classifier
(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8).
All results below use **Protocol A: unseen-account generalization**
(600 train / 200 val / 200 test accounts, 5 seeds [0–4], split seed 99).
Full results: `final_v04/method_summary.csv`, `final_v04/metrics.json`.

### 6.1 Classification Baseline (v0.4)

The static XGBoost model (trained on pre-drift windows 0–4, evaluated on
all 20 test windows) achieves:

| Metric | Value |
|---|---|
| Temporal Mean Macro-F1 | **0.8650 ± 0.0010** (5 seeds) |
| Pre-Drift F1 | 0.8682 |
| During-Drift F1 | 0.8663 |
| Post-Drift F1 | 0.8595 |
| F1 Degradation (pre − worst) | 0.0213 |
| ROC-AUC (all test windows, seed 0) | 0.9421 |

Source: `final_v04/method_summary.csv`, `final_v04/metrics.json`.

**Ceiling diagnostic (from `ceiling_diagnostic_v2/`):**
Account-wise XGB F1 = 0.825 (honest), random-row XGB F1 = 0.993 (leaked),
temporal-disjoint mean F1 = 0.789. The Protocol A score of 0.865 is
above the account-wise ceiling estimate because the 3-way split uses
600 train accounts vs. 200 test (more training data), and the difficulty
tier distribution includes easy accounts.

**Per-difficulty-tier (static model, seed 0):**

| Tier | Pre-Drift F1 | All-Windows F1 | n_bot (test) |
|---|---|---|---|
| Easy | 0.9602 | 0.9621 | 31 |
| Realistic | 0.8890 | 0.8773 | 39 |
| Hard | 0.6749 | 0.7001 | 30 |

Source: `final_v04/per_tier_results.csv`.

### 6.2 Drift-Induced Degradation (v0.4)

The v0.4 dataset has higher baseline separability than v0.3 (static F1
0.865 vs. 0.792 for the 700/300 v0.3 split). This means behavioral drift
causes less absolute F1 degradation on v0.4, reducing the observable
adaptation advantage. This is a known property of the v0.4 configuration
and is documented in `final_v04/PROTOCOL.md` as Outcome B.

### 6.3 Drift Detection and Feature Validation (v0.4)

The v0.4 localized hybrid detector (tuned thresholds: GROUP_KS=0.06,
FRAC=0.35, CLUSTER=0.06, PRED_RISK=0.03) achieves:

- da_localized: 1.0 retrains on average (4/5 seeds beat periodic_4)
- da_localized_nodelay: 2.0 retrains on average (1/5 seeds beat periodic_4)
- All methods: 0 false alarms before drift onset (window 5)

Source: `final_v04/drift_detection_results.csv`,
`final_v04/adaptation_costs.csv`, `final_v04/drift_detection_signal.png`.

Feature-level validation confirms that targeted features show higher KS
statistics than non-targeted features across all scenarios. See
`final_v04/per_scenario_results.csv` for per-scenario breakdowns.



---

## 7. Limitations

1. **Synthetic data.** BotDriftBench is not real longitudinal Twitter/X
   data. Temporal trajectories are generated, not observed. Results on
   synthetic data may not transfer to real-world deployment.

2. **Approximated correlations.** Feature distributions are fitted
   marginally. Inter-feature correlations are approximated through the
   Markov carry-forward mechanism but are not fitted from the joint
   empirical distribution.

3. **Single magnitude.** Only drift magnitude 1.0 is provided. The
   benchmark's sensitivity analysis across magnitudes [0.25, 0.50, 2.00]
   is not yet included.

4. **No human drift.** Drift is injected only into bot accounts. Legitimate
   user behavior evolution — which occurs in real platforms — is not
   modeled.

5. **Single platform.** Calibration is from Twitter/X data. The feature
   semantics may not generalize to other social media platforms.

6. **Cross-dataset divergence.** TwiBot-22 and Cresci-2017 disagree on
   several feature distributions (e.g., posting_regularity KS = 0.996,
   emoji_ratio KS = 0.841). These divergences are due to era differences,
   platform changes, and collection methodology, not purely behavioral
   differences. The benchmark uses TwiBot-22 parameters, which may not
   represent other eras.

7. **Fixed drift patterns.** Drift is modeled as a gradual linear ramp.
   Real-world behavioral changes may be abrupt, periodic, or
   non-monotonic.

8. **Label simplification.** Binary labels (human/bot) do not capture the
   diversity of bot types (e.g., spam bots, follower bots, astroturfing
   accounts, AI-generated content bots).

9. **Shared human population.** All drift scenarios share the same 150
   test humans. Per-scenario evaluation subsets are therefore not fully
   independent; degradation patterns may partially reflect shared human
   feature variation rather than scenario-specific bot drift.

10. **Small control-adjusted effects.** Control-adjusted F1 drops are
    mostly near zero or negative, suggesting that measured degradation
    is dominated by class imbalance and evaluation noise rather than
    large drift-specific effects at the current magnitude.

---

## 8. Ethical Considerations

### 8.1 Data Licensing

BotDriftBench v0.3 is released under CC BY 4.0. The calibration source,
TwiBot-22, is licensed under CC BY-NC-ND 4.0. We do not redistribute raw
TwiBot-22 data. Only derived aggregate statistics (fitted distribution
parameters) are used in the generation process.

### 8.2 No Personal Data

BotDriftBench contains no real user data. All accounts are synthetically
generated. Account IDs are randomly assigned integers with no correspondence
to real Twitter/X user IDs.

### 8.3 Potential for Misuse

Bot detection benchmarks could theoretically be used to develop more
evasive bots by studying which features detectors rely on. We note that
the SHAP feature importance ranking (which identifies influential
detection features) is already publicly available in the bot detection
literature for many classifiers, so BotDriftBench does not provide
fundamentally new information in this regard.

### 8.4 Benchmark Limitations Disclosure

We explicitly label BotDriftBench as synthetic data throughout all
documentation and publications. Users should not interpret results on
this benchmark as evidence of real-world deployment effectiveness.

---

## 9. Reproducibility

### 9.1 Dataset Files (v0.4)

| File | Description | Size |
|---|---|---|
| `dataset_v04/botdriftbench_v0.4.csv` | Main dataset (20,000 rows × 39 columns) | 6.76 MB |
| `dataset_v04/split_assignments.json` | Account-disjoint 3-way split (seed 99) | 25 KB |
| `dataset_v04/split_assignments.csv` | Same split in CSV format | 16 KB |
| `dataset_v04/generation_config_v04.json` | Full generation configuration | 5 KB |
| `dataset_v04/MANIFEST.md` | Dataset specification and SHA-256 | 4 KB |

SHA-256: `4b20d673eb3fc994f652bcd9ea5042f90f50d9d03372c181b112f85143bd8f21`

### 9.2 Generation Code

The generation script (`generate_dataset_v04.py`) is provided for full
reproducibility. Running the script with the configuration in
`dataset_v04/generation_config_v04.json` is expected to reproduce the
dataset. The SHA-256 checksum above can be used to verify an exact match.

### 9.3 Evaluation Code

The primary evaluation script is `run_final_v04_complete.py` (augmented
version producing all required outputs) and `run_final_v04.py` (original).
The pre-registered protocol is in `final_v04/PROTOCOL.md` (registered
2026-09-17, before test-set evaluation).

A Colab-compatible notebook (`Drift_Aware_Bot_Detection_v03.ipynb`) is
provided for interactive exploration; update the data path to
`dataset_v04/botdriftbench_v0.4.csv` and use
`dataset_v04/split_assignments.json` for the split.

### 9.4 Suggested Evaluation Protocol (v0.4)

For consistent cross-study comparison on BotDriftBench v0.4:

1. Load `dataset_v04/split_assignments.json` for the account assignments.
   Do **not** re-split randomly; use the pre-defined 600/200/200 split.
2. Verify SHA-256 of `dataset_v04/botdriftbench_v0.4.csv`.
3. Verify account disjointness (train ∩ val = ∅, train ∩ test = ∅,
   val ∩ test = ∅).
4. Train on train-account pre-drift windows (0–4) only.
5. Tune DA thresholds on validation set only; do not use test set for tuning.
6. Evaluate on test accounts at each window (Protocol A — primary).
7. Report Protocol B (known-account temporal) separately and clearly
   labelled; do not compare Protocol A and B F1 values directly.
8. Report multi-seed results (seeds 0–4).
9. Report per-difficulty-tier results using the `difficulty_tier` column.
10. Report per-scenario results using the `drift_scenario` column.


---

## 10. Conclusion

**BotDriftBench v0.4** provides a controlled, reproducible environment for
evaluating drift-aware bot detection methods. By fitting feature distributions
from TwiBot-22, injecting drift at known onset points with specified
magnitude, and providing a pre-registered 3-way account-disjoint evaluation
protocol (600/200/200, split seed 99), the benchmark enables systematic
evaluation of drift detection accuracy, feature attribution, and adaptation
strategies.

v0.4 improvements over v0.3 include: reduced measurement noise (4% relative)
for realistic monthly aggregate behavior, strengthened archetype separation
for clearer class discrimination, a `difficulty_tier` column for stratified
analysis, and a formally pre-registered evaluation protocol with separate
validation and test sets to prevent threshold leakage.

A known property of v0.4 is that its higher baseline separability reduces
the drift-induced degradation magnitude, making adaptation advantage harder
to demonstrate. The pre-registered protocol documents this as a benchmark
property (Outcome B), not a failure of the methodology.

As synthetic data with approximated correlations and simplified drift
patterns, BotDriftBench does not replace evaluation on real-world
longitudinal data. Rather, it provides a standardized testbed for
developing and comparing drift-aware methods before deployment. The
benchmark, generation code, and evaluation scripts are released under
CC BY 4.0 to support reproducible research in drift-aware social bot
detection.


---

## References

[1] S. Feng, H. Wan, N. Wang, J. Li, and M. Luo, "TwiBot-22: Towards
    graph-based Twitter bot detection," in *Proc. NeurIPS*, 2022.

[2] O. Varol, E. Ferrara, C. A. Davis, F. Menczer, and A. Flammini,
    "Online human-bot interactions: Detection, estimation, and
    characterization," in *Proc. AAAI ICWSM*, 2017.

[3] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi,
    "The paradigm-shift of social spambots: Evidence, theories, and tools
    for the arms race," in *Proc. WWW*, 2017.

[4] C. Hays, Z. Schutzman, M. Raghavan, E. Walk, and P. Zimmer, "Bots
    don't sit still: A longitudinal study of bot behavioral change on
    social media," arXiv preprint, 2025. [Reference ID pending]

[5] E. Ferrara, O. Varol, C. Davis, F. Menczer, and A. Flammini, "The
    rise of social bots," *Communications of the ACM*, vol. 59, no. 7,
    pp. 96–104, 2016.

[6] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia,
    "A survey on concept drift adaptation," *ACM Computing Surveys*,
    vol. 46, no. 4, 2014.

[7] S. Feng, H. Wan, N. Wang, and M. Luo, "TwiBot-20: A comprehensive
    Twitter bot detection benchmark," in *Proc. CIKM*, 2021.

[8] S. Liao, Z. Tan, R. Li, and H. Wan, "MGTAB: A multi-relational
    graph-based Twitter account detection benchmark," arXiv preprint,
    2023.

[9] S. Cresci, "A decade of social bot detection," *Communications of
    the ACM*, vol. 63, no. 10, pp. 72–83, 2020.

[10] A. Rauchfleisch and J. Kaiser, "The false positive problem of
     automatic bot detection in social science research," *PLoS ONE*,
     vol. 15, no. 10, 2020.

[11] N. Patki, R. Wedge, and K. Veeramachaneni, "The Synthetic Data
     Vault," in *Proc. IEEE DSAA*, 2016.

[12] G. O. Campos, A. Zimek, J. Sander, R. J. G. B. Campello,
     B. Micenkova, E. Schubert, I. Assent, and M. E. Houle, "On the
     evaluation of unsupervised outlier detection: Measures, datasets,
     and an empirical study," *Data Mining and Knowledge Discovery*,
     vol. 30, no. 4, 2016.

[13] W. N. Street and Y. Kim, "A streaming ensemble algorithm (SEA)
     for large-scale classification," in *Proc. ACM KDD*, 2001.

[14] G. Hulten, L. Spencer, and P. Domingos, "Mining time-changing data
     streams," in *Proc. ACM KDD*, 2001.

[15] J. C. Schlimmer and R. H. Granger, "Incremental learning from
     noisy data," *Machine Learning*, vol. 1, no. 3, 1986.
