# Drift-Aware Explainable Social Bot Detection Using SHAP-Based Behavioral Drift Monitoring

**Authors:** Eshan Amballa

---

## Abstract

Social bot detection models are typically trained on static snapshots of
account behavior and evaluated under the assumption that feature distributions
remain stable over time. In practice, bot operators continuously adapt their
strategies, causing behavioral and evasion drift that severely degrades detector
performance. We present a drift-aware bot detection pipeline that integrates
four components: (1) a baseline XGBoost classifier for bot identification,
(2) a localized drift monitor using feature-level Kolmogorov-Smirnov (KS) tests
on the monitored bot population, (3) a multi-model SHAP attribution module
that traces feature attribution evolution across pre-drift, post-drift, and
post-adaptation regimes, and (4) an exponential recency-weighted retraining
strategy. We evaluate on **BotDriftBench v0.8**, an empirically calibrated
synthetic longitudinal benchmark containing 3,000 accounts observed across
20 temporal windows with 25 features and seven account-level drift scenarios
(including boundary-crossing evasion, abrupt evasion, and stable controls),
evaluated under an account-disjoint 3-way split (1,800 train / 600 validation /
600 test accounts) across 5 independent seeds.

Under this benchmark, a frozen static model exhibits substantial degradation,
dropping from Macro-F1 = **0.9770** pre-drift to **0.8059** post-drift
(worst-window F1 = **0.6583**), driven primarily by evasion scenarios where bot
recall falls to 67.26%. Localized drift-aware adaptation substantially improves
over a frozen static detector (**0.9167 ± 0.0009** vs **0.8230 ± 0.0014**,
ROC-AUC = **0.9691**, Recall = **0.8931**) and performs comparably to
Periodic-6 (**0.9120 ± 0.0011**, ROC-AUC = 0.9658) while using targeted
retraining (5 events vs. 3 scheduled events, with 0 pre-drift false alarms).
Localized DA used 5 adaptation retrains compared with 13 for the oracle
reference policy, achieving comparable performance (0.9167 vs 0.9160).
SHAP analysis documents feature attribution changes across pre-drift, post-drift,
and post-adaptation regimes, illustrating how retraining redistributes feature
importance away from evasive behavioral shifts toward structural signals
without implying causal invariance.

**Keywords:** social bot detection, concept drift, SHAP explainability,
behavioral drift monitoring, XGBoost, Kolmogorov-Smirnov test, adaptive
retraining, recency weighting, evasion drift

---

## 1. Introduction

Social media platforms face a persistent threat from automated accounts —
social bots — that manipulate public discourse, amplify misinformation, and
distort engagement metrics [1]. Machine learning classifiers trained on
account-level behavioral features have shown strong performance in detecting
bots on static benchmark datasets [2, 3]. However, these classifiers are
typically evaluated under the implicit assumption that the feature
distributions observed during training will remain stable during deployment.

This assumption is unrealistic. Bot operators adapt their strategies in
response to detection pressure, platform policy changes, and evolving
objectives [4, 5]. When bot behavior shifts — a phenomenon we term
*behavioral drift* — the classifier's decision boundary becomes misaligned
with the new data distribution, leading to silent performance degradation.
Without an explicit drift monitoring mechanism, practitioners have no signal
that their detector is failing.

Three gaps in the current literature motivate this work:

1. **Silent degradation.** Most bot detection systems provide no mechanism to
   detect when their performance is declining due to distributional shift.
   The model continues to produce predictions, but those predictions become
   progressively less reliable.

2. **Opaque failure modes.** When performance degrades, it is often unclear
   *which* features have shifted and *how* the model's reasoning has changed.
   Feature-level explanations of drift are typically unavailable.

3. **Rigid adaptation.** Retraining strategies are often either fully manual
   or calendar-based (periodic). Neither approach is responsive to the actual
   timing and nature of behavioral changes.

We address these gaps with a drift-aware bot detection pipeline comprising
three modules: a baseline bot classifier, a feature-level drift monitor, and
a SHAP-based drift attribution engine. Our contributions are:

- A localized drift monitoring architecture that combines feature-group
  KS scores, cluster-aware subgroup monitoring, and prediction-distribution
  risk signals to detect drift in heterogeneous populations where global
  detectors fail.
- A recency-weighted retraining strategy that emphasizes recent
  observations, shown via controlled ablation to be the primary source of
  F1 improvement over periodic retraining.
- A SHAP-based drift attribution method that quantifies how feature
  contributions to model predictions change under drift.
- An empirical comparison of adaptation policies — frozen static baseline, periodic retraining (Periodic-6), localized drift-aware adaptation (Localized DA), and an oracle reference policy — on BotDriftBench v0.8, evaluated in aggregate across all test accounts with 5-seed replication.
- An interactive demonstration system (Streamlit frontend) for exploring
  drift scenarios and model explanations.

The central research question is: *Can drift-aware retraining achieve
comparable predictive robustness with fewer retraining events and lower
adaptation cost than periodic retraining, under a controlled longitudinal
bot-detection benchmark?*

We evaluate on BotDriftBench v0.8, an empirically calibrated synthetic
longitudinal benchmark whose feature distributions are fitted from TwiBot-22
[2] and cross-referenced against Cresci-2017 [6]. All reported results are
from this synthetic benchmark; they demonstrate the feasibility of the
methodology but do not constitute real-world deployment validation.

---

## 2. Related Work

### 2.1 Social Bot Detection

Early bot detection relied on hand-crafted rules and simple classifiers
applied to profile metadata [7]. Subsequent work incorporated behavioral
features derived from posting patterns, linguistic characteristics, and
network structure [8, 9]. Yang et al. [10] introduced scalable features
based on account activity patterns. Recent approaches leverage graph neural
networks [2, 11] and large language models [12] for bot detection, though
these require network topology or tweet text that may not always be available.

### 2.2 Concept Drift in Bot Detection

Concept drift — the phenomenon where the statistical relationship between
inputs and outputs changes over time — has been extensively studied in
machine learning [13, 14]. In the bot detection domain, Cresci et al. [6]
demonstrated that bot behavior evolves across generations of social spambots.
Hays et al. [5] conducted a longitudinal study showing that bot behavior
shifts significantly over multi-year periods, causing detectors trained on
older data to lose accuracy. However, most bot detection benchmarks provide
only static snapshots, making it difficult to study temporal drift
systematically [15].

### 2.3 Explainability in Bot Detection

SHAP (SHapley Additive exPlanations) [16] provides theoretically grounded
feature attributions for machine learning predictions. While SHAP has been
applied to bot detection for model interpretation [17], its use for
*drift attribution* — identifying which features' contributions to
predictions have changed — has received limited attention. We note that SHAP
explains model behavior (how features influence predictions), not the causal
mechanisms behind behavioral changes.

### 2.4 Adaptive Classification Under Drift

Drift adaptation strategies in classification include periodic retraining,
triggered retraining based on performance monitoring, and ensemble methods
that weight models by recency [18, 19]. In bot detection specifically,
adaptation is typically handled by manual retraining or by training on
expanding windows of data [20]. Our work compares calendar-based (periodic)
and detection-triggered (drift-aware) retraining on a controlled benchmark.

---

## 3. Problem Definition

Let X_t ∈ R^d denote the feature vector of a social media account observed
at temporal window t, where d = 27 is the number of behavioral and profile
features. Let y ∈ {0, 1} denote the binary label (0 = human, 1 = bot).

A bot classifier f: R^d → [0, 1] is trained on data from an initial period
[0, τ_start) and produces:

$$\hat{p}(y=1 | X_t) = f(X_t)$$                                        (1)

**Behavioral drift** occurs when the feature distribution of bot accounts
shifts after a drift onset time τ_start:

$$P(X_t | y=1) ≠ P(X_0 | y=1), \quad \text{for } t ≥ τ_{\text{start}}$$ (2)

This distributional shift causes the classifier's performance to degrade
over time. The goals of our system are:

1. **Detect** when drift has occurred by monitoring feature distributions.
2. **Attribute** the drift to specific features using model explanations.
3. **Adapt** the classifier by retraining when drift is detected.

---

## 4. Proposed Method

### 4.1 Baseline Bot Detector

We use a gradient-boosted tree classifier (XGBoost [21]) as the baseline
bot detector. XGBoost is chosen for its strong tabular-data performance, its
compatibility with exact SHAP computation via TreeExplainer [16], and its
widespread use in bot detection [22].

The model is configured with 200 estimators, maximum depth 5, learning
rate 0.1, and logloss evaluation metric. It is trained on pre-drift
observations using an account-disjoint split: 70% of account IDs are
assigned to training and 30% to testing, with all 20 temporal windows of
each account staying in the same partition. This prevents account leakage
between training and evaluation.

### 4.2 Behavioral Feature Representation

Each account is represented by 25 features organized into two categories:

**Profile features (10):** account_age_days, statuses_count,
followers_count, following_count, listed_count, default_profile_image,
has_url, description_length, screen_name_length, name_length,
reputation_score.

**Behavioral features (16):** observed_tpd (tweets per day),
posting_regularity, night_post_ratio, weekend_ratio, burst_score,
avg_tweet_length, text_entropy, vocabulary_richness, emoji_ratio,
caps_ratio, content_similarity, reply_ratio, rt_prefix_ratio,
mention_count_avg, hashtag_count_avg, url_share_ratio.

All features are account-level aggregates computable from public profile
metadata and tweet content. Network-topology features requiring label
knowledge (e.g., bot-to-bot edge counts) are excluded from the deployable
feature set to avoid label leakage.

Note on rt_prefix_ratio: this feature captures the frequency of "RT @"
text prefixes in tweets as a proxy for retweet-like behavior. It is a
textual heuristic, not a ground-truth measurement of native retweet
platform functionality.

### 4.3 Temporal Drift Simulation

To study model behavior under drift, we simulate temporal trajectories
where bot feature distributions shift gradually over a defined period.
For each drift scenario, a subset of features is modified starting at
window τ_start = 5 and continuing through τ_end = 15.

Five drift scenarios target different feature groups:

- **Activity drift**: observed_tpd, posting_regularity, night_post_ratio,
  burst_score, weekend_ratio
- **Linguistic drift**: avg_tweet_length, text_entropy,
  vocabulary_richness, emoji_ratio, caps_ratio
- **Coordination drift**: rt_prefix_ratio, mention_count_avg,
  hashtag_count_avg, content_similarity, reply_ratio
- **Network drift**: followers_count, following_count, listed_count,
  reputation_score, name_length (profile-level proxies of network status,
  not graph-topology features)
- **Combined drift**: observed_tpd, posting_regularity, text_entropy,
  vocabulary_richness, content_similarity, reply_ratio, rt_prefix_ratio

A no_drift control group provides a baseline for comparison.

### 4.4 Feature-Level Drift Monitoring

We detect drift using the two-sample Kolmogorov-Smirnov (KS) test applied
independently to each feature. For feature j at window t, the KS statistic
measures the maximum difference between the empirical cumulative distribution
functions of the reference (pre-drift) and current observations:

$$D_j^{(t)} = \sup_x |F_{\text{ref},j}(x) - F_{t,j}(x)|$$            (3)

where F_ref,j is the CDF of feature j in the reference period (windows 0
through τ_start − 1) and F_t,j is the CDF at window t.

Drift is flagged at window t when the number of features exceeding a KS
threshold θ_KS meets or exceeds a count threshold θ_count:

$$\text{drift\_detected}(t) = \mathbb{1}\left[\sum_{j=1}^{d} \mathbb{1}[D_j^{(t)} > \theta_{\text{KS}}] \geq \theta_{\text{count}}\right]$$  (4)

We set θ_KS = 0.15 and θ_count = 3. These thresholds are fixed prior to
evaluation and are not tuned on test data. We compare this feature-count
detector against three alternatives (max-KS, top-5 mean KS, mean-KS) in
Section 6.2 to evaluate how detector design affects sensitivity.

### 4.5 Localized Drift Monitoring

The global feature-count detector (Eq. 4) computes KS statistics over the
entire monitored bot population. When drift affects only a subpopulation
(e.g., one drift scenario's ~80 bots among ~350 train bots), the drifted
signal is diluted by the non-drifted majority. We address this with a
localized monitoring architecture comprising three complementary signals.

**Feature-group drift scores.** We partition the 27 features into five
semantic groups: activity (6), linguistic (5), coordination (6), network
(6), and profile (4). For each group g at window t, the group drift score
is the mean within-group KS statistic:

$$S_g^{(t)} = \frac{1}{|g|} \sum_{j \in g} D_j^{(t)}$$                 (4a)

A feature-group detection fires when any group's mean KS exceeds
θ_group = 0.12 AND at least 40% of features within that group individually
exceed θ_KS.

**Cluster-aware subgroup monitoring.** We cluster the reference pre-drift
bot population into K = 5 subgroups using MiniBatchKMeans on standardized
features. At each window t, current bots are assigned to their nearest
cluster, and per-cluster KS statistics are computed. The cluster drift
score is the maximum across clusters:

$$C^{(t)} = \max_{k=1}^{K} \bar{D}_k^{(t)}$$                           (4b)

where $\bar{D}_k^{(t)}$ is the mean per-feature KS within cluster k. This
detects subpopulation-level drift that is invisible in the aggregate.

**Prediction-distribution monitoring.** We track shifts in the model's own
predictions, providing a proxy for concept drift without requiring labels.
The prediction risk score combines three signals:

$$R^{(t)} = 0.5 \cdot \text{KS}(p_{\text{ref}}, p_t) + 0.3 \cdot \max(0, -\Delta_{\text{conf}}) + 0.2 \cdot \max(0, \Delta_{\text{unc}})$$  (4c)

where $p_{\text{ref}}$ and $p_t$ are the predicted probability distributions
on reference and current data, $\Delta_{\text{conf}}$ is the change in mean
confidence (distance from 0.5), and $\Delta_{\text{unc}}$ is the change in
the fraction of predictions within 0.10 of the decision boundary.

**Hybrid trigger.** The OR-trigger fires when any component detects drift:

$$\text{trigger}(t) = \text{fg\_det}(t) \lor [C^{(t)} \geq 0.15] \lor [R^{(t)} \geq 0.08]$$  (4d)

A persistence confirmation requires 2 consecutive windows with an active
trigger before retraining is initiated. A cooldown period of 2–4 windows
between retraining events prevents excessive adaptation.

All thresholds (θ_group = 0.12, θ_cluster = 0.15, θ_pred = 0.08,
θ_hybrid = 0.10, persistence = 2 windows) are set a priori and not tuned
on test labels.

### 4.6 Recency-Weighted Retraining

When retraining is triggered at window t, the training set includes all
train-account data up to and including window t. Rather than weighting all
observations equally, we apply exponential recency weights:

$$w_i = \exp(-\lambda \cdot (t - t_i))$$                                (4e)

where $t_i$ is the temporal window of observation i and λ = **0.7** is the
decay rate (tuned on validation). This down-weights older observations that may reflect
pre-drift distributions while preserving them for regularization.

### 4.7 SHAP-Based Drift Attribution

To identify which features are responsible for changes in model behavior
under drift, we compute SHAP values [16] for bot observations before and
during drift. For each feature j, the SHAP value φ_j(x) represents its
additive contribution to the model's prediction for input x:

$$f(x) = \phi_0 + \sum_{j=1}^{d} \phi_j(x)$$                          (5)

where φ_0 is the base value (average prediction).

We define the SHAP drift attribution for feature j at peak drift window
t* as the change in mean absolute SHAP value:

$$\Delta\Phi_j = \frac{1}{|B_{t^*}|}\sum_{x \in B_{t^*}} |\phi_j(x)| - \frac{1}{|B_{\text{ref}}|}\sum_{x \in B_{\text{ref}}} |\phi_j(x)|$$  (6)

where B_ref is the set of bot observations in the reference period and
B_t* is the set of bot observations at the peak drift window (determined
by the highest mean KS statistic). A large |ΔΦ_j| indicates that
feature j's contribution to predictions changed substantially under drift.

This attribution is explanatory: it identifies features whose model
contributions changed, but does not establish that those features caused
the behavioral drift.

### 4.8 Drift-Aware Adaptation

We compare four adaptation strategies:

**Static (S_static):** The model trained on pre-drift data is never
retrained. This serves as the degradation baseline.

**Periodic (S_periodic):** The model is retrained at fixed intervals
(every k windows) regardless of whether drift has been detected.

**Global Drift-Aware (S_global):** The model is retrained only when the
global feature-count drift monitor (Eq. 4) triggers a detection.

**Localized Drift-Aware (S_local):** The model is retrained when the
localized hybrid trigger (Eq. 4d) fires with persistence confirmation.
Retraining uses recency-weighted data (Eq. 4e):

$$\text{If trigger\_confirmed}(t): \quad f_{t+1} \leftarrow \text{train}(\{(X_s, y_s, w_s)\}_{s \leq t})$$  (7)

where $w_s = \exp(-0.3 \cdot (t - t_s))$ are the recency weights.

Retraining uses only train-account data up to the current window; test
account labels are never used for training or drift detection.

We quantify adaptation effectiveness using the F1 degradation during the
drift period:

$$\Delta F_1 = \bar{F}_1^{\text{pre}} - \bar{F}_1^{\text{during}}$$     (8)

where the bars denote averages over the pre-drift (windows 0–4) and
during-drift (windows 5–15) periods respectively.

---

## 5. Experimental Setup

### 5.1 Dataset

We evaluate on **BotDriftBench v0.8**, an empirically calibrated synthetic
longitudinal benchmark. The dataset contains 3,000 accounts (1,500 human,
1,500 bot) observed across 20 temporal windows, producing 60,000
account-window observations with 25 features each.

Per-feature marginal distributions are fitted from TwiBot-22 [2] and
cross-referenced against Cresci-2017 [6]. Core profile features
(followers_count, friends_count, listed_count, account_age_days) evolve
through a dedicated organic growth process across windows rather than artificial
interpolation, preserving natural monotonic aging and compound growth dynamics.
Complete training-human rows are sampled to preserve empirical cross-feature
correlations.

### 5.2 Train/Validation/Test Protocol

The evaluation adheres strictly to an account-disjoint protocol:

- **Account-disjoint 3-way split:** 1,800 accounts (900 human, 900 bot) for
  training; 600 accounts (300 human, 300 bot) for validation; 600 accounts
  (300 human, 300 bot) for testing. All 20 windows of each account stay in
  the same partition. Train, validation, and test account sets are strictly disjoint.
- **Multi-seed replication:** Five independent random seeds (2025, 2026, 2027,
  2028, 2029).
- **Classifier:** XGBoost baseline with fixed hyperparameters across all
  evaluations. Central classification decision cutoff is set to 0.5.
- **Validation-only detector tuning:** KS threshold (θ_KS = 0.06) and fraction
  threshold (θ_frac = 0.20) are tuned strictly on the validation set, with
  cooldown = 2 and confirmation persistence = 2.
- **Strict separation:** Test accounts are strictly held-out and evaluated once;
  test accounts are never used for drift monitoring or model retraining.

### 5.3 Drift Scenarios

Seven account-level drift scenarios are represented among bot accounts:
1. `stable_control` (13%): Bot behavior remains stationary over time.
2. `behavioral_gradual` (17%): Slow, organic behavioral shifts.
3. `gradual_evasion` (26%): Continuous parameter shift toward human feature space.
4. `abrupt_evasion` (10%): Step-function transition toward human-like activity.
5. `localized_evasion` (8%): Targeted manipulation of specific feature groups.
6. `recurring_behavior` (6%): Cyclic shifts between bot and human modalities.
7. `boundary_crossing_evasion` (20%): Aggressive evasion designed to cross static decision boundaries.

### 5.4 Baselines & Adaptation Policies

We compare four adaptation strategies:
1. **Static Baseline:** Model trained on pre-drift windows (0–1) and frozen throughout deployment.
2. **Periodic-6:** Model retrained every 6 windows (retrained at windows 6, 12, 18).
3. **Localized DA:** Drift-aware adaptation triggered by localized KS monitoring on the train bot population with recency-weighted retraining.
4. **Oracle DA:** Reference boundary policy retraining on every window where drift is active (windows 5–17).

### 5.5 Metrics

- **Classification:** Macro-F1, ROC-AUC, Precision, Recall, Accuracy.
- **Drift Detection:** Detection delay (windows from earliest onset to first trigger), false alarms before drift, true detection count.
- **Adaptation Cost:** Number of adaptation retraining events and average retraining interval.

---

## 6. Results

All results in this section are from **BotDriftBench v0.8** using an
account-disjoint 3-way split (1,800 train / 600 validation / 600 test accounts)
evaluated across 5 independent seeds (2025–2029). Source files: `results_v08/aggregate_summary.csv`,
`results_v08/seed_results.csv`, `results_v08/window_results.csv`, `results_v08/scenario_results.csv`,
`results_v08/retraining_events.csv`, and `results_v08/detection_metrics.csv`.

### 6.1 Baseline Classification Performance

**TABLE II: Static Baseline Performance Across Regimes (5 Seeds)**

| Regime | Macro-F1 (mean ± std) | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **Pre-Drift (w0–1)** | 0.9770 ± 0.0008 | 0.9961 | 0.9785 | 0.9754 | 0.9770 |
| **Post-Drift (w2–19)** | 0.8059 ± 0.0015 | 0.8819 | 0.9782 | 0.6394 | 0.8201 |
| **Full Temporal Mean** | 0.8230 ± 0.0014 | 0.8934 | 0.9783 | 0.6726 | 0.8358 |
| **Worst Window (w18)** | 0.6583 ± 0.0032 | 0.7420 | 0.9760 | 0.3540 | 0.6820 |

Source: `results_v08/aggregate_summary.csv`, `results_v08/window_results.csv`.

The static XGBoost baseline exhibits strong initial discrimination (Pre-drift F1 = 0.9770).
However, as evasion drift unfolds, recall degrades precipitously to 63.94% post-drift and
35.40% at window 18, showing that static detectors suffer severe silent failure under evasion.

### 6.2 Drift Detection Performance

The localized KS detector monitors the train bot population across temporal windows
using validation-tuned parameters (θ_KS = 0.06, θ_frac = 0.20, cooldown = 2, confirmation = 2).

**TABLE III: Localized KS Detection Metrics**

| Metric | Target Value | Empirical Value | Status |
|---|---|---|---|
| **Earliest Drift Onset** | Window 2 (excluding controls) | Window 2 | Verified |
| **First Detection Trigger** | Window 5 | Window 5 | Verified |
| **Detection Delay** | ≤ 4 windows | **3 windows** | Verified |
| **Pre-Drift False Alarms** | 0 events (w0–1) | **0 events** | Verified |
| **True Detection Retrains** | Windows 5, 8, 11, 14, 17 | **5 events** | Verified |
| **Average Retraining Interval** | 3.0 windows | **3.0 windows** | Verified |

Source: `results_v08/detection_metrics.csv`, `results_v08/retraining_events.csv`.

The detector achieves a timely detection delay of 3 windows without triggering false alarms
during pre-drift windows (0 pre-drift false alarms across all seeds).

### 6.3 Scenario-Wise Degradation and Recovery

**TABLE IV: Scenario-Wise Policy Performance (Test Accounts, Seed 2025)**

| Scenario | Test Bots | Static F1 | Periodic-6 F1 | Localized DA F1 | Static Recall | Localized DA Recall | DA vs Static (ΔF1) |
|---|---|---|---|---|---|---|---|
| **abrupt_evasion** | 38 | 0.3386 | 0.4591 | **0.4708** | 0.5118 | **0.8895** | +0.1322 |
| **boundary_crossing_evasion** | 51 | 0.3182 | 0.4346 | **0.4463** | 0.4667 | **0.8059** | +0.1281 |
| **gradual_evasion** | 63 | 0.3512 | 0.4606 | **0.4677** | 0.5413 | **0.8786** | +0.1165 |
| **recurring_behavior** | 14 | 0.4203 | 0.4521 | **0.4542** | 0.7250 | **0.8321** | +0.0339 |
| **localized_evasion** | 27 | 0.4556 | 0.4833 | **0.4847** | 0.8370 | **0.9407** | +0.0291 |
| **behavioral_gradual** | 163 | 0.9246 | **0.9453** | 0.9381 | 0.8226 | 0.9245 | +0.0134 |
| **stable_control** | 244 | **0.9677** | 0.9133 | 0.9058 | 0.9631 | 0.9762 | -0.0619 |

Source: `results_v08/scenario_results.csv`.

In the evasion scenarios (`boundary_crossing_evasion`, `abrupt_evasion`, `gradual_evasion`),
the static model degrades severely, with recall falling below 55%. Localized DA restores bot recall
to 80.6%–89.0%, generating substantial scenario-level gains of +0.116 to +0.132 Macro-F1.

### 6.4 Aggregate Adaptation Comparison

**TABLE V: Aggregate Policy Comparison — BotDriftBench v0.8 (5 seeds)**

| Policy | Macro-F1 (mean ± std) | ROC-AUC | Precision | Recall | Accuracy | Worst-Window F1 | Retrains | False Alarms |
|---|---|---|---|---|---|---|---|---|
| **Static Baseline** | 0.8230 ± 0.0014 | 0.8934 | 0.9783 | 0.6726 | 0.8358 | 0.6583 | 0.0 | 0.0 |
| **Periodic-6** | 0.9120 ± 0.0011 | 0.9658 | 0.9370 | 0.8767 | 0.9125 | 0.8723 | 3.0 | 0.0 |
| **da_localized** | **0.9167 ± 0.0009** | **0.9691** | 0.9312 | **0.8931** | **0.9170** | **0.8796** | 5.0 | 0.0 |
| **oracle_da** | 0.9160 ± 0.0011 | 0.9703 | 0.9269 | 0.8958 | 0.9163 | 0.8680 | 13.0 | 0.0 |

Source: `results_v08/aggregate_summary.csv`.

**Key empirical conclusions:**
1. **Recovery over Static:** Localized drift-aware adaptation substantially improves over a frozen static detector (0.9167 vs 0.8230), recovering +0.0937 Macro-F1 and raising bot recall from 67.26% to 89.31%.
2. **Comparison with Periodic-6:** Localized DA performs comparably to Periodic-6 (0.9167 vs 0.9120 Macro-F1, 0.9691 vs 0.9658 ROC-AUC) while using targeted retraining driven by localized drift triggers (5 events vs 3 scheduled events, with 0 pre-drift false alarms).
3. **Comparison with Oracle DA:** Localized DA used 5 adaptation retrains compared with 13 for the oracle reference policy, achieving comparable performance (0.9167 vs 0.9160). This demonstrates that continuous retraining every window yields diminishing returns once targeted adaptation captures primary distribution shifts.

### 6.5 Explainability Analysis: Multi-Model SHAP Attribution

To understand how adaptation modifies detector reasoning across deployment phases,
we computed SHAP feature attributions across three checkpoint models:
(1) `static_pre_drift`: Baseline model evaluated on pre-drift windows (0–1).
(2) `static_post_drift`: Frozen baseline model evaluated on post-drift windows (2–19).
(3) `localized_da_post`: Adapted model evaluated on post-drift windows (2–19).

**TABLE VI: Top-10 Feature Importance Across Models (Mean Absolute SHAP)**

| Feature | Static Pre-Drift | Static Post-Drift | Localized DA Post | Post Shift (Static) | Adaptation Recovery Shift |
|---|---|---|---|---|---|
| **content_similarity** | 1.3368 | 1.3017 | 0.1656 | -0.0351 | -1.1360 |
| **posting_regularity** | 1.1577 | 1.1710 | 0.1912 | +0.0133 | -0.9799 |
| **url_ratio** | 1.0957 | 1.1051 | 0.5539 | +0.0094 | -0.5511 |
| **conversation_depth** | 0.8406 | 0.8092 | 0.3868 | -0.0314 | -0.4224 |
| **listed_count** | 0.8006 | 0.8264 | **1.2554** | +0.0258 | **+0.4290** |
| **burst_score** | 0.7771 | 0.7585 | 0.2061 | -0.0186 | -0.5524 |
| **reply_ratio** | 0.6685 | 0.6467 | 0.1573 | -0.0218 | -0.4894 |
| **hashtag_count_avg** | 0.6588 | 0.6177 | 0.5308 | -0.0410 | -0.0869 |
| **followers_count** | 0.2809 | 0.3029 | **0.7745** | +0.0220 | **+0.4716** |
| **account_age_days** | 0.2028 | 0.2289 | **0.6062** | +0.0260 | **+0.3773** |

Source: `results_v08/shap/shap_importance_comparison.csv`.

**Interpretation:** SHAP analysis documents feature attribution changes across pre-drift,
post-drift, and post-adaptation regimes. Under pre-drift conditions, the detector relied
heavily on behavioral features such as `content_similarity` (1.337) and `posting_regularity` (1.158).
When bots engaged in evasion (lowering repetitive posting and randomizing timing),
the static model's reliance on these features led to missed detections.
Following localized adaptation with recency weighting, the retrained model shifted its primary
reliance toward structural profile attributes—`listed_count` (+0.429), `followers_count` (+0.472),
and `account_age_days` (+0.377)—which are constrained by organic growth and difficult for bots to rapidly forge.

---

## 7. Discussion

### 7.1 Drift Detection: Sensitivity Depends on Detector Design

The detector comparison (Table IIIc) reveals a central finding: different
detection criteria produce dramatically different sensitivity profiles on
the same underlying feature-distribution data. The feature-count detector
(≥3 features with individual KS > 0.15) achieves 63–82% detection across
all five drift scenarios. A mean-KS detector averaging across all 27
features detects 0% of drift windows in any scenario, because most
features are unaffected by each scenario's targeted drift, diluting the
signal below threshold. A max-KS detector is highly sensitive to
network_drift (100%) but misses linguistic and combined drift.

This demonstrates that *not every meaningful behavioral drift produces
a large aggregate distributional shift*. The drift injected by most
scenarios produces moderate shifts in 5–7 targeted features (individual
KS 0.09–0.15), which are detectable by counting features above threshold
but invisible to aggregate statistics. This is consistent with the broader
concept-drift literature distinguishing changes in P(X) from changes in
P(Y|X): a marginal feature-distribution detector may miss concept drift
or decision-boundary drift that does not manifest as large covariate shift.

A single global threshold is insufficient for all drift types. The
benchmark reveals that detector design is itself a research variable,
not merely an engineering choice. Future work should evaluate
SHAP-weighted drift scores (weighting features by their model importance),
prediction-distribution monitors, and combined triggers using both feature
drift and model-performance proxy signals.

The drift detection protocol monitors the *train population* across time:
comparing train-account bot distributions at the reference period (pre-drift)
to train-account bot distributions at each subsequent window. This detects
temporal distributional shifts within the same population rather than
cross-group differences between train and test accounts.

Feature-level validation confirms that targeted features show 1.8–2.8x
higher KS statistics than non-targeted features (Table IIIb), providing
evidence that the detector responds to the specific features shifted by
each scenario rather than spurious distributional noise.

The fixed thresholds (θ_KS = 0.15, θ_count = 3) were not tuned on the
evaluation data. However, they were chosen based on preliminary analysis
during benchmark development, so their effectiveness on this specific
benchmark may be partially confounded.

### 7.2 Adaptation Tradeoffs: From Global Failure to Localized Success

**Global detection fails in mixed populations.** The aggregate evaluation
(Table V, Figures 4a–4b) reveals a structural limitation of global
covariate-drift monitoring: the detector's sensitivity depends on the
proportion of the monitored population that is drifting. In per-scenario
evaluation (Section 6.2), the feature-count detector achieves 63–82%
detection with 1–3 window delay. In aggregate evaluation, detection delay
increases to 8 windows and all global DA policies converge to ~1.4
retraining events. Periodic-4 retrains at windows 5, 9, 13, 17 regardless
of drift state, covering the entire drift period. Global DA policies
retrain only after window 13, missing 8 windows of accumulated drift.

**Localized monitoring recovers early detection.** The localized redesign
(Section 6.4.4) addresses signal dilution by decomposing monitoring into
subpopulation-aware components. Cluster-aware monitoring detects drift
from window 6 by tracking per-cluster feature distributions, detecting
shifts within behavioral subgroups that are invisible in the population
aggregate. Prediction-distribution monitoring adds a complementary signal
from window 10 by tracking shifts in the model's own predicted
probabilities, providing a proxy for concept drift without requiring
labels. The OR-trigger combination achieves 40% detection from drift onset
and 68–100% detection from window 10 onward (Table Ve).

**Recency weighting, not detection, drives the F1 improvement.** The most
informative finding is the ablation between hybrid_recency (F1 = 0.8072)
and hybrid_or without recency (F1 = 0.8022). Both policies use identical
detection triggers and retrain at the same windows (6.2 retrains). The
+0.005 F1 difference comes entirely from exponential recency weighting
(λ = 0.3) of training data. This suggests that *how* the model is
retrained matters more than *when* retraining is triggered — an insight
consistent with the broader adaptive learning literature on sample
weighting [18, 19].

Recency weighting works because older training data, while still
representative of some accounts, contains pre-drift distributions that
dilute the model's adaptation to current behavior. By down-weighting
(but not discarding) older observations, the retrained model more
closely tracks the evolving distribution.

**The tradeoff: F1 vs. retraining cost.** The localized controller
achieves higher F1 than Periodic-4 (+0.0028, 5/5 seeds) but uses more
retraining events (6.2 vs. 4.0). The improvement is in predictive
performance and interpretability, not retraining efficiency. A cooldown-4
variant (5.2 retrains, F1 = 0.8067, 5/5 seeds) partially narrows the
retrain gap while maintaining the F1 advantage. The prediction-only
detector (3.0 retrains, F1 = 0.7999) achieves fewer retrains than
Periodic-4 but lower F1, illustrating the cost-accuracy tradeoff.

**Feature-group detection fails completely in aggregate.** Despite its
success in per-scenario evaluation (Section 6.2), the feature-group
detector never fires in aggregate because the mixed-population dilution
prevents any individual group's mean KS from reaching the 0.12 threshold.
This is a structural limitation: the threshold that avoids false alarms
on non-drifted groups also prevents detection of diluted drift. Adaptive
thresholds or group-specific baselines may address this in future work.

The per-scenario ablation (Table Vc) remains informative as a diagnostic:
it shows that controller design (cooldown, episode logic, budget caps) is
a separate variable from detector design, and produces meaningfully
different tradeoffs when detection works. The localized evaluation
(Table Vd) extends this by showing which detection components provide
signal in the aggregate case.

### 7.3 Explainability Value

SHAP attribution provides interpretable diagnostics that go beyond binary
drift detection. Knowing that content_similarity's model contribution
changed by +0.450 under network drift (Table VIII), for instance, is more
actionable than simply knowing that drift occurred. Practitioners can use this
information to understand *how* the model's reasoning is shifting and
investigate whether the change reflects genuine behavioral evolution or
data quality issues.

However, SHAP values explain model behavior, not the underlying behavioral
mechanisms. A large SHAP change for a feature indicates that the feature's
role in predictions shifted, not necessarily that the feature itself is the
root cause of the drift.

---

## 8. Limitations

1. **Synthetic evaluation only.** All results are from BotDriftBench v0.8,
   an empirically calibrated synthetic benchmark. While marginal distributions
   are fitted from TwiBot-22 and Cresci-2017, performance on live platform data
   may differ due to unmodeled behavioral complexities and open-world dynamics.

2. **Controlled scenario space.** BotDriftBench v0.8 evaluates seven defined
   drift scenarios. Real-world bot evasion may exhibit unanticipated multi-modal
   shifts or simultaneous coordinated campaigns.

3. **Replication scope.** Evaluation is based on 5 independent random seeds
   (2025–2029) across 3,000 accounts (600 test accounts). While performance
   variance across seeds is low (± 0.0009 Macro-F1 for Localized DA), larger
   cohorts and additional seeds would further refine statistical confidence bounds.

4. **Covariate drift vs. concept drift.** The detector monitors feature-distribution
   shifts P_t(X) ≠ P_0(X) on the monitored population. While effective for
   behavioral shifts, pure label-definition changes without covariate shift
   require held-out supervisory labels.

5. **Detection-adaptation coupling.** Localized DA achieved 5 adaptation retrains
   relative to 3 for Periodic-6. The tradeoff between calendar-based simplicity
   and event-driven responsiveness depends on operational infrastructure costs.

---

## 9. Conclusion

We presented a drift-aware bot detection pipeline that integrates
localized Kolmogorov-Smirnov drift monitoring, recency-weighted retraining,
and multi-model SHAP attribution. Evaluated on **BotDriftBench v0.8**
(3,000 accounts, 20 temporal windows, 25 features, account-disjoint 3-way split, 5 seeds):

1. **Static Degradation:** A frozen static classifier suffers severe degradation under
   longitudinal evasion, dropping from Macro-F1 = 0.9770 pre-drift to 0.8059 post-drift
   (worst-window F1 = 0.6583), driven by a collapse in bot recall to 67.26%.
2. **Adaptive Recovery:** Localized drift-aware adaptation substantially improves over
   a frozen static detector (0.9167 vs 0.8230) and performs comparably to Periodic-6 (0.9120)
   while using targeted retraining.
3. **Retraining Efficiency:** Localized DA used 5 adaptation retrains compared with 13 for
   the oracle reference policy, demonstrating that event-driven adaptation captures the
   majority of available performance gains with substantially fewer retraining interventions.
4. **Attribution Auditability:** SHAP analysis documents feature attribution changes across
   pre-drift, post-drift, and post-adaptation regimes, showing how adaptation redistributes
   reliance toward structural attributes when behavioral signals are evaded.

These findings demonstrate the empirical viability of localized drift monitoring and
recency-weighted retraining for maintaining detector robustness under adversarial evasion.

---

## 10. References

[1] C. Shao, G. L. Ciampaglia, O. Varol, K. Yang, A. Flammini, and
    F. Menczer, "The spread of low-credibility content by social bots,"
    *Nature Communications*, vol. 9, no. 1, 2018.

[2] S. Feng, H. Wan, N. Wang, J. Li, and M. Luo, "TwiBot-22: Towards
    graph-based Twitter bot detection," in *Proc. NeurIPS*, 2022.

[3] O. Varol, E. Ferrara, C. A. Davis, F. Menczer, and A. Flammini,
    "Online human-bot interactions: Detection, estimation, and
    characterization," in *Proc. AAAI ICWSM*, 2017.

[4] E. Ferrara, O. Varol, C. Davis, F. Menczer, and A. Flammini, "The
    rise of social bots," *Communications of the ACM*, vol. 59, no. 7,
    pp. 96–104, 2016.

[5] C. Hays, Z. Schutzman, M. Raghavan, E. Walk, and P. Zimmer, "Bots
    don't sit still: A longitudinal study of bot behavioral change on
    social media," arXiv preprint, 2025. [Reference ID pending]

[6] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi,
    "The paradigm-shift of social spambots: Evidence, theories, and tools
    for the arms race," in *Proc. WWW*, 2017.

[7] C. A. Davis, O. Varol, E. Ferrara, A. Flammini, and F. Menczer,
    "BotOrNot: A system to evaluate the credibility of Twitter accounts,"
    in *Proc. WWW Companion*, 2016.

[8] Z. Yang, C. Wilson, X. Wang, T. Gao, B. Y. Zhao, and Y. Dai,
    "Uncovering social network sybils in the wild," *ACM TKDD*, vol. 8,
    no. 1, 2014.

[9] N. Chavoshi, H. Hamooni, and A. Mueen, "DeBot: Twitter bot detection
    via warped correlation," in *Proc. IEEE ICDM*, 2016.

[10] K.-C. Yang, O. Varol, C. A. Davis, E. Ferrara, A. Flammini, and
     F. Menczer, "Arming the public with artificial intelligence to
     counter social bots," *Human Behavior and Emerging Technologies*,
     vol. 1, no. 1, 2019.

[11] S. Feng, H. Wan, N. Wang, and M. Luo, "BotRGCN: Twitter bot
     detection with relational graph convolutional networks," in *Proc.
     ASONAM*, 2021.

[12] S. Feng, Z. Tan, R. Li, and M. Luo, "Heterogeneity-aware Twitter
     bot detection with relational graph transformers," in *Proc. AAAI*,
     2022.

[13] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. Bouchachia,
     "A survey on concept drift adaptation," *ACM Computing Surveys*,
     vol. 46, no. 4, 2014.

[14] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang, "Learning
     under concept drift: A review," *IEEE TKDE*, vol. 31, no. 12, 2019.

[15] S. Cresci, "A decade of social bot detection," *Communications of
     the ACM*, vol. 63, no. 10, pp. 72–83, 2020.

[16] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting
     model predictions," in *Proc. NeurIPS*, 2017.

[17] M. Knauth, "Language-agnostic Twitter bot detection," in *Proc.
     RANLP*, 2019.

[18] A. Bifet and R. Gavalda, "Learning from time-changing data with
     adaptive windowing," in *Proc. SDM*, 2007.

[19] G. Ditzler, M. Roveri, C. Alippi, and R. Polikar, "Learning in
     nonstationary environments: A survey," *IEEE CIM*, vol. 10, no. 4,
     2015.

[20] J. P. Dickerson, V. Kagan, and V. S. Subrahmanian, "Using sentiment
     to detect bots on Twitter: Are humans more opinionated than bots?,"
     in *Proc. ASONAM*, 2014.

[21] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system,"
     in *Proc. ACM KDD*, 2016.

[22] S. Kudugunta and E. Ferrara, "Deep neural networks for bot
     detection," *Information Sciences*, vol. 467, pp. 312–322, 2018.

---

## TABLE I: Benchmark Configuration — BotDriftBench v0.8

| Parameter | Value | Description |
|---|---|---|
| Total accounts | 3,000 | 1,500 human, 1,500 bot |
| Account split | 1,800 / 600 / 600 | 60% train, 20% validation, 20% test (account-disjoint) |
| Temporal windows | 20 | Windows 0–19 |
| Pre-drift windows | 0–1 | Initial stable training reference |
| Post-drift windows | 2–19 | Onset window 2, active evasion windows |
| Feature count | 25 | 10 profile features, 15 behavioral features |
| Seeds | 5 seeds | 2025, 2026, 2027, 2028, 2029 |
| Decision cutoff | 0.5 | Fixed probability threshold across all evaluations |
| Drift detector | Localized KS | θ_KS = 0.06, θ_frac = 0.20, cooldown = 2, confirmation = 2 |
