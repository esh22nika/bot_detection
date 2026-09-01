# BotDriftBench v0.1 — Data Dictionary

## Overview

**BotDriftBench** is a controlled, reproducible benchmark for evaluating social-bot
detection systems under parameterized behavioral shifts. Each row represents one
account observed at one time window — accounts are **longitudinal trajectories**
where state carries forward via a Markov process, not independent random samples.

The dataset provides ground-truth labeled behavioral drift across five scenarios,
enabling rigorous evaluation of:
- Concept drift detection algorithms
- SHAP-based feature importance monitoring
- Model retraining trigger mechanisms
- Comparative analysis of feature-space vs prediction-space vs SHAP-space drift detection

---

## Mathematical Framework

### Evolution model

For continuous feature *j* of account *i* at time *t*:

    X_{i,j,t} = X_{i,j,t-1} + μ_{trend,j} + ε_{i,j,t}

where:
- `μ_{trend,j}` is the natural per-step trend (differs for human vs bot)
- `ε ~ N(0, σ_{natural,j})` provides organic window-to-window variation

For human accounts, ~2% per window experience a "life event" (activity change,
interest shift, or posting break), introducing larger perturbations to prevent
the trivial "anything different = bot" shortcut.

### Drift injection

For bot accounts in drift scenario *s*, during the transition period [τ_start, τ_end):

**Additive features:**

    X_{i,j,t} = X_{i,j,t-1} + μ_{trend,j} + (δ_{s,j} · M / N) + ε_{i,j,t}

**Multiplicative features** (log-scale features like tweets_per_day, burst_score):

    X_{i,j,t} = X_{i,j,t-1} · (1 + δ_{s,j} · M / N + noise) + μ_{trend,j} + ε_{i,j,t}

where:
- `δ_{s,j}` is the base drift vector for scenario *s*, feature *j*
- `M` is the drift magnitude parameter (--drift-magnitude, default 1.0)
- `N` = τ_end - τ_start (number of transition steps)

After τ_end, no further drift is applied — the state carries forward with only
natural evolution and noise.

### Change point definition

    P_t(X_j | Y=bot, scenario=s) =
        P_0(X_j | Y=bot)                              if t < τ_start
        P_0(X_j | Y=bot) + Σ δ_{s,j} · M / N         if τ_start ≤ t < τ_end (accumulating)
        P_0(X_j | Y=bot) + δ_{s,j} · M               if t ≥ τ_end (plateau + noise)

### Drift magnitude as experimental parameter

The `--drift-magnitude` flag scales all δ values:

| Magnitude | Interpretation | Expected detectability |
|-----------|---------------|----------------------|
| 0.1 | Subtle | Hard to detect; may not reach significance |
| 0.25 | Mild | Detectable with sensitive methods |
| 0.5 | Moderate | Clearly detectable; realistic scenario |
| 1.0 | Standard | Strong, unambiguous drift (default) |
| 2.0 | Extreme | For stress-testing detection algorithms |

---

## Files

| File | Description | Rows |
|------|-------------|------|
| `botdriftbench_v0.1.csv` | Main dataset (all observations) | 20,000 |
| `account_metadata.csv` | Per-account summary | 1,000 |
| `drift_ground_truth.csv` | Only drift-active observations | ~6,240 |
| `window_summary_statistics.csv` | Per-window aggregate stats | 20 |
| `generation_config.json` | Full parameters, distributions, references | — |

---

## Main Dataset Schema

### Identification & Temporal Columns

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | string | `H_XXXXX` (human) or `B_XXXXX` (bot) |
| `label` | string | `"human"` or `"bot"` |
| `time_window` | int | 0–19, each ~30 days |
| `window_date` | date | Calendar date (YYYY-MM-DD), from 2023-01-01 |

### Drift Metadata

| Column | Type | Description |
|--------|------|-------------|
| `drift_scenario` | string | Scenario name or `"none"` for humans / no_drift bots |
| `drift_active` | int | 1 if drift is active at this observation |
| `drift_type` | string | `"none"` or `"gradual"` |
| `drift_progress` | float | 0.0 (onset) to 1.0 (fully drifted) |
| `drifted_features` | string | Pipe-separated feature names affected |

### Feature Columns (30 total)

#### Profile Features (12)

| Feature | Range | Human Dist | Bot Dist | Evidence |
|---------|-------|-----------|---------|----------|
| `account_age_days` | >0 | LogN(6.5, 1.1) | LogN(6.0, 1.3) | Cresci 2017 |
| `statuses_count` | >=0 | LogN(7.5, 1.7) | LogN(7.6, 1.8) | TwiBot-22 |
| `followers_count` | >=0 | LogN(4.5, 2.2) | LogN(4.0, 2.3) | Cresci 2017 |
| `friends_count` | >=0 | LogN(4.8, 1.7) | LogN(5.0, 1.7) | TwiBot-20 |
| `favourites_count` | >=0 | LogN(5.8, 2.3) | LogN(5.0, 2.5) | TwiBot-22 |
| `listed_count` | >=0 | LogN(1.3, 1.6) | LogN(0.9, 1.5) | Cresci 2017 |
| `default_profile` | {0,1} | Bern(0.20) | Bern(0.30) | TwiBotX 2026 |
| `default_profile_image` | {0,1} | Bern(0.10) | Bern(0.18) | Cresci 2017 |
| `has_url` | {0,1} | Bern(0.28) | Bern(0.20) | TwiBotX 2026 |
| `verified` | {0,1} | Bern(0.02) | Bern(0.003) | TwiBot-22 |
| `description_length` | >=0 | N(70, 45) | N(50, 45) | TwiBotX 2026 |
| `screen_name_length` | >=3 | N(10.5, 3) | N(11.5, 3) | Cresci 2017 |

#### Behavioral Features (10)

| Feature | Range | Human Dist | Bot Dist | Evidence |
|---------|-------|-----------|---------|----------|
| `tweets_per_day` | >=0.01 | LogN(1.0, 1.3) | LogN(1.5, 1.3) | TwiBot-22 |
| `retweet_ratio` | [0,1] | Beta(3, 4) | Beta(4, 3.5) | Cresci 2017 |
| `reply_ratio` | [0,1] | Beta(2.5, 5) | Beta(2, 6) | TwiBotX 2026 |
| `mention_count_avg` | >=0 | Exp(1.3) | Exp(1.6) | TwiBot-20 |
| `hashtag_count_avg` | >=0 | Exp(0.9) | Exp(1.2) | Cresci 2017 |
| `url_share_ratio` | [0,1] | Beta(2, 6) | Beta(2.5, 5) | TwiBotX 2026 |
| `posting_regularity` | [0.05,1] | Beta(3, 3.5) | Beta(4, 2.5) | Hays 2025 |
| `night_post_ratio` | [0,1] | Beta(2.5, 7) | Beta(3, 5.5) | Hays 2025 |
| `weekend_ratio` | [0,1] | N(0.28, 0.08) | N(0.29, 0.07) | TwiBot-22 |
| `burst_score` | >=1 | LogN(0.5, 0.7) | LogN(0.8, 0.8) | Cresci 2017 |

#### Content Features (6)

| Feature | Range | Human Dist | Bot Dist | Evidence |
|---------|-------|-----------|---------|----------|
| `avg_tweet_length` | >=10 | N(110, 45) | N(95, 42) | TwiBotX 2026 |
| `vocabulary_richness` | [0.05,1] | Beta(4, 3.5) | Beta(3.5, 4) | MDPI 2026 |
| `text_entropy` | [0.5,6] | N(3.9, 0.7) | N(3.6, 0.8) | TwiBotX 2026 |
| `emoji_ratio` | [0,0.3] | Exp(0.022) | Exp(0.018) | TwiBot-22 |
| `caps_ratio` | [0,1] | Beta(1.5, 14) | Beta(1.6, 13) | TwiBotX 2026 |
| `content_similarity` | [0,1] | Beta(2.5, 5) | Beta(3, 4) | Cresci 2017 |

#### Network Features (2)

| Feature | Range | Human Dist | Bot Dist | Evidence |
|---------|-------|-----------|---------|----------|
| `follower_friend_ratio` | >0 | LogN(0.0, 1.3) | LogN(-0.3, 1.4) | Cresci 2017 |
| `reputation_score` | [0,1] | Beta(3, 3.5) | Beta(2.5, 4) | TwiBot-20 |

**Design note:** Distribution gaps are intentionally narrow — modern LLM-era
bots overlap significantly with humans on many features. Individual feature KS
statistics range from 0.08-0.28, reflecting realistic per-feature discriminability.
Classifier difficulty comes from combining 30 features, not from any single feature
being a silver bullet.

### Account Archetypes

Each account is assigned an archetype that coherently shifts multiple features
together, creating realistic cross-feature correlations. Without archetypes,
independently sampling each feature allows classifiers to exploit impossible
feature combinations (e.g., a "spam bot" with high vocabulary richness).

#### Human Archetypes

| Archetype | Weight | Key traits |
|-----------|--------|------------|
| Power user | 15% | High volume, many followers, rich vocabulary |
| Casual | 40% | Baseline distributions, no modifiers |
| Lurker | 20% | Very low posting, high retweet ratio |
| New account | 15% | Young account, sparse profile, few connections |
| Professional | 10% | Regular posting, URL sharing, longer descriptions |

#### Bot Archetypes

| Archetype | Weight | Key traits |
|-----------|--------|------------|
| Spam bot | 15% | High volume, URL/hashtag heavy, repetitive content |
| Social bot | 30% | Conversational, customized profile, diverse text |
| Fake follower | 15% | Nearly dormant, follows aggressively, low content |
| Political bot | 15% | Hashtag/retweet heavy, bursty, coordinated mentions |
| Sophisticated bot | 20% | Human-like on most features, hard to distinguish |

### Measurement Noise

All observed feature values include per-observation measurement noise simulating
real-world feature extraction uncertainty:
- Bounded [0,1] features: additive N(0, 0.03)
- Unbounded features: multiplicative N(0, 0.08) (8% relative noise)

---

## Drift Scenarios

### D1: `activity_drift` (Gradual)
**Rationale:** Bots adapt posting schedules to evade temporal detection heuristics.
**Drift vector (δ):**

| Feature | δ | Application | Cumulative effect |
|---------|---|-------------|-------------------|
| tweets_per_day | -0.50 | Multiplicative | ~50% volume reduction |
| posting_regularity | -0.25 | Additive | Becomes less metronomic |
| night_post_ratio | -0.15 | Additive | Reduces 24/7 activity |
| burst_score | -0.40 | Multiplicative | ~40% burst reduction |
| weekend_ratio | +0.03 | Additive | Mimics human weekend pattern |

**Literature:** Hays et al. (2025): bot posting patterns become less distinguishable over time.

### D2: `linguistic_drift` (Gradual)
**Rationale:** Bots adopt LLM-generated text, increasing linguistic sophistication.
**Drift vector (δ):**

| Feature | δ | Application | Cumulative effect |
|---------|---|-------------|-------------------|
| text_entropy | +0.80 | Additive | Higher text complexity |
| vocabulary_richness | +0.15 | Additive | More diverse word usage |
| content_similarity | -0.20 | Additive | Less content repetition |
| avg_tweet_length | +30.0 | Additive | Longer, more elaborate tweets |
| caps_ratio | -0.03 | Additive | Less "shouting" |

**Literature:** BotEvo (2026): LLM-equipped bots produce higher entropy, richer vocabulary.

### D3: `coordination_drift` (Gradual)
**Rationale:** Bots shift engagement to mimic authentic interaction patterns.
**Drift vector (δ):**

| Feature | δ | Application | Cumulative effect |
|---------|---|-------------|-------------------|
| reply_ratio | +0.12 | Additive | More conversational |
| retweet_ratio | -0.15 | Additive | Fewer bulk amplification |
| mention_count_avg | -0.80 | Additive | Fewer mass mentions |
| hashtag_count_avg | -0.60 | Additive | Less hashtag spam |
| url_share_ratio | -0.10 | Additive | Fewer promotional URLs |

**Literature:** TwiBotX (2026); TwiBot-22: engagement ratios are key discriminators.

### D4: `network_drift` (Gradual)
**Rationale:** Bot profiles mature — custom images, bios, URLs, grown followers.
**Drift vector (δ):**

| Feature | δ | Application | Cumulative effect |
|---------|---|-------------|-------------------|
| default_profile | -0.50 | Binary flip prob | ~50% switch to custom |
| default_profile_image | -0.40 | Binary flip prob | ~40% add real avatar |
| has_url | +0.45 | Binary flip prob | ~45% add profile URL |
| description_length | +35.0 | Additive | Longer, more complete bios |
| followers_count | +0.50 | Multiplicative | ~50% follower growth |
| listed_count | +3.0 | Additive | More list inclusions |
| reputation_score | +0.12 | Additive | Improved follower ratio |

**Literature:** Cresci (2017): paradigm-shift from sparse to sophisticated profiles.

### D5: `combined_drift` (Gradual)
**Rationale:** Simultaneous multi-axis evolution — realistic sophisticated adversary.
**Drift vector (δ):**

| Feature | δ | Application | Cumulative effect |
|---------|---|-------------|-------------------|
| tweets_per_day | -0.35 | Multiplicative | ~35% volume reduction |
| posting_regularity | -0.18 | Additive | Less regular |
| text_entropy | +0.50 | Additive | Higher text complexity |
| vocabulary_richness | +0.10 | Additive | More diverse |
| content_similarity | -0.15 | Additive | Less repetitive |
| reply_ratio | +0.08 | Additive | More replies |
| retweet_ratio | -0.10 | Additive | Fewer retweets |

**Literature:** BotEvo (2026); Hays (2025): real bot evolution is multi-dimensional.

### Control: `no_drift`
No injected drift. Serves as control group — subject only to natural evolution and noise.

---

## Temporal Structure

| Windows | Phase | Description |
|---------|-------|-------------|
| 0–4 | Pre-drift baseline | All accounts follow base distributions + natural noise |
| 5–14 | Drift transition | Gradual drift injection; progress 0.0→1.0 |
| 15–19 | Post-drift plateau | Drift complete; state carries forward with noise only |

---

## Human Behavioral Variation

Human accounts are not static. They undergo:
1. **Natural trend:** organic growth in followers, statuses, favourites
2. **Window-to-window noise:** proportional to feature scale
3. **Life events:** ~2% of humans per window experience one of:
   - Activity change (tweets_per_day scaled 0.5×–2×)
   - Interest shift (hashtag/vocab/entropy perturbation)
   - Posting break (tweets_per_day scaled 0.1×–0.5×)

This prevents the trivial baseline of "anything different from initial = bot."

---

## Validation Summary (v0.1 — 1,000 accounts)

| Question | Result | Evidence |
|----------|--------|----------|
| Q1: Plausible separation? | PASS | All 6 key features KS p < 10^-7 |
| Q2: Drift visible? | PASS | All 5 scenarios significant KS shifts |
| Q3: Baseline detector works? | PASS | XGBoost F1=0.970, AUC=0.994 on pre-drift |
| Q4: Performance degrades? | PASS | F1: 0.990->0.942 (4.8% drop), AUC: 0.999->0.985 |
| Q5: Change point found? | PASS | Feature drift at w=13 (delay 8), prediction at w=5 (delay 0) |
| Q6: SHAP interpretable? | PASS | 7/10 top SHAP-changing features are known drifted |

### Note on classifier difficulty
The pre-drift F1 of 0.970 reflects realistic difficulty — modern bots overlap
significantly with humans on most individual features (per-feature KS 0.08-0.28).
The classifier succeeds by combining many weak signals, not by relying on any
single discriminating feature. This is achieved through:
1. Narrow distribution gaps between human and bot marginals
2. Account archetypes creating cross-feature correlations
3. Measurement noise simulating real-world feature extraction uncertainty
4. Sophisticated bot archetype (20% of bots) designed to be near-indistinguishable

---

## Intended Use

- Evaluating concept drift detection under controlled behavioral shifts
- Benchmarking SHAP-based vs feature-distribution vs prediction drift monitoring
- Testing model retraining trigger mechanisms
- Comparing drift detection sensitivity across magnitude settings

### Not Intended For

- Training production bot detection systems (synthetic, not real accounts)
- Making claims about real bot prevalence on specific platforms
- Replacing evaluation on real public datasets (use alongside Cresci-2017, TwiBot-22)

---

## Reproducibility

```bash
python generate_dataset.py --seed 42 --accounts 1000 --windows 20 --drift-magnitude 1.0
python validate_dataset.py
```

Requires: Python 3.12+, numpy, pandas, scipy, xgboost, shap, scikit-learn.

---

## Citation

```bibtex
@dataset{botdriftbench2026,
  title   = {BotDriftBench: A Controlled Benchmark for Evaluating Social-Bot
             Detection Under Parameterized Behavioral Drift},
  version = {0.1.0},
  year    = {2026},
  license = {CC BY 4.0},
}
```

## License

CC BY 4.0 — free to share and adapt with attribution.
