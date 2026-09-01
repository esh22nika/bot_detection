# Drift-Aware Bot Detection — Revised Implementation Plan

## Contribution (corrected framing)

**Not:** "We made a dataset" / "We use SHAP" / "We use ADWIN"

**Instead:** A drift-aware bot detection framework that monitors changes in the
model's behavioral decision space and adapts its detector when bot behavior evolves.
The dataset is an enabling research artifact, not the contribution itself.

## Research Question

> Does SHAP-space drift provide earlier or more useful information about behavioral
> evolution than conventional feature-distribution drift or prediction drift?

### Experimental comparison (three drift detection approaches)

| Approach | Method | Signal |
|----------|--------|--------|
| A. Feature drift | KS / PSI on raw feature distributions | Raw input change |
| B. Prediction drift | KS on P(y_hat) | Model output change |
| C. SHAP drift | KS / ADWIN on SHAP value distributions | Decision-space change |

## Constraint Reality

| Resource | Status | Workaround |
|----------|--------|------------|
| Twitter/X API | Unavailable | Synthetic benchmark + public datasets |
| Instagram/Meta API | Unavailable | Not needed (scoped to Twitter-like behavior) |
| Google API | Rate-limited | Not a dependency |

## Dataset Strategy (3-tier)

### Tier A — Controlled Synthetic (BotDriftBench)
- Known change point, drift type, affected features, magnitude
- Purpose: **Can the algorithm detect known drift?**
- Status: v0.1 complete and validated

### Tier B — Real Historical (Cresci-2017, TwiBot-22)
- Download from public repositories
- Purpose: **Does the model work on actual bot data?**

### Tier C — Real + Controlled Evolution
- Start with Tier B empirical distributions, inject controlled drift
- Purpose: **Does the method work when drift is grounded in real behavior?**

## Architecture

```
              REAL / SYNTHETIC DATA
                      │
                      ▼
             Behavioral Modeling
                      │
                      ▼
          ┌───────────────────────┐
          │ Bot Behavior Simulator │
          │  D1: Activity         │
          │  D2: Linguistic       │
          │  D3: Coordination     │
          │  D4: Network          │
          │  D5: Combined         │
          └───────────┬───────────┘
                      │
               controlled stream
                      │
                      ▼
              Detection Model
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
         Prediction            SHAP
            │                   │
            │                   ▼
            │             SHAP trajectory
            │                   │
            └─────────┬─────────┘
                      ▼
               Drift Engine
            ┌─────────┴─────────┐
            ▼         ▼         ▼
      Feature     Prediction   SHAP
       drift       drift       drift
            └─────────┬─────────┘
                      ▼
                Drift Decision
                      │
                      ▼
              Model Adaptation
                      │
                      ▼
                  Model v2
```

## Model Plan (corrected)

| Model | Role | Purpose |
|-------|------|---------|
| XGBoost | Baseline | Standard tree-based bot detector |
| GRU / Temporal Transformer | Advanced | Temporal sequence model over account trajectories |
| Drift-aware temporal model | Core contribution | Adaptive model with retraining triggers |

Comparison: XGBoost vs Temporal vs Drift-Aware Temporal

## Revised 12-Week Roadmap

### Phase 1: Data + Experimental Protocol (Weeks 1-2)
- [x] Proposal finalized
- [x] Schema + drift taxonomy + generation equations
- [x] BotDriftBench v0.1 (1,000 accounts, validated, FROZEN)
- [x] Account archetypes for cross-feature correlations
- [x] Realistic classifier difficulty (F1=0.970, AUC=0.994)
- [x] Multi-magnitude variants (delta=0.25, 0.5, 1.0, 2.0)
- [x] Drift detection delay experiment across magnitudes x seeds
- [x] SHAP drift attribution precision/recall metrics
- [x] Static vs Periodic vs Drift-Aware adaptation experiment
- [ ] Download Cresci-2017 / TwiBot-22
- [ ] Fit empirical distributions to real data
- [ ] BotDriftBench v1.0 (10,000 accounts, empirically grounded)

### Phase 2: Feature Engineering + Baseline (Weeks 3-4)
- [ ] Unified feature extraction pipeline
- [ ] Train/test split strategy (temporal, not random)
- [ ] XGBoost baseline on Tier A + Tier B
- [ ] Performance benchmarks

### Phase 3: Temporal Model (Weeks 5-6)
- [ ] Account trajectory → feature sequence representation
- [ ] GRU or Temporal Transformer architecture
- [ ] Compare: XGBoost vs temporal model

### Phase 4: SHAP Explainability Layer (Week 7)
- [ ] TreeExplainer for XGBoost
- [ ] SHAP computation per time window
- [ ] SHAP value storage + trajectory tracking

### Phase 5: Drift Detection Engine (Weeks 8-9)
- [ ] Three-way comparison: feature / prediction / SHAP drift
- [ ] ADWIN / Page-Hinkley integration
- [ ] Detection delay + false alarm rate evaluation
- [ ] Answer: which drift space gives earliest/most useful signal?

### Phase 6: Model Adaptation (Week 10)
- [ ] Retraining trigger mechanism
- [ ] Before/after adaptation performance comparison
- [ ] Model v2 evaluation on post-drift data

### Phase 7: Dashboard + Paper (Weeks 11-12)
- [ ] Streamlit monitoring dashboard (demonstration layer)
- [ ] Auto-generated drift reports
- [ ] End-to-end evaluation across all 3 tiers
- [ ] Final paper write-up

## Preliminary Experimental Results (v0.1)

### F1 degradation scales with drift magnitude

| Magnitude | F1 pre-drift | F1 post-drift | F1 drop |
|-----------|-------------|---------------|---------|
| 0.25      | 0.999       | 0.965         | 3.5%    |
| 0.50      | 0.999       | 0.959         | 4.1%    |
| 1.00      | 0.999       | 0.945         | 5.4%    |
| 2.00      | 0.999       | 0.922         | 7.8%    |

### SHAP drift attribution

- Precision_SHAP: ~0.73-0.80 (SHAP correctly identifies drifted features)
- Recall_SHAP: ~0.32-0.36 (bounded by top-10 of 22 drifted features)
- Consistent across 3 random seeds

### Drift detection sensitivity (preliminary)

- Prediction drift: most sensitive, detects at onset (but requires labels)
- Feature-space drift: less sensitive, requires stronger signal
- SHAP-space drift: further investigation needed for optimal threshold

### Adaptation experiment v2 (CORE ML CONTRIBUTION — 10 seeds, mean +/- std)

| delta | Static F1_drop | Periodic F1_drop | DA F1_drop | DA Retrains | DA Delay | DA Efficiency |
|-------|---------------|-----------------|-----------|------------|---------|--------------|
| 0.25  | 3.5%+/-0.3%   | 2.1%+/-0.1%     | 2.8%+/-0.5% | 1.1+/-0.7 | 4.0+/-3.3w | 0.018+/-0.009 |
| 0.50  | 4.1%+/-0.3%   | 2.0%+/-0.1%     | 2.6%+/-0.5% | 1.6+/-0.8 | 4.0+/-3.0w | 0.017+/-0.010 |
| 1.00  | 5.4%+/-0.7%   | 2.2%+/-0.2%     | 2.8%+/-0.5% | 2.0+/-0.8 | 2.7+/-2.1w | 0.012+/-0.006 |
| 2.00  | 7.8%+/-1.4%   | 2.0%+/-0.2%     | 3.0%+/-0.4% | 2.3+/-0.6 | 1.8+/-1.3w | 0.005+/-0.002 |

Key findings:
- Static detector degrades monotonically with drift magnitude (3.5-7.8%)
- Periodic retraining recovers F1 (~2% drop) but always uses 4 retrains
- Drift-aware achieves comparable robustness (2.6-3.0% drop) with 50-75% fewer retrains
- Zero false alarms across all 40 configurations
- Detection delay scales inversely with magnitude (4.0w at 0.25, 1.8w at 2.0)
- Recovery time: near-instant after retrain (0-0.2 windows)
- Adaptation efficiency: DA is 2-3x more efficient than periodic per retrain

Causal validation (static model):
- At delta=0.25-1.0: class distance INCREASES post-drift (bots move farther from humans)
  yet F1 still drops -> true concept drift (stale decision boundaries, not class overlap)
- At delta=2.0: class distance decreases AND F1 drops -> both effects compound
- AUC and F1 degrade coherently across all magnitudes

Framing: "Drift-aware adaptation achieves comparable robustness to periodic retraining
while requiring substantially fewer retraining events." (efficiency claim, not superiority)

## What's Been Delivered

BotDriftBench v0.1 (FROZEN):
- 1,000 accounts x 20 windows = 20,000 observations
- True longitudinal trajectories (Markov state carry-forward)
- Account archetypes for cross-feature correlations (5 human + 5 bot types)
- Measurement noise simulating real feature extraction uncertainty
- 5 drift scenarios with mathematical formalization
- Controllable drift magnitude delta in {0.25, 0.5, 1.0, 2.0}
- Human behavioral variation (not static)
- All 6 validation questions PASS (F1=0.970, meaningful degradation)
- Full data dictionary with distribution tables
- Drift detection delay experiment across magnitudes x seeds
- Formal SHAP drift attribution metrics (precision/recall)
- Adaptation experiment v2: Static vs Periodic vs Drift-Aware (2400 rows, 10 seeds)
  - Recovery time, recovery magnitude, adaptation efficiency metrics
  - Causal validation (class distance + AUC + F1 coherence)
  - Zero false alarm rate across all configurations
- Reproducible generator with documented methodology
