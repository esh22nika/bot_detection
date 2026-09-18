# Demo Script: Drift-Aware Bot Detection (3-5 minutes)

## Prerequisites

- Streamlit app running: `python -m streamlit run app.py`
- Model artifacts loaded from `model_artifacts/artifacts.pkl`
- BotDriftBench v0.3 dataset in `dataset_v03/`

---

## Slide 1: The Problem (30 seconds)

**Say**: "Social bot detectors work well when they're first deployed, but bot
operators adapt their behavior over time. When the distribution of bot features
shifts — what we call behavioral drift — the detector's accuracy degrades
silently. There's no alert, no explanation, and no automated response."

**Show**: Title slide or the degradation plot from `final_v03/final_degradation.png`
showing F1 declining across drift scenarios.

---

## Slide 2: Our Approach (30 seconds)

**Say**: "We propose a three-part system: detect drift using statistical
monitoring, explain which features changed using SHAP, and adapt by retraining
only when drift is detected. The key contribution is making the degradation
visible and the response targeted."

**Show**: System architecture diagram or the pipeline overview.

---

## Slide 3: Live Demo — Baseline Prediction (45 seconds)

**Action**: Open the Streamlit app. Navigate to the Predict page.

1. Enter features for a typical human account:
   - account_age_days: 2000, statuses_count: 500, followers_count: 300
   - following_count: 200, description_length: 80
   - Leave behavioral features at moderate values

2. Click Predict. Show the bot probability (should be low for human-like values).

3. Show the SHAP waterfall plot explaining which features pushed the prediction.

**Say**: "Using an account-disjoint evaluation — where the model never sees
test accounts during training — the baseline XGBoost achieves F1 = 0.81.
Each prediction comes with a SHAP explanation showing which features
contributed most."

---

## Slide 4: Live Demo — Bot Account (30 seconds)

**Action**: Change inputs to bot-like values:
   - account_age_days: 30, statuses_count: 5000, followers_count: 10
   - following_count: 1000, content_similarity: 0.9, burst_score: 0.8

1. Click Predict. Show high bot probability.
2. Point out how the SHAP waterfall shifts — different features now dominate.

**Say**: "For a bot-like account, the model flags high content similarity and
unusual posting patterns. The SHAP explanation makes the decision interpretable."

---

## Slide 5: Drift and Degradation (45 seconds)

**Action**: Navigate to the Degradation page. Select a drift scenario
(e.g., "network_drift").

1. Show the F1 curve dropping after window 5 (drift onset).
2. Compare against the no_drift control to isolate the drift-specific effect.
3. Note the aggregate F1 decrease: 0.054 average across scenarios for static.

**Say**: "When bot behavior shifts, the static model degrades. The aggregate
effect is clearest — 0.054 average F1 drop for the static model. Per-scenario
breakdowns are noisier due to small test bot counts, but the control-adjusted
effects confirm that drift detection responds to genuine distributional
changes, not noise."

---

## Slide 6: Drift Detection (45 seconds)

**Action**: Navigate to the Drift Scores page. Select the same scenario.

1. Show the KS statistic rising after drift onset.
2. Point out the detection threshold and when it's crossed.
3. Note the 2-3 window detection delay.

**Say**: "Our KS-statistic monitor tracks per-feature distribution shifts.
It detects drift within 2-3 windows of onset with no false alarms in the
25 evaluated pre-drift scenario-windows — though that's a small sample.
The detection is feature-level, so we know which behavioral dimensions
changed."

---

## Slide 7: Adaptation Comparison (45 seconds)

**Action**: Navigate to the Adaptation page. Select a scenario.

1. Show three strategy curves: static (red), periodic (blue), drift-aware (green).
2. Point out how drift-aware tracks closer to the pre-drift baseline.
3. Show the efficiency summary table.

**Say**: "Both periodic and drift-aware retraining dramatically reduce
degradation compared to static — 0.008 and 0.012 average F1 drop versus
0.054. Periodic actually achieves the smallest degradation with fewer
retraining events. The advantage of drift-aware is not efficiency but
*interpretability* — each retraining comes with a detection signal telling
you which features shifted and by how much."

---

## Slide 8: Limitations and Honest Framing (30 seconds)

**Say**: "Three important caveats. First, these results are on synthetic data
with controlled drift — real-world validation is needed. Second, SHAP explains
feature contributions but does not prove that those features caused the drift.
Third, the Streamlit app is a simulation for demonstration, not a production
Twitter monitoring system."

---

## Key Numbers to Reference

| Metric | Value |
|---|---|
| Baseline F1 (seed 42, account-disjoint) | 0.8115 |
| Baseline F1 (5-seed mean) | 0.8106 +/- 0.0114 |
| Baseline AUC (5-seed mean) | 0.8947 +/- 0.0015 |
| Detection rate | 63.6-81.8% |
| Detection delay | 1-3 windows |
| Pre-drift false alarm rate | 0% |
| Targeted/non-targeted KS ratio | 1.8-2.8x |
| Static avg abs F1 drop | 0.054 |
| Periodic avg abs F1 drop | 0.008 (4.0 retrains) |
| Drift-aware avg abs F1 drop | 0.012 (12.6 retrains) |

## Timing Guide

| Section | Duration | Cumulative |
|---|---|---|
| Problem | 0:30 | 0:30 |
| Approach | 0:30 | 1:00 |
| Baseline demo | 0:45 | 1:45 |
| Bot demo | 0:30 | 2:15 |
| Degradation | 0:45 | 3:00 |
| Detection | 0:45 | 3:45 |
| Adaptation | 0:45 | 4:30 |
| Limitations | 0:30 | 5:00 |

Total: ~5 minutes. For a 3-minute version, combine slides 3-4 and skip
slide 6 (mention detection inline during adaptation).
