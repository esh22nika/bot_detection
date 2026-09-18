# Exported Artifact Integrity Audit: BotDriftBench v0.8

**Audit Date**: 2026-09-19  
**Audit Scope**: Consistency, schema verification, and metric agreement of exported evaluation artifacts  
**Target Benchmark**: BotDriftBench v0.8 (3,000 accounts, 20 windows, 25 features, 60,000 rows)

> [!NOTE]
> **Methodological Scope & Limitation**:  
> The current audit verifies exported artifact consistency, schema integrity, and numerical metric agreement across saved outputs. **Full end-to-end rerun reproducibility was not established by this audit**, as rerun execution of the multi-seed generator and training grid from raw initialization was not re-executed during this static file verification.

---

## Executive Audit Summary

| Check Item | Target Requirement | Empirical Audit Result | Verification Status |
|---|---|---|---|
| **1. Predictions File Shape** | Exactly 36,000 rows (600 test accts x 20 windows x 3 policies) | Exactly **36,000 rows**, 600 unique accounts, 20 windows, 3 policies | **PASSED** |
| **2. Split Integrity & Leakage** | All 600 accounts in `predictions.csv` belong strictly to `test` split | `is_subset(test) = True`, `is_disjoint(train) = True`, `is_disjoint(val) = True` | **PASSED** |
| **3. Decision Threshold Audit** | Central threshold definition `CLASSIFICATION_THRESHOLD = 0.5` | Standard cutoff **0.5** verified across `run_v08_comparison.py` and `predictions.csv`. Validation tuning was strictly applied to KS drift detection (`ks=0.06`, `frac=0.20`), not the classification decision boundary. | **PASSED** |
| **4. Metric Reproduction** | Recomputed Macro-F1 and ROC-AUC from `predictions.csv` vs `aggregate_summary.csv` | Categorized with explicit tolerance brackets: `EXACT_MATCH` ($\le 0.0001$), `CLOSE_MATCH` ($\le 0.0020$). Max F1 difference is **0.00138** (Static). | **PASSED** |
| **5. Retraining Event Accounting** | Consistent accounting of initial training vs adaptation retraining | Initial retrain at window 0 across all policies. Subsequent adaptation retrains: Static = 0; Periodic-6 = 3 (w6, w12, w18); Localized DA = 5 (w5, w8, w11, w14, w17); Oracle reference = 13 | **PASSED** |
| **6. Dynamic Detection Delay Audit** | Delay dynamically calculated from raw data without hardcoding | Earliest drifting onset dynamically computed: **Window 2**; First DA detection trigger dynamically computed: **Window 5**; Computed Detection Delay = **3 windows**; Computed Pre-drift False Alarms = **0** | **PASSED** |
| **7. Multi-Model SHAP Attribution** | Must use exact 25 features and genuine checkpoint models | All 25 features accounted for; SHAP values generated from `static_model.pkl` (pre & post) and `da_localized_final_model.pkl` | **PASSED** |

---

## 1. Metric Reproduction Verification

Below is the verification table comparing metrics computed directly from `predictions.csv` against the authoritative multi-seed `aggregate_summary.csv`:

| policy       |   f1_from_predictions |   f1_in_aggregate_summary |   f1_diff |   auc_from_predictions |   auc_in_aggregate_summary |   auc_diff | status      |
|:-------------|----------------------:|--------------------------:|----------:|-----------------------:|---------------------------:|-----------:|:------------|
| static       |                0.8244 |                    0.823  |   0.00138 |                 0.8943 |                     0.8934 |    0.00094 | CLOSE_MATCH |
| periodic_6   |                0.9119 |                    0.912  |   8e-05   |                 0.9657 |                     0.9658 |    0.00014 | EXACT_MATCH |
| da_localized |                0.9174 |                    0.9167 |   0.00073 |                 0.969  |                     0.9691 |    6e-05   | CLOSE_MATCH |

### Numerical Tolerance Brackets
- **EXACT_MATCH**: Absolute difference $\le 0.0001$ (Periodic-6 F1 difference = 0.00008, Localized DA ROC-AUC difference = 0.00006)
- **CLOSE_MATCH**: Absolute difference $\le 0.0020$ (Static F1 difference = 0.00138, Localized DA F1 difference = 0.00073)
- **DEVIATION**: Absolute difference $> 0.0020$ (None observed)

*Interpretation*: The reproduced metrics closely match the reported five-seed means, with absolute F1 differences up to **0.00138** (Static). These differences are expected because `predictions.csv` represents the primary seed trajectory (seed 2025) while `aggregate_summary.csv` reports the pooled five-seed mean (seeds 2025–2029).

---

## 2. Dynamic Calculation of Detection Delay and False Alarms

The detection delay and false-alarm metrics were dynamically verified from raw event files:
- **Earliest Drift Onset**: Computed as $\min(\text{drift\_onset} \ge 0) = 2$ from `dataset_v08/botdriftbench_v08.csv` (correctly excluding stable-control accounts marked with sentinel `drift_onset = -1`).
- **First Detection Trigger**: Computed as $\min(\text{window})$ where $\text{retrained} = 1$ and $\text{reason} = \text{'drift\_detected'} = 5$ from `results_v08/retraining_events.csv`.
- **Detection Delay**: $\text{First Trigger} - \text{Earliest Onset} = 5 - 2 = \mathbf{3\text{ windows}}$. Matches `results_v08/detection_metrics.csv`.
- **Pre-Drift False Alarms**: Sum of retraining events where $0 < \text{window} < 2 = \mathbf{0}$. Matches `results_v08/detection_metrics.csv`.

---

## 3. Prediction Schema & Split Disjointness

```text
predictions.csv Schema Verification
===================================
Rows: 36,000
Columns: ['account_id', 'time_window', 'split', 'label', 'scenario', 'drift_onset', 'policy', 'prediction', 'probability', 'retrained']
Unique accounts: 600
Unique windows: 20
Policies: ['static', 'periodic_6', 'da_localized']
Split values: ['test']
Labels: {0: 18720, 1: 17280}
```

- **Zero Overlap**: Verified zero account leakage between training (1,800), validation (600), and test (600).
- **Label Invariance**: Confirmed constant ground truth per account across all 20 time windows.

---

## 4. Claims Audit & Scientific Framing Adjustments

To maintain strict methodological honesty across the paper, notebook, and demo application:

1. **Comparison with Periodic-6**:
   - *Original overclaim*: "Localized DA outperforms Periodic-6 in 4/5 seeds."
   - *Defensible framing*: **"Localized drift-aware adaptation substantially improves over a frozen static detector (0.9167 vs 0.8230) and performs comparably to Periodic-6 (0.9120) while using targeted retraining."**
2. **Comparison with Oracle Reference**:
   - *Original overclaim*: "Localized DA matches Oracle adaptation with 2.6x fewer retrains."
   - *Defensible framing*: **"Localized DA used 5 adaptation retrains compared with 13 for the oracle reference policy, achieving comparable performance (0.9167 vs 0.9160)."**
3. **SHAP Feature Attribution Interpretation**:
   - *Original overclaim*: "Proves adaptation shifted reliance toward evasion-invariant features."
   - *Defensible framing*: **"SHAP analysis documents feature attribution changes across pre-drift, post-drift, and post-adaptation regimes, providing an interpretable audit trail of model behavior under drift without implying causal invariance."**
