# System Overview: Drift-Aware Explainable Social Bot Detection

## Problem Statement

Social bot detection models degrade over time as bot operators adapt their
behavior. Traditional approaches train a static classifier and assume the
feature distribution remains stable. When bot behavior drifts, the model's
accuracy declines silently — there is no mechanism to detect, explain, or
respond to the change.

## Proposed System

This project presents a drift-aware bot detection pipeline that:

1. **Detects** behavioral drift using per-feature KS-statistic monitoring
2. **Explains** which features changed using SHAP-based attribution
3. **Adapts** by retraining the classifier only when drift is detected

The system is evaluated on BotDriftBench v0.3, an empirically calibrated
synthetic longitudinal benchmark with controlled drift scenarios.

## Architecture

```
Input: Account features (27 dimensions)
         |
         v
+-------------------+
| XGBoost Classifier|  --> Bot probability + SHAP explanation
+-------------------+
         |
         v
+-------------------+
| KS Drift Monitor  |  --> Per-feature drift scores
+-------------------+
         |
    drift detected?
    /           \
   No            Yes
   |              |
   v              v
 Continue    +-------------------+
             | SHAP Attribution  |  --> Which features drifted
             +-------------------+
                      |
                      v
             +-------------------+
             | Retrain Classifier|  --> Updated model
             +-------------------+
```

## Components

### 1. Baseline Bot Detector

- **Model**: XGBoost (200 trees, max_depth=5, learning_rate=0.1)
- **Features**: 27 account-level features (11 profile + 16 behavioral)
- **Training**: Account-disjoint split (700/300 accounts) on pre-drift data
- **Performance**: F1 = 0.8115, AUC = 0.8946 (seed 42, account-disjoint)

### 2. Drift Detection Module

- **Method**: Two-sample Kolmogorov-Smirnov test per feature
- **Reference**: Bot feature distributions from pre-drift windows
- **Threshold**: KS > 0.15, at least 3 features above threshold
- **Detection delay**: 2-3 temporal windows after drift onset
- **False alarm rate**: 0% in pre-drift period

### 3. SHAP Explanation Engine

- **Method**: TreeExplainer (exact SHAP values for tree models)
- **Use 1**: Feature importance ranking for baseline model
- **Use 2**: Drift attribution — comparing SHAP distributions before/after
  drift to identify which features' contributions changed most
- **Limitation**: SHAP identifies feature contributions, not causal mechanisms

### 4. Adaptation Controller

Three strategies compared:

| Strategy | Trigger | Avg Abs F1 Drop | Retraining Events |
|---|---|---|---|
| Static | Never | 0.054 | 0.0 |
| Periodic | Every 4 windows | 0.008 | 4.0 |
| Drift-Aware | On KS detection | 0.012 | 12.6 |

Both periodic and drift-aware substantially outperform static. Periodic
achieves smaller average F1 degradation with fewer events. The value of
drift-aware is interpretable, targeted response — each retraining is
triggered by a detected shift with SHAP attribution.

### 5. Streamlit Frontend

Interactive demonstration app (`app.py`) providing:

- Single-account bot probability prediction with SHAP waterfall
- Scenario-level degradation visualization
- Drift detection score display
- Adaptation strategy comparison
- Feature importance overview

The frontend is a simulation using pre-computed model artifacts, not a
live Twitter monitoring tool.

## Data Flow

```
BotDriftBench v0.3 CSV (20,000 rows)
  |
  +--> Pre-drift (windows 0-4): train/test split --> baseline model
  |
  +--> All windows (0-19): per-scenario evaluation
  |      |
  |      +--> Degradation curves (Experiment B)
  |      +--> KS drift scores (Experiment C)
  |      +--> SHAP attribution (Experiment D)
  |      +--> Adaptation comparison (Experiment E)
  |
  +--> Model artifacts (artifacts.pkl) --> Streamlit frontend
```

## File Structure

```
bot_detection/
  dataset_v03/
    botdriftbench_v0.3.csv        # Frozen benchmark dataset
    MANIFEST.md                    # Dataset specification
  final_v03/
    final_metrics.json             # All experiment results
    final_*.csv                    # Per-experiment data
    final_*.png                    # Visualizations
    MODEL_CARD.md                  # Model documentation
    FINAL_REPRODUCIBILITY_REPORT.md
  model_artifacts/
    artifacts.pkl                  # Serialized model + results
  project_docs/
    SYSTEM_OVERVIEW.md             # This file
    DEMO_SCRIPT.md                 # Presentation guide
    MODEL_PAPER_RESULTS.md         # Paper-ready results
  app.py                           # Streamlit frontend
  run_final_v03_corrected.py        # Reproducible experiment script (account-disjoint)
  Drift_Aware_Bot_Detection_v03.ipynb  # Colab notebook
```

## Reproducibility

All experiments are reproducible from the frozen CSV:

```bash
python run_final_v03_corrected.py
```

No TwiBot-22 or Cresci-2017 raw files are needed. The dataset generation
pipeline (`generate_dataset_v03.py`) uses pre-computed distribution
parameters stored in `outputs/v03_fitted_distributions.json`.

## Disclaimers

- BotDriftBench v0.3 is synthetic data with empirically calibrated
  distributions, not real longitudinal Twitter/X data
- The Streamlit frontend is an interactive simulation, not live monitoring
- SHAP attribution is explanatory, not causal
- Results are not claimed as state-of-the-art or production-ready
- Drift-aware adaptation is evaluated for targeted drift response,
  not claimed as universally superior to periodic retraining
