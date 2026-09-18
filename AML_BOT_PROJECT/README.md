# AML Bot Detection: Drift-Aware Adaptation Under Behavioral Evolution
**BotDriftBench v0.8 Replay Simulator & Replication Package**

This standalone package contains everything required to interactively run the **Static vs. Drift-Aware Bot Detection Simulator**, explore the complete **BotDriftBench v0.8 dataset**, inspect the **actual trained XGBoost models**, reproduce benchmark evaluation results, and compile the **IEEE research manuscripts**.

---

## 🚀 Quickstart: Running the Interactive Simulator

### 1. Install Dependencies
Ensure you have Python 3.9+ installed. In your terminal / command prompt, run:
```bash
pip install -r requirements.txt
```

### 2. Launch the Simulator
* **Windows (One-Click):** Double-click `run_simulation.bat`
* **Command Line (Windows / Linux / macOS):**
  ```bash
  python app.py
  ```

### 3. Open in Browser
Navigate to:
```
http://127.0.0.1:5050
```
*(The browser will open automatically if you run `run_simulation.bat`).*

---

## 🎮 Simulator Overview & Controls

The simulator interactively demonstrates the core research finding:
> **A static detector performs well before evasion drift, but its detection capability degrades after drift. A drift-aware detector identifies localized feature shifts and adapts dynamically, sustaining high performance throughout the account lifecycle.**

### Controls:
1. **Reset (Window 0):** Resets timeline to baseline window 0.
2. **Before Drift (W0 - W4):** Plays the pre-drift period where both Static and Drift-Aware detectors achieve near-identical baseline performance (~0.93 F1).
3. **Trigger Drift & Adapt (W5 - W19):** Steps through the evasion drift phase. Observe:
   - **Static Detector:** F1 drops from 0.93 down to ~0.76.
   - **Localized Drift-Aware Adaptation:** Detects distribution divergence, triggers adaptation at windows 5, 8, 11, 14, and 17, and recovers performance back to ~0.84 F1.
4. **Auto-Play / Pause:** Continuous benchmark replay across all 20 time windows with configurable playback speed (0.5s - 3s).
5. **Feature Distribution Shift Visualizer:** Select any behavioral feature (e.g. `content_similarity`, `tweet_frequency`, `url_ratio`) to view pre-drift vs. post-drift distribution shifts across Humans, Static Bots, and Drifting Bots.
6. **Live Account Inspector:** Click any account or filter by scenario/label to view true labels, predicted probabilities, and individual feature vectors at any temporal window.

---

## 📁 Repository Structure

```
AML_BOT_PROJECT/
├── app.py                             # Flask application entry point
├── run_simulation.bat                 # 1-click Windows launcher
├── run_simulation.sh                  # 1-click Linux/macOS launcher
├── requirements.txt                   # Dependency specifications
├── test_simulator.py                  # 10 automated unit tests (Run: python test_simulator.py)
│
├── dataset_v08/                       # Complete BotDriftBench v0.8 Dataset
│   ├── botdriftbench_v08.csv          # 60,000 observations (3,000 accounts x 20 windows)
│   ├── account_metadata_v08.csv       # Disjoint splits (1,800 train, 600 val, 600 test)
│   ├── window_metadata_v08.csv        # Temporal window metadata & drift milestones
│   └── generation_manifest_v08.json   # Reproducibility seeds and parameters
│
├── final_models/                      # Authoritative Trained Models
│   ├── static_model.pkl               # Static baseline model (trained on W0)
│   ├── da_localized_model_w*.pkl      # Checkpoints at adaptation windows 5, 8, 11, 14, 17
│   ├── da_localized_final_model.pkl   # Final drift-adapted model
│   ├── periodic6_*.pkl                # Periodic retrained models (for benchmark comparison)
│   └── model_manifest.json            # Model parameters, metadata & SHA256 hashes
│
├── results_v08/                       # Authoritative Evaluation Results
│   ├── predictions.csv                # 36,000 predictions (600 test accounts x 20 windows x 3 policies)
│   ├── aggregate_summary.csv          # Aggregate metrics across seeds
│   ├── window_metrics.csv             # Per-window metrics
│   ├── scenario_results.csv           # Scenario-by-scenario degradation
│   ├── shap_values.parquet            # SHAP feature importance across windows
│   └── shap_rankings.csv              # Feature importance rank shifts
│
├── paper_pictures/                    # 11 Publication Figures (300 DPI)
│   ├── fig1_benchmark_architecture.png
│   ├── fig2_feature_separability.png
│   ├── fig3_temporal_drift_profiles.png
│   ├── fig4_scenario_degradation_heatmap.png
│   ├── fig5_policy_f1_trajectories.png
│   ├── fig6_scenario_policy_breakdown.png
│   ├── fig7_detection_timeline.png
│   ├── fig8_shap_rank_evolution.png
│   └── fig9_fig11_shap_beeswarm_*.png
│
├── simulator/                         # Simulator Backend Engine
│   ├── data_loader.py                 # Grounded data reader
│   ├── engine.py                      # Simulation state machine
│   ├── metrics.py                     # Metric calculation & aggregation
│   └── scenarios.py                   # Evasion scenario definitions
│
├── static/                            # Simulator Frontend Assets
│   ├── css/style.css                  # Responsive UI stylesheet
│   └── js/                            # Chart.js dashboards & inspectors
│
├── templates/
│   └── index.html                     # Simulator web interface
│
├── run_v08_comparison.py              # Full policy evaluation pipeline script
├── generate_dataset_v08.py            # Complete dataset generation script
├── generate_paper_pictures.py         # Figure generation script
├── generate_shap_v08.py               # SHAP values computation script
├── diagnose_v08_degradation.py        # Degradation audit tool
├── Drift_Aware_Bot_Detection_v08.ipynb# Complete self-contained Colab / Jupyter notebook
│
├── model_paper_ieee.tex               # IEEEtran LaTeX manuscript (Model & Policy Paper)
├── dataset_paper_ieee.tex             # IEEEtran LaTeX manuscript (Benchmark & Dataset Paper)
├── references.bib                     # Unified BibTeX citations
├── MODEL_PAPER_DRAFT.md               # Markdown version of Model Paper
├── DATASET_PAPER_DRAFT.md             # Markdown version of Dataset Paper
└── audit_report.md                    # Methodology audit & validation report
```

---

## 🔬 Reproducing Benchmark Results

To re-run the complete benchmark evaluation across all seeds on the frozen v0.8 dataset:
```bash
python run_v08_comparison.py --data dataset_v08 --seed 2025 --seeds 5
```

To re-generate all 11 publication figures:
```bash
python generate_paper_pictures.py
```

To run simulator test suite:
```bash
python test_simulator.py
```

---

## 📄 IEEE LaTeX Paper Compilation

The manuscripts are formatted for standard `IEEEtran`:
* Model Paper: `model_paper_ieee.tex`
* Dataset Paper: `dataset_paper_ieee.tex`
* Bibliography: `references.bib`

Compile using `pdflatex` or `latexmk`:
```bash
pdflatex model_paper_ieee.tex
bibtex model_paper_ieee
pdflatex model_paper_ieee.tex
pdflatex model_paper_ieee.tex
```
