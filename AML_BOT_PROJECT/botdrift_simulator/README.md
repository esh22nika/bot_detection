# BotDriftBench: Static vs Drift-Aware Bot Detection Simulator

Interactive research simulation and benchmark replay demonstrating longitudinal social bot detection under behavioral evasion drift, comparing a frozen **Static Detector** against the proposed **Localized Drift-Aware Adaptation** policy on the **BotDriftBench v0.8** benchmark (3,000 accounts, 20 temporal windows, 25 features).

---

## Architecture Overview

```
botdrift_simulator/
├── app.py                     # Flask web server & REST API
├── simulator/
│   ├── engine.py              # Stateful simulator engine (step, drift, adapt, reset)
│   ├── data_loader.py         # Loads empirical benchmark CSVs and distributions
│   ├── scenarios.py           # Schedule definitions, descriptions, milestone events
│   └── metrics.py             # Metric calculations & delta advantages
├── data/
│   ├── window_results.csv     # Window-by-window Macro-F1, ROC-AUC, Recall (v0.8)
│   ├── predictions.csv        # 600 test accounts long predictions format
│   ├── retraining_events.csv  # Empirical retraining schedule
│   ├── detection_metrics.csv  # Delay (3 windows), false alarms (0), intervals (3.0)
│   ├── scenario_results.csv   # Scenario-wise policy comparisons
│   ├── aggregate_summary.csv  # Headline multi-metric 5-seed results
│   └── feature_distributions.json # Pre-drift vs post-drift feature distributions
├── templates/
│   └── index.html             # Responsive research demo interface
├── static/
│   ├── css/style.css          # Modern academic UI design
│   └── js/
│       ├── charts.js          # Chart.js temporal trajectory & feature distribution
│       ├── simulator.js       # REST API client & DOM synchronization
│       └── app.js             # Event listeners & initialization
├── requirements.txt
└── README.md
```

---

## How to Run

1. Navigate to the simulator directory:
   ```bash
   cd botdrift_simulator
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser to:
   ```
   http://localhost:5050
   ```

---

## Demonstration Workflow

1. **Step 1 — Stable Pre-Drift Environment (`Before Drift`)**:
   - Both detectors operate on pre-drift distributions (Window 0–1).
   - High discrimination is observed for both (**Macro-F1 ~0.975–0.979**, **ROC-AUC ~0.997**).
   - Status: Both `STABLE`. Retraining count: `0`.

2. **Step 2 — Inject Behavioral Drift (`Inject Drift`)**:
   - Drift begins at Window 2 (bot evasion strategies commence).
   - The behavioral feature distribution shifts (e.g. bots lower `content_similarity` and randomize `posting_regularity`).
   - Static detector begins silent degradation without warning.
   - Proposed Drift-Aware detector begins monitoring via localized Kolmogorov-Smirnov tests.

3. **Step 3 — Trigger Adaptive Response (`Run Adaptation` or `Step`)**:
   - At Window 5, the localized KS detector fires (detection delay = 3 windows, 0 pre-drift false alarms).
   - Recency-weighted retraining is executed.
   - Drift-Aware detector recovers performance (**Macro-F1 ~0.931 vs Static 0.927**).

4. **Step 4 — Post-Adaptation Longitudinal Divergence (`Step`)**:
   - Stepping through subsequent windows (6 to 19) demonstrates ongoing divergence.
   - The Static baseline degrades down to **Macro-F1 = 0.6583** at Window 18 (bot recall collapses to 37.9%).
   - The Proposed Drift-Aware detector maintains **Macro-F1 ~0.88–0.92** via targeted adaptations at windows 8, 11, 14, and 17.

---

## REST API Endpoints

- `GET /`: Serves the HTML frontend.
- `GET /api/initial-data`: Returns benchmark overview, all window trajectories, and feature distributions.
- `GET /api/state`: Returns current state (window, mode, metrics, statuses, explanation, event log).
- `POST /api/before-drift`: Resets simulation to stable pre-drift state (Window 0).
- `POST /api/inject-drift`: Activates drift scenario and moves to drift onset (Window 2).
- `POST /api/run-adaptation`: Advances to first adaptation trigger (Window 5).
- `POST /api/step`: Advances simulation by exactly one window.
- `POST /api/reset`: Resets simulation to initial state.
- `GET /api/comparison`: Returns trajectory comparison series and current cursor.
- `GET /api/events`: Returns current timeline events log.
- `GET /api/features/<feature_name>`: Returns binned distribution for selected feature.
