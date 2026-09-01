"""
Drift Detector Threshold Sensitivity Analysis
================================================

Evaluates the drift-aware detector across multiple threshold multipliers
to produce its operating characteristic curve.

For each threshold in {0.5x, 1.0x, 1.5x, 2.0x, 2.5x, 3.0x} of the
pre-drift noise floor:
  - Detection rate: fraction of drifting configurations where drift is detected
  - False alarm rate: fraction of pre-drift windows with false alarms
  - Detection delay: mean windows from tau_start to first detection
  - Number of retrains

Output:
  - dataset/threshold_sensitivity.csv
  - Printed operating characteristic table
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from generate_dataset import generate_trajectories, ALL_FEATURES

FEATURES = ALL_FEATURES


def train_model(X, y, seed=42):
    return XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=seed, verbosity=0
    ).fit(X, y)


def run_with_threshold(df, n_windows, tau_start, threshold_mult, cooldown=3, lookback=5):
    """Run drift-aware detection with a specific threshold multiplier."""
    pre = df[df["time_window"] < tau_start]
    model = train_model(pre[FEATURES].values, (pre["label"] == "bot").astype(int).values)
    ref_probs = model.predict_proba(pre[FEATURES].values)[:, 1]

    # Calibrate noise floor
    pre_ks_vals = []
    for w in range(1, tau_start):
        w_df = df[df["time_window"] == w]
        w_probs = model.predict_proba(w_df[FEATURES].values)[:, 1]
        stat, _ = ks_2samp(ref_probs, w_probs)
        pre_ks_vals.append(stat)

    noise_mean = np.mean(pre_ks_vals)
    noise_std = np.std(pre_ks_vals)
    ks_threshold = noise_mean + threshold_mult * noise_std

    false_alarm_windows = 0
    pre_drift_checks = 0
    detection_window = None
    n_retrains = 0
    last_retrain = -999

    f1_scores = []

    for w in range(n_windows):
        w_df = df[df["time_window"] == w]
        X_w = w_df[FEATURES].values
        y_true = (w_df["label"] == "bot").astype(int).values
        y_pred = model.predict(X_w)
        f1_scores.append(f1_score(y_true, y_pred))

        if (w - last_retrain) >= cooldown:
            w_probs = model.predict_proba(X_w)[:, 1]
            stat, p = ks_2samp(ref_probs, w_probs)
            drift_detected = stat > ks_threshold and p < 0.001

            if w < tau_start:
                pre_drift_checks += 1
                if drift_detected:
                    false_alarm_windows += 1

            if drift_detected:
                if detection_window is None and w >= tau_start:
                    detection_window = w

                start_w = max(0, w - lookback)
                recent = df[(df["time_window"] >= start_w) & (df["time_window"] < w)]
                y_recent = (recent["label"] == "bot").astype(int).values
                if len(set(y_recent)) < 2:
                    continue
                model = train_model(recent[FEATURES].values, y_recent)
                ref_probs = model.predict_proba(recent[FEATURES].values)[:, 1]
                last_retrain = w
                n_retrains += 1

    f1_pre = np.mean(f1_scores[:tau_start])
    f1_post = np.mean(f1_scores[15:])
    detection_delay = (detection_window - tau_start) if detection_window is not None else None
    false_alarm_rate = false_alarm_windows / max(pre_drift_checks, 1)

    return {
        "detected": detection_window is not None,
        "detection_delay": detection_delay,
        "false_alarm_rate": false_alarm_rate,
        "false_alarm_count": false_alarm_windows,
        "n_retrains": n_retrains,
        "f1_pre": f1_pre,
        "f1_post": f1_post,
        "f1_drop": f1_pre - f1_post,
        "ks_threshold": ks_threshold,
    }


def run_experiment():
    magnitudes = [0.25, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1001, 1337, 2024, 3141, 4242, 5555]
    threshold_mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_accounts = 1000
    n_windows = 20
    tau_start = 5
    tau_end = 15

    all_results = []

    for mag in magnitudes:
        for seed in seeds:
            print(f"  Generating mag={mag}, seed={seed}...")
            df, _ = generate_trajectories(
                n_accounts, n_windows, seed, mag,
                tau_start, tau_end, None
            )

            for mult in threshold_mults:
                result = run_with_threshold(df, n_windows, tau_start, mult)
                result["magnitude"] = mag
                result["seed"] = seed
                result["threshold_mult"] = mult
                all_results.append(result)

    # Grand summary
    print("\n" + "=" * 110)
    print("THRESHOLD SENSITIVITY -- OPERATING CHARACTERISTIC (mean +/- std across 10 seeds)")
    print("=" * 110)

    print(f"\n{'Mag':>5} {'Mult':>6} {'DetRate':>8} {'FAR':>12} {'Delay':>12} "
          f"{'Retrains':>12} {'F1_drop':>14}")
    print("-" * 110)

    for mag in magnitudes:
        for mult in threshold_mults:
            rows = [r for r in all_results
                    if r["magnitude"] == mag and r["threshold_mult"] == mult]

            det_rate = np.mean([r["detected"] for r in rows])
            fars = [r["false_alarm_rate"] for r in rows]
            delays = [r["detection_delay"] for r in rows if r["detection_delay"] is not None]
            retrains = [r["n_retrains"] for r in rows]
            f1_drops = [r["f1_drop"] for r in rows]

            far_str = f"{np.mean(fars):.3f}+/-{np.std(fars):.3f}"
            delay_str = f"{np.mean(delays):.1f}+/-{np.std(delays):.1f}" if delays else "N/D"
            ret_str = f"{np.mean(retrains):.1f}+/-{np.std(retrains):.1f}"
            f1_str = f"{np.mean(f1_drops):.4f}+/-{np.std(f1_drops):.4f}"

            print(f"{mag:>5.2f} {mult:>6.1f}x {det_rate:>8.0%} {far_str:>12} "
                  f"{delay_str:>12} {ret_str:>12} {f1_str:>14}")
        print()

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("dataset/threshold_sensitivity.csv", index=False)
    print(f"Results saved to dataset/threshold_sensitivity.csv ({len(all_results)} rows)")


if __name__ == "__main__":
    run_experiment()
