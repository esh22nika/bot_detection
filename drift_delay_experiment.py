"""
Drift Detection Delay Experiment
=================================

For each (drift_type x magnitude x seed), measures:
  - Feature-space drift detection delay (KS-test on raw features)
  - Prediction-space drift detection delay (KS-test on P(bot))
  - SHAP-space drift detection delay (KS-test on SHAP values)
  - SHAP drift attribution precision/recall

Research question: Which observable signal provides the earliest
indication that a bot detector is becoming obsolete?
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from generate_dataset import generate_trajectories, ALL_FEATURES, DRIFT_SCENARIOS

FEATURES = ALL_FEATURES
BINARY_FEATURES = {"default_profile", "default_profile_image", "has_url", "verified"}

KNOWN_DRIFT_FEATS_BY_SCENARIO = {}
for name, sdef in DRIFT_SCENARIOS.items():
    if sdef["drift_mode"] != "none":
        KNOWN_DRIFT_FEATS_BY_SCENARIO[name] = set(sdef["affected_features"])

ALL_KNOWN_DRIFT_FEATS = set()
for feats in KNOWN_DRIFT_FEATS_BY_SCENARIO.values():
    ALL_KNOWN_DRIFT_FEATS |= feats

KS_THRESHOLD = 0.001


def train_baseline(df, tau_start):
    pre = df[df["time_window"] < tau_start]
    X = pre[FEATURES].values
    y = (pre["label"] == "bot").astype(int).values
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0
    )
    model.fit(X, y)
    return model


def _calibrated_detect(signals_by_window, tau_start, direction="increase"):
    """Detect first window where signal exceeds pre-drift range.

    Uses mean + 2*std of pre-drift windows as threshold (for direction="increase")
    or mean - 2*std (for direction="decrease").
    """
    pre_vals = [s for w, s in sorted(signals_by_window.items()) if w < tau_start and w > 0]
    if len(pre_vals) < 2:
        return None

    mu = np.mean(pre_vals)
    sigma = np.std(pre_vals)
    if sigma < 1e-10:
        sigma = abs(mu) * 0.05 + 1e-6

    if direction == "increase":
        threshold = mu + 2.5 * sigma
        for w in sorted(signals_by_window.keys()):
            if w >= tau_start and signals_by_window[w] > threshold:
                return w
    else:
        threshold = mu - 2.5 * sigma
        for w in sorted(signals_by_window.keys()):
            if w >= tau_start and signals_by_window[w] < threshold:
                return w
    return None


def detect_feature_drift(df, tau_start, tau_end):
    """Feature-space drift: track mean KS between bot/human distributions per window."""
    tracked = [f for f in FEATURES if f not in BINARY_FEATURES]
    signals = {}

    for w in range(df["time_window"].max() + 1):
        w_df = df[df["time_window"] == w]
        humans = w_df[w_df["label"] == "human"]
        bots = w_df[w_df["label"] == "bot"]
        ks_vals = []
        for feat in tracked:
            stat, _ = ks_2samp(humans[feat].values, bots[feat].values)
            ks_vals.append(stat)
        signals[w] = np.mean(ks_vals)

    detection_w = _calibrated_detect(signals, tau_start, direction="decrease")
    return (detection_w - tau_start) if detection_w is not None else None


def detect_prediction_drift(df, model, tau_start):
    """Prediction drift: track model confidence gap between bot/human per window."""
    signals = {}

    for w in range(df["time_window"].max() + 1):
        w_df = df[df["time_window"] == w]
        probs = model.predict_proba(w_df[FEATURES].values)[:, 1]
        y = (w_df["label"] == "bot").values
        bot_probs = probs[y]
        human_probs = probs[~y]
        stat, _ = ks_2samp(bot_probs, human_probs)
        signals[w] = stat

    detection_w = _calibrated_detect(signals, tau_start, direction="decrease")
    return (detection_w - tau_start) if detection_w is not None else None


def detect_shap_drift(df, model, tau_start):
    """SHAP drift: track mean |SHAP| divergence between bot/human per window."""
    explainer = shap.TreeExplainer(model)
    signals = {}

    for w in range(df["time_window"].max() + 1):
        w_df = df[df["time_window"] == w]
        shap_vals = explainer.shap_values(w_df[FEATURES].values)
        y = (w_df["label"] == "bot").values

        bot_shap = np.abs(shap_vals[y]).mean(axis=0)
        human_shap = np.abs(shap_vals[~y]).mean(axis=0)
        divergence = np.mean(np.abs(bot_shap - human_shap))
        signals[w] = divergence

    detection_w = _calibrated_detect(signals, tau_start, direction="decrease")
    return (detection_w - tau_start) if detection_w is not None else None


def compute_shap_attribution(df, model, tau_start):
    explainer = shap.TreeExplainer(model)

    pre_windows = df[df["time_window"] < tau_start]
    post_windows = df[df["time_window"] >= 15]

    pre_sample = pre_windows.sample(min(500, len(pre_windows)), random_state=42)
    post_sample = post_windows.sample(min(500, len(post_windows)), random_state=42)

    pre_shap = explainer.shap_values(pre_sample[FEATURES].values)
    post_shap = explainer.shap_values(post_sample[FEATURES].values)

    pre_mean = np.abs(pre_shap).mean(axis=0)
    post_mean = np.abs(post_shap).mean(axis=0)
    change = np.abs(post_mean - pre_mean)

    top_k = 10
    top_indices = np.argsort(change)[-top_k:][::-1]
    top_feats = {FEATURES[i] for i in top_indices}

    tp = len(top_feats & ALL_KNOWN_DRIFT_FEATS)
    fp = len(top_feats - ALL_KNOWN_DRIFT_FEATS)
    fn = len(ALL_KNOWN_DRIFT_FEATS - top_feats)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def measure_f1_degradation(df, model, tau_start):
    from sklearn.metrics import f1_score

    pre = df[df["time_window"] < tau_start]
    y_pre = (pre["label"] == "bot").astype(int)
    pred_pre = model.predict(pre[FEATURES].values)
    f1_pre = f1_score(y_pre, pred_pre)

    post = df[df["time_window"] >= 15]
    y_post = (post["label"] == "bot").astype(int)
    pred_post = model.predict(post[FEATURES].values)
    f1_post = f1_score(y_post, pred_post)

    return f1_pre, f1_post


def run_experiment():
    magnitudes = [0.25, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456]
    n_accounts = 1000
    n_windows = 20
    tau_start = 5
    tau_end = 15
    campaign_windows = None

    results = []

    for mag in magnitudes:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  Magnitude={mag}, Seed={seed}")
            print(f"{'='*60}")

            df, scenarios = generate_trajectories(
                n_accounts, n_windows, seed, mag,
                tau_start, tau_end, campaign_windows
            )

            model = train_baseline(df, tau_start)

            print("  Detecting feature drift...")
            feat_delay = detect_feature_drift(df, tau_start, tau_end)

            print("  Detecting prediction drift...")
            pred_delay = detect_prediction_drift(df, model, tau_start)

            print("  Detecting SHAP drift...")
            shap_delay = detect_shap_drift(df, model, tau_start)

            print("  Computing SHAP attribution...")
            prec, rec = compute_shap_attribution(df, model, tau_start)

            print("  Measuring F1 degradation...")
            f1_pre, f1_post = measure_f1_degradation(df, model, tau_start)

            row = {
                "magnitude": mag,
                "seed": seed,
                "feature_delay": feat_delay,
                "prediction_delay": pred_delay,
                "shap_delay": shap_delay,
                "shap_precision": round(prec, 3),
                "shap_recall": round(rec, 3),
                "f1_pre": round(f1_pre, 4),
                "f1_post": round(f1_post, 4),
                "f1_drop": round(f1_pre - f1_post, 4),
            }
            results.append(row)

            d = lambda v: f"{v:2d}" if v is not None else " --"
            print(f"  Feature delay: {d(feat_delay)}  "
                  f"Prediction delay: {d(pred_delay)}  "
                  f"SHAP delay: {d(shap_delay)}")
            print(f"  SHAP Prec={prec:.3f}  Rec={rec:.3f}")
            print(f"  F1: {f1_pre:.4f} -> {f1_post:.4f} (drop {f1_pre - f1_post:.4f})")

    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 80)
    print("DRIFT DETECTION DELAY EXPERIMENT — RESULTS")
    print("=" * 80)

    # Summary table: mean across seeds
    print("\nMean detection delay (windows after onset) by magnitude:")
    print(f"{'Magnitude':>10}  {'Feature':>8}  {'Predict':>8}  {'SHAP':>8}  "
          f"{'F1_pre':>7}  {'F1_post':>8}  {'F1_drop':>8}  "
          f"{'SHAP_P':>7}  {'SHAP_R':>7}")
    print("-" * 90)

    for mag in magnitudes:
        mag_rows = [r for r in results if r["magnitude"] == mag]

        feat_delays = [r["feature_delay"] for r in mag_rows if r["feature_delay"] is not None]
        pred_delays = [r["prediction_delay"] for r in mag_rows if r["prediction_delay"] is not None]
        shap_delays = [r["shap_delay"] for r in mag_rows if r["shap_delay"] is not None]

        feat_mean = f"{np.mean(feat_delays):.1f}" if feat_delays else "  N/D"
        pred_mean = f"{np.mean(pred_delays):.1f}" if pred_delays else "  N/D"
        shap_mean = f"{np.mean(shap_delays):.1f}" if shap_delays else "  N/D"

        f1_pre_mean = np.mean([r["f1_pre"] for r in mag_rows])
        f1_post_mean = np.mean([r["f1_post"] for r in mag_rows])
        f1_drop_mean = np.mean([r["f1_drop"] for r in mag_rows])
        prec_mean = np.mean([r["shap_precision"] for r in mag_rows])
        rec_mean = np.mean([r["shap_recall"] for r in mag_rows])

        print(f"{mag:>10.2f}  {feat_mean:>8}  {pred_mean:>8}  {shap_mean:>8}  "
              f"{f1_pre_mean:>7.4f}  {f1_post_mean:>8.4f}  {f1_drop_mean:>8.4f}  "
              f"{prec_mean:>7.3f}  {rec_mean:>7.3f}")

    # Per-seed detail
    print("\n\nPer-seed detail:")
    print(f"{'Mag':>5}  {'Seed':>5}  {'Feat':>5}  {'Pred':>5}  {'SHAP':>5}  "
          f"{'F1pre':>7}  {'F1post':>7}  {'Drop':>7}  {'P_SHAP':>7}  {'R_SHAP':>7}")
    print("-" * 80)

    for r in results:
        fd = f"{r['feature_delay']:5d}" if r['feature_delay'] is not None else "  N/D"
        pd_ = f"{r['prediction_delay']:5d}" if r['prediction_delay'] is not None else "  N/D"
        sd = f"{r['shap_delay']:5d}" if r['shap_delay'] is not None else "  N/D"
        print(f"{r['magnitude']:>5.2f}  {r['seed']:>5d}  {fd}  {pd_}  {sd}  "
              f"{r['f1_pre']:>7.4f}  {r['f1_post']:>7.4f}  {r['f1_drop']:>7.4f}  "
              f"{r['shap_precision']:>7.3f}  {r['shap_recall']:>7.3f}")

    results_df.to_csv("dataset/drift_delay_experiment.csv", index=False)
    print(f"\nResults saved to dataset/drift_delay_experiment.csv")

    # Save as JSON for programmatic use
    with open("dataset/drift_delay_experiment.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_experiment()
