"""
Model Adaptation Experiment (v2)
=================================

Compares three adaptation strategies on BotDriftBench:

  Model A (Static):      Train on pre-drift, never update
  Model B (Periodic):    Retrain every N windows on recent data
  Model C (Drift-Aware): Monitor prediction drift, retrain when triggered

Per-window metrics:
  - F1, AUC, Accuracy
  - Bot-human distribution distance (causal validation)

Per-configuration summary metrics:
  - F1 drop (pre vs post drift)
  - Recovery time: windows from first retrain to F1 >= 0.95 * F1_baseline
  - Recovery magnitude: (F1_adapted - F1_drift) / (F1_baseline - F1_drift)
  - Adaptation efficiency: F1_recovered / n_retrains
  - Detection delay (drift-aware only)
  - False alarm rate (drift-aware only)
  - Number of retrains

Grand summary reports mean +/- std across 10 seeds.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy.stats import ks_2samp
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from generate_dataset import generate_trajectories, ALL_FEATURES

FEATURES = ALL_FEATURES


def train_model(X, y, seed=42):
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=seed, verbosity=0
    )
    model.fit(X, y)
    return model


def evaluate_window(model, w_df):
    X = w_df[FEATURES].values
    y_true = (w_df["label"] == "bot").astype(int).values
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
    }


def compute_class_distance(w_df):
    """Mean KS distance between bot and human feature distributions."""
    bots = w_df[w_df["label"] == "bot"]
    humans = w_df[w_df["label"] == "human"]
    ks_vals = []
    for feat in FEATURES:
        stat, _ = ks_2samp(bots[feat].values, humans[feat].values)
        ks_vals.append(stat)
    return float(np.mean(ks_vals))


def detect_prediction_drift(model, ref_probs, w_df, threshold_ks, threshold_p=0.001):
    X = w_df[FEATURES].values
    w_probs = model.predict_proba(X)[:, 1]
    stat, p = ks_2samp(ref_probs, w_probs)
    return stat > threshold_ks and p < threshold_p, stat, p


def run_static(df, n_windows, tau_start):
    pre = df[df["time_window"] < tau_start]
    model = train_model(pre[FEATURES].values, (pre["label"] == "bot").astype(int).values)

    results = []
    for w in range(n_windows):
        w_df = df[df["time_window"] == w]
        metrics = evaluate_window(model, w_df)
        metrics["window"] = w
        metrics["strategy"] = "static"
        metrics["retrained"] = False
        metrics["class_distance"] = compute_class_distance(w_df)
        results.append(metrics)
    return results


def run_periodic(df, n_windows, tau_start, retrain_interval=3, lookback=5):
    pre = df[df["time_window"] < tau_start]
    model = train_model(pre[FEATURES].values, (pre["label"] == "bot").astype(int).values)

    results = []
    last_retrain = -1
    for w in range(n_windows):
        w_df = df[df["time_window"] == w]

        if w >= tau_start and (w - tau_start) % retrain_interval == 0 and w > tau_start:
            start_w = max(0, w - lookback)
            recent = df[(df["time_window"] >= start_w) & (df["time_window"] < w)]
            model = train_model(recent[FEATURES].values,
                                (recent["label"] == "bot").astype(int).values)
            last_retrain = w

        metrics = evaluate_window(model, w_df)
        metrics["window"] = w
        metrics["strategy"] = "periodic"
        metrics["retrained"] = (w == last_retrain)
        metrics["class_distance"] = compute_class_distance(w_df)
        results.append(metrics)
    return results


def run_drift_aware(df, n_windows, tau_start, cooldown=3, lookback=5):
    pre = df[df["time_window"] < tau_start]
    model = train_model(pre[FEATURES].values, (pre["label"] == "bot").astype(int).values)
    ref_probs = model.predict_proba(pre[FEATURES].values)[:, 1]

    pre_ks_vals = []
    for w in range(1, tau_start):
        w_df = df[df["time_window"] == w]
        w_probs = model.predict_proba(w_df[FEATURES].values)[:, 1]
        stat, _ = ks_2samp(ref_probs, w_probs)
        pre_ks_vals.append(stat)

    ks_threshold = (np.mean(pre_ks_vals) + 2.0 * np.std(pre_ks_vals)) if pre_ks_vals else 0.10

    results = []
    last_retrain = -999
    detection_window = None
    n_retrains = 0
    false_alarms = 0

    for w in range(n_windows):
        w_df = df[df["time_window"] == w]
        triggered = False
        drift_stat = 0.0

        if (w - last_retrain) >= cooldown:
            drift_detected, drift_stat, _ = detect_prediction_drift(
                model, ref_probs, w_df, ks_threshold
            )
            if drift_detected:
                if w < tau_start:
                    false_alarms += 1
                triggered = True
                if detection_window is None and w >= tau_start:
                    detection_window = w

                start_w = max(0, w - lookback)
                recent = df[(df["time_window"] >= start_w) & (df["time_window"] < w)]
                model = train_model(recent[FEATURES].values,
                                    (recent["label"] == "bot").astype(int).values)
                ref_probs = model.predict_proba(recent[FEATURES].values)[:, 1]
                last_retrain = w
                n_retrains += 1

        metrics = evaluate_window(model, w_df)
        metrics["window"] = w
        metrics["strategy"] = "drift_aware"
        metrics["retrained"] = triggered
        metrics["drift_stat"] = round(drift_stat, 4)
        metrics["class_distance"] = compute_class_distance(w_df)
        results.append(metrics)

    return results, detection_window, n_retrains, ks_threshold, false_alarms


def compute_summary(results_by_strategy, tau_start, tau_end):
    summaries = {}
    for strategy, results in results_by_strategy.items():
        pre = [r for r in results if r["window"] < tau_start]
        post = [r for r in results if r["window"] >= tau_end]
        all_post_drift = [r for r in results if r["window"] >= tau_start]

        f1_pre = np.mean([r["f1"] for r in pre])
        f1_post = np.mean([r["f1"] for r in post])
        f1_min = min(r["f1"] for r in all_post_drift) if all_post_drift else f1_pre
        auc_pre = np.mean([r["auc"] for r in pre])
        auc_post = np.mean([r["auc"] for r in post])
        n_retrains = sum(1 for r in results if r.get("retrained", False))

        cd_pre = np.mean([r["class_distance"] for r in pre])
        cd_post = np.mean([r["class_distance"] for r in post])

        # Recovery time: windows from first retrain to F1 >= 0.95 * F1_baseline
        recovery_time = None
        retrain_windows = [r["window"] for r in results if r.get("retrained")]
        if retrain_windows:
            first_retrain = min(retrain_windows)
            for r in results:
                if r["window"] >= first_retrain and r["f1"] >= f1_pre * 0.95:
                    recovery_time = r["window"] - first_retrain
                    break

        # Recovery magnitude: (F1_adapted - F1_drift) / (F1_baseline - F1_drift)
        recovery_magnitude = None
        if f1_pre > f1_min and f1_min < f1_pre:
            recovery_magnitude = (f1_post - f1_min) / (f1_pre - f1_min)
            recovery_magnitude = min(max(recovery_magnitude, 0.0), 1.5)

        # Adaptation efficiency: F1_recovered / n_retrains
        adapt_efficiency = None
        if n_retrains > 0:
            f1_recovered = f1_post - f1_min
            adapt_efficiency = f1_recovered / n_retrains

        summaries[strategy] = {
            "f1_pre": round(f1_pre, 4),
            "f1_post": round(f1_post, 4),
            "f1_drop": round(f1_pre - f1_post, 4),
            "f1_min": round(f1_min, 4),
            "auc_pre": round(auc_pre, 4),
            "auc_post": round(auc_post, 4),
            "class_dist_pre": round(cd_pre, 4),
            "class_dist_post": round(cd_post, 4),
            "n_retrains": n_retrains,
            "recovery_time": recovery_time,
            "recovery_magnitude": round(recovery_magnitude, 4) if recovery_magnitude is not None else None,
            "adapt_efficiency": round(adapt_efficiency, 4) if adapt_efficiency is not None else None,
        }
    return summaries


def run_experiment():
    magnitudes = [0.25, 0.5, 1.0, 2.0]
    seeds = [42, 123, 456, 789, 1001, 1337, 2024, 3141, 4242, 5555]
    n_accounts = 1000
    n_windows = 20
    tau_start = 5
    tau_end = 15

    all_results = []
    all_summaries = []

    for mag in magnitudes:
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"  Magnitude={mag}, Seed={seed}")
            print(f"{'='*70}")

            df, _ = generate_trajectories(
                n_accounts, n_windows, seed, mag,
                tau_start, tau_end, None
            )

            print("  [A] Static model...")
            static_results = run_static(df, n_windows, tau_start)

            print("  [B] Periodic retraining (every 3 windows)...")
            periodic_results = run_periodic(df, n_windows, tau_start,
                                            retrain_interval=3, lookback=5)

            print("  [C] Drift-aware adaptation...")
            da_results, detect_w, n_retrains, ks_thresh, false_alarms = run_drift_aware(
                df, n_windows, tau_start, cooldown=3, lookback=5
            )

            detect_delay = (detect_w - tau_start) if detect_w is not None else None

            strategies = {
                "static": static_results,
                "periodic": periodic_results,
                "drift_aware": da_results,
            }
            summaries = compute_summary(strategies, tau_start, tau_end)

            print(f"\n  Results (mag={mag}, seed={seed}):")
            print(f"  {'Strategy':<14} {'F1_pre':>7} {'F1_post':>8} {'F1_drop':>8} "
                  f"{'F1_min':>7} {'Retr':>5} {'RecT':>5} {'RecM':>6} {'AdEf':>7}")

            for strat, s in summaries.items():
                rec_t = f"{s['recovery_time']}w" if s['recovery_time'] is not None else "N/A"
                rec_m = f"{s['recovery_magnitude']:.2f}" if s['recovery_magnitude'] is not None else "N/A"
                eff = f"{s['adapt_efficiency']:.4f}" if s['adapt_efficiency'] is not None else "N/A"
                print(f"  {strat:<14} {s['f1_pre']:>7.4f} {s['f1_post']:>8.4f} "
                      f"{s['f1_drop']:>8.4f} {s['f1_min']:>7.4f} "
                      f"{s['n_retrains']:>5} {rec_t:>5} {rec_m:>6} {eff:>7}")

            if detect_delay is not None:
                print(f"  Drift-aware: delay={detect_delay}w, "
                      f"false_alarms={false_alarms}, threshold={ks_thresh:.4f}")
            else:
                print(f"  Drift-aware: no detection, "
                      f"false_alarms={false_alarms}, threshold={ks_thresh:.4f}")

            for strat, res_list in strategies.items():
                for r in res_list:
                    all_results.append({
                        "magnitude": mag,
                        "seed": seed,
                        "strategy": strat,
                        "window": r["window"],
                        "f1": round(r["f1"], 4),
                        "auc": round(r["auc"], 4),
                        "acc": round(r["acc"], 4),
                        "class_distance": round(r["class_distance"], 4),
                        "retrained": r.get("retrained", False),
                    })

            for strat, s in summaries.items():
                row = dict(s)
                row["magnitude"] = mag
                row["seed"] = seed
                row["strategy"] = strat
                if strat == "drift_aware":
                    row["detection_delay"] = detect_delay
                    row["false_alarms"] = false_alarms
                all_summaries.append(row)

    # ===== GRAND SUMMARY =====
    print("\n\n" + "=" * 120)
    print("ADAPTATION EXPERIMENT v2 -- GRAND SUMMARY (mean +/- std across 10 seeds)")
    print("=" * 120)

    header = (f"{'Mag':>5}  {'Strategy':<14} {'F1_drop':>12} {'F1_min':>12} "
              f"{'Retrains':>12} {'RecTime':>10} {'RecMag':>12} {'AdaptEff':>14} "
              f"{'ClassDist':>14}")
    print(f"\n{header}")
    print("-" * 120)

    for mag in magnitudes:
        for strat in ["static", "periodic", "drift_aware"]:
            s_rows = [s for s in all_summaries
                      if s["magnitude"] == mag and s["strategy"] == strat]

            f1_drops = [s["f1_drop"] for s in s_rows]
            f1_mins = [s["f1_min"] for s in s_rows]
            retrains = [s["n_retrains"] for s in s_rows]
            rec_times = [s["recovery_time"] for s in s_rows if s["recovery_time"] is not None]
            rec_mags = [s["recovery_magnitude"] for s in s_rows if s["recovery_magnitude"] is not None]
            efficiencies = [s["adapt_efficiency"] for s in s_rows if s["adapt_efficiency"] is not None]
            cd_posts = [s["class_dist_post"] for s in s_rows]

            f1d = f"{np.mean(f1_drops):.4f}+/-{np.std(f1_drops):.4f}"
            f1m = f"{np.mean(f1_mins):.4f}+/-{np.std(f1_mins):.4f}"
            ret = f"{np.mean(retrains):.1f}+/-{np.std(retrains):.1f}"
            rt = f"{np.mean(rec_times):.1f}+/-{np.std(rec_times):.1f}" if rec_times else "N/A"
            rm = f"{np.mean(rec_mags):.2f}+/-{np.std(rec_mags):.2f}" if rec_mags else "N/A"
            ae = f"{np.mean(efficiencies):.4f}+/-{np.std(efficiencies):.4f}" if efficiencies else "N/A"
            cd = f"{np.mean(cd_posts):.4f}+/-{np.std(cd_posts):.4f}"

            print(f"{mag:>5.2f}  {strat:<14} {f1d:>12} {f1m:>12} "
                  f"{ret:>12} {rt:>10} {rm:>12} {ae:>14} {cd:>14}")

        # Drift-aware detection delay for this magnitude
        da_rows = [s for s in all_summaries
                   if s["magnitude"] == mag and s["strategy"] == "drift_aware"]
        delays = [s.get("detection_delay") for s in da_rows if s.get("detection_delay") is not None]
        fa = [s.get("false_alarms", 0) for s in da_rows]
        if delays:
            print(f"       DA detect delay: {np.mean(delays):.1f}+/-{np.std(delays):.1f}w, "
                  f"false alarms: {np.mean(fa):.1f}+/-{np.std(fa):.1f}")
        else:
            print(f"       DA: no detection across seeds")
        print()

    # ===== CAUSAL VALIDATION =====
    print("\n" + "=" * 80)
    print("CAUSAL VALIDATION: Drift magnitude -> Class distance -> AUC -> F1")
    print("=" * 80)

    for mag in magnitudes:
        mag_rows = [r for r in all_results
                    if r["magnitude"] == mag and r["strategy"] == "static"]

        pre_cd, post_cd = [], []
        pre_auc, post_auc = [], []
        pre_f1, post_f1 = [], []
        for seed in seeds:
            seed_rows = [r for r in mag_rows if r["seed"] == seed]
            pre = [r for r in seed_rows if r["window"] < tau_start]
            post = [r for r in seed_rows if r["window"] >= tau_end]
            pre_cd.append(np.mean([r["class_distance"] for r in pre]))
            post_cd.append(np.mean([r["class_distance"] for r in post]))
            pre_auc.append(np.mean([r["auc"] for r in pre]))
            post_auc.append(np.mean([r["auc"] for r in post]))
            pre_f1.append(np.mean([r["f1"] for r in pre]))
            post_f1.append(np.mean([r["f1"] for r in post]))

        cd_change = np.mean(pre_cd) - np.mean(post_cd)
        auc_change = np.mean(pre_auc) - np.mean(post_auc)
        f1_change = np.mean(pre_f1) - np.mean(post_f1)

        print(f"\n  delta={mag:.2f}:")
        print(f"    Class distance:  {np.mean(pre_cd):.4f} -> {np.mean(post_cd):.4f}  "
              f"(change: {cd_change:+.4f})")
        print(f"    AUC:             {np.mean(pre_auc):.4f} -> {np.mean(post_auc):.4f}  "
              f"(change: {auc_change:+.4f})")
        print(f"    F1 (static):     {np.mean(pre_f1):.4f} -> {np.mean(post_f1):.4f}  "
              f"(change: {f1_change:+.4f})")
        if cd_change > 0 and auc_change > 0 and f1_change > 0:
            print(f"    --> COHERENT: all three degrade together")
        else:
            print(f"    --> CHECK: not all metrics degrade in same direction")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("dataset/adaptation_experiment_v2.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("dataset/adaptation_summary_v2.csv", index=False)

    n_configs = len(magnitudes) * len(seeds)
    print(f"\n\nPer-window results: dataset/adaptation_experiment_v2.csv "
          f"({len(all_results)} rows)")
    print(f"Per-config summaries: dataset/adaptation_summary_v2.csv "
          f"({len(all_summaries)} rows)")
    print(f"({len(magnitudes)} magnitudes x {len(seeds)} seeds x 3 strategies "
          f"x {n_windows} windows)")


if __name__ == "__main__":
    run_experiment()
