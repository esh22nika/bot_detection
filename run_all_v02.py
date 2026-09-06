"""
BotDriftBench v0.2 — Complete Experiment Pipeline
==================================================
Runs all experiments on the v0.2 calibrated dataset, saves every result
to CSV, and generates matplotlib figures for the final report.

Output directory: results_v02/
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from generate_dataset import generate_trajectories, ALL_FEATURES

FEATURES = ALL_FEATURES
OUT = "results_v02"
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1001, 1337, 2024, 3141, 4242, 5555]
MAGNITUDES = [0.25, 0.5, 1.0, 2.0]
THRESHOLD_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.alpha": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
COLORS = {"static": "#E74C3C", "periodic": "#3498DB", "drift_aware": "#2ECC71"}


def train_model(X, y, seed=42):
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      eval_metric="logloss", random_state=seed, verbosity=0)
    m.fit(X, y)
    return m


def gen_data(seed=42, magnitude=1.0, n_accounts=1000, n_windows=20):
    df, scenarios = generate_trajectories(
        n_accounts=n_accounts, n_windows=n_windows, seed=seed,
        drift_magnitude=magnitude, tau_start=5, tau_end=15,
        campaign_windows=set(),
    )
    df["label_binary"] = (df["label"] == "bot").astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Baseline Validation (Q3-Q6)
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: BASELINE VALIDATION ON v0.2")
    print("=" * 65)

    df = pd.read_csv("dataset/botdriftbench_v0.2.csv",
                      dtype={"drifted_features": str}, low_memory=False)
    df["label_binary"] = (df["label"] == "bot").astype(int)
    tau = 5

    # Q3: Baseline detector
    pre = df[df["time_window"] < tau]
    X = pre[FEATURES].values
    y = pre["label_binary"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    baseline = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    print(f"  Baseline F1={baseline['f1']:.4f}  AUC={baseline['auc']:.4f}")

    # Q4: Per-window degradation
    window_rows = []
    for w in sorted(df["time_window"].unique()):
        wdf = df[df["time_window"] == w]
        Xw = wdf[FEATURES].values
        yw = wdf["label_binary"].values
        yp = model.predict(Xw)
        ypr = model.predict_proba(Xw)[:, 1]
        window_rows.append({
            "window": w,
            "f1": f1_score(yw, yp),
            "auc": roc_auc_score(yw, ypr),
            "accuracy": accuracy_score(yw, yp),
            "n_drift": int(wdf["drift_active"].sum()),
        })
    win_df = pd.DataFrame(window_rows)

    pre_f1 = win_df[win_df["window"] < tau]["f1"].mean()
    post_f1 = win_df[win_df["window"] >= 15]["f1"].mean()
    pre_auc = win_df[win_df["window"] < tau]["auc"].mean()
    post_auc = win_df[win_df["window"] >= 15]["auc"].mean()
    baseline["pre_f1"] = pre_f1
    baseline["post_f1"] = post_f1
    baseline["f1_drop"] = pre_f1 - post_f1
    baseline["pre_auc"] = pre_auc
    baseline["post_auc"] = post_auc
    baseline["auc_drop"] = pre_auc - post_auc
    print(f"  Pre-drift F1={pre_f1:.4f}  Post-drift F1={post_f1:.4f}  Drop={pre_f1 - post_f1:.4f}")
    print(f"  Pre-drift AUC={pre_auc:.4f}  Post-drift AUC={post_auc:.4f}  Drop={pre_auc - post_auc:.4f}")

    pd.DataFrame([baseline]).to_csv(f"{OUT}/baseline_metrics.csv", index=False)
    win_df.to_csv(f"{OUT}/window_performance.csv", index=False)

    # Q5: Drift detection
    ref = df[df["time_window"] == 0]
    ref_probs = model.predict_proba(ref[FEATURES].values)[:, 1]
    detect_rows = []
    for w in sorted(df["time_window"].unique()):
        wdf = df[df["time_window"] == w]
        wp = model.predict_proba(wdf[FEATURES].values)[:, 1]
        ks, p = stats.ks_2samp(ref_probs, wp)
        detect_rows.append({"window": w, "ks_prediction": ks, "p_value": p})
    detect_df = pd.DataFrame(detect_rows)
    detect_df.to_csv(f"{OUT}/drift_detection_by_window.csv", index=False)

    # Q6: SHAP analysis
    explainer = shap.TreeExplainer(model)
    check_windows = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
    shap_rows = []
    for w in check_windows:
        wdf = df[df["time_window"] == w]
        sample = wdf.sample(min(200, len(wdf)), random_state=42)
        sv = explainer.shap_values(sample[FEATURES].values)
        abs_mean = np.abs(sv).mean(axis=0)
        shap_rows.append({"window": w, **{FEATURES[i]: abs_mean[i] for i in range(len(FEATURES))}})
    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(f"{OUT}/shap_by_window.csv", index=False)

    pre_shap = shap_df[shap_df["window"] < tau][FEATURES].mean()
    post_shap = shap_df[shap_df["window"] >= 15][FEATURES].mean()
    shap_change = (post_shap - pre_shap).abs().sort_values(ascending=False)

    known_drift = {
        "tweets_per_day", "posting_regularity", "night_post_ratio",
        "burst_score", "weekend_ratio",
        "text_entropy", "vocabulary_richness", "content_similarity",
        "avg_tweet_length", "caps_ratio",
        "reply_ratio", "retweet_ratio", "mention_count_avg",
        "hashtag_count_avg", "url_share_ratio",
        "default_profile", "default_profile_image", "has_url",
        "description_length", "followers_count", "listed_count",
        "reputation_score",
    }
    prec_recall_rows = []
    for k in [5, 10, 15, 20, 30]:
        top_k = set(shap_change.index[:k])
        tp = len(top_k & known_drift)
        prec = tp / k
        rec = tp / len(known_drift)
        f1_k = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        prec_recall_rows.append({"k": k, "precision": prec, "recall": rec, "f1": f1_k, "tp": tp, "fp": k - tp})
    pd.DataFrame(prec_recall_rows).to_csv(f"{OUT}/shap_precision_recall.csv", index=False)

    shap_attr = pd.DataFrame({
        "feature": shap_change.index,
        "shap_change": shap_change.values,
        "is_drift_feature": [f in known_drift for f in shap_change.index],
    })
    shap_attr.to_csv(f"{OUT}/shap_attribution.csv", index=False)

    # --- PLOTS ---
    # Fig 1: F1/AUC over windows
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(win_df["window"], win_df["f1"], "o-", color="#2ECC71", label="F1", linewidth=2)
    ax.plot(win_df["window"], win_df["auc"], "s-", color="#3498DB", label="AUC", linewidth=2)
    ax.axvline(x=tau, color="#E74C3C", linestyle="--", alpha=0.7, label="Drift onset")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Over Time (Static Model on v0.2)")
    ax.legend()
    ax.set_ylim(0.85, 1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_performance_over_time.png")
    plt.close(fig)

    # Fig 2: Prediction drift KS
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(detect_df["window"], detect_df["ks_prediction"], "D-", color="#9B59B6", linewidth=2)
    ax.axvline(x=tau, color="#E74C3C", linestyle="--", alpha=0.7, label="Drift onset")
    ax.set_xlabel("Time Window")
    ax.set_ylabel("KS Statistic")
    ax.set_title("Prediction Distribution Drift (KS vs Window 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_prediction_drift.png")
    plt.close(fig)

    # Fig 3: SHAP attribution bar
    top15 = shap_attr.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ECC71" if d else "#BDC3C7" for d in top15["is_drift_feature"]]
    ax.barh(range(len(top15)), top15["shap_change"].values, color=colors)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|SHAP Change| (post-drift - pre-drift)")
    ax.set_title("Top 15 Features by SHAP Change (green = known drift feature)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_shap_attribution.png")
    plt.close(fig)

    print("  Validation complete. Plots saved.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Drift Magnitude Scaling
# ═══════════════════════════════════════════════════════════════════════════

def run_drift_scaling():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: DRIFT MAGNITUDE SCALING (3 seeds x 4 magnitudes)")
    print("=" * 65)

    rows = []
    for mag in MAGNITUDES:
        for seed in SEEDS[:3]:
            df = gen_data(seed=seed, magnitude=mag)
            pre = df[df["time_window"] < 5]
            X_tr, X_te, y_tr, y_te = train_test_split(
                pre[FEATURES].values, pre["label_binary"].values,
                test_size=0.3, random_state=seed, stratify=pre["label_binary"].values)
            model = train_model(X_tr, y_tr, seed=seed)

            f1_pre = f1_score(y_te, model.predict(X_te))
            post = df[df["time_window"] >= 15]
            f1_post = f1_score(post["label_binary"].values, model.predict(post[FEATURES].values))

            explainer = shap.TreeExplainer(model)
            pre_sample = pre.sample(min(200, len(pre)), random_state=seed)
            post_sample = post.sample(min(200, len(post)), random_state=seed)
            sv_pre = np.abs(explainer.shap_values(pre_sample[FEATURES].values)).mean(axis=0)
            sv_post = np.abs(explainer.shap_values(post_sample[FEATURES].values)).mean(axis=0)
            shap_change = np.abs(sv_post - sv_pre)
            top5 = set(np.array(FEATURES)[np.argsort(-shap_change)[:5]])

            known = {"tweets_per_day", "posting_regularity", "night_post_ratio",
                     "burst_score", "text_entropy", "vocabulary_richness",
                     "content_similarity", "reply_ratio", "retweet_ratio",
                     "mention_count_avg", "hashtag_count_avg", "url_share_ratio",
                     "default_profile", "default_profile_image", "has_url",
                     "description_length", "followers_count", "listed_count",
                     "reputation_score", "weekend_ratio", "avg_tweet_length", "caps_ratio"}
            tp5 = len(top5 & known)

            ref_probs = model.predict_proba(pre[FEATURES].values)[:, 1]
            post_probs = model.predict_proba(post[FEATURES].values)[:, 1]
            ks_pred, _ = stats.ks_2samp(ref_probs, post_probs)

            rows.append({
                "magnitude": mag, "seed": seed,
                "f1_pre": round(f1_pre, 4), "f1_post": round(f1_post, 4),
                "f1_drop": round(f1_pre - f1_post, 4),
                "prediction_ks": round(ks_pred, 4),
                "shap_precision_at_5": tp5 / 5,
            })
            print(f"  mag={mag} seed={seed}: F1 {f1_pre:.4f}->{f1_post:.4f} drop={f1_pre-f1_post:.4f}")

    scale_df = pd.DataFrame(rows)
    scale_df.to_csv(f"{OUT}/drift_scaling.csv", index=False)

    # Fig 4: F1 drop by magnitude
    fig, ax = plt.subplots(figsize=(7, 4))
    means = scale_df.groupby("magnitude")["f1_drop"].agg(["mean", "std"]).reset_index()
    ax.bar(range(len(means)), means["mean"], yerr=means["std"],
           color=["#AED6F1", "#85C1E9", "#5DADE2", "#2E86C1"], capsize=5, edgecolor="#2C3E50")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([f"{m}" for m in means["magnitude"]])
    ax.set_xlabel("Drift Magnitude")
    ax.set_ylabel("F1 Drop")
    ax.set_title("Classifier Degradation by Drift Magnitude")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_magnitude_scaling.png")
    plt.close(fig)
    print("  Scaling experiment complete.")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Threshold Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def run_threshold():
    print("\n" + "=" * 65)
    print("EXPERIMENT 3: THRESHOLD SENSITIVITY (10 seeds x 4 mags x 6 thresholds)")
    print("=" * 65)

    rows = []
    total = len(SEEDS) * len(MAGNITUDES) * len(THRESHOLD_MULTS)
    done = 0
    for seed in SEEDS:
        for mag in MAGNITUDES:
            df = gen_data(seed=seed, magnitude=mag)
            pre = df[df["time_window"] < 5]
            X_tr, X_te, y_tr, y_te = train_test_split(
                pre[FEATURES].values, pre["label_binary"].values,
                test_size=0.3, random_state=seed, stratify=pre["label_binary"].values)
            model = train_model(X_tr, y_tr, seed=seed)

            f1_pre = f1_score(y_te, model.predict(X_te))

            ref_probs = model.predict_proba(df[df["time_window"] == 0][FEATURES].values)[:, 1]
            noise_ks = []
            for w in range(1, 5):
                wp = model.predict_proba(df[df["time_window"] == w][FEATURES].values)[:, 1]
                ks, _ = stats.ks_2samp(ref_probs, wp)
                noise_ks.append(ks)
            noise_floor = np.mean(noise_ks)

            for tmult in THRESHOLD_MULTS:
                threshold = noise_floor * tmult
                detected = False
                detection_window = None
                false_alarms = 0
                n_retrains = 0

                current_model = model
                current_ref = ref_probs.copy()

                for w in range(5, 20):
                    wdf = df[df["time_window"] == w]
                    wp = current_model.predict_proba(wdf[FEATURES].values)[:, 1]
                    ks, _ = stats.ks_2samp(current_ref, wp)

                    n_drifting = int(wdf["drift_active"].sum())

                    if ks > threshold:
                        if n_drifting > 0 and not detected:
                            detected = True
                            detection_window = w
                        elif n_drifting == 0:
                            false_alarms += 1

                        n_retrains += 1
                        recent = df[df["time_window"].isin([w-1, w])]
                        X_r = recent[FEATURES].values
                        y_r = recent["label_binary"].values
                        if len(np.unique(y_r)) > 1:
                            current_model = train_model(X_r, y_r, seed=seed)
                            current_ref = current_model.predict_proba(wdf[FEATURES].values)[:, 1]

                post = df[df["time_window"] >= 15]
                f1_post = f1_score(post["label_binary"].values,
                                   current_model.predict(post[FEATURES].values))

                rows.append({
                    "seed": seed, "magnitude": mag, "threshold_mult": tmult,
                    "ks_threshold": round(threshold, 6),
                    "detected": detected,
                    "detection_delay": (detection_window - 5) if detected else None,
                    "false_alarms": false_alarms,
                    "n_retrains": n_retrains,
                    "f1_pre": round(f1_pre, 4),
                    "f1_post": round(f1_post, 4),
                    "f1_drop": round(f1_pre - f1_post, 4),
                })
                done += 1

        print(f"  Progress: {done}/{total} ({done*100//total}%)")

    thresh_df = pd.DataFrame(rows)
    thresh_df.to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)

    # Fig 5: Threshold operating characteristics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    summary = thresh_df.groupby("threshold_mult").agg(
        detect_rate=("detected", "mean"),
        avg_delay=("detection_delay", lambda x: x.dropna().mean() if x.notna().any() else 0),
        avg_fa=("false_alarms", "mean"),
        avg_drop=("f1_drop", "mean"),
    ).reset_index()

    ax1.plot(summary["threshold_mult"], summary["detect_rate"], "o-", color="#2ECC71", linewidth=2, label="Detection Rate")
    ax1b = ax1.twinx()
    ax1b.plot(summary["threshold_mult"], summary["avg_fa"], "s-", color="#E74C3C", linewidth=2, label="Avg False Alarms")
    ax1.set_xlabel("Threshold Multiplier")
    ax1.set_ylabel("Detection Rate", color="#2ECC71")
    ax1b.set_ylabel("Avg False Alarms", color="#E74C3C")
    ax1.set_title("Detection Rate vs False Alarms")
    ax1.set_ylim(0.5, 1.05)

    ax2.plot(summary["threshold_mult"], summary["avg_delay"], "D-", color="#9B59B6", linewidth=2, label="Avg Delay")
    ax2b = ax2.twinx()
    ax2b.plot(summary["threshold_mult"], summary["avg_drop"], "^-", color="#E67E22", linewidth=2, label="Avg F1 Drop")
    ax2.set_xlabel("Threshold Multiplier")
    ax2.set_ylabel("Avg Detection Delay (windows)", color="#9B59B6")
    ax2b.set_ylabel("Avg F1 Drop", color="#E67E22")
    ax2.set_title("Detection Delay vs F1 Drop")

    fig.suptitle("Threshold Sensitivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_threshold_sensitivity.png")
    plt.close(fig)
    print("  Threshold experiment complete.")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Adaptation Strategy Comparison (10 seeds x 4 magnitudes)
# ═══════════════════════════════════════════════════════════════════════════

def run_adaptation():
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: ADAPTATION COMPARISON (10 seeds x 4 magnitudes x 3 strategies)")
    print("=" * 65)

    rows = []
    total = len(SEEDS) * len(MAGNITUDES)
    done = 0

    for seed in SEEDS:
        for mag in MAGNITUDES:
            df = gen_data(seed=seed, magnitude=mag)
            pre = df[df["time_window"] < 5]
            X_tr, X_te, y_tr, y_te = train_test_split(
                pre[FEATURES].values, pre["label_binary"].values,
                test_size=0.3, random_state=seed, stratify=pre["label_binary"].values)
            base_model = train_model(X_tr, y_tr, seed=seed)
            f1_pre = f1_score(y_te, base_model.predict(X_te))

            ref_probs = base_model.predict_proba(df[df["time_window"] == 0][FEATURES].values)[:, 1]
            noise_ks = []
            for w in range(1, 5):
                wp = base_model.predict_proba(df[df["time_window"] == w][FEATURES].values)[:, 1]
                ks, _ = stats.ks_2samp(ref_probs, wp)
                noise_ks.append(ks)
            noise_floor = np.mean(noise_ks)
            threshold = noise_floor * 2.0

            for strategy in ["static", "periodic", "drift_aware"]:
                current_model = base_model
                current_ref = ref_probs.copy()
                n_retrains = 0
                false_alarms = 0
                detection_delay = None
                detected = False
                f1_windows = []

                for w in range(5, 20):
                    wdf = df[df["time_window"] == w]
                    Xw = wdf[FEATURES].values
                    yw = wdf["label_binary"].values

                    f1_w = f1_score(yw, current_model.predict(Xw))
                    f1_windows.append({"window": w, "f1": f1_w})

                    should_retrain = False
                    if strategy == "periodic" and w % 4 == 0:
                        should_retrain = True
                    elif strategy == "drift_aware":
                        wp = current_model.predict_proba(Xw)[:, 1]
                        ks, _ = stats.ks_2samp(current_ref, wp)
                        if ks > threshold:
                            should_retrain = True
                            n_drifting = int(wdf["drift_active"].sum())
                            if n_drifting > 0 and not detected:
                                detected = True
                                detection_delay = w - 5
                            elif n_drifting == 0:
                                false_alarms += 1

                    if should_retrain:
                        n_retrains += 1
                        recent = df[df["time_window"].isin([w-1, w])]
                        Xr = recent[FEATURES].values
                        yr = recent["label_binary"].values
                        if len(np.unique(yr)) > 1:
                            current_model = train_model(Xr, yr, seed=seed)
                            current_ref = current_model.predict_proba(Xw)[:, 1]

                post = df[df["time_window"] >= 15]
                f1_post = f1_score(post["label_binary"].values,
                                   current_model.predict(post[FEATURES].values))

                rows.append({
                    "seed": seed, "magnitude": mag, "strategy": strategy,
                    "f1_pre": round(f1_pre, 4), "f1_post": round(f1_post, 4),
                    "f1_drop": round(f1_pre - f1_post, 4),
                    "n_retrains": n_retrains,
                    "detection_delay": detection_delay,
                    "false_alarms": false_alarms,
                })

            done += 1
            if done % 5 == 0:
                print(f"  Progress: {done}/{total} ({done*100//total}%)")

    adapt_df = pd.DataFrame(rows)
    adapt_df.to_csv(f"{OUT}/adaptation_results.csv", index=False)

    # Summary
    summary_rows = []
    for mag in MAGNITUDES:
        for strat in ["static", "periodic", "drift_aware"]:
            s = adapt_df[(adapt_df["magnitude"] == mag) & (adapt_df["strategy"] == strat)]
            summary_rows.append({
                "magnitude": mag, "strategy": strat,
                "f1_drop_mean": round(s["f1_drop"].mean(), 4),
                "f1_drop_std": round(s["f1_drop"].std(), 4),
                "retrains_mean": round(s["n_retrains"].mean(), 1),
                "retrains_std": round(s["n_retrains"].std(), 1),
                "delay_mean": round(s["detection_delay"].mean(), 1) if s["detection_delay"].notna().any() else None,
                "false_alarms_mean": round(s["false_alarms"].mean(), 1),
            })
    pd.DataFrame(summary_rows).to_csv(f"{OUT}/adaptation_summary.csv", index=False)

    # Fig 6: Adaptation comparison grouped bar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MAGNITUDES))
    width = 0.25
    for i, strat in enumerate(["static", "periodic", "drift_aware"]):
        means = [adapt_df[(adapt_df["magnitude"] == m) & (adapt_df["strategy"] == strat)]["f1_drop"].mean()
                 for m in MAGNITUDES]
        stds = [adapt_df[(adapt_df["magnitude"] == m) & (adapt_df["strategy"] == strat)]["f1_drop"].std()
                for m in MAGNITUDES]
        label = strat.replace("_", "-").title()
        ax1.bar(x + i * width, means, width, yerr=stds, label=label,
                color=COLORS[strat], capsize=3, edgecolor="#2C3E50", alpha=0.85)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([str(m) for m in MAGNITUDES])
    ax1.set_xlabel("Drift Magnitude")
    ax1.set_ylabel("F1 Drop")
    ax1.set_title("F1 Degradation by Strategy")
    ax1.legend()

    for i, strat in enumerate(["static", "periodic", "drift_aware"]):
        means = [adapt_df[(adapt_df["magnitude"] == m) & (adapt_df["strategy"] == strat)]["n_retrains"].mean()
                 for m in MAGNITUDES]
        label = strat.replace("_", "-").title()
        ax2.bar(x + i * width, means, width, label=label,
                color=COLORS[strat], edgecolor="#2C3E50", alpha=0.85)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(m) for m in MAGNITUDES])
    ax2.set_xlabel("Drift Magnitude")
    ax2.set_ylabel("Number of Retrains")
    ax2.set_title("Retraining Events by Strategy")
    ax2.legend()

    fig.suptitle("Adaptation Strategy Comparison (10 seeds)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_adaptation_comparison.png")
    plt.close(fig)

    # Fig 7: Efficiency scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for strat in ["static", "periodic", "drift_aware"]:
        s = adapt_df[adapt_df["strategy"] == strat]
        ax.scatter(s["n_retrains"], s["f1_drop"],
                   color=COLORS[strat], alpha=0.6, s=40,
                   label=strat.replace("_", "-").title(), edgecolors="#2C3E50", linewidth=0.5)
    ax.set_xlabel("Number of Retrains")
    ax.set_ylabel("F1 Drop")
    ax.set_title("Retraining Efficiency: F1 Drop vs Retraining Cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_efficiency_scatter.png")
    plt.close(fig)

    print("  Adaptation experiment complete.")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Calibration quality (already have data, just plot)
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration_plot():
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: CALIBRATION QUALITY VISUALIZATION")
    print("=" * 65)

    comp = pd.read_csv("dataset/v02_vs_cresci_comparison.csv")

    # Fig 8: v0.1 vs v0.2 KS comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    features = comp["feature"].values
    y_pos = np.arange(len(features))
    ax.barh(y_pos - 0.2, comp["v01_ks_human"], 0.35, label="v0.1 KS (human)",
            color="#E74C3C", alpha=0.7, edgecolor="#2C3E50")
    ax.barh(y_pos + 0.2, comp["v02_ks_human"], 0.35, label="v0.2 KS (human)",
            color="#2ECC71", alpha=0.7, edgecolor="#2C3E50")
    ax.axvline(x=0.3, color="#95A5A6", linestyle="--", alpha=0.7, label="KS < 0.3 threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("KS Statistic (lower = better match to Cresci-2017)")
    ax.set_title("Distribution Calibration: v0.1 vs v0.2 (Human Features)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_calibration_comparison.png")
    plt.close(fig)
    print("  Calibration plot saved.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 65)
    print("BotDriftBench v0.2 — Complete Experiment Pipeline")
    print("=" * 65)

    run_validation()
    run_drift_scaling()
    run_threshold()
    run_adaptation()
    run_calibration_plot()

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUT}/")
    print(f"{'=' * 65}")

    for f in sorted(os.listdir(OUT)):
        sz = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f:45s} {sz:>10,} bytes")
