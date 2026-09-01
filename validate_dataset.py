"""
BotDriftBench v0.1 — Six-Question Validation
=============================================

Answers the 6 critical questions before scaling to full dataset:

Q1. Does the data look like plausible bot/human behavior?
Q2. Does the injected drift visibly exist?
Q3. Can a baseline detector detect bots?
Q4. Does its performance degrade after drift?
Q5. Can a drift detector find the known change point?
Q6. Does SHAP actually change in an interpretable way?

Usage:
    python validate_dataset.py
"""

import json
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────

FEATURES = [
    "account_age_days", "statuses_count", "followers_count",
    "friends_count", "favourites_count", "listed_count",
    "default_profile", "default_profile_image", "has_url",
    "verified", "description_length", "screen_name_length",
    "tweets_per_day", "retweet_ratio", "reply_ratio",
    "mention_count_avg", "hashtag_count_avg", "url_share_ratio",
    "posting_regularity", "night_post_ratio", "weekend_ratio",
    "burst_score",
    "avg_tweet_length", "vocabulary_richness", "text_entropy",
    "emoji_ratio", "caps_ratio", "content_similarity",
    "follower_friend_ratio", "reputation_score",
]


def load_dataset(path="dataset/botdriftbench_v0.1.csv"):
    df = pd.read_csv(path, dtype={"drifted_features": str}, low_memory=False)
    df["label_binary"] = (df["label"] == "bot").astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Q3: Can a baseline detector detect bots?
# ═══════════════════════════════════════════════════════════════════════════

def q3_baseline_detector(df, tau_start=5):
    print("\n" + "=" * 65)
    print("Q3: CAN A BASELINE DETECTOR DETECT BOTS?")
    print("=" * 65)

    pre = df[df["time_window"] < tau_start]
    X = pre[FEATURES].values
    y = pre["label_binary"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nTrained on pre-drift data (windows 0–{tau_start-1})")
    print(f"  Train size: {len(X_train):,}")
    print(f"  Test size:  {len(X_test):,}")
    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    verdict = "PASS" if auc > 0.85 else "FAIL"
    print(f"\n  Verdict: [{verdict}] — {'Detector works' if verdict == 'PASS' else 'Detector too weak'}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Q4: Does its performance degrade after drift?
# ═══════════════════════════════════════════════════════════════════════════

def q4_performance_degradation(df, model, tau_start=5):
    print("\n" + "=" * 65)
    print("Q4: DOES PERFORMANCE DEGRADE AFTER DRIFT?")
    print("=" * 65)

    results = []
    for w in sorted(df["time_window"].unique()):
        w_df = df[df["time_window"] == w]
        X_w = w_df[FEATURES].values
        y_w = w_df["label_binary"].values

        y_pred = model.predict(X_w)
        y_prob = model.predict_proba(X_w)[:, 1]

        acc = accuracy_score(y_w, y_pred)
        f1 = f1_score(y_w, y_pred)
        auc = roc_auc_score(y_w, y_prob)

        results.append({
            "window": w,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "n_drift": int(w_df["drift_active"].sum()),
        })

    results_df = pd.DataFrame(results)

    pre_auc = results_df[results_df["window"] < tau_start]["auc"].mean()
    post_auc = results_df[results_df["window"] >= 15]["auc"].mean()
    degradation = pre_auc - post_auc

    print(f"\n  Window-by-window AUC (pre-drift baseline model applied):")
    print(f"  {'Window':>6} {'AUC':>7} {'F1':>7} {'Acc':>7} {'Drifting':>8}")
    print(f"  {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for _, r in results_df.iterrows():
        marker = " ←" if r["window"] == tau_start else ""
        print(f"  {int(r['window']):6d} {r['auc']:7.4f} {r['f1']:7.4f} "
              f"{r['accuracy']:7.4f} {int(r['n_drift']):8d}{marker}")

    print(f"\n  Pre-drift AUC (mean windows 0–{tau_start-1}):  {pre_auc:.4f}")
    print(f"  Post-drift AUC (mean windows 15–19):           {post_auc:.4f}")
    print(f"  Degradation:                                   {degradation:.4f}")

    verdict = "PASS" if degradation > 0.01 else "FAIL"
    print(f"\n  Verdict: [{verdict}] — "
          f"{'Performance degrades as expected' if verdict == 'PASS' else 'No degradation detected'}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# Q5: Can a drift detector find the known change point?
# ═══════════════════════════════════════════════════════════════════════════

def q5_drift_detection(df, model, tau_start=5):
    print("\n" + "=" * 65)
    print("Q5: CAN A DRIFT DETECTOR FIND THE CHANGE POINT?")
    print("=" * 65)

    print("\n  Using 3 drift detection approaches:")
    print("    A. Raw feature distribution drift (KS-test)")
    print("    B. Prediction distribution drift")
    print("    C. SHAP distribution drift (preview — full in Q6)")

    # ── A. Feature-space drift ──
    print(f"\n  [A] Feature-Space Drift (KS-test vs window 0 reference)")
    ref = df[df["time_window"] == 0]
    ref_bots = ref[ref["label"] == "bot"]
    track_feats = ["posting_regularity", "text_entropy", "reply_ratio", "content_similarity"]

    print(f"\n  {'Window':>6}", end="")
    for f in track_feats:
        print(f"  {f[:12]:>12}", end="")
    print()

    first_detection_a = None
    for w in sorted(df["time_window"].unique()):
        w_bots = df[(df["time_window"] == w) & (df["label"] == "bot")]
        print(f"  {w:6d}", end="")
        any_sig = False
        for f in track_feats:
            ks, p = stats.ks_2samp(ref_bots[f].values, w_bots[f].values)
            marker = "*" if p < 0.001 else " "
            print(f"  {ks:11.3f}{marker}", end="")
            if p < 0.001:
                any_sig = True
        if any_sig and first_detection_a is None and w >= tau_start:
            first_detection_a = w
        print()

    if first_detection_a is not None:
        delay_a = first_detection_a - tau_start
        print(f"\n  Feature drift first detected at window {first_detection_a} "
              f"(delay = {delay_a} windows after onset)")
    else:
        print(f"\n  Feature drift not detected (may need larger sample)")

    # ── B. Prediction drift ──
    print(f"\n  [B] Prediction Drift (KS-test on P(bot) vs window 0)")
    ref_probs = model.predict_proba(ref[FEATURES].values)[:, 1]
    first_detection_b = None

    for w in sorted(df["time_window"].unique()):
        w_df = df[df["time_window"] == w]
        w_probs = model.predict_proba(w_df[FEATURES].values)[:, 1]
        ks, p = stats.ks_2samp(ref_probs, w_probs)
        marker = " ***" if p < 0.001 else ""
        print(f"    Window {w:2d}: KS={ks:.3f}  p={p:.1e}{marker}")
        if p < 0.001 and first_detection_b is None and w >= tau_start:
            first_detection_b = w

    if first_detection_b is not None:
        delay_b = first_detection_b - tau_start
        print(f"\n  Prediction drift first detected at window {first_detection_b} "
              f"(delay = {delay_b})")

    # ── Summary ──
    print(f"\n  Change point detection summary:")
    print(f"    True onset:              window {tau_start}")
    print(f"    Feature drift detected:  window {first_detection_a} "
          f"(delay {first_detection_a - tau_start if first_detection_a else 'N/A'})")
    print(f"    Prediction drift detect: window {first_detection_b} "
          f"(delay {first_detection_b - tau_start if first_detection_b else 'N/A'})")


# ═══════════════════════════════════════════════════════════════════════════
# Q6: Does SHAP actually change in an interpretable way?
# ═══════════════════════════════════════════════════════════════════════════

def q6_shap_evolution(df, model, tau_start=5):
    print("\n" + "=" * 65)
    print("Q6: DOES SHAP CHANGE IN AN INTERPRETABLE WAY?")
    print("=" * 65)

    explainer = shap.TreeExplainer(model)

    # Compute SHAP for selected windows
    check_windows = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
    shap_means = []

    print(f"\n  Computing SHAP values for {len(check_windows)} windows...")
    for w in check_windows:
        w_df = df[df["time_window"] == w]
        # Sample to keep SHAP computation fast
        sample = w_df.sample(min(200, len(w_df)), random_state=42)
        X_sample = sample[FEATURES].values
        sv = explainer.shap_values(X_sample)
        abs_mean = np.abs(sv).mean(axis=0)
        shap_means.append({
            "window": w,
            **{FEATURES[i]: abs_mean[i] for i in range(len(FEATURES))},
        })

    shap_df = pd.DataFrame(shap_means)

    # Track top features
    top_pre = shap_df[shap_df["window"] < tau_start][FEATURES].mean().nlargest(8)
    top_post = shap_df[shap_df["window"] >= 15][FEATURES].mean().nlargest(8)

    print(f"\n  Top 8 features by mean |SHAP| — PRE-DRIFT (windows 0–4):")
    for feat, val in top_pre.items():
        print(f"    {feat:30s} {val:.4f}")

    print(f"\n  Top 8 features by mean |SHAP| — POST-DRIFT (windows 15+):")
    for feat, val in top_post.items():
        print(f"    {feat:30s} {val:.4f}")

    # SHAP trajectory for key drifting features
    print(f"\n  SHAP importance trajectory for drifting features:")
    drift_feats = ["posting_regularity", "text_entropy", "content_similarity",
                   "reply_ratio", "tweets_per_day", "vocabulary_richness"]
    print(f"  {'Window':>6}", end="")
    for f in drift_feats:
        print(f"  {f[:12]:>12}", end="")
    print()

    for _, row in shap_df.iterrows():
        w = int(row["window"])
        marker = " <- tau" if w == tau_start else ""
        print(f"  {w:6d}", end="")
        for f in drift_feats:
            print(f"  {row[f]:12.4f}", end="")
        print(marker)

    # SHAP drift detection
    print(f"\n  [C] SHAP-Space Drift (KS-test on SHAP distributions vs window 0)")
    ref_w = df[df["time_window"] == 0].sample(min(200, len(df[df["time_window"] == 0])), random_state=42)
    ref_shap = explainer.shap_values(ref_w[FEATURES].values)

    first_detection_c = None
    for w in check_windows:
        if w == 0:
            continue
        w_sample = df[df["time_window"] == w].sample(
            min(200, len(df[df["time_window"] == w])), random_state=42
        )
        w_shap = explainer.shap_values(w_sample[FEATURES].values)

        # Aggregate KS across all features
        ks_vals = []
        for j in range(len(FEATURES)):
            ks, _ = stats.ks_2samp(ref_shap[:, j], w_shap[:, j])
            ks_vals.append(ks)
        mean_ks = np.mean(ks_vals)
        max_ks = np.max(ks_vals)
        max_feat = FEATURES[np.argmax(ks_vals)]

        sig = " ***" if mean_ks > 0.15 else ""
        print(f"    Window {w:2d}: mean_KS={mean_ks:.3f}  max_KS={max_ks:.3f} "
              f"({max_feat}){sig}")

        if mean_ks > 0.15 and first_detection_c is None and w >= tau_start:
            first_detection_c = w

    if first_detection_c is not None:
        print(f"\n  SHAP drift first detected at window {first_detection_c} "
              f"(delay = {first_detection_c - tau_start})")

    # Interpretability check
    print(f"\n  Interpretability Check:")
    print(f"  Do SHAP changes match the KNOWN drifted features?")

    pre_shap = shap_df[shap_df["window"] < tau_start][FEATURES].mean()
    post_shap = shap_df[shap_df["window"] >= 15][FEATURES].mean()
    shap_change = (post_shap - pre_shap).abs().nlargest(10)

    known_drift_feats = {
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

    # Full ranked list
    full_change = (post_shap - pre_shap).abs().sort_values(ascending=False)

    print(f"\n  Top 15 features with largest |SHAP change|:")
    for i, (feat, change) in enumerate(full_change.items()):
        if i >= 15:
            break
        is_known = feat in known_drift_feats
        marker = "KNOWN DRIFT" if is_known else "  (stable)"
        print(f"    {feat:30s} D={change:.4f}  {marker}")

    # Precision@k / Recall@k at multiple thresholds
    n_known = len(known_drift_feats)
    print(f"\n  SHAP Drift Attribution — Precision@k / Recall@k:")
    print(f"    (ground truth: {n_known} features affected by drift)")
    print(f"    {'k':>5}  {'P@k':>6}  {'R@k':>6}  {'F1@k':>6}  {'TP':>4}  {'FP':>4}")
    print(f"    {'---':>5}  {'---':>6}  {'---':>6}  {'---':>6}  {'--':>4}  {'--':>4}")

    matches = 0
    for k in [5, 10, 15, 20, len(FEATURES)]:
        top_k_feats = set(full_change.index[:k])
        tp = len(top_k_feats & known_drift_feats)
        fp = len(top_k_feats - known_drift_feats)
        prec = tp / k if k > 0 else 0
        rec = tp / n_known if n_known > 0 else 0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
        label = f"@{k}" if k < len(FEATURES) else "@all"
        print(f"    {label:>5}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {tp:>4}  {fp:>4}")
        if k == 10:
            matches = tp

    verdict = "PASS" if matches >= 5 else "NEEDS REVIEW"
    print(f"  Verdict: [{verdict}]")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("BotDriftBench v0.1 — Six-Question Validation")
    print("=" * 65)

    df = load_dataset()

    with open("dataset/generation_config.json") as f:
        config = json.load(f)
    tau_start = config["parameters"]["tau_start"]

    print(f"\nLoaded: {len(df):,} observations, "
          f"{df['account_id'].nunique():,} accounts, "
          f"{df['time_window'].nunique()} windows")
    print(f"Change point tau_start = {tau_start}")

    # Q1 and Q2 are already validated by generate_dataset.py
    print("\nQ1: PLAUSIBLE BOT/HUMAN BEHAVIOR? → Already validated (see generation output)")
    print("Q2: DRIFT VISIBLY EXISTS? → Already validated (see generation output)")

    # Q3–Q6
    model = q3_baseline_detector(df, tau_start)
    q4_performance_degradation(df, model, tau_start)
    q5_drift_detection(df, model, tau_start)
    q6_shap_evolution(df, model, tau_start)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
