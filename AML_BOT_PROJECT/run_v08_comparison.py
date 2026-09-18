"""
BotDriftBench v0.8 - Full Policy Comparison (Step 4)
======================================================
Compares 5 retraining policies on account-disjoint test accounts:
  1. static          - never retrain
  2. periodic_4      - retrain every 4 windows
  3. periodic_6      - retrain every 6 windows
  4. da_localized    - drift-aware: retrain when drift detected
  5. oracle_da       - oracle: retrain at true drift onset (upper bound)

Drift detection uses a simple localized KS-based signal on training bots.
Thresholds tuned on validation set only (no test data used for tuning).

Usage:
    python run_v08_comparison.py --data dataset_v08 --seed 2025 --seeds 5
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

XGBOOST_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    eval_metric="logloss", verbosity=0,
)

META_COLS = {
    "account_id", "window", "label", "archetype", "scenario",
    "drift_type", "difficulty_tier", "drift_onset", "drift_duration",
    "drift_severity", "drift_progress", "is_drifting", "drift_phase", "split",
}

STABLE_CONTROL_ONSET = -1
KS_THRESHOLD = 0.12
KS_FRAC_THRESHOLD = 0.30
RECENCY_DECAY = 0.7


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def roc_auc_safe(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


# ============================================================
# Train helper
# ============================================================

def train_xgb(X, y, seed: int, sample_weights=None) -> XGBClassifier:
    clf = XGBClassifier(random_state=seed, **XGBOOST_PARAMS)
    clf.fit(X, y, sample_weight=sample_weights)
    return clf


def recency_weights(windows: np.ndarray, current_window: int, decay: float) -> np.ndarray:
    """Exponential recency weighting - higher weight for more recent windows."""
    age = current_window - windows
    return np.exp(-decay * age.astype(float))


# ============================================================
# Drift detection (localized KS on training bots)
# ============================================================

def detect_drift_ks(
    train_bots_ref: pd.DataFrame,
    train_bots_cur: pd.DataFrame,
    feat_cols: list[str],
    ks_thresh: float,
    frac_thresh: float,
) -> bool:
    """Returns True if drift detected on training bots between reference and current window."""
    if len(train_bots_ref) == 0 or len(train_bots_cur) == 0:
        return False
    n_drifted = 0
    for col in feat_cols:
        ks, _ = ks_2samp(train_bots_ref[col].dropna(), train_bots_cur[col].dropna())
        if ks > ks_thresh:
            n_drifted += 1
    return (n_drifted / len(feat_cols)) >= frac_thresh


# ============================================================
# Policy runners
# ============================================================

def run_static(
    train_pre: pd.DataFrame, test_df: pd.DataFrame,
    feat_cols: list[str], seed: int,
) -> dict:
    clf = train_xgb(train_pre[feat_cols].values, train_pre["label"].values, seed)
    results = []
    for w in sorted(test_df["window"].unique()):
        wdf = test_df[test_df["window"] == w]
        y_true = wdf["label"].values
        y_prob = clf.predict_proba(wdf[feat_cols].values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({"window": int(w), "f1_macro": macro_f1(y_true, y_pred),
                        "roc_auc": roc_auc_safe(y_true, y_prob)})
    return {"policy": "static", "retrains": 0, "false_alarms": 0, "per_window": results}


def run_periodic(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    feat_cols: list[str], seed: int,
    period: int, train_ids: set, global_min_onset: int,
) -> dict:
    """Periodic retraining every period windows using all train data up to current window."""
    clf = train_xgb(
        train_df[train_df["window"] < global_min_onset][feat_cols].values,
        train_df[train_df["window"] < global_min_onset]["label"].values,
        seed,
    )
    results = []
    retrains = 0
    max_w = int(test_df["window"].max())

    for w in range(max_w + 1):
        # Retrain at each multiple of period
        if w > 0 and w % period == 0:
            train_avail = train_df[train_df["window"] <= w]
            if len(train_avail) > 0:
                weights = recency_weights(
                    train_avail["window"].values, w, RECENCY_DECAY
                )
                clf = train_xgb(
                    train_avail[feat_cols].values,
                    train_avail["label"].values,
                    seed, weights
                )
                retrains += 1

        wdf = test_df[test_df["window"] == w]
        if len(wdf) == 0:
            continue
        y_true = wdf["label"].values
        y_prob = clf.predict_proba(wdf[feat_cols].values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({"window": int(w), "f1_macro": macro_f1(y_true, y_pred),
                        "roc_auc": roc_auc_safe(y_true, y_prob)})

    return {"policy": f"periodic_{period}", "retrains": retrains, "false_alarms": 0,
            "per_window": results}


def run_da_localized(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    feat_cols: list[str], seed: int,
    train_ids: set, global_min_onset: int,
    ks_thresh: float = KS_THRESHOLD, frac_thresh: float = KS_FRAC_THRESHOLD,
) -> dict:
    """Drift-aware retraining: retrain when localized KS detects drift on train bots."""
    train_bots_ref = train_df[
        (train_df["label"] == 1) & (train_df["window"] < global_min_onset)
    ]
    clf = train_xgb(
        train_df[train_df["window"] < global_min_onset][feat_cols].values,
        train_df[train_df["window"] < global_min_onset]["label"].values,
        seed,
    )

    results = []
    retrains = 0
    false_alarms = 0
    max_w = int(test_df["window"].max())
    retrain_cooldown = 0  # windows since last retrain

    for w in range(max_w + 1):
        # Check for drift using train bots only
        if retrain_cooldown == 0:
            train_bots_cur = train_df[
                (train_df["label"] == 1) & (train_df["window"] == w)
            ]
            drift_detected = detect_drift_ks(
                train_bots_ref, train_bots_cur, feat_cols, ks_thresh, frac_thresh
            )
            if drift_detected:
                # Is this before drift onset? (false alarm)
                if w < global_min_onset:
                    false_alarms += 1
                # Retrain on all train data up to now
                train_avail = train_df[train_df["window"] <= w]
                if len(train_avail) > 0:
                    weights = recency_weights(
                        train_avail["window"].values, w, RECENCY_DECAY
                    )
                    clf = train_xgb(
                        train_avail[feat_cols].values,
                        train_avail["label"].values,
                        seed, weights
                    )
                    retrains += 1
                    retrain_cooldown = 2  # enforce minimum gap
        else:
            retrain_cooldown -= 1

        wdf = test_df[test_df["window"] == w]
        if len(wdf) == 0:
            continue
        y_true = wdf["label"].values
        y_prob = clf.predict_proba(wdf[feat_cols].values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({"window": int(w), "f1_macro": macro_f1(y_true, y_pred),
                        "roc_auc": roc_auc_safe(y_true, y_prob)})

    return {"policy": "da_localized", "retrains": retrains, "false_alarms": false_alarms,
            "per_window": results}


def run_oracle(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    feat_cols: list[str], seed: int,
    train_ids: set, global_min_onset: int, meta: pd.DataFrame,
) -> dict:
    """Oracle: retrain exactly at each account's actual drift onset (upper bound)."""
    # Find the oracle retrain windows (unique onset windows among train accounts)
    train_meta = meta[meta["account_id"].isin(train_ids)].copy()
    onset_windows = sorted(
        train_meta.loc[
            train_meta["drift_onset"] != STABLE_CONTROL_ONSET, "drift_onset"
        ].dropna().astype(int).unique()
    )

    clf = train_xgb(
        train_df[train_df["window"] < global_min_onset][feat_cols].values,
        train_df[train_df["window"] < global_min_onset]["label"].values,
        seed,
    )
    onset_set = set(onset_windows)
    results = []
    retrains = 0
    max_w = int(test_df["window"].max())

    for w in range(max_w + 1):
        if w in onset_set:
            train_avail = train_df[train_df["window"] <= w]
            if len(train_avail) > 0:
                weights = recency_weights(
                    train_avail["window"].values, w, RECENCY_DECAY
                )
                clf = train_xgb(
                    train_avail[feat_cols].values,
                    train_avail["label"].values,
                    seed, weights
                )
                retrains += 1

        wdf = test_df[test_df["window"] == w]
        if len(wdf) == 0:
            continue
        y_true = wdf["label"].values
        y_prob = clf.predict_proba(wdf[feat_cols].values)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results.append({"window": int(w), "f1_macro": macro_f1(y_true, y_pred),
                        "roc_auc": roc_auc_safe(y_true, y_prob)})

    return {"policy": "oracle_da", "retrains": retrains, "false_alarms": 0,
            "per_window": results}


# ============================================================
# Multi-seed runner
# ============================================================

def run_all_seeds(data_dir: Path, n_seeds: int, base_seed: int) -> None:
    out_dir = data_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    main_csvs = sorted(data_dir.glob("botdriftbench_v08*.csv"))
    if not main_csvs:
        main_csvs = [f for f in sorted(data_dir.glob("*.csv"))
                     if "metadata" not in f.name and "diagnostic" not in str(f)]
    df = pd.read_csv(main_csvs[0])
    print(f"Loaded: {main_csvs[0].name}  ({len(df):,} rows)")

    meta_csvs = sorted(data_dir.glob("account_metadata_v08*.csv"))
    if not meta_csvs:
        meta_csvs = sorted(data_dir.glob("account_metadata*.csv"))
    meta = pd.read_csv(meta_csvs[0])

    # Auto-detect features
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Features: {len(feat_cols)}")

    train_ids = set(df[df["split"] == "train"]["account_id"].unique())
    val_ids   = set(df[df["split"] == "validation"]["account_id"].unique())
    test_ids  = set(df[df["split"] == "test"]["account_id"].unique())
    print(f"Train: {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}")

    # Global min drift onset (from drifting accounts only)
    drifting_onsets = meta[meta["drift_onset"] != STABLE_CONTROL_ONSET]["drift_onset"].dropna()
    global_min_onset = int(drifting_onsets.min()) if len(drifting_onsets) > 0 else 5
    print(f"Global min drift onset: window {global_min_onset}")
    print(f"Pre-drift = windows 0 to {global_min_onset - 1}")

    # Tune DA thresholds on VALIDATION set (not test)
    print("\nTuning DA thresholds on validation set...")
    val_df    = df[df["account_id"].isin(val_ids)]
    train_df  = df[df["account_id"].isin(train_ids)]
    train_pre = train_df[train_df["window"] < global_min_onset]

    best_ks, best_frac, best_f1 = KS_THRESHOLD, KS_FRAC_THRESHOLD, -1.0
    for ks in [0.06, 0.09, 0.12, 0.15, 0.20]:
        for frac in [0.20, 0.25, 0.30, 0.35, 0.40]:
            res = run_da_localized(
                pd.concat([train_df, val_df]).copy(),
                val_df, feat_cols, base_seed,
                train_ids | val_ids, global_min_onset, ks, frac,
            )
            f1_vals = [r["f1_macro"] for r in res["per_window"]
                       if r["window"] >= global_min_onset]
            if f1_vals:
                avg = float(np.mean(f1_vals))
                if avg > best_f1:
                    best_f1, best_ks, best_frac = avg, ks, frac
    print(f"  Best val post-drift F1={best_f1:.4f}  KS={best_ks}  frac={best_frac}")

    # Run multi-seed comparison on TEST
    test_df = df[df["account_id"].isin(test_ids)]
    policies = ["static", "periodic_4", "periodic_6", "da_localized", "oracle_da"]
    all_results = {p: [] for p in policies}

    for s_idx in range(n_seeds):
        seed = base_seed + s_idx
        print(f"\n--- Seed {seed} (run {s_idx+1}/{n_seeds}) ---")

        res_static = run_static(train_pre, test_df, feat_cols, seed)
        res_p4     = run_periodic(train_df, test_df, feat_cols, seed, 4, train_ids, global_min_onset)
        res_p6     = run_periodic(train_df, test_df, feat_cols, seed, 6, train_ids, global_min_onset)
        res_da     = run_da_localized(train_df, test_df, feat_cols, seed,
                                      train_ids, global_min_onset, best_ks, best_frac)
        res_oracle = run_oracle(train_df, test_df, feat_cols, seed, train_ids, global_min_onset, meta)

        for name, res in [("static", res_static), ("periodic_4", res_p4),
                          ("periodic_6", res_p6), ("da_localized", res_da),
                          ("oracle_da", res_oracle)]:
            f1_all = [r["f1_macro"] for r in res["per_window"]]
            f1_pre  = [r["f1_macro"] for r in res["per_window"] if r["window"] < global_min_onset]
            f1_post = [r["f1_macro"] for r in res["per_window"] if r["window"] >= global_min_onset]
            all_results[name].append({
                "seed": seed,
                "tmean_f1":    float(np.mean(f1_all))  if f1_all  else float("nan"),
                "pre_f1":     float(np.mean(f1_pre))   if f1_pre  else float("nan"),
                "post_f1":    float(np.mean(f1_post))  if f1_post else float("nan"),
                "worst_f1":   float(np.min(f1_all))    if f1_all  else float("nan"),
                "retrains":   res["retrains"],
                "false_alarms": res["false_alarms"],
                "per_window": res["per_window"],
            })
            print(f"  {name:20s}: tmean={np.mean(f1_all):.4f}  pre={np.mean(f1_pre):.4f}  "
                  f"post={np.mean(f1_post):.4f}  retrains={res['retrains']}  "
                  f"fa={res['false_alarms']}")

    # Aggregate
    print()
    print("=" * 70)
    print("AGGREGATE RESULTS ({n_seeds} seeds)")
    print("=" * 70)
    agg_rows = []
    for policy, seed_results in all_results.items():
        tmeans   = [r["tmean_f1"]  for r in seed_results]
        pre_f1s  = [r["pre_f1"]    for r in seed_results]
        post_f1s = [r["post_f1"]   for r in seed_results]
        worst_f1s= [r["worst_f1"]  for r in seed_results]
        retrains  = [r["retrains"] for r in seed_results]
        agg_rows.append({
            "policy":     policy,
            "tmean_f1":   round(float(np.mean(tmeans)), 4),
            "tmean_std":  round(float(np.std(tmeans)),  4),
            "pre_f1":     round(float(np.nanmean(pre_f1s)), 4),
            "post_f1":    round(float(np.mean(post_f1s)), 4),
            "post_f1_std":round(float(np.std(post_f1s)), 4),
            "worst_f1":   round(float(np.mean(worst_f1s)), 4),
            "retrains":   round(float(np.mean(retrains)), 1),
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "aggregate_summary.csv", index=False)

    print(f"\n{'Policy':<22} {'TmeanF1':>9} {'Std':>7} {'PreF1':>7} {'PostF1':>8} {'WorstF1':>8} {'Retrains':>9}")
    print("-" * 78)
    for _, row in agg_df.iterrows():
        print(f"  {row['policy']:<20} {row['tmean_f1']:>9.4f} {row['tmean_std']:>7.4f} "
              f"{row['pre_f1']:>7.4f} {row['post_f1']:>8.4f} "
              f"{row['worst_f1']:>8.4f} {row['retrains']:>9.1f}")

    # Head-to-head DA vs best periodic
    da_tmeans = [r["tmean_f1"] for r in all_results["da_localized"]]
    p4_tmeans = [r["tmean_f1"] for r in all_results["periodic_4"]]
    p6_tmeans = [r["tmean_f1"] for r in all_results["periodic_6"]]
    bp_tmeans = [max(p4, p6) for p4, p6 in zip(p4_tmeans, p6_tmeans)]
    deltas = [da - bp for da, bp in zip(da_tmeans, bp_tmeans)]
    wins   = sum(1 for d in deltas if d > 0)
    mean_delta = float(np.mean(deltas))
    print()
    print(f"Head-to-head: DA vs best periodic (p4/p6)")
    print(f"  Mean delta:  {mean_delta:+.4f}")
    print(f"  Seed wins:   {wins}/{n_seeds}")
    print()

    static_tmeans = [r["tmean_f1"] for r in all_results["static"]]
    da_vs_static  = [da - st for da, st in zip(da_tmeans, static_tmeans)]
    wins_vs_static = sum(1 for d in da_vs_static if d > 0)
    print(f"Head-to-head: DA vs static")
    print(f"  Mean delta:  {float(np.mean(da_vs_static)):+.4f}")
    print(f"  Seed wins:   {wins_vs_static}/{n_seeds}")
    print()

    # Save seed-level results
    seed_rows = []
    for policy, seed_results in all_results.items():
        for r in seed_results:
            seed_rows.append({
                "policy": policy, "seed": r["seed"],
                "tmean_f1": r["tmean_f1"], "pre_f1": r["pre_f1"],
                "post_f1": r["post_f1"], "worst_f1": r["worst_f1"],
                "retrains": r["retrains"], "false_alarms": r["false_alarms"],
            })
    pd.DataFrame(seed_rows).to_csv(out_dir / "seed_results.csv", index=False)

    # Save per-window window_results (averaged across seeds)
    pw_rows = []
    for policy, seed_results in all_results.items():
        by_window = {}
        for r in seed_results:
            for pw in r["per_window"]:
                w = pw["window"]
                if w not in by_window:
                    by_window[w] = []
                by_window[w].append(pw["f1_macro"])
        for w in sorted(by_window.keys()):
            pw_rows.append({"policy": policy, "window": w,
                "f1_macro_mean": round(float(np.mean(by_window[w])), 4),
                "f1_macro_std":  round(float(np.std(by_window[w])),  4)})
    pd.DataFrame(pw_rows).to_csv(out_dir / "window_results.csv", index=False)

    # Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pw_df_all = pd.DataFrame(pw_rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        for policy in policies:
            pdf = pw_df_all[pw_df_all["policy"] == policy]
            ax.plot(pdf["window"], pdf["f1_macro_mean"], "-o", ms=4, label=policy, lw=1.5)
            ax.fill_between(pdf["window"],
                            pdf["f1_macro_mean"] - pdf["f1_macro_std"],
                            pdf["f1_macro_mean"] + pdf["f1_macro_std"],
                            alpha=0.08)
        ax.axvline(global_min_onset - 0.5, color="red", lw=1.5, ls="--",
                   label=f"Min drift onset (w{global_min_onset})")
        ax.set_xlabel("Window")
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Policy Comparison - BotDriftBench v0.8 ({n_seeds} seeds)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "policy_comparison_f1_curves.png", dpi=150)
        plt.close(fig)
        print("policy_comparison_f1_curves.png  [OK]")

        # Bar chart: mean post-drift F1
        fig, ax = plt.subplots(figsize=(8, 4))
        bar_df = agg_df.sort_values("post_f1", ascending=True)
        colors = ["steelblue" if p != "da_localized" else "crimson"
                  for p in bar_df["policy"]]
        ax.barh(bar_df["policy"], bar_df["post_f1"], color=colors, alpha=0.8,
                xerr=bar_df["post_f1_std"], capsize=4)
        ax.set_xlabel("Mean Post-Drift F1 (windows >= min_onset)")
        ax.set_title("Policy Comparison - Post-Drift F1 (v0.8)")
        ax.axvline(bar_df["post_f1"].max(), color="black", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(out_dir / "policy_comparison_postdrift_bar.png", dpi=150)
        plt.close(fig)
        print("policy_comparison_postdrift_bar.png  [OK]")

    except ImportError:
        print("matplotlib not available; skipping plots")

    print(f"\nAll comparison outputs saved to: {out_dir}")
    print("  aggregate_summary.csv")
    print("  seed_results.csv")
    print("  window_results.csv")


def main():
    p = argparse.ArgumentParser(
        description="BotDriftBench v0.8 - Full policy comparison"
    )
    p.add_argument("--data",    default="dataset_v08")
    p.add_argument("--seed",    type=int, default=2025)
    p.add_argument("--seeds",   type=int, default=5, help="Number of random seeds")
    args = p.parse_args()
    print("=" * 70)
    print("BotDriftBench v0.8 - Full Policy Comparison (Step 4)")
    print("=" * 70)
    run_all_seeds(Path(args.data), n_seeds=args.seeds, base_seed=args.seed)


if __name__ == "__main__":
    main()
