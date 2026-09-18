"""
BotDriftBench v0.8 - Frozen-Model Degradation Diagnostic (fixed)
=================================================================
Trains static XGBoost on pre-drift TRAIN accounts (windows before the
earliest drift onset among drifting accounts), then evaluates the frozen
model on all TEST windows and reports:
  - temporal mean F1
  - pre-drift F1 / post-drift F1 / F1 degradation
  - worst-window F1 / worst-case degradation
  - bot-probability shift
  - per-scenario degradation
  - per-tier degradation

Usage:
    python diagnose_v08_degradation.py --data dataset_v08 --seed 2025
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

XGBOOST_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    eval_metric="logloss", verbosity=0,
)

# Non-feature metadata columns - everything else is a feature
META_COLS = {
    "account_id", "window", "label", "archetype", "scenario",
    "drift_type", "difficulty_tier", "drift_onset", "drift_duration",
    "drift_severity", "drift_progress", "is_drifting", "drift_phase", "split",
}

STABLE_CONTROL_ONSET = -1  # sentinel value used by generator for no-drift accounts


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def roc_auc_safe(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def run_diagnostic(data_dir: Path, seed: int) -> None:
    out_dir = data_dir / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load -----------------------------------------------------------
    main_csvs = sorted(data_dir.glob("botdriftbench_v08*.csv"))
    if not main_csvs:
        main_csvs = [f for f in sorted(data_dir.glob("*.csv"))
                     if "metadata" not in f.name and "diagnostic" not in str(f)]
    if not main_csvs:
        raise FileNotFoundError(f"No dataset CSV in {data_dir}")
    df = pd.read_csv(main_csvs[0])
    print(f"Loaded: {main_csvs[0].name}  ({len(df):,} rows, {len(df.columns)} cols)")

    meta_csvs = sorted(data_dir.glob("account_metadata_v08*.csv"))
    if not meta_csvs:
        meta_csvs = sorted(data_dir.glob("account_metadata*.csv"))
    meta = pd.read_csv(meta_csvs[0]) if meta_csvs else None
    if meta is not None:
        print(f"Loaded: {meta_csvs[0].name}  ({len(meta):,} accounts)")

    # --- Feature columns (auto-detect) ----------------------------------
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Features used: {len(feat_cols)}")
    print(f"  {feat_cols}")

    # --- Splits ---------------------------------------------------------
    if "split" not in df.columns:
        raise ValueError("'split' column not found in main CSV.")
    train_ids = set(df[df["split"] == "train"]["account_id"].unique())
    test_ids  = set(df[df["split"] == "test" ]["account_id"].unique())
    print(f"Train accounts: {len(train_ids):,}  |  Test accounts: {len(test_ids):,}")

    # --- Global min drift onset (exclude stable_control accounts) -------
    # drift_onset == STABLE_CONTROL_ONSET (-1) is the sentinel for no-drift accounts.
    # Pre-drift window boundary = min onset among accounts that actually drift.
    if "drift_onset" in df.columns:
        drifting_onsets = df[
            (df["drift_onset"] != STABLE_CONTROL_ONSET) & (df["window"] == 0)
        ]["drift_onset"].dropna()
        if len(drifting_onsets) == 0:
            print("[WARN] No drifting accounts found; using onset=5 as fallback")
            global_min_onset = 5
        else:
            global_min_onset = int(drifting_onsets.min())
    elif meta is not None and "drift_onset" in meta.columns:
        drifting_onsets = meta[meta["drift_onset"] != STABLE_CONTROL_ONSET]["drift_onset"].dropna()
        global_min_onset = int(drifting_onsets.min()) if len(drifting_onsets) > 0 else 5
    else:
        global_min_onset = 5
    print(f"Global min drift onset (drifting accounts only): window {global_min_onset}")
    print(f"Pre-drift = windows 0 to {global_min_onset - 1}")

    # --- Train static model on pre-drift train accounts -----------------
    train_pre = df[
        (df["account_id"].isin(train_ids)) & (df["window"] < global_min_onset)
    ]
    print(f"Training rows: {len(train_pre):,}")
    if len(train_pre) == 0:
        raise ValueError(
            f"Zero training rows! global_min_onset={global_min_onset} "
            f"may be wrong. Check drift_onset column values."
        )
    lvc = train_pre["label"].value_counts().to_dict()
    print(f"Label distribution in training: {lvc}")

    clf = XGBClassifier(random_state=seed, **XGBOOST_PARAMS)
    clf.fit(train_pre[feat_cols].values, train_pre["label"].values)
    print("XGBoost trained (frozen from now on).")

    # Evaluate on train pre-drift itself to confirm it is non-trivial
    y_tr = train_pre["label"].values
    y_tr_pred = (clf.predict_proba(train_pre[feat_cols].values)[:, 1] >= 0.5).astype(int)
    print(f"Train-set F1 (sanity): {macro_f1(y_tr, y_tr_pred):.4f}")
    print()

    # --- Per-window evaluation on TEST accounts -------------------------
    test_df = df[df["account_id"].isin(test_ids)].copy()

    # Merge difficulty_tier if missing
    if "difficulty_tier" not in test_df.columns and meta is not None and "difficulty_tier" in meta.columns:
        test_df = test_df.merge(meta[["account_id", "difficulty_tier"]],
                                on="account_id", how="left")

    pw_rows = []
    for w in sorted(test_df["window"].unique()):
        wdf = test_df[test_df["window"] == w]
        y_true = wdf["label"].values
        X_w    = wdf[feat_cols].values
        y_prob = clf.predict_proba(X_w)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        bm = y_true == 1
        pw_rows.append({
            "window": int(w),
            "n_accounts": int(len(wdf)),
            "f1_macro":        round(macro_f1(y_true, y_pred), 4),
            "roc_auc":         round(roc_auc_safe(y_true, y_prob), 4),
            "mean_bot_prob":   round(float(y_prob[bm].mean()),   4) if bm.any()  else None,
            "mean_human_prob": round(float(y_prob[~bm].mean()),  4) if (~bm).any() else None,
            "is_pre_drift":    int(w) < global_min_onset,
        })

    pw_df = pd.DataFrame(pw_rows)
    pw_df.to_csv(out_dir / "per_window_f1.csv", index=False)

    pre_mask  = pw_df["window"] < global_min_onset
    post_mask = pw_df["window"] >= global_min_onset

    pre_f1   = pw_df.loc[pre_mask,  "f1_macro"].mean()
    post_f1  = pw_df.loc[post_mask, "f1_macro"].mean()
    worst_f1 = pw_df["f1_macro"].min()
    deg      = pre_f1 - post_f1
    worst_deg = pre_f1 - worst_f1
    tmean    = pw_df["f1_macro"].mean()
    pre_bp   = pw_df.loc[pre_mask,  "mean_bot_prob"].dropna().mean()
    post_bp  = pw_df.loc[post_mask, "mean_bot_prob"].dropna().mean()

    if   np.isnan(deg): verdict = "UNDETERMINED - no pre-drift test windows (onset at window 0?)"
    elif deg < 0.02:    verdict = "INSUFFICIENT - drift does not meaningfully degrade classifier"
    elif deg < 0.05:    verdict = "MARGINAL - small but detectable degradation"
    elif deg < 0.15:    verdict = "MODERATE - drift has real effect; adaptation has value"
    else:               verdict = "STRONG - substantial degradation; benchmark tests model failure well"

    print("=" * 60)
    print("STATIC MODEL - FROZEN DEGRADATION REPORT")
    print("=" * 60)
    print(f"  Temporal mean F1 (all windows):   {tmean:.4f}")
    print(f"  Pre-drift F1  (windows 0..{global_min_onset-1}):     {pre_f1:.4f}")
    print(f"  Post-drift F1 (windows {global_min_onset}..{int(pw_df['window'].max())}):   {post_f1:.4f}")
    print(f"  F1 degradation (pre - post):      {deg:+.4f}")
    print(f"  Worst-window F1:                  {worst_f1:.4f}")
    print(f"  Worst-case degradation:           {worst_deg:+.4f}")
    print(f"  Bot prob pre-drift:               {pre_bp:.4f}")
    print(f"  Bot prob post-drift:              {post_bp:.4f}")
    print(f"  Bot prob shift:                   {post_bp - pre_bp:+.4f}")
    print(f"  Verdict: {verdict}")
    print()

    print("Per-window F1:")
    print(pw_df[["window","f1_macro","roc_auc","mean_bot_prob","mean_human_prob"]].to_string(index=False))
    print()

    # --- Per-scenario ---------------------------------------------------
    scen_rows = []
    if "scenario" in test_df.columns:
        for scen, sdf in test_df.groupby("scenario"):
            pre_r  = sdf[sdf["window"] < global_min_onset]
            post_r = sdf[sdf["window"] >= global_min_onset]
            def _f(rows):
                if len(rows) == 0:
                    return float("nan")
                return macro_f1(rows["label"].values,
                    (clf.predict_proba(rows[feat_cols].values)[:, 1] >= 0.5).astype(int))
            pref, postf = _f(pre_r), _f(post_r)
            scen_rows.append({
                "scenario": scen,
                "n_accounts": int(sdf["account_id"].nunique()),
                "pre_drift_f1":  round(pref, 4),
                "post_drift_f1": round(postf, 4),
                "f1_degradation": round(pref - postf, 4)
                    if not (np.isnan(pref) or np.isnan(postf)) else float("nan"),
            })
        scen_df = pd.DataFrame(scen_rows).sort_values("f1_degradation", ascending=False)
        scen_df.to_csv(out_dir / "per_scenario_degradation.csv", index=False)
        print("Per-scenario degradation:")
        print(scen_df.to_string(index=False))
        print()
    else:
        scen_df = pd.DataFrame()

    # --- Per-tier -------------------------------------------------------
    tier_col = next((c for c in ["difficulty_tier","tier"] if c in test_df.columns), None)
    tier_rows = []
    if tier_col:
        for tier, tdf in test_df.groupby(tier_col):
            pre_r  = tdf[tdf["window"] < global_min_onset]
            post_r = tdf[tdf["window"] >= global_min_onset]
            def _f2(rows):
                if len(rows) == 0:
                    return float("nan")
                return macro_f1(rows["label"].values,
                    (clf.predict_proba(rows[feat_cols].values)[:, 1] >= 0.5).astype(int))
            pref, postf = _f2(pre_r), _f2(post_r)
            tier_rows.append({
                "tier": tier,
                "n_accounts": int(tdf["account_id"].nunique()),
                "pre_drift_f1":  round(pref, 4),
                "post_drift_f1": round(postf, 4),
                "f1_degradation": round(pref - postf, 4)
                    if not (np.isnan(pref) or np.isnan(postf)) else float("nan"),
            })
        tier_df = pd.DataFrame(tier_rows).sort_values("f1_degradation", ascending=False)
        tier_df.to_csv(out_dir / "per_tier_degradation.csv", index=False)
        print("Per-tier degradation:")
        print(tier_df.to_string(index=False))
        print()
    else:
        tier_df = pd.DataFrame()

    # --- Summary JSON ---------------------------------------------------
    summary = {
        "dataset_dir": str(data_dir),
        "seed": seed,
        "global_min_drift_onset": global_min_onset,
        "pre_drift_windows": f"0 to {global_min_onset - 1}",
        "n_train_accounts":  len(train_ids),
        "n_test_accounts":   len(test_ids),
        "n_train_pre_drift_rows": len(train_pre),
        "n_features": len(feat_cols),
        "temporal_mean_f1":      round(float(tmean), 4),
        "pre_drift_f1":          round(float(pre_f1),  4),
        "post_drift_f1":         round(float(post_f1), 4),
        "f1_degradation":        round(float(deg), 4) if not np.isnan(deg) else None,
        "worst_window_f1":       round(float(worst_f1), 4),
        "worst_case_degradation":round(float(worst_deg), 4) if not np.isnan(worst_deg) else None,
        "bot_prob_pre_drift":    round(float(pre_bp),  4) if not np.isnan(pre_bp)  else None,
        "bot_prob_post_drift":   round(float(post_bp), 4) if not np.isnan(post_bp) else None,
        "bot_prob_shift":        round(float(post_bp - pre_bp), 4)
                                     if not (np.isnan(pre_bp) or np.isnan(post_bp)) else None,
        "verdict": verdict,
        "feature_cols": feat_cols,
    }
    with open(out_dir / "degradation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("degradation_summary.json  [OK]")

    # --- Plots ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # F1 over time
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(pw_df["window"], pw_df["f1_macro"], "b-o", ms=5, label="Test Macro-F1")
        if global_min_onset > 0:
            ax.axvspan(-0.5, global_min_onset - 0.5, alpha=0.08, color="green",
                       label=f"Pre-drift (w0..{global_min_onset-1})")
        ax.axvline(global_min_onset - 0.5, color="red", lw=1.5, ls="--",
                   label=f"Min drift onset (w{global_min_onset})")
        if not np.isnan(pre_f1):
            ax.axhline(pre_f1, color="green", lw=1, ls=":",
                       label=f"Pre-drift mean F1={pre_f1:.4f}")
        ax.axhline(post_f1, color="orange", lw=1, ls=":",
                   label=f"Post-drift mean F1={post_f1:.4f}")
        ax.set_xlabel("Window")
        ax.set_ylabel("Macro-F1")
        ax.set_title(
            f"Frozen Static Model - BotDriftBench v0.8\n"
            f"Degradation={deg:+.4f}  |  {verdict[:40]}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        lo = max(0, pw_df["f1_macro"].min() - 0.05)
        hi = min(1, pw_df["f1_macro"].max() + 0.05)
        ax.set_ylim(lo, hi)
        fig.tight_layout()
        fig.savefig(out_dir / "frozen_model_f1_curve.png", dpi=150)
        plt.close(fig)
        print("frozen_model_f1_curve.png  [OK]")

        # Bot probability shift
        fig, ax = plt.subplots(figsize=(11, 4))
        bp_s = pw_df["mean_bot_prob"].ffill()
        hp_s = pw_df["mean_human_prob"].ffill()
        ax.plot(pw_df["window"], bp_s, "r-o", ms=5, label="Mean P(bot) for bots")
        ax.plot(pw_df["window"], hp_s, "b-o", ms=5, label="Mean P(bot) for humans")
        ax.axvline(global_min_onset - 0.5, color="gray", lw=1.5, ls="--",
                   label="Min drift onset")
        ax.set_xlabel("Window")
        ax.set_ylabel("Mean predicted P(bot)")
        ax.set_title("Bot Probability Shift Over Time - Frozen Static Model")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "bot_probability_shift.png", dpi=150)
        plt.close(fig)
        print("bot_probability_shift.png  [OK]")

        # Per-scenario bar chart
        if len(scen_df) > 0:
            sdf_plot = scen_df.dropna(subset=["f1_degradation"]).sort_values("f1_degradation")
            if len(sdf_plot) > 0:
                fig, ax = plt.subplots(figsize=(10, max(4, len(sdf_plot) * 0.6)))
                colors = ["red" if d > 0 else "steelblue" for d in sdf_plot["f1_degradation"]]
                bars = ax.barh(sdf_plot["scenario"], sdf_plot["f1_degradation"],
                               color=colors, alpha=0.75)
                ax.axvline(0, color="black", lw=0.8)
                ax.set_xlabel("F1 Degradation (pre - post,  red = classifier gets worse)")
                ax.set_title("Per-Scenario F1 Degradation - Frozen Model (v0.8)")
                for bar, val in zip(bars, sdf_plot["f1_degradation"]):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{val:+.4f}", va="center", fontsize=8)
                ax.grid(True, alpha=0.3, axis="x")
                fig.tight_layout()
                fig.savefig(out_dir / "scenario_degradation_heatmap.png", dpi=150)
                plt.close(fig)
                print("scenario_degradation_heatmap.png  [OK]")

    except ImportError:
        print("matplotlib not available; skipping plots")

    print(f"\nAll outputs saved to: {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description="BotDriftBench v0.8 frozen-model degradation diagnostic"
    )
    p.add_argument("--data",    default="dataset_v08",
                   help="Path to dataset_v08 directory")
    p.add_argument("--seed",    type=int, default=2025)
    p.add_argument("--windows", type=int, default=20)
    args = p.parse_args()
    print("=" * 60)
    print("BotDriftBench v0.8 - Degradation Diagnostic (fixed)")
    print("=" * 60)
    run_diagnostic(Path(args.data), args.seed)


if __name__ == "__main__":
    main()
