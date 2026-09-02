"""
Calibrate BotDriftBench v0.2 generator parameters from Cresci-2017 empirical data.

Reads dataset/cresci2017_features.csv (output of real_data_analysis.py),
fits distribution parameters for each feature, and outputs the calibrated
INIT_DISTRIBUTIONS dict ready to paste into generate_dataset.py.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
import json

warnings.filterwarnings("ignore")

FEATURES_CSV = "dataset/cresci2017_features.csv"

DIST_SPEC = {
    "account_age_days":      "lognormal",
    "statuses_count":        "lognormal",
    "followers_count":       "lognormal",
    "friends_count":         "lognormal",
    "favourites_count":      "lognormal",
    "listed_count":          "lognormal",
    "default_profile":       "bernoulli",
    "default_profile_image": "bernoulli",
    "has_url":               "bernoulli",
    "verified":              "bernoulli",
    "description_length":    "normal",
    "screen_name_length":    "normal",
    "tweets_per_day":        "lognormal",
    "retweet_ratio":         "beta",
    "reply_ratio":           "beta",
    "mention_count_avg":     "lognormal",
    "hashtag_count_avg":     "lognormal",
    "url_share_ratio":       "beta",
    "posting_regularity":    "beta",
    "night_post_ratio":      "beta",
    "weekend_ratio":         "normal",
    "burst_score":           "lognormal",
    "avg_tweet_length":      "normal",
    "vocabulary_richness":   "beta",
    "text_entropy":          "normal",
    "emoji_ratio":           "exponential",
    "caps_ratio":            "beta",
    "content_similarity":    "beta",
    "follower_friend_ratio": "lognormal",
    "reputation_score":      "beta",
}


def fit_lognormal(data):
    """Fit lognormal(mu, sigma) to positive data."""
    pos = data[data > 0]
    if len(pos) < 10:
        return (0.0, 1.0)
    log_data = np.log(pos)
    mu = float(np.mean(log_data))
    sigma = float(np.std(log_data, ddof=1))
    sigma = max(sigma, 0.1)
    return (round(mu, 2), round(sigma, 2))


def fit_beta(data):
    """Fit beta(a, b) to data in [0, 1]."""
    clipped = np.clip(data, 0.001, 0.999)
    clipped = clipped[(clipped > 0.001) & (clipped < 0.999)]
    if len(clipped) < 10:
        mean = float(np.mean(data))
        if mean <= 0 or mean >= 1:
            return (2.0, 2.0)
        var = max(float(np.var(data)), 0.001)
        common = mean * (1 - mean) / var - 1
        common = max(common, 2.0)
        return (round(mean * common, 2), round((1 - mean) * common, 2))
    a, b, loc, scale = stats.beta.fit(clipped, floc=0, fscale=1)
    a = max(round(a, 2), 0.5)
    b = max(round(b, 2), 0.5)
    return (a, b)


def fit_normal(data):
    """Fit normal(mu, sigma) to data."""
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    sigma = max(sigma, 0.1)
    return (round(mu, 1), round(sigma, 1))


def fit_exponential(data):
    """Fit exponential(lambda) — parameterized as mean."""
    pos = data[data > 0]
    if len(pos) < 10:
        return (float(np.mean(data)),)
    lam = float(np.mean(pos))
    return (round(lam, 4),)


def fit_bernoulli(data):
    """Fit bernoulli(p) to binary data."""
    p = float(np.mean(data))
    return (round(p, 4),)


FITTERS = {
    "lognormal": fit_lognormal,
    "beta": fit_beta,
    "normal": fit_normal,
    "exponential": fit_exponential,
    "bernoulli": fit_bernoulli,
}


def ks_check(data, dist_type, params):
    """Compute KS statistic between data and the fitted distribution."""
    n = min(len(data), 5000)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(data), n, replace=False)
    sample = data.values[sample_idx] if hasattr(data, 'values') else data[sample_idx]

    if dist_type == "lognormal":
        synthetic = rng.lognormal(params[0], params[1], n)
    elif dist_type == "beta":
        synthetic = rng.beta(params[0], params[1], n)
    elif dist_type == "normal":
        synthetic = rng.normal(params[0], params[1], n)
    elif dist_type == "exponential":
        synthetic = rng.exponential(params[0], n)
    elif dist_type == "bernoulli":
        synthetic = rng.binomial(1, params[0], n).astype(float)
    else:
        return 1.0

    ks_stat, _ = stats.ks_2samp(sample, synthetic)
    return round(ks_stat, 4)


def classify_overlap(ks_stat):
    if ks_stat < 0.10:
        return "near-identical"
    elif ks_stat < 0.20:
        return "heavy overlap"
    elif ks_stat < 0.35:
        return "moderate overlap"
    else:
        return "distinct"


def main():
    print("=" * 70)
    print("BotDriftBench v0.2 — Distribution Calibration from Cresci-2017")
    print("=" * 70)

    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df):,} accounts from {FEATURES_CSV}")
    print(f"  Humans: {(df['label'] == 'human').sum():,}")
    print(f"  Bots:   {(df['label'] == 'bot').sum():,}")
    print()

    humans = df[df["label"] == "human"]
    bots = df[df["label"] == "bot"]

    results = {}
    print(f"{'Feature':<25} {'Dist':<12} {'Human Params':<20} {'Bot Params':<20} "
          f"{'KS_h':<8} {'KS_b':<8} {'H-B overlap':<15}")
    print("-" * 110)

    for feat, dist_type in DIST_SPEC.items():
        if feat not in df.columns:
            print(f"  {feat:<25} MISSING")
            continue

        h_data = humans[feat].dropna()
        b_data = bots[feat].dropna()

        if len(h_data) < 10 or len(b_data) < 10:
            print(f"  {feat:<25} insufficient data (h={len(h_data)}, b={len(b_data)})")
            continue

        fitter = FITTERS[dist_type]
        h_params = fitter(h_data.values)
        b_params = fitter(b_data.values)

        ks_h = ks_check(h_data, dist_type, h_params)
        ks_b = ks_check(b_data, dist_type, b_params)

        # Compute human-vs-bot separability via KS
        ks_hb, _ = stats.ks_2samp(
            h_data.sample(min(2000, len(h_data)), random_state=42).values,
            b_data.sample(min(2000, len(b_data)), random_state=42).values,
        )
        overlap = classify_overlap(ks_hb)

        results[feat] = {
            "dist_type": dist_type,
            "human_params": h_params,
            "bot_params": b_params,
            "ks_fit_human": ks_h,
            "ks_fit_bot": ks_b,
            "ks_human_vs_bot": round(ks_hb, 4),
            "overlap": overlap,
            "evidence": f"Cresci 2017 ({overlap})",
        }

        h_str = str(h_params)
        b_str = str(b_params)
        print(f"  {feat:<25} {dist_type:<12} {h_str:<20} {b_str:<20} "
              f"{ks_h:<8} {ks_b:<8} {overlap:<15}")

    # Generate the Python dict output
    print("\n\n" + "=" * 70)
    print("CALIBRATED INIT_DISTRIBUTIONS (copy into generate_dataset.py)")
    print("=" * 70)
    print()
    print("INIT_DISTRIBUTIONS = {")

    groups = {
        "Profile": ["account_age_days", "statuses_count", "followers_count",
                     "friends_count", "favourites_count", "listed_count",
                     "default_profile", "default_profile_image", "has_url",
                     "verified", "description_length", "screen_name_length"],
        "Behavioral": ["tweets_per_day", "retweet_ratio", "reply_ratio",
                        "mention_count_avg", "hashtag_count_avg", "url_share_ratio",
                        "posting_regularity", "night_post_ratio", "weekend_ratio",
                        "burst_score"],
        "Content": ["avg_tweet_length", "vocabulary_richness", "text_entropy",
                     "emoji_ratio", "caps_ratio", "content_similarity"],
        "Network": ["follower_friend_ratio", "reputation_score"],
    }

    for group_name, feats in groups.items():
        print(f"    # --- {group_name} ---")
        for feat in feats:
            if feat not in results:
                continue
            r = results[feat]
            h = r["human_params"]
            b = r["bot_params"]
            ev = r["evidence"]
            padding = max(1, 25 - len(feat))
            print(f'    "{feat}":{" " * padding}'
                  f'("{r["dist_type"]}", {h},  {b},  "{ev}"),')
        print()

    print("}")

    # Save calibration report
    rows = []
    for feat, r in results.items():
        rows.append({
            "feature": feat,
            "dist_type": r["dist_type"],
            "human_params": str(r["human_params"]),
            "bot_params": str(r["bot_params"]),
            "ks_fit_human": r["ks_fit_human"],
            "ks_fit_bot": r["ks_fit_bot"],
            "ks_human_vs_bot": r["ks_human_vs_bot"],
            "overlap": r["overlap"],
        })
    report_df = pd.DataFrame(rows)
    report_df.to_csv("dataset/v02_calibration_report.csv", index=False)
    print(f"\nSaved calibration report to dataset/v02_calibration_report.csv")

    # Summary statistics
    good_fits = sum(1 for r in results.values()
                    if r["ks_fit_human"] < 0.3 and r["ks_fit_bot"] < 0.3)
    total = len(results)
    print(f"\nFit quality: {good_fits}/{total} features have KS < 0.3 for both classes")

    overlaps = {}
    for r in results.values():
        cat = r["overlap"]
        overlaps[cat] = overlaps.get(cat, 0) + 1
    print("Human-bot separability:")
    for cat in ["near-identical", "heavy overlap", "moderate overlap", "distinct"]:
        print(f"  {cat}: {overlaps.get(cat, 0)} features")


if __name__ == "__main__":
    main()
