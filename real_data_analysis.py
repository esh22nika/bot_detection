"""
Real-Data Calibration: Cresci-2017
====================================

Extracts features from Cresci-2017 matching our 30-feature schema,
computes empirical distributions and correlations, and compares
against BotDriftBench v0.1 synthetic data.

Feature mapping:
  - Profile features: directly from users.csv (12 features)
  - Behavioral features: computed from tweets.csv (10 features)
  - Content features: computed from tweet text (6 features)
  - Network features: derived from profile counts (2 features)

Output:
  - dataset/cresci2017_features.csv (extracted features + labels)
  - dataset/cresci2017_vs_synthetic.csv (distribution comparison)
  - Printed report of real vs synthetic statistics
"""

import os
import sys
import re
import zipfile
import io
import math
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy, wasserstein_distance
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

CRESCI_ZIP = "dataset/cresci-2017.csv.zip"

CATEGORIES = {
    "genuine_accounts": "human",
    "social_spambots_1": "bot",
    "social_spambots_2": "bot",
    "social_spambots_3": "bot",
    "traditional_spambots_1": "bot",
    "traditional_spambots_2": "bot",
    "traditional_spambots_3": "bot",
    "traditional_spambots_4": "bot",
    "fake_followers": "bot",
}

SKIP = {"crowdflower_results", "readme", "read.me"}


def load_cresci_data(zip_path):
    """Extract users and tweets from the nested zip structure."""
    all_users = []
    all_tweets = []

    with zipfile.ZipFile(zip_path, "r") as outer:
        for entry in outer.namelist():
            basename = os.path.basename(entry).replace(".csv", "").replace(".CSV", "").replace(".zip", "").replace(".ZIP", "")

            if any(s in basename.lower() for s in SKIP):
                continue

            label = None
            for cat, lbl in CATEGORIES.items():
                if cat in basename.lower() or basename.lower() in cat:
                    label = lbl
                    break

            if label is None:
                # Try partial matching
                for cat, lbl in CATEGORIES.items():
                    if any(part in basename.lower() for part in cat.split("_")):
                        label = lbl
                        break

            if label is None:
                print(f"  Skipping unrecognized entry: {entry}")
                continue

            data = outer.read(entry)

            # Each .csv in the outer zip is itself a zip containing users.csv and tweets.csv
            try:
                with zipfile.ZipFile(io.BytesIO(data), "r") as inner:
                    inner_names = inner.namelist()
                    print(f"  {basename} ({label}): inner files = {inner_names}")

                    for inner_name in inner_names:
                        if "users" in inner_name.lower():
                            try:
                                users_df = pd.read_csv(
                                    io.BytesIO(inner.read(inner_name)),
                                    encoding="utf-8",
                                    on_bad_lines="skip",
                                )
                            except Exception:
                                users_df = pd.read_csv(
                                    io.BytesIO(inner.read(inner_name)),
                                    encoding="latin-1",
                                    on_bad_lines="skip",
                                )
                            users_df["_label"] = label
                            users_df["_category"] = basename
                            all_users.append(users_df)
                            print(f"    users: {len(users_df)} rows, cols: {list(users_df.columns[:10])}")

                        elif "tweets" in inner_name.lower() or "tweet" in inner_name.lower():
                            try:
                                tweets_df = pd.read_csv(
                                    io.BytesIO(inner.read(inner_name)),
                                    encoding="utf-8",
                                    on_bad_lines="skip",
                                )
                            except Exception:
                                tweets_df = pd.read_csv(
                                    io.BytesIO(inner.read(inner_name)),
                                    encoding="latin-1",
                                    on_bad_lines="skip",
                                )
                            tweets_df["_label"] = label
                            tweets_df["_category"] = basename
                            all_tweets.append(tweets_df)
                            print(f"    tweets: {len(tweets_df)} rows")

            except zipfile.BadZipFile:
                print(f"  {basename}: not a valid inner zip, trying as CSV...")
                try:
                    df = pd.read_csv(io.BytesIO(data), encoding="latin-1", on_bad_lines="skip")
                    if any(c in df.columns for c in ["followers_count", "friends_count"]):
                        df["_label"] = label
                        df["_category"] = basename
                        all_users.append(df)
                        print(f"    users (flat CSV): {len(df)} rows")
                    else:
                        df["_label"] = label
                        df["_category"] = basename
                        all_tweets.append(df)
                        print(f"    tweets (flat CSV): {len(df)} rows")
                except Exception as e:
                    print(f"    Could not parse: {e}")

    users = pd.concat(all_users, ignore_index=True) if all_users else pd.DataFrame()
    tweets = pd.concat(all_tweets, ignore_index=True) if all_tweets else pd.DataFrame()

    print(f"\nTotal users: {len(users)}")
    print(f"Total tweets: {len(tweets)}")

    return users, tweets


def find_user_id_col(df):
    for col in ["id", "user_id", "userid", "ID"]:
        if col in df.columns:
            return col
    return None


def extract_profile_features(users):
    """Extract the 12 profile features from users.csv."""
    features = pd.DataFrame()
    uid = find_user_id_col(users)
    if uid:
        features["user_id"] = users[uid]

    features["label"] = users["_label"]
    features["category"] = users["_category"]

    # Direct numeric fields
    for col in ["statuses_count", "followers_count", "friends_count",
                 "favourites_count", "listed_count"]:
        if col in users.columns:
            features[col] = pd.to_numeric(users[col], errors="coerce").fillna(0).astype(float)
        else:
            features[col] = 0.0

    # Account age
    if "created_at" in users.columns:
        ref_date = pd.Timestamp("2017-07-01", tz="UTC")
        parsed = pd.to_datetime(users["created_at"], errors="coerce", format="mixed", utc=True)
        features["account_age_days"] = (ref_date - parsed).dt.days.clip(lower=1).astype(float)
    else:
        features["account_age_days"] = np.nan

    # Binary features
    for col in ["default_profile", "default_profile_image", "verified", "geo_enabled"]:
        if col in users.columns:
            features[col] = users[col].apply(
                lambda x: 1.0 if str(x).strip().lower() in ("true", "1", "yes") else 0.0
            )
        else:
            features[col] = 0.0

    # has_url
    if "url" in users.columns:
        features["has_url"] = users["url"].apply(
            lambda x: 0.0 if pd.isna(x) or str(x).strip() == "" else 1.0
        )
    else:
        features["has_url"] = 0.0

    # Description length
    if "description" in users.columns:
        features["description_length"] = users["description"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        ).astype(float)
    else:
        features["description_length"] = 0.0

    # Screen name length
    if "screen_name" in users.columns:
        features["screen_name_length"] = users["screen_name"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        ).astype(float)
    else:
        features["screen_name_length"] = 0.0

    # Derived: tweets_per_day
    if "account_age_days" in features.columns:
        features["tweets_per_day"] = features["statuses_count"] / features["account_age_days"].clip(lower=1)
    else:
        features["tweets_per_day"] = 0.0

    # Network features
    features["follower_friend_ratio"] = (
        features["followers_count"] / features["friends_count"].clip(lower=1)
    )
    features["reputation_score"] = (
        features["followers_count"] /
        (features["followers_count"] + features["friends_count"]).clip(lower=1)
    )

    return features


def extract_tweet_features(tweets, user_features):
    """Compute behavioral and content features from tweets, per user."""
    uid_col = None
    for col in ["user_id", "userid", "id", "ID"]:
        if col in tweets.columns:
            uid_col = col
            break

    if uid_col is None:
        print("WARNING: No user_id column found in tweets. Skipping tweet features.")
        return user_features

    text_col = None
    for col in ["text", "tweet_text", "full_text", "content"]:
        if col in tweets.columns:
            text_col = col
            break

    tweet_time_col = None
    for col in ["created_at", "timestamp", "date"]:
        if col in tweets.columns:
            tweet_time_col = col
            break

    # Group tweets by user
    grouped = tweets.groupby(uid_col)

    results = {}
    for user_id, group in grouped:
        n = len(group)
        row = {}

        # Behavioral features from tweet metadata
        if "retweeted_status_id" in group.columns:
            row["retweet_ratio"] = group["retweeted_status_id"].notna().mean() if n > 0 else 0.0
        elif "retweeted" in group.columns:
            rt = group["retweeted"].apply(
                lambda x: 1 if str(x).strip().lower() in ("true", "1") else 0
            )
            row["retweet_ratio"] = rt.mean() if n > 0 else 0.0
        elif "retweet_count" in group.columns:
            row["retweet_ratio"] = (pd.to_numeric(group["retweet_count"], errors="coerce").fillna(0) > 0).mean()
        else:
            row["retweet_ratio"] = 0.0

        # Reply ratio (tweets starting with @)
        if text_col:
            texts = group[text_col].dropna().astype(str)
            row["reply_ratio"] = texts.apply(lambda t: 1 if t.startswith("@") else 0).mean() if len(texts) > 0 else 0.0

            # Mention count
            row["mention_count_avg"] = texts.apply(lambda t: len(re.findall(r"@\w+", t))).mean() if len(texts) > 0 else 0.0

            # Hashtag count
            row["hashtag_count_avg"] = texts.apply(lambda t: len(re.findall(r"#\w+", t))).mean() if len(texts) > 0 else 0.0

            # URL share ratio
            row["url_share_ratio"] = texts.apply(
                lambda t: 1 if re.search(r"https?://", t) else 0
            ).mean() if len(texts) > 0 else 0.0

            # Content features
            lengths = texts.apply(len)
            row["avg_tweet_length"] = lengths.mean() if len(texts) > 0 else 0.0

            # Vocabulary richness (type-token ratio)
            all_words = " ".join(texts).lower().split()
            if len(all_words) > 0:
                row["vocabulary_richness"] = min(len(set(all_words)) / len(all_words), 1.0)
            else:
                row["vocabulary_richness"] = 0.0

            # Text entropy
            char_counts = Counter(" ".join(texts).lower())
            total_chars = sum(char_counts.values())
            if total_chars > 0:
                probs = np.array([c / total_chars for c in char_counts.values()])
                row["text_entropy"] = float(entropy(probs, base=2))
            else:
                row["text_entropy"] = 0.0

            # Emoji ratio
            emoji_pattern = re.compile(
                "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
                "\U00002702-\U000027B0\U0000FE00-\U0000FE0F]"
            )
            row["emoji_ratio"] = texts.apply(
                lambda t: len(emoji_pattern.findall(t)) / max(len(t), 1)
            ).mean() if len(texts) > 0 else 0.0

            # Caps ratio
            row["caps_ratio"] = texts.apply(
                lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
            ).mean() if len(texts) > 0 else 0.0

            # Content similarity (pairwise Jaccard on word sets, sample for speed)
            if len(texts) >= 2:
                sample_texts = texts.sample(min(50, len(texts)), random_state=42).values
                word_sets = [set(t.lower().split()) for t in sample_texts]
                sim_sum, sim_count = 0.0, 0
                for i in range(len(word_sets)):
                    for j in range(i + 1, min(i + 5, len(word_sets))):
                        union = word_sets[i] | word_sets[j]
                        if len(union) > 0:
                            sim_sum += len(word_sets[i] & word_sets[j]) / len(union)
                            sim_count += 1
                row["content_similarity"] = sim_sum / sim_count if sim_count > 0 else 0.0
            else:
                row["content_similarity"] = 0.0
        else:
            for feat in ["reply_ratio", "mention_count_avg", "hashtag_count_avg",
                         "url_share_ratio", "avg_tweet_length", "vocabulary_richness",
                         "text_entropy", "emoji_ratio", "caps_ratio", "content_similarity"]:
                row[feat] = np.nan

        # Temporal features
        if tweet_time_col:
            try:
                times = pd.to_datetime(group[tweet_time_col], errors="coerce").dropna()
                if len(times) >= 2:
                    times_sorted = times.sort_values()
                    intervals = times_sorted.diff().dt.total_seconds().dropna()
                    if len(intervals) > 0 and intervals.std() > 0:
                        row["posting_regularity"] = min(1.0 / (1.0 + intervals.std() / 3600), 1.0)
                    else:
                        row["posting_regularity"] = 1.0

                    hours = times.dt.hour
                    row["night_post_ratio"] = ((hours >= 0) & (hours < 6)).mean()
                    row["weekend_ratio"] = (times.dt.dayofweek >= 5).mean()

                    # Burst score: max tweets in any 1-hour window / total
                    times_sorted_idx = times_sorted.reset_index(drop=True)
                    if len(times_sorted_idx) >= 5:
                        hour_counts = times.dt.floor("h").value_counts()
                        row["burst_score"] = float(hour_counts.max()) / len(times)
                    else:
                        row["burst_score"] = 0.0
                else:
                    row["posting_regularity"] = 0.5
                    row["night_post_ratio"] = 0.0
                    row["weekend_ratio"] = 0.0
                    row["burst_score"] = 0.0
            except Exception:
                row["posting_regularity"] = np.nan
                row["night_post_ratio"] = np.nan
                row["weekend_ratio"] = np.nan
                row["burst_score"] = np.nan
        else:
            row["posting_regularity"] = np.nan
            row["night_post_ratio"] = np.nan
            row["weekend_ratio"] = np.nan
            row["burst_score"] = np.nan

        results[user_id] = row

    tweet_df = pd.DataFrame.from_dict(results, orient="index")
    tweet_df.index.name = "user_id"

    # Merge with user features
    uid_feat = "user_id" if "user_id" in user_features.columns else None
    if uid_feat:
        merged = user_features.merge(tweet_df, left_on="user_id", right_index=True, how="left")
    else:
        merged = user_features.copy()
        for col in tweet_df.columns:
            merged[col] = np.nan

    return merged


def compare_distributions(real_df, synthetic_path):
    """Compare real vs synthetic feature distributions."""
    syn_df = pd.read_csv(synthetic_path)
    syn_w0 = syn_df[syn_df["time_window"] == 0].copy()

    our_features = [
        "account_age_days", "statuses_count", "followers_count", "friends_count",
        "favourites_count", "listed_count", "default_profile", "default_profile_image",
        "has_url", "verified", "description_length", "screen_name_length",
        "tweets_per_day", "retweet_ratio", "reply_ratio", "mention_count_avg",
        "hashtag_count_avg", "url_share_ratio", "posting_regularity",
        "night_post_ratio", "weekend_ratio", "burst_score",
        "avg_tweet_length", "vocabulary_richness", "text_entropy",
        "emoji_ratio", "caps_ratio", "content_similarity",
        "follower_friend_ratio", "reputation_score",
    ]

    results = []
    print("\n" + "=" * 100)
    print("REAL vs SYNTHETIC DISTRIBUTION COMPARISON")
    print("=" * 100)
    print(f"\n{'Feature':<25} {'Real Human':>12} {'Real Bot':>12} {'Syn Human':>12} "
          f"{'Syn Bot':>12} {'KS(H)':>8} {'KS(B)':>8} {'Match':>6}")
    print("-" * 100)

    for feat in our_features:
        if feat not in real_df.columns:
            print(f"{feat:<25} {'N/A':>12}")
            continue

        r_human = real_df[real_df["label"] == "human"][feat].dropna()
        r_bot = real_df[real_df["label"] == "bot"][feat].dropna()

        if feat not in syn_w0.columns:
            print(f"{feat:<25} real only (not in synthetic)")
            continue

        s_human = syn_w0[syn_w0["label"] == "human"][feat].dropna()
        s_bot = syn_w0[syn_w0["label"] == "bot"][feat].dropna()

        if len(r_human) < 5 or len(r_bot) < 5:
            continue

        ks_h = ks_2samp(r_human, s_human).statistic if len(s_human) > 5 else np.nan
        ks_b = ks_2samp(r_bot, s_bot).statistic if len(s_bot) > 5 else np.nan

        match = "OK" if (ks_h < 0.3 and ks_b < 0.3) else "DIFF"

        print(f"{feat:<25} {r_human.median():>12.2f} {r_bot.median():>12.2f} "
              f"{s_human.median():>12.2f} {s_bot.median():>12.2f} "
              f"{ks_h:>8.3f} {ks_b:>8.3f} {match:>6}")

        results.append({
            "feature": feat,
            "real_human_median": round(float(r_human.median()), 4),
            "real_human_std": round(float(r_human.std()), 4),
            "real_bot_median": round(float(r_bot.median()), 4),
            "real_bot_std": round(float(r_bot.std()), 4),
            "syn_human_median": round(float(s_human.median()), 4),
            "syn_bot_median": round(float(s_bot.median()), 4),
            "ks_human": round(float(ks_h), 4) if not np.isnan(ks_h) else None,
            "ks_bot": round(float(ks_b), 4) if not np.isnan(ks_b) else None,
        })

    return pd.DataFrame(results)


def compute_correlations(df, label):
    """Compute correlation matrix for numeric features."""
    subset = df[df["label"] == label]
    numeric_cols = [c for c in subset.columns if subset[c].dtype in [np.float64, np.int64]
                    and c not in ["user_id", "time_window"]]
    return subset[numeric_cols].corr()


def run_analysis():
    if not os.path.exists(CRESCI_ZIP):
        print(f"ERROR: {CRESCI_ZIP} not found.")
        print("Download from: https://botometer.osome.iu.edu/bot-repository/datasets.html")
        sys.exit(1)

    print("=" * 70)
    print("CRESCI-2017 REAL DATA ANALYSIS")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1] Loading Cresci-2017 from zip...")
    users, tweets = load_cresci_data(CRESCI_ZIP)

    print(f"\nUsers columns: {list(users.columns)}")
    print(f"Tweets columns: {list(tweets.columns[:15])}")

    # Label distribution
    print(f"\nUser label distribution:")
    for cat in users["_category"].unique():
        sub = users[users["_category"] == cat]
        print(f"  {cat}: {len(sub)} ({sub['_label'].iloc[0]})")

    # Step 2: Extract profile features
    print("\n[2] Extracting profile features...")
    features = extract_profile_features(users)
    print(f"  Profile features extracted: {len(features)} users")

    # Step 3: Extract tweet features
    print("\n[3] Extracting tweet-derived features (this may take a while)...")
    if len(tweets) > 0:
        features = extract_tweet_features(tweets, features)
        print(f"  Tweet features merged: {features.shape}")
    else:
        print("  No tweets loaded, skipping tweet features")

    # Step 4: Summary statistics
    print("\n[4] Feature summary by label:")
    numeric_feats = [c for c in features.columns
                     if features[c].dtype in [np.float64, np.int64]
                     and c not in ["user_id"]]

    for label in ["human", "bot"]:
        subset = features[features["label"] == label]
        print(f"\n  === {label.upper()} (n={len(subset)}) ===")
        for feat in sorted(numeric_feats):
            vals = subset[feat].dropna()
            if len(vals) > 0:
                print(f"    {feat:<25} median={vals.median():>10.2f}  "
                      f"mean={vals.mean():>10.2f}  std={vals.std():>10.2f}")

    # Step 5: Save features
    out_path = "dataset/cresci2017_features.csv"
    features.to_csv(out_path, index=False)
    print(f"\n[5] Features saved to {out_path} ({len(features)} rows)")

    # Step 6: Compare with synthetic
    syn_path = "dataset/botdriftbench_v0.1.csv"
    if os.path.exists(syn_path):
        print("\n[6] Comparing real vs synthetic distributions...")
        comparison = compare_distributions(features, syn_path)
        comparison.to_csv("dataset/cresci2017_vs_synthetic.csv", index=False)
        print(f"\n  Comparison saved to dataset/cresci2017_vs_synthetic.csv")

        # Summary
        n_ok = (comparison["ks_human"].dropna() < 0.3).sum()
        n_diff = (comparison["ks_human"].dropna() >= 0.3).sum()
        print(f"\n  Distribution match (KS < 0.3): {n_ok} features")
        print(f"  Distribution mismatch (KS >= 0.3): {n_diff} features")
        print(f"  -> These mismatches identify where v0.2 calibration is needed")
    else:
        print(f"\n[6] Synthetic data not found at {syn_path}, skipping comparison")

    # Step 7: Per-category archetype analysis
    print("\n[7] Per-category archetype analysis (real behavioral archetypes)...")
    print("=" * 100)

    key_features = [
        "followers_count", "friends_count", "statuses_count", "tweets_per_day",
        "retweet_ratio", "reply_ratio", "url_share_ratio", "posting_regularity",
        "burst_score", "avg_tweet_length", "vocabulary_richness", "content_similarity",
        "follower_friend_ratio", "reputation_score", "description_length",
        "account_age_days", "has_url", "verified",
    ]
    available_feats = [f for f in key_features if f in features.columns]

    categories = sorted(features["category"].dropna().unique())
    cat_profiles = {}

    print(f"\n{'Category':<28} {'N':>5}", end="")
    for feat in available_feats[:12]:
        print(f" {feat[:12]:>12}", end="")
    print()
    print("-" * 180)

    for cat in categories:
        sub = features[features["category"] == cat]
        label = sub["label"].iloc[0] if len(sub) > 0 else "?"
        n = len(sub)

        profile = {}
        print(f"{cat:<28} {n:>5}", end="")
        for feat in available_feats[:12]:
            vals = sub[feat].dropna()
            med = vals.median() if len(vals) > 0 else np.nan
            profile[feat] = {
                "median": float(med) if not np.isnan(med) else None,
                "mean": float(vals.mean()) if len(vals) > 0 else None,
                "std": float(vals.std()) if len(vals) > 0 else None,
            }
            print(f" {med:>12.2f}" if not np.isnan(med) else f" {'N/A':>12}", end="")
        print(f"  ({label})")

        cat_profiles[cat] = profile

    # Remaining features
    print(f"\n{'Category':<28} {'N':>5}", end="")
    for feat in available_feats[12:]:
        print(f" {feat[:15]:>15}", end="")
    print()
    print("-" * 120)

    for cat in categories:
        sub = features[features["category"] == cat]
        label = sub["label"].iloc[0] if len(sub) > 0 else "?"
        print(f"{cat:<28} {len(sub):>5}", end="")
        for feat in available_feats[12:]:
            vals = sub[feat].dropna()
            med = vals.median() if len(vals) > 0 else np.nan
            print(f" {med:>15.2f}" if not np.isnan(med) else f" {'N/A':>15}", end="")
        print(f"  ({label})")

    # Inter-category KS distances (how different are the bot types from each other?)
    bot_cats = [c for c in categories if features[features["category"] == c]["label"].iloc[0] == "bot"]
    human_cats = [c for c in categories if features[features["category"] == c]["label"].iloc[0] == "human"]

    if len(bot_cats) >= 2:
        print(f"\n  Inter-category KS distances (bot types vs each other):")
        print(f"  {'Cat1':<25} {'Cat2':<25}", end="")
        for feat in available_feats[:8]:
            print(f" {feat[:10]:>10}", end="")
        print()

        for i, c1 in enumerate(bot_cats):
            for c2 in bot_cats[i+1:]:
                s1 = features[features["category"] == c1]
                s2 = features[features["category"] == c2]
                print(f"  {c1:<25} {c2:<25}", end="")
                for feat in available_feats[:8]:
                    v1 = s1[feat].dropna()
                    v2 = s2[feat].dropna()
                    if len(v1) >= 5 and len(v2) >= 5:
                        ks = ks_2samp(v1, v2).statistic
                        print(f" {ks:>10.3f}", end="")
                    else:
                        print(f" {'N/A':>10}", end="")
                print()

    # Step 8: Correlation comparison
    print("\n[8] Computing correlation structure...")
    for label in ["human", "bot"]:
        corr = compute_correlations(features, label)
        if len(corr) > 0:
            pairs = []
            for i in range(len(corr)):
                for j in range(i + 1, len(corr)):
                    if not np.isnan(corr.iloc[i, j]):
                        pairs.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            print(f"\n  Top 10 correlations ({label}):")
            for f1, f2, r in pairs[:10]:
                print(f"    {f1} <-> {f2}: {r:.3f}")

    # Per-category correlation structure
    print("\n  Per-category top correlations:")
    for cat in bot_cats[:4]:
        sub = features[features["category"] == cat]
        numeric_cols = [c for c in available_feats if sub[c].dtype in [np.float64, np.int64]]
        if len(sub) < 10 or len(numeric_cols) < 3:
            continue
        corr = sub[numeric_cols].corr()
        pairs = []
        for i in range(len(corr)):
            for j in range(i + 1, len(corr)):
                if not np.isnan(corr.iloc[i, j]):
                    pairs.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n    {cat} (n={len(sub)}):")
        for f1, f2, r in pairs[:5]:
            print(f"      {f1} <-> {f2}: {r:.3f}")

    # Step 9: Category separability (pairwise XGBoost macro-F1 matrix)
    print("\n[9] Category separability (pairwise XGBoost macro-F1)...")
    print("=" * 80)

    sep_matrix = {}
    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            if i >= j:
                continue
            s1 = features[features["category"] == c1][available_feats].dropna(axis=1, how="all")
            s2 = features[features["category"] == c2][available_feats].dropna(axis=1, how="all")
            common_cols = list(set(s1.columns) & set(s2.columns))
            if len(common_cols) < 3 or len(s1) < 10 or len(s2) < 10:
                continue

            combined = pd.concat([s1[common_cols], s2[common_cols]], ignore_index=True)
            for col in common_cols:
                combined[col] = combined[col].fillna(combined[col].median())
            X = combined.values
            y = np.array([0]*len(s1) + [1]*len(s2))

            rng = np.random.RandomState(42)
            idx = rng.permutation(len(X))
            split = int(0.7 * len(X))
            X_tr, X_te = X[idx[:split]], X[idx[split:]]
            y_tr, y_te = y[idx[:split]], y[idx[split:]]

            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            try:
                model = XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, verbosity=0
                ).fit(X_tr, y_tr)
                macro_f1 = f1_score(y_te, model.predict(X_te), average="macro")
                sep_matrix[(c1, c2)] = macro_f1
                sep_matrix[(c2, c1)] = macro_f1
            except Exception:
                continue

    # Print as matrix
    short_names = {c: c[:15] for c in categories}
    print(f"\n  {'':>16}", end="")
    for c in categories:
        print(f" {short_names[c]:>15}", end="")
    print()

    for c1 in categories:
        print(f"  {short_names[c1]:>15}", end="")
        for c2 in categories:
            if c1 == c2:
                print(f" {'---':>15}", end="")
            elif (c1, c2) in sep_matrix:
                f1_val = sep_matrix[(c1, c2)]
                print(f" {f1_val:>15.3f}", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()

    # Also print sorted list
    print(f"\n  Sorted by separability:")
    pairs_sorted = sorted(set((min(k), max(k), v) for k, v in sep_matrix.items()),
                          key=lambda x: x[2], reverse=True)
    for c1, c2, f1_val in pairs_sorted:
        sep = "easy" if f1_val > 0.9 else "moderate" if f1_val > 0.7 else "hard"
        print(f"    {c1:<25} vs {c2:<25} macro-F1={f1_val:.3f}  ({sep})")

    # Step 10: Archetype matching (real category -> closest synthetic archetype)
    print("\n[10] Archetype matching: real category -> synthetic archetype...")
    print("=" * 80)

    syn_path = "dataset/botdriftbench_v0.1.csv"
    if os.path.exists(syn_path):
        syn_df = pd.read_csv(syn_path)
        syn_w0 = syn_df[syn_df["time_window"] == 0]

        if "archetype" in syn_w0.columns:
            syn_archetypes = syn_w0["archetype"].unique()
            match_feats = [f for f in available_feats
                           if f in syn_w0.columns and f not in ["verified", "has_url", "default_profile", "default_profile_image"]]

            print(f"\n  Matching on {len(match_feats)} continuous features")
            print(f"  Synthetic archetypes: {list(syn_archetypes)}")

            alpha = 0.7  # weight for marginal vs correlation similarity

            for cat in categories:
                real_sub = features[features["category"] == cat]
                if len(real_sub) < 5:
                    continue

                # Real correlation matrix
                real_vals = real_sub[match_feats].dropna(axis=1, how="all")
                real_avail = [f for f in match_feats if f in real_vals.columns]
                real_corr = real_sub[real_avail].corr() if len(real_avail) >= 3 else None

                best_arch = None
                best_dist = float("inf")
                arch_details = {}

                for arch in syn_archetypes:
                    syn_sub = syn_w0[syn_w0["archetype"] == arch]
                    if len(syn_sub) < 5:
                        continue

                    # Marginal similarity (mean KS)
                    total_ks = 0
                    n_compared = 0
                    for feat in real_avail:
                        rv = real_sub[feat].dropna()
                        sv = syn_sub[feat].dropna() if feat in syn_sub.columns else pd.Series(dtype=float)
                        if len(rv) >= 5 and len(sv) >= 5:
                            total_ks += ks_2samp(rv, sv).statistic
                            n_compared += 1

                    d_marginal = total_ks / max(n_compared, 1)

                    # Correlation similarity (Frobenius norm of difference)
                    d_corr = np.nan
                    if real_corr is not None and len(real_avail) >= 3:
                        syn_corr_feats = [f for f in real_avail if f in syn_sub.columns]
                        if len(syn_corr_feats) >= 3:
                            syn_corr = syn_sub[syn_corr_feats].corr()
                            common_corr = list(set(real_corr.columns) & set(syn_corr.columns))
                            if len(common_corr) >= 3:
                                rc = real_corr.loc[common_corr, common_corr].values
                                sc = syn_corr.loc[common_corr, common_corr].values
                                rc = np.nan_to_num(rc, nan=0.0)
                                sc = np.nan_to_num(sc, nan=0.0)
                                d_corr = np.linalg.norm(rc - sc, "fro") / len(common_corr)

                    if np.isnan(d_corr):
                        d_total = d_marginal
                    else:
                        d_total = alpha * d_marginal + (1 - alpha) * d_corr

                    arch_details[arch] = {
                        "d_marginal": round(d_marginal, 4),
                        "d_corr": round(d_corr, 4) if not np.isnan(d_corr) else None,
                        "d_total": round(d_total, 4),
                    }
                    if d_total < best_dist:
                        best_dist = d_total
                        best_arch = arch

                label = real_sub["label"].iloc[0]
                print(f"\n  {cat} ({label}, n={len(real_sub)}):")
                print(f"    Best match: {best_arch} (D_total={best_dist:.4f})")
                sorted_archs = sorted(arch_details.items(), key=lambda x: x[1]["d_total"])
                for arch, d in sorted_archs[:5]:
                    marker = " <-- best" if arch == best_arch else ""
                    corr_str = f"{d['d_corr']:.4f}" if d["d_corr"] is not None else "N/A"
                    print(f"      {arch:<25} marginal={d['d_marginal']:.4f}  "
                          f"corr={corr_str}  total={d['d_total']:.4f}{marker}")

                if best_dist > 0.5:
                    print(f"    WARNING: No close synthetic match (D_total={best_dist:.4f} > 0.5)")
                    print(f"    -> Consider adding a new synthetic archetype for this category")
        else:
            print("  No 'archetype' column in synthetic data, skipping matching")
    else:
        print(f"  Synthetic data not found at {syn_path}, skipping")

    # Step 11: Feature availability per category
    print("\n[11] Feature availability per category...")
    print("=" * 80)
    all_feats = available_feats
    print(f"\n  {'Category':<28} {'N':>5} {'feats_available':>16} {'tweet_feats':>12}")
    print("  " + "-" * 65)

    tweet_derived = ["retweet_ratio", "reply_ratio", "mention_count_avg", "hashtag_count_avg",
                     "url_share_ratio", "posting_regularity", "night_post_ratio", "weekend_ratio",
                     "burst_score", "avg_tweet_length", "vocabulary_richness", "text_entropy",
                     "emoji_ratio", "caps_ratio", "content_similarity"]

    for cat in categories:
        sub = features[features["category"] == cat]
        n_avail = sum(1 for f in all_feats if sub[f].notna().sum() > len(sub) * 0.1)
        n_tweet = sum(1 for f in tweet_derived if f in sub.columns and sub[f].notna().sum() > len(sub) * 0.1)
        print(f"  {cat:<28} {len(sub):>5} {n_avail:>16} {n_tweet:>12}")

    # Save per-category profiles
    print("\n[12] Saving all outputs...")
    profile_rows = []
    for cat in categories:
        sub = features[features["category"] == cat]
        label = sub["label"].iloc[0] if len(sub) > 0 else "?"
        row = {"category": cat, "label": label, "n_accounts": len(sub)}
        for feat in available_feats:
            vals = sub[feat].dropna()
            if len(vals) > 0:
                row[f"{feat}_median"] = round(float(vals.median()), 4)
                row[f"{feat}_mean"] = round(float(vals.mean()), 4)
                row[f"{feat}_std"] = round(float(vals.std()), 4)
                row[f"{feat}_q25"] = round(float(vals.quantile(0.25)), 4)
                row[f"{feat}_q75"] = round(float(vals.quantile(0.75)), 4)
        profile_rows.append(row)

    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv("dataset/cresci2017_category_profiles.csv", index=False)
    print(f"  Category profiles saved to dataset/cresci2017_category_profiles.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey outputs:")
    print(f"  dataset/cresci2017_features.csv      ({len(features)} accounts, all features)")
    print(f"  dataset/cresci2017_vs_synthetic.csv   (real vs synthetic comparison)")
    print(f"  dataset/cresci2017_category_profiles.csv (per-category statistics)")


if __name__ == "__main__":
    run_analysis()
