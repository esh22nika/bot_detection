"""
BotDriftBench v0.2 — Drift Injection Benchmark for Social Bot Detection
========================================================================

Generates longitudinal synthetic account trajectories with mathematically
defined behavioral drift for evaluating drift-aware bot detection systems.

Design principles (addressing review feedback):
  1. TRUE TRAJECTORIES: each account carries persistent state forward via
     a Markov process — X_{i,t} = X_{i,t-1} + drift + noise
  2. HUMANS EVOLVE TOO: human accounts undergo natural behavioral variation
     so drift detection cannot shortcut on "anything different = bot"
  3. DRIFT IS PARAMETERIZED: each drift scenario defines a shift magnitude δ
     as an experimental variable, controllable across runs
  4. MATHEMATICAL FORMALIZATION: drift is defined as a change in the
     conditional distribution P_t(X|Y) at a known change point τ

Mathematical framework:
  For feature j of account i at time t:

    X_{i,j,t} = X_{i,j,t-1} + μ_{evolution,j} + ε_{i,j,t}

  where ε ~ N(0, σ_{natural,j}) provides organic variation.

  For bot accounts under drift scenario s, after change point τ:

    X_{i,j,t} = X_{i,j,t-1} + μ_{evolution,j} + δ_{s,j} · Δ_t + ε_{i,j,t}

  where δ_{s,j} is the per-feature drift magnitude and Δ_t is the
  time-step contribution (1/(τ_end - τ_start) for gradual drift, or
  a full step for sudden/recurring drift).

  Drift magnitude δ is an experimental parameter:
    δ ∈ {0.1, 0.25, 0.5, 1.0} × base_drift_vector

Usage:
    python generate_dataset.py                          # v0.2: 1000 accounts
    python generate_dataset.py --accounts 10000         # full scale
    python generate_dataset.py --drift-magnitude 0.5    # weaker drift
    python generate_dataset.py --drift-magnitude 2.0    # stronger drift

References:
  Cresci et al. (2017) — profile feature distributions
  Yang et al. (TwiBot-20, 2020) — behavioral feature ranges
  Feng et al. (TwiBot-22, 2022) — large-scale bot characteristics
  Hays et al. (2025) — longitudinal bot behavior evolution
  BotEvo (2026) — LLM-enhanced bot behavior patterns
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_SCHEMA = {
    "profile": [
        "account_age_days", "statuses_count", "followers_count",
        "friends_count", "favourites_count", "listed_count",
        "default_profile", "default_profile_image", "has_url",
        "verified", "description_length", "screen_name_length",
    ],
    "behavioral": [
        "tweets_per_day", "retweet_ratio", "reply_ratio",
        "mention_count_avg", "hashtag_count_avg", "url_share_ratio",
        "posting_regularity", "night_post_ratio", "weekend_ratio",
        "burst_score",
    ],
    "content": [
        "avg_tweet_length", "vocabulary_richness", "text_entropy",
        "emoji_ratio", "caps_ratio", "content_similarity",
    ],
    "network": [
        "follower_friend_ratio", "reputation_score",
    ],
}

ALL_FEATURES = []
FEATURE_GROUP_MAP = {}
for group, feats in FEATURE_SCHEMA.items():
    ALL_FEATURES.extend(feats)
    for f in feats:
        FEATURE_GROUP_MAP[f] = group

BINARY_FEATURES = {"default_profile", "default_profile_image", "has_url", "verified"}

BOUNDED_01 = {
    "retweet_ratio", "reply_ratio", "url_share_ratio", "posting_regularity",
    "night_post_ratio", "weekend_ratio", "vocabulary_richness", "emoji_ratio",
    "caps_ratio", "content_similarity", "reputation_score",
}

NON_NEGATIVE = {
    "account_age_days", "statuses_count", "followers_count", "friends_count",
    "favourites_count", "listed_count", "tweets_per_day", "mention_count_avg",
    "hashtag_count_avg", "burst_score", "avg_tweet_length", "description_length",
    "screen_name_length", "follower_friend_ratio",
}


# ═══════════════════════════════════════════════════════════════════════════
# INITIAL DISTRIBUTION PARAMETERS
#
# IMPORTANT: Distribution gaps are intentionally NARROWER than published
# "cleanest" separability to produce realistic classifier difficulty (F1
# ~0.90-0.97 pre-drift). Real-world bot detection is hard because
# sophisticated bots overlap with humans on many features.
#
# Each entry: (distribution_type, params_human, params_bot, evidence)
# ═══════════════════════════════════════════════════════════════════════════

INIT_DISTRIBUTIONS = {
    # --- Profile --- (calibrated from Cresci-2017 empirical fits)
    "account_age_days":      ("lognormal", (7.54, 0.37), (7.42, 0.32), "Cresci 2017 (moderate overlap)"),
    "statuses_count":        ("lognormal", (8.55, 1.82), (4.03, 2.08), "Cresci 2017 (distinct)"),
    "followers_count":       ("lognormal", (5.83, 1.45), (3.32, 2.17), "Cresci 2017 (distinct)"),
    "friends_count":         ("lognormal", (5.78, 1.11), (5.26, 1.45), "Cresci 2017 (moderate overlap)"),
    "favourites_count":      ("lognormal", (6.91, 2.14), (1.36, 1.81), "Cresci 2017 (distinct)"),
    "listed_count":          ("lognormal", (1.69, 1.45), (1.63, 1.34), "Cresci 2017 (distinct)"),
    "default_profile":       ("bernoulli", (0.20,),      (0.30,),      "TwiBotX 2026 (heavy overlap)"),
    "default_profile_image": ("bernoulli", (0.10,),      (0.18,),      "Cresci 2017 (heavy overlap)"),
    "has_url":               ("bernoulli", (0.3644,),    (0.2151,),    "Cresci 2017 (heavy overlap)"),
    "verified":              ("bernoulli", (0.02,),      (0.003,),     "TwiBot-22"),
    "description_length":    ("normal",    (63.8, 50.9), (43.7, 53.2), "Cresci 2017 (distinct)"),
    "screen_name_length":    ("normal",    (10.9, 2.5),  (12.1, 2.2),  "Cresci 2017 (moderate overlap)"),
    # --- Behavioral --- (calibrated from Cresci-2017 empirical fits)
    "tweets_per_day":        ("lognormal", (1.0, 1.74),  (-3.24, 2.08),"Cresci 2017 (distinct)"),
    "retweet_ratio":         ("beta",      (1.27, 3.62), (0.5, 6.52),  "Cresci 2017 (distinct)"),
    "reply_ratio":           ("beta",      (1.63, 4.19), (1.21, 2.77), "Cresci 2017 (distinct)"),
    "mention_count_avg":     ("lognormal", (-0.34, 0.55),(-1.91, 1.64),"Cresci 2017 (distinct)"),
    "hashtag_count_avg":     ("lognormal", (-2.0, 1.21), (-2.25, 0.81),"Cresci 2017 (moderate overlap)"),
    "url_share_ratio":       ("beta",      (1.35, 3.63), (0.5, 1.39),  "Cresci 2017 (distinct)"),
    "posting_regularity":    ("beta",      (0.71, 3.9),  (0.5, 3.1),   "Cresci 2017 (distinct)"),
    "night_post_ratio":      ("beta",      (2.01, 4.59), (1.81, 6.05), "Cresci 2017 (moderate overlap)"),
    "weekend_ratio":         ("normal",    (0.3, 0.1),   (0.3, 0.1),   "Cresci 2017 (heavy overlap)"),
    "burst_score":           ("lognormal", (-4.43, 0.8), (-3.01, 1.13),"Cresci 2017 (distinct)"),
    # --- Content --- (calibrated from Cresci-2017 empirical fits)
    "avg_tweet_length":      ("normal",    (73.6, 20.0), (71.0, 16.5), "Cresci 2017 (heavy overlap)"),
    "vocabulary_richness":   ("beta",      (6.07, 14.64),(2.78, 1.98), "Cresci 2017 (distinct)"),
    "text_entropy":          ("normal",    (4.7, 0.2),   (4.6, 0.3),   "Cresci 2017 (distinct)"),
    "emoji_ratio":           ("exponential", (0.022,),   (0.018,),     "TwiBot-22 (near-identical)"),
    "caps_ratio":            ("beta",      (6.93, 61.94),(4.21, 44.23),"Cresci 2017 (moderate overlap)"),
    "content_similarity":    ("beta",      (3.57, 156.15),(1.49, 58.93),"Cresci 2017 (heavy overlap)"),
    # --- Network --- (calibrated from Cresci-2017 empirical fits)
    "follower_friend_ratio": ("lognormal", (0.05, 1.06), (-1.91, 1.44),"Cresci 2017 (distinct)"),
    "reputation_score":      ("beta",      (2.49, 2.4),  (0.83, 3.27), "Cresci 2017 (distinct)"),
}


# ═══════════════════════════════════════════════════════════════════════════
# ACCOUNT ARCHETYPES
#
# Each label has multiple archetypes. Archetypes create cross-feature
# correlations by coherently shifting feature groups together rather than
# sampling every feature independently.
#
# archetype_modifier[feature] is ADDED to the base sample (for continuous)
# or replaces the probability (for binary). Not all features are modified —
# unmodified features keep their marginal sample.
# ═══════════════════════════════════════════════════════════════════════════

HUMAN_ARCHETYPES = {
    "power_user": {
        "weight": 0.15,
        "modifiers": {
            "tweets_per_day": ("multiply", 3.0),
            "followers_count": ("multiply", 5.0),
            "statuses_count": ("multiply", 3.0),
            "favourites_count": ("multiply", 2.0),
            "listed_count": ("multiply", 4.0),
            "reply_ratio": ("add", 0.10),
            "vocabulary_richness": ("add", 0.08),
            "text_entropy": ("add", 0.3),
            "description_length": ("add", 30),
            "has_url": ("set_prob", 0.65),
            "default_profile": ("set_prob", 0.03),
            "reputation_score": ("add", 0.15),
        },
    },
    "casual": {
        "weight": 0.40,
        "modifiers": {},
    },
    "lurker": {
        "weight": 0.20,
        "modifiers": {
            "tweets_per_day": ("multiply", 0.15),
            "statuses_count": ("multiply", 0.2),
            "reply_ratio": ("add", -0.05),
            "retweet_ratio": ("add", 0.08),
            "posting_regularity": ("multiply", 0.3),
            "burst_score": ("multiply", 0.5),
            "content_similarity": ("multiply", 0.5),
        },
    },
    "new_account": {
        "weight": 0.15,
        "modifiers": {
            "account_age_days": ("multiply", 0.05),
            "statuses_count": ("multiply", 0.1),
            "followers_count": ("multiply", 0.05),
            "friends_count": ("multiply", 0.1),
            "favourites_count": ("multiply", 0.1),
            "listed_count": ("multiply", 0.02),
            "default_profile": ("set_prob", 0.50),
            "default_profile_image": ("set_prob", 0.30),
            "has_url": ("set_prob", 0.10),
            "description_length": ("add", -40),
            "reputation_score": ("add", -0.15),
        },
    },
    "professional": {
        "weight": 0.10,
        "modifiers": {
            "tweets_per_day": ("multiply", 1.5),
            "url_share_ratio": ("add", 0.15),
            "hashtag_count_avg": ("add", 0.5),
            "has_url": ("set_prob", 0.80),
            "description_length": ("add", 40),
            "posting_regularity": ("add", 0.15),
            "night_post_ratio": ("add", -0.05),
            "vocabulary_richness": ("add", 0.05),
            "content_similarity": ("add", 0.05),
        },
    },
}

BOT_ARCHETYPES = {
    "spam_bot": {
        "weight": 0.15,
        "modifiers": {
            "tweets_per_day": ("multiply", 4.0),
            "url_share_ratio": ("add", 0.20),
            "hashtag_count_avg": ("add", 1.5),
            "content_similarity": ("multiply", 8.0),
            "posting_regularity": ("add", 0.15),
            "burst_score": ("multiply", 2.0),
            "vocabulary_richness": ("add", -0.10),
            "reply_ratio": ("add", -0.05),
        },
    },
    "social_bot": {
        "weight": 0.30,
        "modifiers": {
            "reply_ratio": ("add", 0.08),
            "mention_count_avg": ("add", 0.5),
            "vocabulary_richness": ("add", 0.05),
            "text_entropy": ("add", 0.2),
            "content_similarity": ("multiply", 0.5),
            "default_profile": ("set_prob", 0.15),
            "default_profile_image": ("set_prob", 0.10),
            "description_length": ("add", 25),
            "has_url": ("set_prob", 0.30),
        },
    },
    "fake_follower": {
        "weight": 0.15,
        "modifiers": {
            "tweets_per_day": ("multiply", 0.15),
            "statuses_count": ("multiply", 0.1),
            "friends_count": ("multiply", 3.0),
            "followers_count": ("multiply", 0.05),
            "retweet_ratio": ("add", 0.10),
            "posting_regularity": ("multiply", 0.3),
            "content_similarity": ("multiply", 4.0),
            "burst_score": ("multiply", 0.3),
        },
    },
    "political_bot": {
        "weight": 0.15,
        "modifiers": {
            "hashtag_count_avg": ("add", 2.0),
            "retweet_ratio": ("add", 0.15),
            "mention_count_avg": ("add", 1.0),
            "night_post_ratio": ("add", 0.05),
            "content_similarity": ("multiply", 4.0),
            "burst_score": ("multiply", 1.5),
        },
    },
    "sophisticated_bot": {
        "weight": 0.20,
        "modifiers": {
            "vocabulary_richness": ("add", 0.15),
            "text_entropy": ("add", 0.5),
            "content_similarity": ("multiply", 0.3),
            "reply_ratio": ("add", 0.12),
            "posting_regularity": ("multiply", 0.2),
            "night_post_ratio": ("add", -0.12),
            "burst_score": ("multiply", 0.4),
            "default_profile": ("set_prob", 0.08),
            "default_profile_image": ("set_prob", 0.05),
            "has_url": ("set_prob", 0.45),
            "description_length": ("add", 45),
            "followers_count": ("multiply", 4.0),
            "listed_count": ("multiply", 3.0),
            "reputation_score": ("add", 0.15),
            "tweets_per_day": ("multiply", 0.3),
            "retweet_ratio": ("add", -0.10),
            "hashtag_count_avg": ("multiply", 0.5),
        },
    },
}


def sample_initial(feat_name, label, n, rng):
    """Sample n initial values for a feature given the label."""
    dist_type, h_params, b_params, _ = INIT_DISTRIBUTIONS[feat_name]
    params = h_params if label == "human" else b_params

    if dist_type == "lognormal":
        return rng.lognormal(params[0], params[1], n)
    elif dist_type == "normal":
        return rng.normal(params[0], params[1], n)
    elif dist_type == "beta":
        return rng.beta(params[0], params[1], n)
    elif dist_type == "exponential":
        return rng.exponential(params[0], n)
    elif dist_type == "bernoulli":
        return (rng.random(n) < params[0]).astype(float)
    raise ValueError(f"Unknown distribution: {dist_type}")


def assign_archetypes(n, label, rng):
    """Assign each of n accounts to an archetype. Returns list of archetype names."""
    archetypes = HUMAN_ARCHETYPES if label == "human" else BOT_ARCHETYPES
    names = list(archetypes.keys())
    weights = np.array([archetypes[a]["weight"] for a in names])
    weights /= weights.sum()
    return rng.choice(names, size=n, p=weights)


def apply_archetype_modifiers(feat_arrays, archetype_assignments, label, rng):
    """Apply archetype-specific modifiers to create cross-feature correlations."""
    archetypes = HUMAN_ARCHETYPES if label == "human" else BOT_ARCHETYPES
    n = len(archetype_assignments)

    for arch_name, arch_def in archetypes.items():
        mask = archetype_assignments == arch_name
        n_arch = mask.sum()
        if n_arch == 0:
            continue

        for feat, (op, val) in arch_def["modifiers"].items():
            if op == "multiply":
                jitter = rng.uniform(0.7, 1.3, n_arch)
                feat_arrays[feat][mask] *= val * jitter
            elif op == "add":
                jitter = rng.normal(0, abs(val) * 0.2, n_arch)
                feat_arrays[feat][mask] += val + jitter
            elif op == "set_prob":
                feat_arrays[feat][mask] = (rng.random(n_arch) < val).astype(float)

    return feat_arrays


# ═══════════════════════════════════════════════════════════════════════════
# NATURAL EVOLUTION PARAMETERS
# How much each feature naturally drifts per time step (for BOTH labels).
# Expressed as fraction of the feature's typical scale.
# ═══════════════════════════════════════════════════════════════════════════

NATURAL_EVOLUTION = {
    # Profile features: slow organic growth
    "account_age_days":      {"human_trend": 30.0,  "bot_trend": 30.0,  "noise_scale": 0.0},
    "statuses_count":        {"human_trend": 0.005, "bot_trend": 0.008, "noise_scale": 0.015},
    "followers_count":       {"human_trend": 0.003, "bot_trend": 0.001, "noise_scale": 0.01},
    "friends_count":         {"human_trend": 0.002, "bot_trend": 0.003, "noise_scale": 0.01},
    "favourites_count":      {"human_trend": 0.004, "bot_trend": 0.001, "noise_scale": 0.01},
    "listed_count":          {"human_trend": 0.002, "bot_trend": 0.0,   "noise_scale": 0.02},
    "default_profile":       {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.0},
    "default_profile_image": {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.0},
    "has_url":               {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.0},
    "verified":              {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.0},
    "description_length":    {"human_trend": 0.5,   "bot_trend": 0.0,   "noise_scale": 3.0},
    "screen_name_length":    {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.0},
    # Behavioral features: moderate variation
    "tweets_per_day":        {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.05},
    "retweet_ratio":         {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.02},
    "reply_ratio":           {"human_trend": 0.001, "bot_trend": 0.0,   "noise_scale": 0.02},
    "mention_count_avg":     {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.03},
    "hashtag_count_avg":     {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.03},
    "url_share_ratio":       {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.02},
    "posting_regularity":    {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.02},
    "night_post_ratio":      {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.015},
    "weekend_ratio":         {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.01},
    "burst_score":           {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.04},
    # Content features: humans evolve language over time too
    "avg_tweet_length":      {"human_trend": 0.3,   "bot_trend": 0.0,   "noise_scale": 5.0},
    "vocabulary_richness":   {"human_trend": 0.001, "bot_trend": 0.0,   "noise_scale": 0.015},
    "text_entropy":          {"human_trend": 0.005, "bot_trend": 0.0,   "noise_scale": 0.04},
    "emoji_ratio":           {"human_trend": 0.001, "bot_trend": 0.0,   "noise_scale": 0.003},
    "caps_ratio":            {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.005},
    "content_similarity":    {"human_trend": 0.0,   "bot_trend": 0.0,   "noise_scale": 0.015},
    # Network features: slow organic change
    "follower_friend_ratio": {"human_trend": 0.001, "bot_trend": 0.0,   "noise_scale": 0.02},
    "reputation_score":      {"human_trend": 0.001, "bot_trend": 0.0,   "noise_scale": 0.01},
}


# ═══════════════════════════════════════════════════════════════════════════
# DRIFT SCENARIO DEFINITIONS
#
# Each scenario defines:
#   - affected_features: which features shift
#   - drift_vector: the BASE δ per feature (multiplied by --drift-magnitude)
#   - drift_mode: "gradual" | "sudden" | "recurring"
#   - literature_basis: citation justification
#
# Mathematical definition:
#   P_t(X_j | Y=bot, scenario=s) =
#     P_0(X_j | Y=bot)                            if t < τ_start
#     P_0(X_j | Y=bot) + δ_{s,j} · progress(t)   if τ_start ≤ t ≤ τ_end
#     P_0(X_j | Y=bot) + δ_{s,j}                  if t > τ_end
#
#   where progress(t) = (t - τ_start) / (τ_end - τ_start)
# ═══════════════════════════════════════════════════════════════════════════

DRIFT_SCENARIOS = {
    "no_drift": {
        "description": "Control group — no injected drift beyond natural evolution.",
        "drift_mode": "none",
        "affected_features": [],
        "drift_vector": {},
        "literature_basis": "Baseline comparison group",
    },
    "activity_drift": {
        "description": (
            "D1 — Activity drift: posting frequency, timing, and volume patterns "
            "shift as bots adapt schedules to evade temporal detection heuristics."
        ),
        "drift_mode": "gradual",
        "affected_features": [
            "tweets_per_day", "posting_regularity", "night_post_ratio",
            "burst_score", "weekend_ratio",
        ],
        "drift_vector": {
            "tweets_per_day":     -0.50,   # multiplicative: total shrink ~50% over drift period
            "posting_regularity": -0.25,   # additive: become less metronomic
            "night_post_ratio":   -0.15,   # additive: reduce 24/7 posting
            "burst_score":        -0.40,   # multiplicative: reduce burst intensity ~40%
            "weekend_ratio":      +0.03,   # additive: mimic human weekend pattern
        },
        "multiplicative_features": {"tweets_per_day", "burst_score"},
        "literature_basis": "Hays et al. (2025): bot posting patterns become less distinguishable over time",
    },
    "linguistic_drift": {
        "description": (
            "D2 — Linguistic drift: bots adopt LLM-generated text, increasing "
            "vocabulary diversity, text entropy, and tweet length while reducing "
            "content repetitiveness."
        ),
        "drift_mode": "gradual",
        "affected_features": [
            "text_entropy", "vocabulary_richness", "content_similarity",
            "avg_tweet_length", "caps_ratio",
        ],
        "drift_vector": {
            "text_entropy":       +0.80,   # absolute shift in entropy
            "vocabulary_richness":+0.15,   # absolute shift in richness
            "content_similarity": -0.20,   # less repetitive
            "avg_tweet_length":   +30.0,   # longer tweets
            "caps_ratio":         -0.03,   # less shouting
        },
        "multiplicative_features": set(),
        "literature_basis": "BotEvo (2026): LLM-equipped bots produce higher entropy, richer vocabulary text",
    },
    "coordination_drift": {
        "description": (
            "D3 — Coordination drift: bots shift engagement patterns to mimic "
            "authentic interaction — more replies, fewer bulk retweets, reduced "
            "hashtag/URL spam."
        ),
        "drift_mode": "gradual",
        "affected_features": [
            "reply_ratio", "retweet_ratio", "mention_count_avg",
            "hashtag_count_avg", "url_share_ratio",
        ],
        "drift_vector": {
            "reply_ratio":        +0.12,   # more replies
            "retweet_ratio":      -0.15,   # fewer bulk retweets
            "mention_count_avg":  -0.80,   # fewer mass mentions
            "hashtag_count_avg":  -0.60,   # fewer hashtags
            "url_share_ratio":    -0.10,   # fewer URLs
        },
        "multiplicative_features": set(),
        "literature_basis": "TwiBotX (2026); TwiBot-22: engagement ratios are key discriminators",
    },
    "network_drift": {
        "description": (
            "D4 — Network/profile drift: bot profiles mature — custom images, "
            "bios, URLs added, follower counts grown organically, reputation "
            "scores improve."
        ),
        "drift_mode": "gradual",
        "affected_features": [
            "default_profile", "default_profile_image", "has_url",
            "description_length", "followers_count", "listed_count",
            "reputation_score",
        ],
        "drift_vector": {
            "default_profile":       -0.50,  # total: flip probability across drift period
            "default_profile_image": -0.40,  # total: flip probability across drift period
            "has_url":               +0.45,  # total: flip probability across drift period
            "description_length":    +35.0,  # total additive across drift period
            "followers_count":       +0.50,  # total multiplicative growth across drift period
            "listed_count":          +3.0,   # total additive across drift period
            "reputation_score":      +0.12,  # total additive across drift period
        },
        "multiplicative_features": {"followers_count"},
        "literature_basis": "Cresci (2017): paradigm-shift from sparse to sophisticated profiles",
    },
    "combined_drift": {
        "description": (
            "D5 — Combined/adversarial drift: simultaneous evolution across "
            "activity, linguistic, and coordination dimensions. Simulates a "
            "sophisticated adversary adapting multiple behavioral axes at once."
        ),
        "drift_mode": "gradual",
        "affected_features": [
            "tweets_per_day", "posting_regularity",
            "text_entropy", "vocabulary_richness", "content_similarity",
            "reply_ratio", "retweet_ratio",
        ],
        "drift_vector": {
            "tweets_per_day":      -0.35,   # multiplicative: total shrink ~35%
            "posting_regularity":  -0.18,
            "text_entropy":        +0.50,
            "vocabulary_richness": +0.10,
            "content_similarity":  -0.15,
            "reply_ratio":         +0.08,
            "retweet_ratio":       -0.10,
        },
        "multiplicative_features": {"tweets_per_day"},
        "literature_basis": "BotEvo (2026); Hays (2025): real bot evolution is multi-dimensional",
    },
}

SCENARIO_NAMES = list(DRIFT_SCENARIOS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# TRAJECTORY GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def clip_feature(values, feat_name):
    """Enforce domain constraints on a feature array."""
    if feat_name in BINARY_FEATURES:
        return values
    if feat_name in BOUNDED_01:
        np.clip(values, 0, 1, out=values)
    if feat_name in NON_NEGATIVE:
        np.maximum(values, 0, out=values)
    if feat_name == "tweets_per_day":
        np.maximum(values, 0.01, out=values)
    elif feat_name == "burst_score":
        np.maximum(values, 0.001, out=values)
    elif feat_name == "screen_name_length":
        np.maximum(values, 3.0, out=values)
    elif feat_name == "text_entropy":
        np.clip(values, 0.5, 6.0, out=values)
    return values


def generate_trajectories(n_accounts, n_windows, seed, drift_magnitude,
                          tau_start, tau_end, campaign_windows):
    """Generate longitudinal account trajectories with drift injection.

    Each account's state at time t depends on its state at t-1 (Markov property).
    """
    rng = np.random.default_rng(seed)

    n_human = n_accounts // 2
    n_bot = n_accounts - n_human

    # --- Assign scenarios to bots ---
    bots_per_scenario = n_bot // len(SCENARIO_NAMES)
    remainder = n_bot - bots_per_scenario * len(SCENARIO_NAMES)
    scenario_list = []
    for i, s in enumerate(SCENARIO_NAMES):
        count = bots_per_scenario + (1 if i < remainder else 0)
        scenario_list.extend([s] * count)
    rng.shuffle(scenario_list)

    labels = np.array(["human"] * n_human + ["bot"] * n_bot)
    account_ids = ([f"H_{i:05d}" for i in range(n_human)]
                   + [f"B_{i:05d}" for i in range(n_bot)])
    scenarios = np.array(["none"] * n_human + list(scenario_list))

    is_human = np.arange(n_accounts) < n_human
    is_bot = ~is_human
    scenario_masks = {s: (scenarios == s) for s in SCENARIO_NAMES}

    # --- Assign archetypes for cross-feature correlations ---
    print("  Assigning account archetypes...")
    human_archetypes = assign_archetypes(n_human, "human", rng)
    bot_archetypes = assign_archetypes(n_bot, "bot", rng)
    archetype_labels = np.concatenate([human_archetypes, bot_archetypes])

    # --- Sample initial state (t=0) ---
    print("  Sampling initial account states...")
    state = {}
    for feat in ALL_FEATURES:
        h_vals = sample_initial(feat, "human", n_human, rng)
        b_vals = sample_initial(feat, "bot", n_bot, rng)
        state[feat] = np.concatenate([h_vals, b_vals])

    # Apply archetype modifiers BEFORE clipping — creates cross-feature correlations
    h_arrays = {feat: state[feat][:n_human] for feat in ALL_FEATURES}
    b_arrays = {feat: state[feat][n_human:] for feat in ALL_FEATURES}
    apply_archetype_modifiers(h_arrays, human_archetypes, "human", rng)
    apply_archetype_modifiers(b_arrays, bot_archetypes, "bot", rng)
    for feat in ALL_FEATURES:
        state[feat] = np.concatenate([h_arrays[feat], b_arrays[feat]])
        state[feat] = clip_feature(state[feat], feat)

    # --- Generate trajectories window by window ---
    all_window_dfs = []
    start_date = datetime(2023, 1, 1)

    for w in range(n_windows):
        if w > 0:
            # ── STEP 1: Natural evolution (both humans and bots) ──
            for feat in ALL_FEATURES:
                if feat in BINARY_FEATURES:
                    continue

                evo = NATURAL_EVOLUTION[feat]
                noise_scale = evo["noise_scale"]

                # Trend component
                human_trend = np.full(n_accounts, 0.0)
                human_trend[is_human] = evo["human_trend"]
                human_trend[is_bot] = evo["bot_trend"]

                # For count features, trend is multiplicative
                if feat in ("statuses_count", "followers_count", "friends_count",
                            "favourites_count", "listed_count"):
                    state[feat] *= (1 + human_trend)
                elif feat == "account_age_days":
                    state[feat] += human_trend
                else:
                    state[feat] += human_trend

                # Noise component (proportional for large-scale features,
                # absolute for bounded features)
                if feat in BOUNDED_01:
                    noise = rng.normal(0, noise_scale, n_accounts)
                elif feat in ("account_age_days",):
                    noise = np.zeros(n_accounts)
                else:
                    noise = state[feat] * rng.normal(0, noise_scale, n_accounts)

                state[feat] += noise

            # ── STEP 2: Human-specific natural variation events ──
            # Humans occasionally change behavior (life events, interests shift)
            # ~2% of humans per window have a behavioral "event"
            human_event_mask = np.zeros(n_accounts, dtype=bool)
            n_events = max(1, int(n_human * 0.02))
            event_indices = rng.choice(n_human, n_events, replace=False)
            human_event_mask[event_indices] = True

            for idx in event_indices:
                event_type = rng.choice(["activity_change", "interest_shift", "break"])
                if event_type == "activity_change":
                    state["tweets_per_day"][idx] *= rng.uniform(0.5, 2.0)
                    state["night_post_ratio"][idx] += rng.normal(0, 0.05)
                elif event_type == "interest_shift":
                    state["hashtag_count_avg"][idx] *= rng.uniform(0.5, 1.5)
                    state["vocabulary_richness"][idx] += rng.normal(0, 0.03)
                    state["text_entropy"][idx] += rng.normal(0, 0.1)
                elif event_type == "break":
                    state["tweets_per_day"][idx] *= rng.uniform(0.1, 0.5)

            # ── STEP 3: Drift injection for bot scenarios ──
            drift_active = np.zeros(n_accounts, dtype=int)
            drift_type_arr = np.full(n_accounts, "none", dtype=object)
            drift_progress_arr = np.zeros(n_accounts)

            for scenario_name, scenario_def in DRIFT_SCENARIOS.items():
                if scenario_def["drift_mode"] == "none":
                    continue

                mask = scenario_masks[scenario_name]
                n_masked = mask.sum()
                if n_masked == 0:
                    continue

                drift_vec = scenario_def["drift_vector"]

                if scenario_def["drift_mode"] == "gradual":
                    raw_progress = (w - tau_start) / max(1, (tau_end - tau_start))
                    progress = min(1.0, max(0.0, raw_progress))

                    # Mark as drifting from onset onward (state was changed)
                    if w >= tau_start:
                        drift_active[mask] = 1
                        drift_type_arr[mask] = "gradual"
                        drift_progress_arr[mask] = progress

                    # Only apply drift deltas during the transition period
                    if w < tau_start or w >= tau_end:
                        continue

                    mult_feats = scenario_def.get("multiplicative_features", set())
                    n_drift_steps = max(1, tau_end - tau_start)

                    for feat, base_delta in drift_vec.items():
                        delta = base_delta * drift_magnitude / n_drift_steps

                        if feat in BINARY_FEATURES:
                            flip_prob = abs(delta)
                            flips = rng.random(n_masked) < flip_prob
                            target = 0.0 if delta < 0 else 1.0
                            state[feat][mask] = np.where(
                                flips, target, state[feat][mask]
                            )
                        elif feat in mult_feats:
                            factor = 1 + delta * (1 + rng.normal(0, 0.1, n_masked))
                            state[feat][mask] *= factor
                        else:
                            noise = rng.normal(0, abs(delta) * 0.15, n_masked)
                            state[feat][mask] += delta + noise

            # Clip everything after drift
            for feat in ALL_FEATURES:
                state[feat] = clip_feature(state[feat], feat)

        else:
            # Window 0: no drift
            drift_active = np.zeros(n_accounts, dtype=int)
            drift_type_arr = np.full(n_accounts, "none", dtype=object)
            drift_progress_arr = np.zeros(n_accounts)

        # ── Build output for this window ──
        window_date = (start_date + timedelta(days=w * 30)).strftime("%Y-%m-%d")

        drifted_feat_strs = np.full(n_accounts, "", dtype=object)
        for scenario_name, scenario_def in DRIFT_SCENARIOS.items():
            if scenario_def["drift_mode"] == "none":
                continue
            mask = scenario_masks[scenario_name] & (drift_active == 1)
            if mask.any():
                feat_str = "|".join(scenario_def["affected_features"])
                drifted_feat_strs[mask] = feat_str

        window_data = {
            "account_id": account_ids,
            "label": labels,
            "archetype": archetype_labels,
            "time_window": w,
            "window_date": window_date,
            "drift_scenario": scenarios,
            "drift_active": drift_active,
            "drift_type": drift_type_arr,
            "drift_progress": np.round(drift_progress_arr, 4),
            "drifted_features": drifted_feat_strs,
        }
        for feat in ALL_FEATURES:
            observed = state[feat].copy()
            if feat not in BINARY_FEATURES:
                if feat in BOUNDED_01:
                    meas_noise = rng.normal(0, 0.03, n_accounts)
                else:
                    meas_noise = observed * rng.normal(0, 0.08, n_accounts)
                observed = observed + meas_noise
                observed = clip_feature(observed, feat)
            window_data[feat] = np.round(observed, 6)

        all_window_dfs.append(pd.DataFrame(window_data))

        if (w + 1) % 5 == 0 or w == 0:
            drift_count = drift_active.sum()
            print(f"  Window {w:2d} ({window_date}): {drift_count} accounts drifting")

    print("  Concatenating trajectories...")
    df = pd.concat(all_window_dfs, ignore_index=True)
    return df, scenarios


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate(df, tau_start):
    print("\n" + "=" * 65)
    print("VALIDATION REPORT")
    print("=" * 65)

    print(f"\nDataset: {df.shape[0]:,} obs  |  {df['account_id'].nunique():,} accounts  |  "
          f"{df['time_window'].nunique()} windows")

    # --- Label distribution ---
    print(f"\nAccounts by label:")
    for label in ["human", "bot"]:
        n = df[df["label"] == label]["account_id"].nunique()
        print(f"  {label}: {n:,}")

    # --- Scenario distribution ---
    print(f"\nBot drift scenarios:")
    bot_df = df[df["label"] == "bot"]
    for s, n in bot_df.groupby("drift_scenario")["account_id"].nunique().items():
        print(f"  {s}: {n:,}")

    # --- Drift prevalence ---
    drift_obs = int(df["drift_active"].sum())
    print(f"\nDrift: {drift_obs:,} / {len(df):,} observations ({drift_obs/len(df):.1%})")

    # --- Q1: Plausible separation ---
    print(f"\n--- Q1: Bot/Human Separability (pre-drift, windows 0–{tau_start-1}) ---")
    pre = df[df["time_window"] < tau_start]
    key_feats = ["tweets_per_day", "posting_regularity", "content_similarity",
                 "vocabulary_richness", "follower_friend_ratio", "reply_ratio"]
    for feat in key_feats:
        h = pre[pre["label"] == "human"][feat].values
        b = pre[pre["label"] == "bot"][feat].values
        ks, p = stats.ks_2samp(h, b)
        tag = "OK" if p < 0.001 else "WEAK"
        print(f"  {feat:25s} KS={ks:.3f}  p={p:.1e}  [{tag}]")

    # --- Q2: Drift visibility ---
    print(f"\n--- Q2: Drift Visibility (pre vs post for each scenario) ---")
    max_w = df["time_window"].max()
    for scenario_name, scenario_def in DRIFT_SCENARIOS.items():
        if scenario_def["drift_mode"] == "none":
            continue
        sc = df[(df["drift_scenario"] == scenario_name) & (df["label"] == "bot")]
        pre_d = sc[sc["time_window"] < tau_start]
        post_d = sc[sc["time_window"] >= max_w - 2]
        print(f"\n  [{scenario_name}]")
        for feat in scenario_def["affected_features"][:3]:
            if feat in BINARY_FEATURES:
                print(f"    {feat}: {pre_d[feat].mean():.3f} -> {post_d[feat].mean():.3f}")
            else:
                ks, p = stats.ks_2samp(pre_d[feat].values, post_d[feat].values)
                d = "UP" if post_d[feat].mean() > pre_d[feat].mean() else "DOWN"
                print(f"    {feat}: KS={ks:.3f}  p={p:.1e}  [{d}]  "
                      f"({pre_d[feat].mean():.3f} -> {post_d[feat].mean():.3f})")

    # --- Trajectory sanity: single account trace ---
    print(f"\n--- Sample Trajectory (one drifting bot) ---")
    ling_bots = df[df["drift_scenario"] == "linguistic_drift"]["account_id"].unique()
    if len(ling_bots) > 0:
        sample_id = ling_bots[0]
        trace = df[df["account_id"] == sample_id][
            ["time_window", "drift_active", "drift_progress",
             "text_entropy", "vocabulary_richness", "content_similarity"]
        ]
        print(f"  Account: {sample_id}")
        for _, row in trace.iterrows():
            w = int(row["time_window"])
            da = int(row["drift_active"])
            dp = row["drift_progress"]
            te = row["text_entropy"]
            vr = row["vocabulary_richness"]
            cs = row["content_similarity"]
            marker = " <-- drift" if da else ""
            print(f"    w={w:2d}  entropy={te:.3f}  vocab={vr:.3f}  sim={cs:.3f}  "
                  f"progress={dp:.2f}{marker}")

    # --- Range check ---
    print(f"\n--- Feature Range Check ---")
    issues = 0
    for feat in BOUNDED_01:
        if df[feat].min() < -0.001 or df[feat].max() > 1.001:
            print(f"  FAIL {feat}: [{df[feat].min():.4f}, {df[feat].max():.4f}]")
            issues += 1
    for feat in NON_NEGATIVE:
        if df[feat].min() < -0.001:
            print(f"  FAIL {feat}: min={df[feat].min():.4f}")
            issues += 1
    print(f"  {'All ranges valid' if issues == 0 else f'{issues} issues'}.")


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_all(df, scenarios, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)

    # Main dataset
    main_path = os.path.join(output_dir, "botdriftbench_v0.2.csv")
    df.to_csv(main_path, index=False)
    size_mb = os.path.getsize(main_path) / 1e6
    print(f"\nSaved: {main_path} ({size_mb:.1f} MB)")

    # Account metadata
    acc = df.groupby("account_id").agg(
        label=("label", "first"),
        drift_scenario=("drift_scenario", "first"),
        windows_with_drift=("drift_active", "sum"),
    ).reset_index()
    acc.to_csv(os.path.join(output_dir, "account_metadata.csv"), index=False)
    print(f"Saved: account_metadata.csv")

    # Drift ground truth
    gt = df[df["drift_active"] == 1][[
        "account_id", "time_window", "window_date",
        "drift_scenario", "drift_type", "drift_progress", "drifted_features"
    ]]
    gt.to_csv(os.path.join(output_dir, "drift_ground_truth.csv"), index=False)
    print(f"Saved: drift_ground_truth.csv ({len(gt):,} rows)")

    # Window summary
    rows = []
    for w in sorted(df["time_window"].unique()):
        wdf = df[df["time_window"] == w]
        row = {
            "time_window": w,
            "window_date": wdf["window_date"].iloc[0],
            "n_observations": len(wdf),
            "n_drift_active": int(wdf["drift_active"].sum()),
        }
        for feat in ALL_FEATURES:
            row[f"{feat}_mean"] = round(wdf[feat].mean(), 4)
            row[f"{feat}_std"] = round(wdf[feat].std(), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "window_summary_statistics.csv"), index=False
    )
    print(f"Saved: window_summary_statistics.csv")

    # Generation config
    config = {
        "dataset_name": "BotDriftBench",
        "version": "0.2.0",
        "description": (
            "A controlled, reproducible benchmark for evaluating social-bot "
            "detection under parameterized behavioral shifts. Feature distributions "
            "are empirically calibrated from Cresci-2017 (14,368 accounts, 6.6M tweets). "
            "Each account is a longitudinal trajectory with Markov state carry-forward."
        ),
        "generated_at": datetime.now().isoformat(),
        "parameters": {
            "seed": args.seed,
            "n_accounts": args.accounts,
            "n_time_windows": args.windows,
            "drift_magnitude": args.drift_magnitude,
            "tau_start": args.tau_start,
            "tau_end": args.tau_end,
        },
        "statistics": {
            "n_total_observations": len(df),
            "n_human_accounts": int((acc["label"] == "human").sum()),
            "n_bot_accounts": int((acc["label"] == "bot").sum()),
            "n_features": len(ALL_FEATURES),
        },
        "mathematical_framework": {
            "evolution_model": "X_{i,j,t} = X_{i,j,t-1} + trend_j + noise_{i,j,t}",
            "drift_model": (
                "X_{i,j,t} = X_{i,j,t-1} + trend_j + "
                "delta_{s,j} * magnitude * per_step + noise_{i,j,t}"
            ),
            "noise_distribution": "N(0, sigma_{natural,j})",
            "drift_magnitude_param": args.drift_magnitude,
            "change_point_tau_start": args.tau_start,
            "change_point_tau_end": args.tau_end,
        },
        "temporal_coverage": {
            "start_date": "2023-01-01",
            "end_date": (datetime(2023, 1, 1)
                         + timedelta(days=(args.windows - 1) * 30)).strftime("%Y-%m-%d"),
            "window_interval_days": 30,
        },
        "feature_groups": dict(FEATURE_SCHEMA),
        "drift_scenarios": {
            k: {
                "description": v["description"],
                "drift_mode": v["drift_mode"],
                "affected_features": v["affected_features"],
                "drift_vector": v["drift_vector"],
                "multiplicative_features": sorted(v.get("multiplicative_features", set())),
                "literature_basis": v["literature_basis"],
            }
            for k, v in DRIFT_SCENARIOS.items()
        },
        "initial_distributions": {
            feat: {
                "type": info[0],
                "human_params": list(info[1]),
                "bot_params": list(info[2]),
                "evidence": info[3],
            }
            for feat, info in INIT_DISTRIBUTIONS.items()
        },
        "references": [
            "Cresci, S., et al. (2017). The paradigm-shift of social spambots. WWW.",
            "Yang, K.-C., et al. (2020). Scalable and Generalizable Social Bot Detection. AAAI. (TwiBot-20)",
            "Feng, S., et al. (2022). TwiBot-22: Towards Graph-Based Twitter Bot Detection. NeurIPS.",
            "Hays, C., et al. (2025). Bots Don't Sit Still: A Longitudinal Study. arXiv.",
            "BotEvo (2026). LLM-Driven Social Bot Detection via Behavioral Evolution. Springer.",
            "TwiBotX (2026). Interpretable ML with SHAP. Springer.",
            "MDPI (2026). Explainable Feature-Based Approach for Social Bots.",
        ],
        "license": "CC BY 4.0",
    }
    config_path = os.path.join(output_dir, "generation_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: generation_config.json")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BotDriftBench: Drift Injection Benchmark for Bot Detection"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accounts", type=int, default=1000,
                        help="Total accounts (default: 1000 for v0.2)")
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--drift-magnitude", type=float, default=1.0,
                        help="Drift magnitude multiplier (0.1=subtle, 1.0=standard, 2.0=strong)")
    parser.add_argument("--tau-start", type=int, default=5,
                        help="Drift onset window (default: 5)")
    parser.add_argument("--tau-end", type=int, default=15,
                        help="Drift completion window (default: 15)")
    parser.add_argument("--output-dir", type=str, default="dataset")
    args = parser.parse_args()

    print("=" * 65)
    print("BotDriftBench v0.2 — Dataset Generator")
    print("=" * 65)
    print(f"  Seed:            {args.seed}")
    print(f"  Accounts:        {args.accounts:,}")
    print(f"  Time windows:    {args.windows}")
    print(f"  Drift magnitude: {args.drift_magnitude}")
    print(f"  Change point:    tau=[{args.tau_start}, {args.tau_end}]")
    print(f"  Expected obs:    {args.accounts * args.windows:,}")
    print()

    df, scenarios = generate_trajectories(
        n_accounts=args.accounts,
        n_windows=args.windows,
        seed=args.seed,
        drift_magnitude=args.drift_magnitude,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        campaign_windows=set(),
    )

    validate(df, args.tau_start)
    save_all(df, scenarios, args.output_dir, args)

    print("\n" + "=" * 65)
    print("Generation complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
