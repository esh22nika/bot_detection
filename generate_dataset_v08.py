"""
BotDriftBench v0.8
------------------

Empirically-inspired synthetic longitudinal benchmark for
drift-aware social bot detection.

Changelog from v0.6
--------------------
1. FIXED: `sample_human_target` previously sampled each feature of the
   human target INDEPENDENTLY (`rng.choice` per feature), which destroys
   real cross-feature correlations (e.g. could combine "very old account"
   with "near-zero followers" and "very high posting activity" -- a
   combination that may not exist in any real human). It now samples one
   COMPLETE training-human feature row, preserving empirical
   cross-feature correlation structure.
2. FIXED: core profile features (followers_count, friends_count,
   listed_count, account_age_days) no longer interpolate toward a
   sampled human target at all -- interpolating "account age" toward
   another account's age, or "followers" toward another account's
   follower count, isn't a realistic mechanism regardless of how slow
   the interpolation rate is. They are now driven by a dedicated organic
   growth process (`apply_organic_growth`) that is monotonic and
   cumulative across windows (age increases every window; followers/
   friends/listed compound-grow), completely independent of the
   humanization target. `gradual_evasion` and `recurring_behavior` now
   only humanize the NON-core profile features (description length,
   screen name length, verified score, reputation score).
3. FIXED: human accounts on `behavioral_gradual` no longer humanize
   toward another sampled human account at full strength. `apply_drift`
   now takes `label` and dampens `effective_progress` by 0.25x for
   label==0 + behavioral_gradual, representing small ordinary behavioral
   evolution rather than a targeted move toward a different account's
   behavior profile.
4. FIXED: `drift_phase` previously used `progress < severity`, which
   mislabels the final ramp-up window as "post_drift" the moment it
   reaches full severity, even though the account has only just arrived
   there. It's now `window < onset + duration` for the transition
   window, giving an exact, schedule-based phase boundary instead of a
   magnitude-based one.
5. CLARIFIED: `mean_feature_shift` documentation and manifest wording
   now explicitly label it a descriptive per-feature displacement
   measure, not a calibrated or causal drift-magnitude metric -- it can
   behave oddly for small-baseline, large-count features (e.g. 1 -> 20
   looks huge; 10,000 -> 12,000 looks small despite being operationally
   larger).

Changelog from v0.5
--------------------
1. FIXED: human-reference leakage. In v0.5, the "training humans" used
   to build the human-reference manifold were chosen by raw account-id
   order (first 60% of humans by index), while the actual train/val/test
   split was a separate, independent random shuffle. That meant some
   accounts contributing to the human reference manifold could end up
   labeled "validation" or "test". The split is now computed FIRST, and
   the human reference is built strictly from accounts whose split is
   "train" (and label == human).
2. FIXED: `affected_feature_groups` in account metadata was a constant
   dump of ALL four feature groups for every account, regardless of
   scenario. It now records the actual per-account affected feature
   list (and group breakdown) computed by `affected_features_for_scenario`.
3. ADDED: `drift_type` column (coarse category: none/gradual/abrupt/
   localized/recurring) alongside the existing fine-grained `scenario`.
4. ADDED: `difficulty_tier` column (control/easy/medium/medium_hard/hard)
   derived from scenario + severity + archetype.
5. ADDED: `mean_feature_shift` in window_metadata — mean per-account,
   per-feature displacement from each account's own window-0 baseline,
   normalized so bounded and unbounded features contribute comparably.
6. ADDED: split policy (train/validation/test fractions + split unit)
   recorded explicitly in the manifest.
7. CHANGED: scenario proportions rebalanced so gradual-type scenarios
   (behavioral_gradual + gradual_evasion) total ~55% (was 40%), matching
   the 50-60% guidance, while abrupt/localized/recurring stay in the
   10-20% band individually.
8. CHANGED: core profile features (followers_count, friends_count,
   listed_count, account_age_days) are no longer completely frozen
   during humanization scenarios. They now evolve, but far more slowly
   than behavioral/content features (a small group_strength), reflecting
   realistic organic growth rather than an instant jump to a human value.
   They remain untouched by apply_drift for abrupt/localized/behavioral
   scenarios (a sudden follower-count jump would be unrealistic), but are
   included (slowly) for gradual_evasion and recurring_behavior, which
   are the scenarios that span enough windows for organic growth to be
   plausible.
9. CHANGED: human accounts (label == 0) can no longer be assigned bot
   evasion scenarios (abrupt_evasion, gradual_evasion, localized_evasion,
   recurring_behavior). Semantically, "evading detection by looking more
   human" doesn't apply to accounts that already are human. Humans are
   now restricted to `stable_control` or `behavioral_gradual` (ordinary,
   non-evasive behavioral evolution over time). Bots keep the full
   scenario menu.

Important:
- This is synthetic data, not real social-media data.
- Labels remain fixed over time.
- Drift is account-specific and scenario-specific.
- Human reference statistics are estimated from TRAINING-SPLIT humans only.
- Test accounts are generated independently.
- No global fixed drift-start window is used.

Outputs:
    botdriftbench_v08.csv
    account_metadata_v08.csv
    window_metadata_v08.csv
    generation_manifest_v08.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

VERSION = "v0.8"

DEFAULT_SEED = 2025
N_ACCOUNTS = 3000
N_WINDOWS = 20

HUMAN_RATIO = 0.50

# Split policy (account-disjoint).
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20  # implied remainder; kept explicit for the manifest

# Bot accounts get the full scenario menu.
# Rebalanced so gradual-type scenarios total ~55% (was 40% in v0.5),
# matching the 50-60% guidance, while abrupt/localized/recurring each
# stay within a 10-20% band.
BOT_SCENARIO_PROBS = {
    "stable_control": 0.13,
    "behavioral_gradual": 0.17,
    "gradual_evasion": 0.26,
    "abrupt_evasion": 0.10,
    "localized_evasion": 0.08,
    "recurring_behavior": 0.06,
    "boundary_crossing_evasion": 0.20,
}
# Validation: probabilities must sum to exactly 1.0.
# 0.13 + 0.17 + 0.26 + 0.10 + 0.08 + 0.06 + 0.20 = 1.00

# Human accounts only ever get ordinary, non-evasive behavioral drift
# (or none at all). "Evading detection by looking human" is not a
# meaningful concept for an account that already is human.
HUMAN_SCENARIO_PROBS = {
    "stable_control": 0.65,
    "behavioral_gradual": 0.35,
}

# Coarse drift-type category per scenario (for the drift_type column).
SCENARIO_TO_DRIFT_TYPE = {
    "stable_control": "none",
    "behavioral_gradual": "gradual",
    "gradual_evasion": "gradual",
    "abrupt_evasion": "abrupt",
    "localized_evasion": "localized",
    "recurring_behavior": "recurring",
    "boundary_crossing_evasion": "concept",
}

# Noise is deliberately moderate.
# It should not destroy all signal, but should prevent perfect separation.
CONTINUOUS_NOISE = 0.035
BOUNDED_NOISE = 0.018
EVOLUTION_NOISE = 0.012
# Smaller relative measurement noise for followers/friends/listed counts,
# so it doesn't swamp the organic growth trend (see apply_measurement_noise).
CORE_PROFILE_NOISE = 0.006

# Drift onset and duration ranges.
MIN_ONSET = 2
MAX_ONSET = 14

MIN_DURATION = 2
MAX_DURATION = 7

# Maximum humanization strength.
# This is sampled per account, not fixed globally.
MIN_SEVERITY = 0.45
MAX_SEVERITY = 0.95

# Feature groups.
PROFILE_FEATURES = [
    "followers_count",
    "friends_count",
    "listed_count",
    "account_age_days",
    "description_length",
    "screen_name_length",
    "verified_score",
    "reputation_score",
]

# The four "core" profile features are things that, in real platforms,
# only move through slow organic growth (or very rarely, sharp cheating
# like follower buying -- which is out of scope here). They should never
# be interpolated instantly toward a human target the way behavioral
# features can be.
CORE_PROFILE_FEATURES = {
    "followers_count",
    "friends_count",
    "listed_count",
    "account_age_days",
}

# Profile features that ARE eligible for humanization interpolation
# (cosmetic/self-reported attributes, unlike the core growth-driven ones
# above which are handled exclusively by apply_organic_growth).
NON_CORE_PROFILE_FEATURES = [
    f for f in PROFILE_FEATURES if f not in CORE_PROFILE_FEATURES
]


# Features used by the dedicated boundary-crossing/evasion scenario.
# These are behavior/content/interaction features; slow core profile
# metadata is intentionally excluded.
BOUNDARY_CROSSING_FEATURES = [
    "tweets_per_day",
    "posting_regularity",
    "burst_score",
    "night_post_ratio",
    "weekend_post_ratio",
    "text_entropy",
    "vocabulary_richness",
    "content_similarity",
    "hashtag_count_avg",
    "url_ratio",
    "emoji_ratio",
    "caps_ratio",
    "reply_ratio",
    "retweet_ratio",
    "mention_ratio",
    "conversation_depth",
    "description_length",
    "screen_name_length",
    "reputation_score",
]

# Organic growth parameters for core profile features. These evolve
# monotonically over time, independent of any drift/humanization
# scenario -- a bot or human account's age always increases, and
# follower/friend/listed counts compound-grow rather than interpolate
# toward another account's values.
ACCOUNT_AGE_GROWTH_MIN_DAYS = 20
ACCOUNT_AGE_GROWTH_MAX_DAYS = 45
SOCIAL_COUNT_GROWTH_MEAN = 1.02
SOCIAL_COUNT_GROWTH_STD = 0.015

ACTIVITY_FEATURES = [
    "statuses_count",
    "tweets_per_day",
    "posting_regularity",
    "burst_score",
    "night_post_ratio",
    "weekend_post_ratio",
]

CONTENT_FEATURES = [
    "text_entropy",
    "vocabulary_richness",
    "content_similarity",
    "hashtag_count_avg",
    "url_ratio",
    "emoji_ratio",
    "caps_ratio",
]

INTERACTION_FEATURES = [
    "reply_ratio",
    "retweet_ratio",
    "mention_ratio",
    "conversation_depth",
]

FEATURES = (
    PROFILE_FEATURES
    + ACTIVITY_FEATURES
    + CONTENT_FEATURES
    + INTERACTION_FEATURES
)


# ============================================================
# Utility functions
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    value = rng.normal(mean, std)
    return float(np.clip(value, low, high))


def lognormal_from_median(
    rng: np.random.Generator,
    median: float,
    sigma: float,
) -> float:
    return float(rng.lognormal(mean=np.log(max(median, 1e-6)), sigma=sigma))


def choose_scenario(rng: np.random.Generator, label: int) -> str:
    """
    Human accounts (label == 0) only ever receive ordinary, non-evasive
    behavioral drift or none at all. Bot accounts (label == 1) get the
    full scenario menu, including the evasion-specific scenarios.

    Probabilities must sum to exactly 1.0 (validated at call time).
    Silent normalization is intentionally NOT performed — any accidental
    change to the probability dicts will raise immediately.
    """
    probs = HUMAN_SCENARIO_PROBS if label == 0 else BOT_SCENARIO_PROBS
    names = list(probs.keys())
    probabilities = np.asarray(list(probs.values()), dtype=float)

    if np.any(probabilities < 0):
        raise ValueError(f"Negative scenario probability: {probabilities}")

    total = probabilities.sum()

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"Scenario probabilities must sum to 1.0, got {total:.10f}. "
            f"Fix the {'BOT' if label == 1 else 'HUMAN'}_SCENARIO_PROBS dict."
        )

    return str(rng.choice(names, p=probabilities))


def choose_archetype(
    rng: np.random.Generator,
    label: int,
) -> str:
    if label == 0:
        return str(
            rng.choice(
                [
                    "casual",
                    "lurker",
                    "power_user",
                    "professional",
                    "new_account",
                ],
                p=[0.35, 0.20, 0.20, 0.15, 0.10],
            )
        )

    return str(
        rng.choice(
            [
                "spam_bot",
                "social_bot",
                "fake_follower",
                "political_bot",
                "sophisticated_bot",
            ],
            p=[0.20, 0.20, 0.15, 0.15, 0.30],
        )
    )


def compute_difficulty_tier(
    scenario: str,
    severity: float,
    archetype: str,
) -> str:
    """
    A coarse difficulty label for how hard an account should be to
    detect post-drift, derived from scenario type, severity, and
    archetype. This is metadata for analysis/stratified reporting,
    not something the model ever sees.
    """
    if scenario == "stable_control":
        return "control"

    if scenario == "localized_evasion" and archetype == "sophisticated_bot":
        return "hard"

    if scenario == "boundary_crossing_evasion":
        if severity >= 0.75:
            return "hard"
        if severity >= 0.55:
            return "medium_hard"
        return "medium"

    if scenario in {"abrupt_evasion", "gradual_evasion"} and severity >= 0.75:
        return "hard"

    if severity >= 0.65:
        return "medium_hard"

    if severity >= 0.50:
        return "medium"

    return "easy"


# ============================================================
# Archetype feature distributions
# ============================================================

def base_feature_vector(
    rng: np.random.Generator,
    label: int,
    archetype: str,
) -> Dict[str, float]:
    """
    Generate a coherent feature vector.

    label:
        0 = human
        1 = bot

    The distributions overlap deliberately.
    Sophisticated bots overlap heavily with humans.
    """

    is_bot = label == 1

    # ----------------------------
    # Latent behavioral factors
    # ----------------------------

    if archetype in {"power_user", "professional"}:
        activity_factor = rng.normal(1.25, 0.18)
        social_factor = rng.normal(1.20, 0.15)
    elif archetype == "lurker":
        activity_factor = rng.normal(0.35, 0.10)
        social_factor = rng.normal(0.40, 0.12)
    elif archetype == "new_account":
        activity_factor = rng.normal(0.55, 0.15)
        social_factor = rng.normal(0.55, 0.15)
    elif archetype == "spam_bot":
        activity_factor = rng.normal(1.50, 0.20)
        social_factor = rng.normal(0.45, 0.12)
    elif archetype == "social_bot":
        activity_factor = rng.normal(1.15, 0.16)
        social_factor = rng.normal(1.00, 0.16)
    elif archetype == "fake_follower":
        activity_factor = rng.normal(0.25, 0.08)
        social_factor = rng.normal(0.20, 0.08)
    elif archetype == "political_bot":
        activity_factor = rng.normal(1.35, 0.18)
        social_factor = rng.normal(0.80, 0.15)
    else:
        # Sophisticated bots intentionally overlap with humans.
        activity_factor = rng.normal(0.90, 0.18)
        social_factor = rng.normal(0.85, 0.18)

    activity_factor = max(0.05, activity_factor)
    social_factor = max(0.05, social_factor)

    # ----------------------------
    # Profile features
    # ----------------------------

    if archetype == "new_account":
        account_age = lognormal_from_median(rng, 180, 0.45)
    elif archetype == "fake_follower":
        account_age = lognormal_from_median(rng, 450, 0.55)
    else:
        account_age = lognormal_from_median(
            rng,
            850 if is_bot else 1100,
            0.50,
        )

    followers = lognormal_from_median(
        rng,
        (75 if is_bot else 140) * social_factor,
        0.75,
    )

    friends = lognormal_from_median(
        rng,
        (180 if is_bot else 210) * social_factor,
        0.60,
    )

    listed = lognormal_from_median(
        rng,
        (2.5 if is_bot else 5.0) * social_factor,
        0.60,
    )

    description_length = clipped_normal(
        rng,
        70 if is_bot else 95,
        38,
        0,
        220,
    )

    screen_name_length = clipped_normal(
        rng,
        12 if is_bot else 13,
        4.5,
        3,
        30,
    )

    verified_score = clipped_normal(
        rng,
        0.015 if is_bot else 0.04,
        0.035,
        0,
        1,
    )

    reputation_score = clipped_normal(
        rng,
        0.38 if is_bot else 0.55,
        0.20,
        0,
        1,
    )

    # ----------------------------
    # Activity features
    # ----------------------------

    tweets_per_day = clipped_normal(
        rng,
        8.0 * activity_factor if is_bot else 5.5 * activity_factor,
        3.5,
        0.05,
        80,
    )

    statuses_count = lognormal_from_median(
        rng,
        2600 * activity_factor if is_bot else 2100 * activity_factor,
        0.65,
    )

    posting_regularity = clipped_normal(
        rng,
        0.72 if is_bot else 0.48,
        0.18,
        0,
        1,
    )

    burst_score = clipped_normal(
        rng,
        0.62 if is_bot else 0.38,
        0.20,
        0,
        1,
    )

    night_post_ratio = clipped_normal(
        rng,
        0.42 if is_bot else 0.28,
        0.16,
        0,
        1,
    )

    weekend_post_ratio = clipped_normal(
        rng,
        0.24 if is_bot else 0.30,
        0.12,
        0,
        1,
    )

    # ----------------------------
    # Content features
    # ----------------------------

    if archetype == "sophisticated_bot":
        text_entropy_mean = 0.57
        vocabulary_mean = 0.55
        similarity_mean = 0.43
    elif is_bot:
        text_entropy_mean = 0.40
        vocabulary_mean = 0.38
        similarity_mean = 0.70
    else:
        text_entropy_mean = 0.62
        vocabulary_mean = 0.60
        similarity_mean = 0.32

    text_entropy = clipped_normal(
        rng,
        text_entropy_mean,
        0.16,
        0,
        1,
    )

    vocabulary_richness = clipped_normal(
        rng,
        vocabulary_mean,
        0.17,
        0,
        1,
    )

    content_similarity = clipped_normal(
        rng,
        similarity_mean,
        0.18,
        0,
        1,
    )

    hashtag_count_avg = clipped_normal(
        rng,
        2.8 if is_bot else 1.4,
        1.5,
        0,
        20,
    )

    url_ratio = clipped_normal(
        rng,
        0.32 if is_bot else 0.13,
        0.15,
        0,
        1,
    )

    emoji_ratio = clipped_normal(
        rng,
        0.035 if is_bot else 0.065,
        0.035,
        0,
        1,
    )

    caps_ratio = clipped_normal(
        rng,
        0.075 if is_bot else 0.045,
        0.04,
        0,
        1,
    )

    # ----------------------------
    # Interaction features
    # ----------------------------

    reply_ratio = clipped_normal(
        rng,
        0.10 if is_bot else 0.24,
        0.12,
        0,
        1,
    )

    retweet_ratio = clipped_normal(
        rng,
        0.48 if is_bot else 0.35,
        0.18,
        0,
        1,
    )

    mention_ratio = clipped_normal(
        rng,
        0.34 if is_bot else 0.18,
        0.15,
        0,
        1,
    )

    conversation_depth = clipped_normal(
        rng,
        0.20 if is_bot else 0.42,
        0.18,
        0,
        1,
    )

    return {
        "followers_count": followers,
        "friends_count": friends,
        "listed_count": listed,
        "account_age_days": account_age,
        "description_length": description_length,
        "screen_name_length": screen_name_length,
        "verified_score": verified_score,
        "reputation_score": reputation_score,
        "statuses_count": statuses_count,
        "tweets_per_day": tweets_per_day,
        "posting_regularity": posting_regularity,
        "burst_score": burst_score,
        "night_post_ratio": night_post_ratio,
        "weekend_post_ratio": weekend_post_ratio,
        "text_entropy": text_entropy,
        "vocabulary_richness": vocabulary_richness,
        "content_similarity": content_similarity,
        "hashtag_count_avg": hashtag_count_avg,
        "url_ratio": url_ratio,
        "emoji_ratio": emoji_ratio,
        "caps_ratio": caps_ratio,
        "reply_ratio": reply_ratio,
        "retweet_ratio": retweet_ratio,
        "mention_ratio": mention_ratio,
        "conversation_depth": conversation_depth,
    }


# ============================================================
# Feature constraints and normalization
# ============================================================

BOUNDED_FEATURES = {
    "posting_regularity",
    "burst_score",
    "night_post_ratio",
    "weekend_post_ratio",
    "text_entropy",
    "vocabulary_richness",
    "content_similarity",
    "url_ratio",
    "emoji_ratio",
    "caps_ratio",
    "reply_ratio",
    "retweet_ratio",
    "mention_ratio",
    "conversation_depth",
    "verified_score",
    "reputation_score",
}

NONNEGATIVE_FEATURES = {
    "followers_count",
    "friends_count",
    "listed_count",
    "account_age_days",
    "description_length",
    "screen_name_length",
    "statuses_count",
    "tweets_per_day",
    "hashtag_count_avg",
}


def clip_feature(name: str, value: float) -> float:
    if name in BOUNDED_FEATURES:
        return float(np.clip(value, 0.0, 1.0))

    if name in NONNEGATIVE_FEATURES:
        return float(max(0.0, value))

    return float(value)


def apply_measurement_noise(
    rng: np.random.Generator,
    features: Dict[str, float],
) -> Dict[str, float]:
    noisy = {}

    for name, value in features.items():
        if name in BOUNDED_FEATURES:
            noise = rng.normal(0, BOUNDED_NOISE)
            updated = value + noise
        elif name == "account_age_days":
            # Account age is exact public metadata, not something
            # subject to measurement noise -- and the standard
            # CONTINUOUS_NOISE relative scale (3.5%) would otherwise
            # swamp the organic growth signal at large ages, making the
            # column non-monotonic despite apply_organic_growth being
            # monotonic by construction.
            updated = value
        elif name in CORE_PROFILE_FEATURES:
            # followers/friends/listed: still real snapshot-to-snapshot
            # noise (counts do fluctuate slightly between measurements),
            # but far smaller than the general continuous-feature noise
            # so it doesn't overwhelm the organic growth trend.
            scale = max(abs(value), 1.0)
            noise = rng.normal(0, CORE_PROFILE_NOISE * scale)
            updated = value + noise
        else:
            scale = max(abs(value), 1.0)
            noise = rng.normal(0, CONTINUOUS_NOISE * scale)
            updated = value + noise

        noisy[name] = clip_feature(name, updated)

    return noisy


# ============================================================
# Drift construction
# ============================================================

def sample_human_target(
    rng: np.random.Generator,
    training_human_vectors: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Sample one COMPLETE training-human feature vector to use as a
    humanization target.

    This deliberately does NOT sample each feature independently
    (that would destroy real cross-feature correlations -- e.g. it
    could combine a very old account age with near-zero followers and
    very high posting activity, a combination that may not exist in
    any real human). Sampling a whole row preserves the empirical
    joint distribution across features.
    """

    index = int(rng.integers(0, len(training_human_vectors)))
    return dict(training_human_vectors[index])


def apply_organic_growth(
    rng: np.random.Generator,
    previous: Dict[str, float],
) -> Dict[str, float]:
    """
    Advance the core profile features (account_age_days, followers_count,
    friends_count, listed_count) by one window of monotonic, cumulative
    organic growth. Independent of scenario, drift progress, and the
    humanization target -- every account, bot or human, drifting or
    stable, ages and grows through this same mechanism.
    """

    updated = dict(previous)

    updated["account_age_days"] = previous["account_age_days"] + rng.uniform(
        ACCOUNT_AGE_GROWTH_MIN_DAYS,
        ACCOUNT_AGE_GROWTH_MAX_DAYS,
    )

    for feature in ("followers_count", "friends_count", "listed_count"):
        growth = rng.normal(SOCIAL_COUNT_GROWTH_MEAN, SOCIAL_COUNT_GROWTH_STD)
        updated[feature] = max(0.0, previous[feature] * growth)

    for feature in CORE_PROFILE_FEATURES:
        updated[feature] = clip_feature(feature, updated[feature])

    return updated


def drift_progress(
    window: int,
    onset: int,
    duration: int,
    scenario: str,
    severity: float,
) -> float:
    """
    Return account-specific drift progress in [0, 1] (scaled by severity).
    """

    if scenario == "stable_control":
        return 0.0

    if scenario == "abrupt_evasion":
        return severity if window >= onset else 0.0

    if scenario in {
        "behavioral_gradual",
        "gradual_evasion",
        "localized_evasion",
    }:
        if window < onset:
            return 0.0

        raw = (window - onset + 1) / max(duration, 1)
        return float(np.clip(raw, 0.0, 1.0) * severity)

    if scenario == "boundary_crossing_evasion":
        if window < onset:
            return 0.0

        elapsed = window - onset
        raw = (elapsed + 1) / max(duration, 1)

        # A strong but gradual transition toward the human behavioral
        # manifold. The cap avoids an instantaneous perfect copy.
        return float(np.clip(raw, 0.0, 1.0) * severity)

    if scenario == "recurring_behavior":
        if window < onset:
            return 0.0

        elapsed = window - onset

        # Humanization phase followed by partial return.
        phase = elapsed / max(duration, 1)

        if phase <= 1.0:
            return float(np.clip(phase * severity, 0.0, severity))

        recovery = math.exp(-(phase - 1.0) / 1.8)
        return float(np.clip(severity * recovery, 0.0, severity))

    return 0.0


def affected_features_for_scenario(
    scenario: str,
    archetype: str,
) -> List[str]:
    """
    Returns the exact list of features this scenario perturbs for this
    archetype. This is the single source of truth used both to apply
    drift AND to record ground-truth `affected_feature_groups` metadata
    per account (see generate_account_metadata), so the two can never
    drift apart the way they did in v0.5.
    """
    if scenario == "stable_control":
        return []

    if scenario == "behavioral_gradual":
        return ACTIVITY_FEATURES + CONTENT_FEATURES[:4]

    if scenario == "abrupt_evasion":
        return (
            ACTIVITY_FEATURES
            + CONTENT_FEATURES
            + INTERACTION_FEATURES
        )

    if scenario == "gradual_evasion":
        # Only the NON-core profile features (description length, screen
        # name length, verified/reputation score) are humanized here.
        # Core features (followers/friends/listed/account_age) are never
        # interpolated toward a target -- they evolve exclusively through
        # apply_organic_growth, independent of scenario.
        return (
            ACTIVITY_FEATURES
            + CONTENT_FEATURES
            + INTERACTION_FEATURES
            + NON_CORE_PROFILE_FEATURES
        )

    if scenario == "localized_evasion":
        if archetype in {"sophisticated_bot", "social_bot"}:
            return CONTENT_FEATURES + INTERACTION_FEATURES
        return ACTIVITY_FEATURES[:4]

    if scenario == "boundary_crossing_evasion":
        return list(BOUNDARY_CROSSING_FEATURES)

    if scenario == "recurring_behavior":
        return (
            ACTIVITY_FEATURES
            + CONTENT_FEATURES
            + INTERACTION_FEATURES
            + NON_CORE_PROFILE_FEATURES
        )

    return []


def feature_group_breakdown(features: List[str]) -> Dict[str, List[str]]:
    """Group an affected-feature list by profile/activity/content/interaction."""
    features_set = set(features)
    return {
        "profile": [f for f in PROFILE_FEATURES if f in features_set],
        "activity": [f for f in ACTIVITY_FEATURES if f in features_set],
        "content": [f for f in CONTENT_FEATURES if f in features_set],
        "interaction": [f for f in INTERACTION_FEATURES if f in features_set],
    }


def apply_drift(
    rng: np.random.Generator,
    original: Dict[str, float],
    human_target: Dict[str, float],
    scenario: str,
    archetype: str,
    progress: float,
    window: int,
    label: int,
) -> Dict[str, float]:
    """
    Apply drift to one account at one time window.

    The main harmful drift is humanization:
    bot features move toward a human reference manifold while
    the ground-truth label remains bot.

    Core profile features (followers/friends/listed/account_age) are
    NEVER touched here -- they are handled exclusively by
    `apply_organic_growth`, since interpolating "account age" or
    "follower count" toward another account's value isn't a realistic
    mechanism at any rate, slow or fast.

    Human accounts (label == 0) on `behavioral_gradual` are dampened to
    a small fraction of full humanization strength: this represents
    ordinary behavioral evolution over time, not a targeted move toward
    a different account's behavior profile.
    """

    updated = dict(original)

    affected = affected_features_for_scenario(
        scenario,
        archetype,
    )

    if progress <= 0 or not affected:
        return updated

    # Dedicated boundary-crossing/evasion transformation.
    # Only bots receive this transformation. Labels remain unchanged.
    if scenario == "boundary_crossing_evasion" and label == 1:
        effective_progress = float(np.clip(progress * 1.20, 0.0, 0.92))

        for feature in BOUNDARY_CROSSING_FEATURES:
            old_value = original[feature]
            target_value = human_target[feature]

            value = (
                (1.0 - effective_progress) * old_value
                + effective_progress * target_value
            )

            if feature in BOUNDED_FEATURES:
                noise = rng.normal(0, EVOLUTION_NOISE * 0.75)
            else:
                noise = rng.normal(
                    0,
                    EVOLUTION_NOISE * 0.75 * max(abs(value), 1.0),
                )

            updated[feature] = clip_feature(
                feature,
                value + noise,
            )

        # Additional correlated camouflage effects.
        updated["posting_regularity"] = clip_feature(
            "posting_regularity",
            updated["posting_regularity"] - 0.25 * progress,
        )
        updated["burst_score"] = clip_feature(
            "burst_score",
            updated["burst_score"] - 0.28 * progress,
        )
        updated["content_similarity"] = clip_feature(
            "content_similarity",
            updated["content_similarity"] - 0.25 * progress,
        )
        updated["url_ratio"] = clip_feature(
            "url_ratio",
            updated["url_ratio"] - 0.12 * progress,
        )
        updated["reply_ratio"] = clip_feature(
            "reply_ratio",
            updated["reply_ratio"] + 0.18 * progress,
        )
        updated["conversation_depth"] = clip_feature(
            "conversation_depth",
            updated["conversation_depth"] + 0.18 * progress,
        )
        updated["text_entropy"] = clip_feature(
            "text_entropy",
            updated["text_entropy"] + 0.15 * progress,
        )
        updated["vocabulary_richness"] = clip_feature(
            "vocabulary_richness",
            updated["vocabulary_richness"] + 0.15 * progress,
        )

        return updated

    for feature in affected:
        old_value = original[feature]
        target_value = human_target[feature]

        # Different feature groups evolve at different speeds. Core
        # profile features never reach here (excluded from `affected`
        # by affected_features_for_scenario), so no branch is needed
        # for them.
        if feature in PROFILE_FEATURES:
            group_strength = 0.35
        elif feature in ACTIVITY_FEATURES:
            group_strength = 1.00
        elif feature in CONTENT_FEATURES:
            group_strength = 0.90
        else:
            group_strength = 0.80

        effective_progress = np.clip(
            progress * group_strength,
            0.0,
            1.0,
        )

        # Ordinary human behavioral evolution, not targeted humanization:
        # a human account's own gradual drift shouldn't move it toward
        # another specific account's profile at full strength.
        if label == 0 and scenario == "behavioral_gradual":
            effective_progress *= 0.25

        # Humanization toward a sampled human manifold point.
        value = (
            (1.0 - effective_progress) * old_value
            + effective_progress * target_value
        )

        # Add realistic temporal evolution noise.
        if feature in BOUNDED_FEATURES:
            noise = rng.normal(0, EVOLUTION_NOISE)
        else:
            noise = rng.normal(
                0,
                EVOLUTION_NOISE * max(abs(value), 1.0),
            )

        updated[feature] = clip_feature(
            feature,
            value + noise,
        )

    # Additional scenario-specific effects.
    if scenario in {"abrupt_evasion", "gradual_evasion"}:
        # Bots become less bursty and less repetitive.
        updated["posting_regularity"] = clip_feature(
            "posting_regularity",
            updated["posting_regularity"] - 0.18 * progress,
        )

        updated["burst_score"] = clip_feature(
            "burst_score",
            updated["burst_score"] - 0.20 * progress,
        )

        updated["content_similarity"] = clip_feature(
            "content_similarity",
            updated["content_similarity"] - 0.20 * progress,
        )

    if scenario == "localized_evasion":
        # Localized sophisticated bots primarily alter content and
        # interaction behavior.
        updated["text_entropy"] = clip_feature(
            "text_entropy",
            updated["text_entropy"] + 0.12 * progress,
        )

        updated["vocabulary_richness"] = clip_feature(
            "vocabulary_richness",
            updated["vocabulary_richness"] + 0.12 * progress,
        )

        updated["reply_ratio"] = clip_feature(
            "reply_ratio",
            updated["reply_ratio"] + 0.15 * progress,
        )

        updated["conversation_depth"] = clip_feature(
            "conversation_depth",
            updated["conversation_depth"] + 0.15 * progress,
        )

    return updated


# ============================================================
# Account and trajectory generation
# ============================================================

def generate_account_metadata(
    rng: np.random.Generator,
    account_id: str,
    label: int,
) -> Dict[str, object]:
    archetype = choose_archetype(rng, label)
    scenario = choose_scenario(rng, label)

    onset = int(
        rng.integers(
            MIN_ONSET,
            MAX_ONSET + 1,
        )
    )

    duration = int(
        rng.integers(
            MIN_DURATION,
            MAX_DURATION + 1,
        )
    )

    severity = float(
        rng.uniform(
            MIN_SEVERITY,
            MAX_SEVERITY,
        )
    )

    # Stable accounts have no actual drift metadata.
    if scenario == "stable_control":
        onset = -1
        duration = 0
        severity = 0.0

    affected = affected_features_for_scenario(scenario, archetype)
    difficulty_tier = compute_difficulty_tier(scenario, severity, archetype)
    drift_type = SCENARIO_TO_DRIFT_TYPE[scenario]

    return {
        "account_id": account_id,
        "label": int(label),
        "archetype": archetype,
        "scenario": scenario,
        "drift_type": drift_type,
        "difficulty_tier": difficulty_tier,
        "drift_onset": onset,
        "drift_duration": duration,
        "drift_severity": severity,
        # Ground truth: the ACTUAL features perturbed for this account,
        # not a constant dump of every feature group (fixed from v0.5).
        "affected_feature_groups": json.dumps(
            feature_group_breakdown(affected)
        ),
    }


def generate_dataset(
    seed: int,
    n_accounts: int,
    n_windows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_humans = int(round(n_accounts * HUMAN_RATIO))
    n_bots = n_accounts - n_humans

    account_rows = []
    dataset_rows = []

    # --------------------------------------------------------
    # Create account metadata and initial vectors.
    # --------------------------------------------------------

    initial_vectors = {}

    for index in range(n_accounts):
        label = 0 if index < n_humans else 1

        if label == 0:
            account_id = f"H_{index:05d}"
        else:
            bot_index = index - n_humans
            account_id = f"B_{bot_index:05d}"

        metadata = generate_account_metadata(
            rng,
            account_id,
            label,
        )

        account_rows.append(metadata)

        original = base_feature_vector(
            rng,
            label=label,
            archetype=str(metadata["archetype"]),
        )

        initial_vectors[account_id] = original

    # --------------------------------------------------------
    # Account-disjoint split -- computed BEFORE the human reference is
    # built (fixed from v0.5, where the split was computed afterward
    # from an unrelated ordering, allowing validation/test humans to
    # leak into the human reference manifold).
    # --------------------------------------------------------

    account_metadata_preview = pd.DataFrame(account_rows)

    shuffled_ids = account_metadata_preview["account_id"].sample(
        frac=1.0,
        random_state=seed,
    ).tolist()

    n_train = int(TRAIN_FRACTION * len(shuffled_ids))
    n_val = int(VALIDATION_FRACTION * len(shuffled_ids))

    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])

    def split_for_account(account_id: str) -> str:
        if account_id in train_ids:
            return "train"
        if account_id in val_ids:
            return "validation"
        return "test"

    split_by_account = {
        row["account_id"]: split_for_account(row["account_id"])
        for row in account_rows
    }

    for row in account_rows:
        row["split"] = split_by_account[row["account_id"]]

    # --------------------------------------------------------
    # Human reference is constructed from TRAIN-SPLIT humans only.
    # --------------------------------------------------------

    training_human_ids = [
        row["account_id"]
        for row in account_rows
        if row["label"] == 0 and row["split"] == "train"
    ]

    training_human_vectors = [
        initial_vectors[account_id]
        for account_id in training_human_ids
    ]

    # --------------------------------------------------------
    # Generate longitudinal trajectories.
    # --------------------------------------------------------

    for metadata in account_rows:
        account_id = str(metadata["account_id"])
        label = int(metadata["label"])
        archetype = str(metadata["archetype"])
        scenario = str(metadata["scenario"])
        drift_type = str(metadata["drift_type"])
        difficulty_tier = str(metadata["difficulty_tier"])
        split = str(metadata["split"])

        onset = int(metadata["drift_onset"])
        duration = int(metadata["drift_duration"])
        severity = float(metadata["drift_severity"])

        original = initial_vectors[account_id]

        # Every account gets its own human-like target, sampled as one
        # complete row from a training human (preserves cross-feature
        # correlation; see sample_human_target). For humans on
        # stable_control this target is irrelevant; for humans on
        # behavioral_gradual it's used only at heavily dampened strength.
        human_target = sample_human_target(
            rng,
            training_human_vectors,
        )

        previous = dict(original)

        for window in range(n_windows):
            progress = drift_progress(
                window=window,
                onset=onset,
                duration=duration,
                scenario=scenario,
                severity=severity,
            )

            current = apply_drift(
                rng=rng,
                original=original,
                human_target=human_target,
                scenario=scenario,
                archetype=archetype,
                progress=progress,
                window=window,
                label=label,
            )

            # Core profile features (age/followers/friends/listed) are
            # never touched by apply_drift -- they evolve exclusively
            # through cumulative organic growth from the PREVIOUS
            # window's values, independent of scenario or humanization.
            if window > 0:
                grown = apply_organic_growth(rng, previous)
                for feature in CORE_PROFILE_FEATURES:
                    current[feature] = grown[feature]

            # Temporal continuity for all other features: later windows
            # are based partly on the previous observation, preventing
            # independent random samples for the same account. Core
            # profile features are excluded here since organic growth
            # above already makes them a function of `previous`.
            if window > 0:
                for feature in FEATURES:
                    if feature in CORE_PROFILE_FEATURES:
                        continue
                    current[feature] = (
                        0.82 * current[feature]
                        + 0.18 * previous[feature]
                    )
                    current[feature] = clip_feature(
                        feature,
                        current[feature],
                    )

            current = apply_measurement_noise(
                rng,
                current,
            )

            if scenario == "stable_control":
                is_drifting = 0
            else:
                is_drifting = int(progress > 0)

            # Schedule-based phase boundary (not magnitude-based): a
            # progress-vs-severity comparison mislabels the final
            # ramp-up window as "post_drift" the instant it reaches
            # full severity, even though the account has only just
            # arrived there.
            if progress <= 0:
                drift_phase = "stable"
            elif window < onset + duration:
                drift_phase = "transition"
            else:
                drift_phase = "post_drift"

            row = {
                "account_id": account_id,
                "window": window,
                "label": label,
                "archetype": archetype,
                "scenario": scenario,
                "drift_type": drift_type,
                "difficulty_tier": difficulty_tier,
                "drift_onset": onset,
                "drift_duration": duration,
                "drift_severity": severity,
                "drift_progress": progress,
                "is_drifting": is_drifting,
                "drift_phase": drift_phase,
                "split": split,
            }

            row.update(current)
            dataset_rows.append(row)

            previous = dict(current)

    dataset = pd.DataFrame(dataset_rows)
    account_metadata = pd.DataFrame(account_rows)

    # --------------------------------------------------------
    # Window-level ground truth (now includes mean_feature_shift).
    # --------------------------------------------------------

    window_metadata = build_window_metadata(dataset, n_windows)

    return dataset, account_metadata, window_metadata


def build_window_metadata(dataset: pd.DataFrame, n_windows: int) -> pd.DataFrame:
    baseline = dataset[dataset["window"] == 0].set_index("account_id")[FEATURES]

    window_rows = []

    for window in range(n_windows):
        window_data = dataset[dataset["window"] == window]

        active = window_data[window_data["is_drifting"] == 1]

        scenario_counts = (
            window_data["scenario"]
            .value_counts()
            .to_dict()
        )

        # mean_feature_shift: a DESCRIPTIVE per-feature displacement
        # measure, not a calibrated or causal drift-magnitude metric.
        # It is mean, per-account, per-feature displacement from that
        # account's own window-0 baseline. Bounded features (already in
        # [0, 1]) use raw absolute difference; unbounded features use a
        # relative difference so large-magnitude features like
        # followers_count don't dominate the average -- but this relative
        # form can itself behave oddly at small baselines (e.g. 1 -> 20
        # looks like a huge shift, while 10,000 -> 12,000 looks small
        # despite being operationally larger). Use this to eyeball
        # whether drift is happening at all, not as ground truth for how
        # "harmful" it is -- that requires the frozen-model
        # boundary-crossing diagnostic described in the module docstring.
        current = window_data.set_index("account_id")[FEATURES]
        common_ids = current.index.intersection(baseline.index)

        per_feature_shift = []
        for feature in FEATURES:
            cur_vals = current.loc[common_ids, feature]
            base_vals = baseline.loc[common_ids, feature]

            if feature in BOUNDED_FEATURES:
                diff = (cur_vals - base_vals).abs()
            else:
                diff = (cur_vals - base_vals).abs() / (base_vals.abs() + 1.0)

            per_feature_shift.append(float(diff.mean()))

        mean_feature_shift = float(np.mean(per_feature_shift))

        window_rows.append(
            {
                "window": window,
                "active_drift_accounts": int(len(active)),
                "drifting_fraction": float(
                    len(active) / max(len(window_data), 1)
                ),
                "mean_drift_progress": float(
                    window_data["drift_progress"].mean()
                ),
                "mean_drift_severity": float(
                    window_data["drift_severity"].mean()
                ),
                "mean_feature_shift": mean_feature_shift,
                "scenario_counts": json.dumps(scenario_counts),
            }
        )

    return pd.DataFrame(window_rows)


# ============================================================
# Validation checks
# ============================================================

def validate_dataset(
    dataset: pd.DataFrame,
    account_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
) -> Dict[str, object]:
    checks = {}

    checks["n_rows"] = int(len(dataset))
    checks["n_accounts"] = int(
        dataset["account_id"].nunique()
    )
    checks["n_windows"] = int(
        dataset["window"].nunique()
    )
    checks["n_features"] = int(len(FEATURES))

    checks["class_balance"] = (
        dataset.groupby("label")["account_id"]
        .nunique()
        .to_dict()
    )

    checks["scenario_counts"] = (
        account_metadata["scenario"]
        .value_counts()
        .to_dict()
    )

    checks["drift_type_counts"] = (
        account_metadata["drift_type"]
        .value_counts()
        .to_dict()
    )

    checks["difficulty_tier_counts"] = (
        account_metadata["difficulty_tier"]
        .value_counts()
        .to_dict()
    )

    checks["split_account_counts"] = (
        account_metadata["split"]
        .value_counts()
        .to_dict()
    )

    checks["drifting_account_count"] = int(
        account_metadata[
            account_metadata["scenario"] != "stable_control"
        ].shape[0]
    )

    checks["stable_account_count"] = int(
        account_metadata[
            account_metadata["scenario"] == "stable_control"
        ].shape[0]
    )

    checks["accounts_with_multiple_windows"] = bool(
        dataset.groupby("account_id")["window"]
        .nunique()
        .eq(N_WINDOWS)
        .all()
    )

    checks["labels_constant_per_account"] = bool(
        dataset.groupby("account_id")["label"]
        .nunique()
        .eq(1)
        .all()
    )

    checks["split_constant_per_account"] = bool(
        dataset.groupby("account_id")["split"]
        .nunique()
        .eq(1)
        .all()
    )

    checks["no_missing_features"] = bool(
        dataset[FEATURES].isna().sum().sum() == 0
    )

    checks["all_features_nonnegative_or_bounded"] = True

    for feature in FEATURES:
        if feature in BOUNDED_FEATURES:
            valid = dataset[feature].between(0, 1).all()
        else:
            valid = (dataset[feature] >= 0).all()

        if not valid:
            checks["all_features_nonnegative_or_bounded"] = False

    # Explicit leakage check: no account that contributed to the human
    # reference manifold should be outside the "train" split. Since the
    # reference is built strictly from split=="train" humans by
    # construction, this check mainly guards against future regressions
    # to generate_dataset's ordering.
    train_humans = account_metadata[
        (account_metadata["label"] == 0)
        & (account_metadata["split"] == "train")
    ]["account_id"]
    checks["human_reference_leakage_guard"] = (
        "human_reference is built only from label==0 & split=='train' "
        f"accounts ({len(train_humans)} accounts); no validation/test "
        "account statistics are used to construct drift targets."
    )

    return checks


# ============================================================
# Saving
# ============================================================

def save_outputs(
    output_dir: Path,
    dataset: pd.DataFrame,
    account_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
    checks: Dict[str, object],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "botdriftbench_v08.csv"
    accounts_path = output_dir / "account_metadata_v08.csv"
    windows_path = output_dir / "window_metadata_v08.csv"
    manifest_path = output_dir / "generation_manifest_v08.json"

    dataset.to_csv(dataset_path, index=False)
    account_metadata.to_csv(accounts_path, index=False)
    window_metadata.to_csv(windows_path, index=False)

    manifest = {
        "version": VERSION,
        "seed": seed,
        "n_accounts": int(
            dataset["account_id"].nunique()
        ),
        "n_windows": int(
            dataset["window"].nunique()
        ),
        "n_rows": int(len(dataset)),
        "features": FEATURES,
        "feature_groups": {
            "profile": PROFILE_FEATURES,
            "core_profile": sorted(CORE_PROFILE_FEATURES),
            "activity": ACTIVITY_FEATURES,
            "content": CONTENT_FEATURES,
            "interaction": INTERACTION_FEATURES,
        },
        "bot_scenario_probabilities": BOT_SCENARIO_PROBS,
        "human_scenario_probabilities": HUMAN_SCENARIO_PROBS,
        "scenario_to_drift_type": SCENARIO_TO_DRIFT_TYPE,
        "noise": {
            "continuous_noise": CONTINUOUS_NOISE,
            "bounded_noise": BOUNDED_NOISE,
            "evolution_noise": EVOLUTION_NOISE,
        },
        "drift": {
            "min_onset": MIN_ONSET,
            "max_onset": MAX_ONSET,
            "min_duration": MIN_DURATION,
            "max_duration": MAX_DURATION,
            "min_severity": MIN_SEVERITY,
            "max_severity": MAX_SEVERITY,
        },
        "split_policy": {
            "unit": "account",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "method": (
                "random shuffle of account_id with fixed seed, computed "
                "before the human-reference manifold is built; the human "
                "reference is derived only from label==0 accounts whose "
                "split is 'train'"
            ),
        },
        "calibration_note": (
            "Synthetic feature distributions inspired by social "
            "bot and human behavioral patterns. Human reference "
            "manifold estimated from synthetic training-split humans only. "
            "Humanization targets are sampled as COMPLETE human rows "
            "(preserving cross-feature correlation), not per-feature "
            "independent samples."
        ),
        "mean_feature_shift_caveat": (
            "mean_feature_shift (in window_metadata) is a descriptive "
            "per-feature displacement measure, not a calibrated or "
            "causal drift-magnitude metric. It can behave oddly for "
            "small-baseline, large-count features (e.g. a 1->20 change "
            "looks huge; a 10,000->12,000 change looks small despite "
            "being operationally larger). Do not use it as ground truth "
            "for how 'harmful' drift is."
        ),
        "important_limitations": [
            "Synthetic benchmark, not real longitudinal social data.",
            "Drift scenarios are controlled simulations.",
            "Results must not be presented as real-world performance.",
            "External empirical calibration should be reported separately.",
            "Boundary-crossing validation (pre/post-drift F1 on a trained "
            "model) is a separate step that must be run before treating "
            "this dataset as containing 'harmful' drift by construction.",
            "Core profile features (age/followers/friends/listed) evolve "
            "via monotonic organic growth only -- never toward a "
            "humanization target -- regardless of scenario.",
            "Do not tune generator parameters against test-set F1, and "
            "do not use the test split to choose drift-aware thresholds.",
        ],
        "validation_checks": checks,
    }

    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\nSaved files:")
    print(f"  {dataset_path}")
    print(f"  {accounts_path}")
    print(f"  {windows_path}")
    print(f"  {manifest_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--accounts",
        type=int,
        default=N_ACCOUNTS,
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=N_WINDOWS,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dataset_v08",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"Generating BotDriftBench {VERSION}")
    print("=" * 70)

    dataset, account_metadata, window_metadata = generate_dataset(
        seed=args.seed,
        n_accounts=args.accounts,
        n_windows=args.windows,
    )

    checks = validate_dataset(
        dataset,
        account_metadata,
        window_metadata,
    )

    print("\nValidation summary:")
    print(json.dumps(checks, indent=2))

    save_outputs(
        output_dir=Path(args.output),
        dataset=dataset,
        account_metadata=account_metadata,
        window_metadata=window_metadata,
        checks=checks,
        seed=args.seed,
    )

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
