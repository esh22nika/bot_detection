DRIFT_ONSET_WINDOW = 2
FIRST_DETECTION_WINDOW = 5
ADAPTATION_WINDOWS = [5, 8, 11, 14, 17]

EXPLANATIONS = {
    'before_drift': (
        "Both detectors operate in a stable environment, so their performance is similar. "
        "The static baseline achieves strong discrimination (F1 = 0.975–0.979) without requiring adaptation."
    ),
    'drift_injected': (
        "The static detector retains its original decision boundary and becomes less effective as behavior changes. "
        "Behavioral drift is active (bot evasion commences). The drift-aware detector monitors distributional shifts via localized KS tests."
    ),
    'adapted': (
        "The drift-aware detector detects the distribution shift and updates the model using targeted adaptation. "
        "With recency-weighted retraining, the adapted model maintains robust detection while the static detector continues to degrade."
    )
}

EVENT_DESCRIPTIONS = {
    0: {
        'type': 'training',
        'badge': 'Initialization',
        'title': 'Initial Baseline Training',
        'description': 'Static and Drift-Aware models initialized on pre-drift window 0–1 training accounts.'
    },
    2: {
        'type': 'drift_onset',
        'badge': 'Drift Begins',
        'title': 'Behavioral Drift Injected',
        'description': 'Bot evasion strategies commence across test cohort; static detector begins silent degradation.'
    },
    5: {
        'type': 'detection',
        'badge': 'Adaptation #1',
        'title': 'Drift Detected & Localized Adaptation #1',
        'description': 'Localized KS test fires (delay = 3 windows, 0 false alarms). Recency-weighted retraining restores performance.'
    },
    8: {
        'type': 'adaptation',
        'badge': 'Adaptation #2',
        'title': 'Localized Adaptation #2 Triggered',
        'description': 'Ongoing evasion detected; model retrained with recency weighting on window 8 data.'
    },
    11: {
        'type': 'adaptation',
        'badge': 'Adaptation #3',
        'title': 'Localized Adaptation #3 Triggered',
        'description': 'Secondary distribution shift detected; model refreshed at window 11.'
    },
    14: {
        'type': 'adaptation',
        'badge': 'Adaptation #4',
        'title': 'Localized Adaptation #4 Triggered',
        'description': 'Late-stage evasion counteracted via targeted retraining at window 14.'
    },
    17: {
        'type': 'adaptation',
        'badge': 'Adaptation #5',
        'title': 'Localized Adaptation #5 Triggered',
        'description': 'Final scheduled adaptation preserves bot recall (>85%) while static model collapses.'
    },
    18: {
        'type': 'worst_case',
        'badge': 'Static Nadir',
        'title': 'Static Baseline Worst Window',
        'description': 'Static detector suffers worst degradation (Macro-F1 drops to 0.6583, recall falls to 37.9%).'
    }
}
