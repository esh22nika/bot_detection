import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings('ignore')

SHAP_OUT = Path('results_v08/shap')
SHAP_OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path('model_artifacts/checkpoints_v08')

DATASET_PATH = Path('dataset_v08/botdriftbench_v08.csv')
df = pd.read_csv(DATASET_PATH)

META_COLS = {
    'account_id', 'window', 'label', 'archetype', 'scenario',
    'drift_type', 'difficulty_tier', 'drift_onset', 'drift_duration',
    'drift_severity', 'drift_progress', 'is_drifting', 'drift_phase', 'split'
}
feat_cols = [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
df['label_binary'] = df['label'].astype(int)

test_df = df[df['split'] == 'test']
pre_test = test_df[test_df['window'] < 2]
post_test = test_df[test_df['window'] >= 2]

X_pre = pre_test[feat_cols].values
X_post = post_test[feat_cols].values

# 1. Load exact models
with open(MODELS_DIR / 'static_model.pkl', 'rb') as f:
    static_model = pickle.load(f)
with open(MODELS_DIR / 'da_localized_final_model.pkl', 'rb') as f:
    da_final_model = pickle.load(f)

def compute_shap(model, X):
    exp = shap.TreeExplainer(model)
    v = exp.shap_values(X)
    if isinstance(v, list): v = v[-1]
    return exp, np.asarray(v)

print('Computing SHAP for 1. Static Pre-drift...')
exp_stat_pre, vals_stat_pre = compute_shap(static_model, X_pre)

print('Computing SHAP for 2. Static Post-drift...')
exp_stat_post, vals_stat_post = compute_shap(static_model, X_post)

print('Computing SHAP for 3. Localized DA Post-adaptation...')
exp_da, vals_da = compute_shap(da_final_model, X_post)

# 1. static_pre_drift_beeswarm.png
plt.figure(figsize=(10, 5.5))
shap.summary_plot(vals_stat_pre, X_pre, feature_names=feat_cols, max_display=12, show=False)
plt.title('SHAP Attribution: Frozen Static Model (Pre-Drift Windows 0–1)', fontsize=12)
plt.tight_layout()
plt.savefig(SHAP_OUT / 'static_pre_drift_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved static_pre_drift_beeswarm.png')

# 2. static_post_drift_beeswarm.png
plt.figure(figsize=(10, 5.5))
shap.summary_plot(vals_stat_post, X_post, feature_names=feat_cols, max_display=12, show=False)
plt.title('SHAP Attribution: Frozen Static Model Under Drift (Post-Drift Windows 2–19)', fontsize=12)
plt.tight_layout()
plt.savefig(SHAP_OUT / 'static_post_drift_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved static_post_drift_beeswarm.png')

# 3. localized_da_beeswarm.png
plt.figure(figsize=(10, 5.5))
shap.summary_plot(vals_da, X_post, feature_names=feat_cols, max_display=12, show=False)
plt.title('SHAP Attribution: Adapted Localized DA Model (Post-Drift Windows 2–19)', fontsize=12)
plt.tight_layout()
plt.savefig(SHAP_OUT / 'localized_da_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved localized_da_beeswarm.png')

# 4. shap_importance_comparison.csv
imp_stat_pre = np.abs(vals_stat_pre).mean(axis=0)
imp_stat_post = np.abs(vals_stat_post).mean(axis=0)
imp_da_post = np.abs(vals_da).mean(axis=0)

comp_df = pd.DataFrame({
    'feature': feat_cols,
    'static_pre_drift': imp_stat_pre,
    'static_post_drift': imp_stat_post,
    'localized_da_post': imp_da_post,
})
comp_df['static_drift_shift'] = comp_df['static_post_drift'] - comp_df['static_pre_drift']
comp_df['adaptation_recovery_shift'] = comp_df['localized_da_post'] - comp_df['static_post_drift']
comp_df = comp_df.sort_values('localized_da_post', ascending=False)
comp_df.to_csv(SHAP_OUT / 'shap_importance_comparison.csv', index=False)
print('Saved shap_importance_comparison.csv')

# 5. shap_rank_change.png
top10 = comp_df.head(10).sort_values('localized_da_post', ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(top10))
h = 0.28
ax.barh(y - h, top10['static_pre_drift'], height=h, color='#3498DB', label='Static (Pre-drift w0–1)', alpha=0.85)
ax.barh(y, top10['static_post_drift'], height=h, color='#E74C3C', label='Static (Post-drift w2–19)', alpha=0.85)
ax.barh(y + h, top10['localized_da_post'], height=h, color='#2ECC71', label='Localized DA (Adapted)', alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(top10['feature'], fontsize=10)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance Evolution: Static Pre vs Static Post vs Localized DA', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(SHAP_OUT / 'shap_rank_change.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved shap_rank_change.png')

# Copy to BotDriftBench_v08 as well
shutil_target = Path('BotDriftBench_v08/results_v08/shap')
shutil_target.mkdir(parents=True, exist_ok=True)
import shutil
for f in SHAP_OUT.glob('*'):
    shutil.copy(f, shutil_target / f.name)

print('All 5 SHAP artifacts successfully created and mirrored to BotDriftBench_v08/results_v08/shap/')
