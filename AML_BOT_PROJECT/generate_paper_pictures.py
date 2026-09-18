import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path('paper_pictures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

print("Generating publication figures for paper_pictures/...")

# -------------------------------------------------------------------------------------------------
# FIGURE 1: Benchmark Architecture & Longitudinal Split Diagram (Dataset Paper Fig 1)
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
ax.axis('off')

# Diagram blocks
# Box 1: Empirical Data Calibration
rect1 = patches.FancyBboxPatch((0.02, 0.55), 0.25, 0.38, boxstyle="round,pad=0.03", ec="#1e3a8a", fc="#eff6ff", lw=1.5)
ax.add_patch(rect1)
ax.text(0.145, 0.85, "Empirical Calibration", ha="center", va="center", fontsize=11, fontweight="bold", color="#1e3a8a")
ax.text(0.145, 0.73, "• TwiBot-22 Subset\n• Cresci-2017 Cross-ref\n• Marginal distribution fits\n• Complete human rows", ha="center", va="center", fontsize=8.5, color="#1e293b")

# Arrow 1
ax.annotate('', xy=(0.32, 0.74), xytext=(0.28, 0.74), arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))

# Box 2: BotDriftBench v0.8 Cohort
rect2 = patches.FancyBboxPatch((0.33, 0.45), 0.34, 0.48, boxstyle="round,pad=0.03", ec="#0f766e", fc="#f0fdfa", lw=1.5)
ax.add_patch(rect2)
ax.text(0.50, 0.85, "BotDriftBench v0.8 Cohort", ha="center", va="center", fontsize=11, fontweight="bold", color="#0f766e")
ax.text(0.50, 0.67, "3,000 Total Accounts (60,000 Rows)\n• 1,500 Bots / 1,500 Humans\n• 20 Temporal Windows (w0–w19)\n• 25 Account Features\n• Organic Monotonic Growth", ha="center", va="center", fontsize=8.5, color="#0f172a")

# Arrow 2
ax.annotate('', xy=(0.72, 0.74), xytext=(0.68, 0.74), arrowprops=dict(arrowstyle="->", lw=2, color="#475569"))

# Box 3: Account-Disjoint Split
rect3 = patches.FancyBboxPatch((0.73, 0.45), 0.25, 0.48, boxstyle="round,pad=0.03", ec="#7e22ce", fc="#faf5ff", lw=1.5)
ax.add_patch(rect3)
ax.text(0.855, 0.85, "Account-Disjoint Split", ha="center", va="center", fontsize=11, fontweight="bold", color="#7e22ce")
ax.text(0.855, 0.67, "Strict 3-Way Partition:\n• Train: 1,800 accts (60%)\n• Val: 600 accts (20%)\n• Test: 600 accts (20%)\nZero test account leakage", ha="center", va="center", fontsize=8.5, color="#1e293b")

# Bottom block: 7 Drift Scenarios
rect_bot = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.32, boxstyle="round,pad=0.02", ec="#b45309", fc="#fffbeb", lw=1.5)
ax.add_patch(rect_bot)
ax.text(0.50, 0.30, "7 Account-Level Longitudinal Drift Scenarios (Sum = 1.00 Exactly)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#b45309")

scen_text = (
    "1. Stable Control (13%)  |  2. Behavioral Gradual (17%)  |  3. Gradual Evasion (26%)\n"
    "4. Abrupt Evasion (10%)  |  5. Localized Evasion (8%)   |  6. Recurring Behavior (6%)\n"
    "7. Boundary-Crossing Evasion (20%)\n"
    "All scenarios follow scheduled phase boundaries: Window < (onset + duration)"
)
ax.text(0.50, 0.16, scen_text, ha="center", va="center", fontsize=8.5, color="#78350f")

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_benchmark_architecture.png', dpi=300)
plt.close()
print("Saved fig1_benchmark_architecture.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 2: Feature Separability Pre-Drift (Dataset Paper Fig 2)
# -------------------------------------------------------------------------------------------------
df = pd.read_csv('dataset_v08/botdriftbench_v08.csv')
pre_df = df[df['window'] < 2]
humans_pre = pre_df[pre_df['label'] == 0]
bots_pre = pre_df[pre_df['label'] == 1]

key_feats = ['content_similarity', 'posting_regularity', 'url_ratio', 'burst_score']
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), dpi=300)

for i, feat in enumerate(key_feats):
    ax = axes[i]
    ax.hist(humans_pre[feat], bins=25, alpha=0.6, color='#2563eb', label='Humans (w0–1)', density=True)
    ax.hist(bots_pre[feat], bins=25, alpha=0.6, color='#dc2626', label='Bots (w0–1)', density=True)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('Feature Value', fontsize=8.5)
    if i == 0:
        ax.set_ylabel('Density', fontsize=8.5)
        ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(True, alpha=0.25)

plt.suptitle('Figure 2: Empirical Baseline Separability Across Key Features (Pre-Drift Windows 0–1)', fontsize=11, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_feature_separability.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig2_feature_separability.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 3: Temporal Feature Drift Across Scenarios (Dataset Paper Fig 3)
# -------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=300)

# 1. Content similarity trajectories across scenarios
ax1 = axes[0]
scenarios = df['scenario'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
for s, c in zip(scenarios, colors):
    sub = df[(df['label'] == 1) & (df['scenario'] == s)].groupby('window')['content_similarity'].mean()
    ax1.plot(sub.index, sub.values, label=s.replace('_', ' ').title(), color=c, lw=1.8)
ax1.axvline(2, color='gray', linestyle='--', label='Onset (w2)')
ax1.set_title('Content Similarity by Scenario', fontsize=10.5, fontweight='bold')
ax1.set_xlabel('Time Window', fontsize=9)
ax1.set_ylabel('Mean Content Similarity', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=7, loc='lower left')

# 2. Posting regularity trajectories
ax2 = axes[1]
for s, c in zip(scenarios, colors):
    sub = df[(df['label'] == 1) & (df['scenario'] == s)].groupby('window')['posting_regularity'].mean()
    ax2.plot(sub.index, sub.values, label=s.replace('_', ' ').title(), color=c, lw=1.8)
ax2.axvline(2, color='gray', linestyle='--', label='Onset (w2)')
ax2.set_title('Posting Regularity by Scenario', fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Time Window', fontsize=9)
ax2.set_ylabel('Mean Posting Regularity', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Figure 3: Longitudinal Behavioral Trajectories Across 7 Scenarios', fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_temporal_drift_profiles.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig3_temporal_drift_profiles.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 4: Scenario Degradation Matrix / Bar (Dataset Paper Fig 4)
# -------------------------------------------------------------------------------------------------
scen_df = pd.read_csv('results_v08/scenario_results.csv')
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

y_pos = np.arange(len(scen_df))
h = 0.26

ax.barh(y_pos - h, scen_df['static_f1'], height=h, label='Static Baseline', color='#ef4444', alpha=0.9)
ax.barh(y_pos, scen_df['periodic6_f1'], height=h, label='Periodic-6', color='#3b82f6', alpha=0.9)
ax.barh(y_pos + h, scen_df['da_localized_f1'], height=h, label='Proposed Localized DA', color='#10b981', alpha=0.9)

ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace('_', ' ').title() for s in scen_df['scenario']], fontsize=9)
ax.set_xlabel('Macro-F1 Score', fontsize=9.5)
ax.set_title('Figure 4: Policy Performance Breakdown by Bot Scenario (Test Accounts)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

for i in range(len(scen_df)):
    diff = scen_df.loc[i, 'da_f1_advantage_vs_static']
    ax.text(0.98, y_pos[i], f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}", 
            va='center', ha='right', fontsize=7.5, fontweight='bold', 
            color='#065f46' if diff > 0 else '#991b1b')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_scenario_degradation_heatmap.png', dpi=300)
plt.close()
print("Saved fig4_scenario_degradation_heatmap.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 5: Policy Macro-F1 Trajectories (Model Paper Fig 1)
# -------------------------------------------------------------------------------------------------
win_df = pd.read_csv('results_v08/window_results.csv')
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

policies = [
    ('static', 'Static Baseline (Frozen)', '#ef4444', '-'),
    ('periodic_6', 'Periodic-6 (Scheduled)', '#3b82f6', '--'),
    ('da_localized', 'Proposed Localized DA', '#10b981', '-'),
    ('oracle_da', 'Theoretical Oracle Reference', '#8b5cf6', ':')
]

for pol, label, col, ls in policies:
    sub = win_df[win_df['policy'] == pol].sort_values('window')
    ax.plot(sub['window'], sub['f1_macro_mean'], label=label, color=col, linestyle=ls, lw=2.2)
    ax.fill_between(sub['window'], 
                    sub['f1_macro_mean'] - sub['f1_macro_std'], 
                    sub['f1_macro_mean'] + sub['f1_macro_std'], 
                    color=col, alpha=0.10)

# Annotate Drift Onset & Adaptation Triggers
ax.axvline(2, color='gray', linestyle='--', alpha=0.7, label='Drift Onset (Window 2)')
ax.scatter([5, 8, 11, 14, 17], [0.9315, 0.9198, 0.9054, 0.8906, 0.8914], 
           color='#047857', s=45, zorder=5, label='Localized DA Retrains')

ax.set_xlabel('Temporal Window', fontsize=10, fontweight='bold')
ax.set_ylabel('Macro-F1 Score', fontsize=10, fontweight='bold')
ax.set_title('Figure 1: Longitudinal Detection Performance (Mean ± 1σ across 5 seeds)', fontsize=11, fontweight='bold')
ax.set_ylim(0.62, 1.00)
ax.set_xticks(range(20))
ax.legend(fontsize=8, loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_policy_f1_trajectories.png', dpi=300)
plt.close()
print("Saved fig5_policy_f1_trajectories.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 6: Scenario Bot Recall Breakdown (Model Paper Fig 2)
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
y_pos = np.arange(len(scen_df))
h = 0.26

ax.barh(y_pos - h, scen_df['static_recall'] * 100, height=h, label='Static Baseline Recall', color='#ef4444', alpha=0.85)
ax.barh(y_pos, scen_df['periodic6_recall'] * 100, height=h, label='Periodic-6 Recall', color='#3b82f6', alpha=0.85)
ax.barh(y_pos + h, scen_df['da_localized_recall'] * 100, height=h, label='Localized DA Recall', color='#10b981', alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels([s.replace('_', ' ').title() for s in scen_df['scenario']], fontsize=9)
ax.set_xlabel('Bot Recall Rate (%)', fontsize=9.5)
ax.set_title('Figure 2: Bot Recall Across Evasion Scenarios: Recovery via Adaptation', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_scenario_policy_breakdown.png', dpi=300)
plt.close()
print("Saved fig6_scenario_policy_breakdown.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 7: Detection Delay & Retraining Timeline (Model Paper Fig 3)
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.8), dpi=300)

# Timeline bar
ax.plot([0, 19], [0, 0], color='#cbd5e1', lw=6, zorder=1)
ax.plot([0, 1], [0, 0], color='#10b981', lw=6, zorder=2, label='Pre-Drift Stable (w0–1)')
ax.plot([2, 19], [0, 0], color='#ef4444', lw=6, zorder=2, label='Drift Active (w2–19)')

# Onset marker
ax.scatter([2], [0], color='#dc2626', s=140, zorder=3, marker='X')
ax.annotate('Earliest Onset\n(Window 2)', xy=(2, 0), xytext=(2, 0.4),
            ha='center', fontsize=8, fontweight='bold', color='#dc2626',
            arrowprops=dict(arrowstyle="->", color="#dc2626"))

# Detection delay bracket
ax.plot([2, 5], [-0.25, -0.25], color='#475569', lw=1.5)
ax.plot([2, 2], [-0.2, -0.3], color='#475569', lw=1.5)
ax.plot([5, 5], [-0.2, -0.3], color='#475569', lw=1.5)
ax.text(3.5, -0.5, 'Detection Delay\n(3 Windows)', ha='center', fontsize=7.5, fontweight='bold', color='#475569')

# Retrain triggers
da_triggers = [5, 8, 11, 14, 17]
ax.scatter(da_triggers, [0]*len(da_triggers), color='#059669', s=100, zorder=4, marker='o')
for w in da_triggers:
    ax.annotate(f'Adapt\n(w{w})', xy=(w, 0), xytext=(w, 0.4),
                ha='center', fontsize=7.5, color='#059669',
                arrowprops=dict(arrowstyle="->", color="#059669"))

ax.set_xlim(-1, 20)
ax.set_ylim(-0.8, 0.8)
ax.set_xticks(range(20))
ax.set_xticklabels([f'W{i}' for i in range(20)], fontsize=8)
ax.set_yticks([])
ax.set_title('Figure 3: Detection Timeline, Delay, and Localized Retraining Events', fontsize=10.5, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_detection_timeline.png', dpi=300)
plt.close()
print("Saved fig7_detection_timeline.png")


# -------------------------------------------------------------------------------------------------
# FIGURE 8: SHAP Feature Importance Rank Evolution (Model Paper Fig 4)
# -------------------------------------------------------------------------------------------------
shap_df = pd.read_csv('results_v08/shap/shap_importance_comparison.csv')
top8 = shap_df.head(8).copy().sort_values('localized_da_post', ascending=True)

fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
y_pos = np.arange(len(top8))
h = 0.26

ax.barh(y_pos - h, top8['static_pre_drift'], height=h, label='Static Pre-Drift (w0–1)', color='#3b82f6', alpha=0.85)
ax.barh(y_pos, top8['static_post_drift'], height=h, label='Static Post-Drift (w2–19)', color='#ef4444', alpha=0.85)
ax.barh(y_pos + h, top8['localized_da_post'], height=h, label='Localized DA Post-Adaptation', color='#10b981', alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(top8['feature'], fontsize=8.5)
ax.set_xlabel('Mean |SHAP value| (Impact on Model Output)', fontsize=9)
ax.set_title('Figure 4: Feature Importance Shifts: Pre-Drift vs Post-Drift vs Adapted', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_shap_rank_evolution.png', dpi=300)
plt.close()
print("Saved fig8_shap_rank_evolution.png")


# -------------------------------------------------------------------------------------------------
# COPY BEESWARM FIGURES (from results_v08/shap/)
# -------------------------------------------------------------------------------------------------
shutil.copy('results_v08/shap/static_pre_drift_beeswarm.png', OUT_DIR / 'fig9_shap_beeswarm_static_pre.png')
shutil.copy('results_v08/shap/static_post_drift_beeswarm.png', OUT_DIR / 'fig10_shap_beeswarm_static_post.png')
shutil.copy('results_v08/shap/localized_da_beeswarm.png', OUT_DIR / 'fig11_shap_beeswarm_da_post.png')
print("Copied fig9, fig10, fig11 beeswarm plots into paper_pictures/")

print("\nAll publication figures successfully created in paper_pictures/!")
