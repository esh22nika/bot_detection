def compute_comparison_metrics(static_m: dict, da_m: dict):
    s_f1 = static_m.get('f1_macro', 0.0)
    d_f1 = da_m.get('f1_macro', 0.0)
    s_auc = static_m.get('roc_auc', 0.0)
    d_auc = da_m.get('roc_auc', 0.0)
    s_rec = static_m.get('recall', 0.0)
    d_rec = da_m.get('recall', 0.0)
    
    f1_diff = round(d_f1 - s_f1, 4)
    auc_diff = round(d_auc - s_auc, 4)
    rec_diff = round(d_rec - s_rec, 4)
    
    return {
        'static_f1': s_f1,
        'drift_aware_f1': d_f1,
        'f1_advantage': f1_diff,
        'static_auc': s_auc,
        'drift_aware_auc': d_auc,
        'auc_advantage': auc_diff,
        'static_recall': s_rec,
        'drift_aware_recall': d_rec,
        'recall_advantage': rec_diff,
    }
