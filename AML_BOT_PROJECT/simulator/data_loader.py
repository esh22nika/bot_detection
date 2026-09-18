import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

class DataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.window_df = pd.read_csv(self.data_dir / 'window_results.csv')
        self.retraining_df = pd.read_csv(self.data_dir / 'retraining_events.csv')
        self.detection_df = pd.read_csv(self.data_dir / 'detection_metrics.csv')
        self.scenario_df = pd.read_csv(self.data_dir / 'scenario_results.csv')
        self.aggregate_df = pd.read_csv(self.data_dir / 'aggregate_summary.csv')
        
        with open(self.data_dir / 'feature_distributions.json', 'r', encoding='utf-8') as f:
            self.feature_distributions = json.load(f)
            
        # Parse window results into lookup maps
        self._parse_window_data()
        
    def _parse_window_data(self):
        # Keep only static and da_localized
        static_rows = self.window_df[self.window_df['policy'] == 'static'].sort_values('window')
        da_rows = self.window_df[self.window_df['policy'] == 'da_localized'].sort_values('window')
        
        self.windows_static = {
            int(r['window']): {
                'window': int(r['window']),
                'f1_macro': round(float(r['f1_macro_mean']), 4),
                'f1_macro_std': round(float(r.get('f1_macro_std', 0.0)), 4),
                'roc_auc': round(float(r['roc_auc_mean']), 4),
                'roc_auc_std': round(float(r.get('roc_auc_std', 0.0)), 4),
                'recall': round(float(r['recall_mean']), 4),
                'precision': round(float(r['precision_mean']), 4),
                'accuracy': round(float(r['accuracy_mean']), 4),
            } for _, r in static_rows.iterrows()
        }
        
        self.windows_da = {
            int(r['window']): {
                'window': int(r['window']),
                'f1_macro': round(float(r['f1_macro_mean']), 4),
                'f1_macro_std': round(float(r.get('f1_macro_std', 0.0)), 4),
                'roc_auc': round(float(r['roc_auc_mean']), 4),
                'roc_auc_std': round(float(r.get('roc_auc_std', 0.0)), 4),
                'recall': round(float(r['recall_mean']), 4),
                'precision': round(float(r['precision_mean']), 4),
                'accuracy': round(float(r['accuracy_mean']), 4),
            } for _, r in da_rows.iterrows()
        }
        
        # Adaptation triggers: find windows where da retrains occurred
        da_events = self.retraining_df[
            (self.retraining_df['policy'] == 'da_localized') & 
            (self.retraining_df['retrained'] == 1)
        ]
        self.da_retrain_windows = sorted(da_events['window'].tolist())
        # Adaptation retrains (post-window 0)
        self.da_adaptation_windows = [w for w in self.da_retrain_windows if w > 0]
        
    def get_window_metrics(self, window: int):
        window = max(0, min(19, int(window)))
        return {
            'window': window,
            'static': self.windows_static.get(window, {}),
            'drift_aware': self.windows_da.get(window, {})
        }
        
    def get_all_trajectories(self):
        static_series = [self.windows_static[w] for w in range(20) if w in self.windows_static]
        da_series = [self.windows_da[w] for w in range(20) if w in self.windows_da]
        return {
            'windows': list(range(20)),
            'static': static_series,
            'drift_aware': da_series,
            'adaptation_windows': self.da_adaptation_windows
        }
        
    def get_feature_distribution(self, feature_name: str):
        return self.feature_distributions.get(feature_name, {})
        
    def get_feature_list(self):
        return list(self.feature_distributions.keys())
