from .data_loader import DataLoader
from .metrics import compute_comparison_metrics
from .scenarios import DRIFT_ONSET_WINDOW, FIRST_DETECTION_WINDOW, ADAPTATION_WINDOWS, EXPLANATIONS, EVENT_DESCRIPTIONS

class SimulatorEngine:
    def __init__(self, data_loader: DataLoader = None):
        self.loader = data_loader if data_loader is not None else DataLoader()
        self.reset()
        
    def reset(self):
        self.mode = "before_drift"
        self.current_window = 0
        self.total_windows = 20
        self.drift_injected = False
        self.drift_detected = False
        self.adaptation_triggered = False
        self.adaptation_completed = False
        self.retraining_count = 0
        self.event_log = []
        self._record_event(0)
        self._update_metrics()
        return self.get_state()
        
    def before_drift(self):
        # Set to window 0 or 1 in stable pre-drift condition
        self.mode = "before_drift"
        self.current_window = 0
        self.drift_injected = False
        self.drift_detected = False
        self.adaptation_triggered = False
        self.adaptation_completed = False
        self.retraining_count = 0
        self.event_log = []
        self._record_event(0)
        self._update_metrics()
        return self.get_state()
        
    def inject_drift(self):
        # Move simulation to onset of drift (window 2)
        self.current_window = DRIFT_ONSET_WINDOW
        self.mode = "drift_injected"
        self.drift_injected = True
        self.drift_detected = False
        self.adaptation_triggered = False
        self.adaptation_completed = False
        self.retraining_count = 0
        self._record_event(2)
        self._update_metrics()
        return self.get_state()
        
    def run_adaptation(self):
        # Move to first adaptation window (window 5)
        self.current_window = FIRST_DETECTION_WINDOW
        self.mode = "adapted"
        self.drift_injected = True
        self.drift_detected = True
        self.adaptation_triggered = True
        self.adaptation_completed = True
        self.retraining_count = 1
        self._record_event(2)
        self._record_event(5)
        self._update_metrics()
        return self.get_state()
        
    def step(self):
        if self.current_window < self.total_windows - 1:
            self.current_window += 1
            
        # Dynamically evaluate environment state based on current window
        if self.current_window >= DRIFT_ONSET_WINDOW:
            self.drift_injected = True
            self._record_event(2)
            
        if self.current_window >= FIRST_DETECTION_WINDOW:
            self.drift_detected = True
            self.adaptation_triggered = True
            self.adaptation_completed = True
            self.mode = "adapted"
            self._record_event(5)
        elif self.current_window >= DRIFT_ONSET_WINDOW:
            self.drift_detected = False
            self.adaptation_triggered = False
            self.adaptation_completed = False
            self.mode = "drift_injected"
        else:
            self.drift_injected = False
            self.drift_detected = False
            self.adaptation_triggered = False
            self.adaptation_completed = False
            self.mode = "before_drift"
            
        # Count adaptation events up to current_window
        self.retraining_count = sum(1 for w in ADAPTATION_WINDOWS if w <= self.current_window)
        
        # Check any intermediate milestone events
        for milestone in [8, 11, 14, 17, 18]:
            if self.current_window >= milestone:
                self._record_event(milestone)
                
        self._update_metrics()
        return self.get_state()

    def _record_event(self, window: int):
        if window in EVENT_DESCRIPTIONS:
            existing = [e['window'] for e in self.event_log]
            if window not in existing:
                item = dict(EVENT_DESCRIPTIONS[window])
                item['window'] = window
                self.event_log.append(item)
                self.event_log.sort(key=lambda x: x['window'])

    def _update_metrics(self):
        wm = self.loader.get_window_metrics(self.current_window)
        s_m = wm['static']
        d_m = wm['drift_aware']
        
        self.static_f1 = s_m.get('f1_macro', 0.0)
        self.static_auc = s_m.get('roc_auc', 0.0)
        self.static_recall = s_m.get('recall', 0.0)
        
        self.drift_aware_f1 = d_m.get('f1_macro', 0.0)
        self.drift_aware_auc = d_m.get('roc_auc', 0.0)
        self.drift_aware_recall = d_m.get('recall', 0.0)
        
        # Determine status strings
        if self.mode == "before_drift":
            self.static_status = "stable"
            self.drift_aware_status = "stable"
            self.explanation = EXPLANATIONS['before_drift']
        elif self.mode == "drift_injected":
            self.static_status = "degraded"
            self.drift_aware_status = "detecting"
            self.explanation = EXPLANATIONS['drift_injected']
        else: # adapted
            self.static_status = "degraded"
            self.drift_aware_status = "adapted"
            self.explanation = EXPLANATIONS['adapted']

    def get_state(self):
        comp = compute_comparison_metrics(
            {'f1_macro': self.static_f1, 'roc_auc': self.static_auc, 'recall': self.static_recall},
            {'f1_macro': self.drift_aware_f1, 'roc_auc': self.drift_aware_auc, 'recall': self.drift_aware_recall}
        )
        return {
            'success': True,
            'mode': self.mode,
            'current_window': self.current_window,
            'total_windows': self.total_windows,
            'drift_injected': self.drift_injected,
            'drift_detected': self.drift_detected,
            'adaptation_triggered': self.adaptation_triggered,
            'adaptation_completed': self.adaptation_completed,
            'static_f1': self.static_f1,
            'static_auc': self.static_auc,
            'static_recall': self.static_recall,
            'drift_aware_f1': self.drift_aware_f1,
            'drift_aware_auc': self.drift_aware_auc,
            'drift_aware_recall': self.drift_aware_recall,
            'f1_advantage': comp['f1_advantage'],
            'auc_advantage': comp['auc_advantage'],
            'recall_advantage': comp['recall_advantage'],
            'static_status': self.static_status,
            'drift_aware_status': self.drift_aware_status,
            'retraining_count': self.retraining_count,
            'explanation': self.explanation,
            'event_log': self.event_log
        }
        
    def get_comparison(self):
        traj = self.loader.get_all_trajectories()
        return {
            'success': True,
            'current_window': self.current_window,
            'drift_onset_window': DRIFT_ONSET_WINDOW,
            'first_detection_window': FIRST_DETECTION_WINDOW,
            'adaptation_windows': ADAPTATION_WINDOWS,
            'trajectories': traj,
            'current_state': self.get_state()
        }
        
    def get_events(self):
        return {
            'success': True,
            'current_window': self.current_window,
            'events': self.event_log
        }
