/**
 * Simulator API Client & State Sync for BotDriftBench
 */

const Simulator = {
  initialData: null,
  currentState: null,

  async init() {
    try {
      const response = await fetch('/api/initial-data');
      const data = await response.json();
      if (!data.success) throw new Error(data.error);

      this.initialData = data;
      this.currentState = data.initial_state;

      // Initialize Performance Chart
      initPerformanceChart(
        data.trajectories.windows,
        data.trajectories.static,
        data.trajectories.drift_aware,
        this.currentState.current_window
      );

      // Initialize Distribution Chart
      initDistributionChart(data.feature_distribution);

      // Render Initial State to DOM
      this.renderState(this.currentState);

    } catch (err) {
      console.error('Simulator Initialization Error:', err);
      alert('Failed to connect to simulator backend: ' + err.message);
    }
  },

  async beforeDrift() {
    try {
      const res = await fetch('/api/before-drift', { method: 'POST' });
      const state = await res.json();
      if (state.success) {
        this.currentState = state;
        this.renderState(state);
      }
    } catch (err) {
      console.error('Error in beforeDrift:', err);
    }
  },

  async injectDrift() {
    try {
      const res = await fetch('/api/inject-drift', { method: 'POST' });
      const state = await res.json();
      if (state.success) {
        this.currentState = state;
        this.renderState(state);
      }
    } catch (err) {
      console.error('Error in injectDrift:', err);
    }
  },

  async runAdaptation() {
    try {
      const res = await fetch('/api/run-adaptation', { method: 'POST' });
      const state = await res.json();
      if (state.success) {
        this.currentState = state;
        this.renderState(state);
      }
    } catch (err) {
      console.error('Error in runAdaptation:', err);
    }
  },

  async step() {
    try {
      const res = await fetch('/api/step', { method: 'POST' });
      const state = await res.json();
      if (state.success) {
        this.currentState = state;
        this.renderState(state);
      }
    } catch (err) {
      console.error('Error in step:', err);
    }
  },

  async reset() {
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      const state = await res.json();
      if (state.success) {
        this.currentState = state;
        this.renderState(state);
      }
    } catch (err) {
      console.error('Error in reset:', err);
    }
  },

  async switchFeature(featureName) {
    try {
      const res = await fetch(`/api/features/${encodeURIComponent(featureName)}`);
      const data = await res.json();
      if (data.success) {
        updateDistributionChart(data.distribution);
      }
    } catch (err) {
      console.error('Error switching feature:', err);
    }
  },

  renderState(state) {
    // 1. Status Bar
    document.getElementById('window-display').textContent = `Window ${state.current_window} / 19`;
    
    const envDisplay = document.getElementById('env-display');
    const driftDisplay = document.getElementById('drift-display');
    const adaptDisplay = document.getElementById('adaptation-display');

    if (state.mode === 'before_drift') {
      envDisplay.textContent = 'Stable (Pre-Drift)';
      envDisplay.style.color = '#059669';
      driftDisplay.textContent = 'Inactive';
      driftDisplay.style.color = '#059669';
      adaptDisplay.textContent = 'None Required';
      adaptDisplay.style.color = '#64748b';
    } else if (state.mode === 'drift_injected') {
      envDisplay.textContent = 'Behavioral Drift Injected';
      envDisplay.style.color = '#dc2626';
      driftDisplay.textContent = 'Active (Evasion underway)';
      driftDisplay.style.color = '#dc2626';
      adaptDisplay.textContent = 'Monitoring KS Signals';
      adaptDisplay.style.color = '#d97706';
    } else { // adapted
      envDisplay.textContent = 'Adaptive Drift Environment';
      envDisplay.style.color = '#2563eb';
      driftDisplay.textContent = 'Detected (Window 5)';
      driftDisplay.style.color = '#2563eb';
      adaptDisplay.textContent = `Adapted (${state.retraining_count} retrain events)`;
      adaptDisplay.style.color = '#059669';
    }

    // 2. Metric Cards: Static
    document.getElementById('static-f1').textContent = state.static_f1.toFixed(4);
    document.getElementById('static-auc').textContent = state.static_auc.toFixed(4);
    document.getElementById('static-recall').textContent = state.static_recall.toFixed(4);
    
    const staticTag = document.getElementById('static-status-tag');
    if (state.static_status === 'stable') {
      staticTag.textContent = 'STABLE';
      staticTag.className = 'status-tag tag-stable';
    } else {
      staticTag.textContent = 'DEGRADED';
      staticTag.className = 'status-tag tag-degraded';
    }

    // 3. Metric Cards: Drift-Aware
    document.getElementById('da-f1').textContent = state.drift_aware_f1.toFixed(4);
    document.getElementById('da-auc').textContent = state.drift_aware_auc.toFixed(4);
    document.getElementById('da-recall').textContent = state.drift_aware_recall.toFixed(4);
    document.getElementById('da-retrains').textContent = `${state.retraining_count} event${state.retraining_count === 1 ? '' : 's'}`;

    const daTag = document.getElementById('da-status-tag');
    if (state.drift_aware_status === 'stable') {
      daTag.textContent = 'STABLE';
      daTag.className = 'status-tag tag-stable';
    } else if (state.drift_aware_status === 'detecting') {
      daTag.textContent = 'MONITORING';
      daTag.className = 'status-tag tag-detecting';
    } else {
      daTag.textContent = 'ADAPTED';
      daTag.className = 'status-tag tag-adapted';
    }

    // Advantage badge
    const advBadge = document.getElementById('da-advantage');
    const diff = state.f1_advantage;
    if (diff > 0.0001) {
      advBadge.textContent = `+${diff.toFixed(4)} F1 advantage`;
      advBadge.style.color = '#059669';
      advBadge.style.background = '#ecfdf5';
    } else if (diff < -0.0001) {
      advBadge.textContent = `${diff.toFixed(4)} F1`;
      advBadge.style.color = '#dc2626';
      advBadge.style.background = '#fef2f2';
    } else {
      advBadge.textContent = 'Parity (0.0000)';
      advBadge.style.color = '#64748b';
      advBadge.style.background = '#f1f5f9';
    }

    // 4. Dynamic Explanation Panel
    document.getElementById('explanation-text').textContent = state.explanation;

    // 5. Update Timeline
    this.renderTimeline(state.event_log, state.current_window);

    // 6. Update Chart Cursor
    if (this.initialData) {
      updatePerformanceCursor(
        state.current_window,
        this.initialData.trajectories.static,
        this.initialData.trajectories.drift_aware
      );
    }
  },

  renderTimeline(events, currentWindow) {
    const container = document.getElementById('timeline-container');
    container.innerHTML = '';

    events.forEach(evt => {
      const card = document.createElement('div');
      card.className = 'timeline-card' + (evt.window === currentWindow ? ' active-event' : '');
      
      let badgeClass = 'badge-init';
      if (evt.type === 'drift_onset') badgeClass = 'badge-drift';
      else if (evt.type === 'detection' || evt.type === 'adaptation') badgeClass = 'badge-adapt';
      else if (evt.type === 'worst_case') badgeClass = 'badge-nadir';

      card.innerHTML = `
        <span class="timeline-badge ${badgeClass}">${evt.badge} (W${evt.window})</span>
        <div class="timeline-content">
          <span class="timeline-title">${evt.title}</span>
          <span class="timeline-desc">${evt.description}</span>
        </div>
      `;
      container.appendChild(card);
    });
  }
};
