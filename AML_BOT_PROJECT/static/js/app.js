/**
 * Application Bootstrap & Event Handlers
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Simulator state & charts
  Simulator.init();

  // Button Listeners
  document.getElementById('btn-before-drift').addEventListener('click', () => {
    Simulator.beforeDrift();
  });

  document.getElementById('btn-inject-drift').addEventListener('click', () => {
    Simulator.injectDrift();
  });

  document.getElementById('btn-run-adaptation').addEventListener('click', () => {
    Simulator.runAdaptation();
  });

  document.getElementById('btn-step').addEventListener('click', () => {
    Simulator.step();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    Simulator.reset();
  });

  // Feature Dropdown
  const featureSelect = document.getElementById('feature-select');
  if (featureSelect) {
    featureSelect.addEventListener('change', (e) => {
      Simulator.switchFeature(e.target.value);
    });
  }
});
