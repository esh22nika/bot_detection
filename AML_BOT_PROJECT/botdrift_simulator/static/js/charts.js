/**
 * Chart management for BotDriftBench Simulator using Chart.js
 */

let performanceChart = null;
let distributionChart = null;

function initPerformanceChart(allWindows, staticData, daData, currentWindow) {
  const ctx = document.getElementById('performanceChart').getContext('2d');
  
  const labels = allWindows.map(w => `W${w}`);
  const staticF1s = staticData.map(d => d.f1_macro);
  const daF1s = daData.map(d => d.f1_macro);

  // Highlighting current window cursor point
  const cursorStatic = staticF1s.map((val, idx) => (idx === currentWindow ? val : null));
  const cursorDA = daF1s.map((val, idx) => (idx === currentWindow ? val : null));

  performanceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Static Detector (Frozen)',
          data: staticF1s,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          borderWidth: 2.5,
          tension: 0.15,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: false
        },
        {
          label: 'Proposed Drift-Aware Detector',
          data: daF1s,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.05)',
          borderWidth: 2.5,
          tension: 0.15,
          pointRadius: 3,
          pointHoverRadius: 6,
          fill: false
        },
        {
          label: 'Current Position (Static)',
          data: cursorStatic,
          borderColor: '#b91c1c',
          backgroundColor: '#ef4444',
          pointRadius: 7,
          pointHoverRadius: 9,
          showLine: false
        },
        {
          label: 'Current Position (DA)',
          data: cursorDA,
          borderColor: '#047857',
          backgroundColor: '#10b981',
          pointRadius: 7,
          pointHoverRadius: 9,
          showLine: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              if (context.datasetIndex > 1) return null; // skip cursor duplicate
              return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          ticks: { font: { family: 'JetBrains Mono', size: 11 } }
        },
        y: {
          min: 0.60,
          max: 1.00,
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          title: {
            display: true,
            text: 'Macro-F1 Score',
            font: { size: 12, weight: '600' }
          },
          ticks: { font: { family: 'JetBrains Mono', size: 11 } }
        }
      }
    }
  });
}

function updatePerformanceCursor(currentWindow, staticData, daData) {
  if (!performanceChart) return;
  
  const staticF1s = staticData.map(d => d.f1_macro);
  const daF1s = daData.map(d => d.f1_macro);
  
  const cursorStatic = staticF1s.map((val, idx) => (idx === currentWindow ? val : null));
  const cursorDA = daF1s.map((val, idx) => (idx === currentWindow ? val : null));

  performanceChart.data.datasets[2].data = cursorStatic;
  performanceChart.data.datasets[3].data = cursorDA;
  performanceChart.update('none');
}

function initDistributionChart(distData) {
  const ctx = document.getElementById('distributionChart').getContext('2d');
  
  distributionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: distData.bin_labels,
      datasets: [
        {
          label: 'Pre-Drift Distribution (w0–1)',
          data: distData.pre_drift,
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: '#2563eb',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Post-Drift Evasion Distribution (w2–19)',
          data: distData.post_drift,
          backgroundColor: 'rgba(239, 68, 68, 0.8)',
          borderColor: '#dc2626',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.parsed.y}%`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { family: 'JetBrains Mono', size: 9 }, maxRotation: 45 }
        },
        y: {
          title: { display: true, text: 'Bot Account Population (%)', font: { size: 11, weight: '600' } },
          ticks: { font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    }
  });

  updateDistributionMeans(distData.pre_mean, distData.post_mean);
}

function updateDistributionChart(distData) {
  if (!distributionChart) return;
  distributionChart.data.labels = distData.bin_labels;
  distributionChart.data.datasets[0].data = distData.pre_drift;
  distributionChart.data.datasets[1].data = distData.post_drift;
  distributionChart.update();
  updateDistributionMeans(distData.pre_mean, distData.post_mean);
}

function updateDistributionMeans(preMean, postMean) {
  const preEl = document.getElementById('pre-mean');
  const postEl = document.getElementById('post-mean');
  if (preEl) preEl.textContent = `${preMean}`;
  if (postEl) postEl.textContent = `${postMean} (Shift: ${(postMean - preMean).toFixed(3)})`;
}
