/**
 * GradFed Showcase Application Logic
 * Interactive Federated Learning Round Simulator & Dashboard
 */

// 1. Real Experimental Trajectory Data (CIFAR-10, ResNet-10, 50 Clients, Non-IID Dirichlet)
const ROUNDS_DATA = [
  { round: 1, acc: 16.78, loss: 2.1801, selected: [11, 1, 2, 19, 5, 30, 48, 22, 41, 3], avgQ: 0.350, noisy: 0 },
  { round: 2, acc: 33.53, loss: 1.7474, selected: [3, 22, 38, 0, 45, 20, 40, 24, 49, 26], avgQ: 0.470, noisy: 0 },
  { round: 3, acc: 40.54, loss: 1.5322, selected: [0, 45, 20, 40, 24, 49, 26, 41, 1, 2], avgQ: 0.510, noisy: 0 },
  { round: 4, acc: 50.04, loss: 1.3481, selected: [22, 3, 41, 0, 1, 2, 11, 20, 45, 36], avgQ: 0.580, noisy: 0 },
  { round: 5, acc: 44.15, loss: 1.5614, selected: [22, 3, 41, 0, 1, 2, 11, 20, 45, 15], avgQ: 0.550, noisy: 0 },
  { round: 6, acc: 47.95, loss: 1.3738, selected: [22, 3, 41, 0, 1, 2, 11, 20, 45, 36], avgQ: 0.580, noisy: 0 },
  { round: 7, acc: 56.25, loss: 1.1516, selected: [22, 3, 41, 0, 1, 2, 11, 20, 45, 19], avgQ: 0.590, noisy: 0 },
  { round: 8, acc: 61.09, loss: 1.0785, selected: [22, 3, 41, 0, 1, 2, 11, 20, 45, 36], avgQ: 0.600, noisy: 0 },
  { round: 9, acc: 61.23, loss: 1.0721, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 10, acc: 61.05, loss: 1.0486, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 11, acc: 64.60, loss: 0.9853, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 12, acc: 64.72, loss: 0.9809, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 13, acc: 64.74, loss: 0.9737, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 14, acc: 67.47, loss: 0.9192, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 15, acc: 67.53, loss: 0.9132, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 16, acc: 68.95, loss: 0.8852, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 17, acc: 68.82, loss: 0.8883, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 18, acc: 69.94, loss: 0.8769, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 19, acc: 70.43, loss: 0.8448, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 20, acc: 70.07, loss: 0.8557, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 21, acc: 71.27, loss: 0.8377, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 22, acc: 71.67, loss: 0.8270, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 23, acc: 71.48, loss: 0.8333, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 24, acc: 72.32, loss: 0.8079, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 25, acc: 72.60, loss: 0.8162, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 26, acc: 71.93, loss: 0.8409, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 27, acc: 74.42, loss: 0.7475, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 28, acc: 73.78, loss: 0.7778, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 29, acc: 73.40, loss: 0.7967, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 },
  { round: 30, acc: 73.18, loss: 0.8123, selected: [0, 45, 41, 20, 1, 22, 11, 2, 3, 36], avgQ: 0.600, noisy: 0 }
];

// 2. Client Base Profiles (50 clients)
const CLIENT_PROFILES = [
  { id: 0, quality: 0.88, participation: 25, type: 'Core High Quality' },
  { id: 1, quality: 0.86, participation: 25, type: 'Core High Quality' },
  { id: 2, quality: 0.84, participation: 25, type: 'Core High Quality' },
  { id: 3, quality: 0.91, participation: 29, type: 'Core High Quality' },
  { id: 4, quality: 0.72, participation: 5,  type: 'Stable' },
  { id: 5, quality: 0.65, participation: 6,  type: 'Stable' },
  { id: 6, quality: 0.58, participation: 4,  type: 'Moderate' },
  { id: 7, quality: 0.45, participation: 3,  type: 'Moderate' },
  { id: 8, quality: 0.62, participation: 5,  type: 'Stable' },
  { id: 9, quality: 0.54, participation: 4,  type: 'Moderate' },
  { id: 10, quality: 0.48, participation: 3, type: 'Moderate' },
  { id: 11, quality: 0.87, participation: 25, type: 'Core High Quality' },
  { id: 12, quality: 0.42, participation: 2, type: 'Moderate' },
  { id: 13, quality: 0.36, participation: 2, type: 'Low Quality' },
  { id: 14, quality: 0.41, participation: 3, type: 'Moderate' },
  { id: 15, quality: 0.68, participation: 8, type: 'Stable' },
  { id: 16, quality: 0.38, participation: 2, type: 'Low Quality' },
  { id: 17, quality: 0.27, participation: 1, type: 'Noisy / Filtered' },
  { id: 18, quality: 0.49, participation: 3, type: 'Moderate' },
  { id: 19, quality: 0.64, participation: 6, type: 'Stable' },
  { id: 20, quality: 0.85, participation: 24, type: 'Core High Quality' },
  { id: 21, quality: 0.46, participation: 3, type: 'Moderate' },
  { id: 22, quality: 0.94, participation: 30, type: 'Top Performer' },
  { id: 23, quality: 0.44, participation: 3, type: 'Moderate' },
  { id: 24, quality: 0.47, participation: 4, type: 'Moderate' },
  { id: 25, quality: 0.71, participation: 5, type: 'Stable' },
  { id: 26, quality: 0.34, participation: 2, type: 'Low Quality' },
  { id: 27, quality: 0.59, participation: 4, type: 'Moderate' },
  { id: 28, quality: 0.24, participation: 1, type: 'Noisy / Filtered' },
  { id: 29, quality: 0.43, participation: 3, type: 'Moderate' },
  { id: 30, quality: 0.61, participation: 5, type: 'Stable' },
  { id: 31, quality: 0.37, participation: 2, type: 'Low Quality' },
  { id: 32, quality: 0.57, participation: 4, type: 'Moderate' },
  { id: 33, quality: 0.22, participation: 1, type: 'Noisy / Filtered' },
  { id: 34, quality: 0.45, participation: 3, type: 'Moderate' },
  { id: 35, quality: 0.41, participation: 2, type: 'Moderate' },
  { id: 36, quality: 0.76, participation: 17, type: 'Core Contributor' },
  { id: 37, quality: 0.35, participation: 2, type: 'Exploration Pool' },
  { id: 38, quality: 0.35, participation: 1, type: 'Exploration Pool' },
  { id: 39, quality: 0.43, participation: 3, type: 'Moderate' },
  { id: 40, quality: 0.35, participation: 2, type: 'Exploration Pool' },
  { id: 41, quality: 0.89, participation: 26, type: 'Core High Quality' },
  { id: 42, quality: 0.32, participation: 1, type: 'Exploration Pool' },
  { id: 43, quality: 0.26, participation: 1, type: 'Noisy / Filtered' },
  { id: 44, quality: 0.31, participation: 1, type: 'Exploration Pool' },
  { id: 45, quality: 0.86, participation: 24, type: 'Core High Quality' },
  { id: 46, quality: 0.35, participation: 1, type: 'Exploration Pool' },
  { id: 47, quality: 0.21, participation: 1, type: 'Noisy / Filtered' },
  { id: 48, quality: 0.35, participation: 3, type: 'Exploration Pool' },
  { id: 49, quality: 0.35, participation: 1, type: 'Exploration Pool' }
];

// State variables
let currentRoundIdx = 0;
let isPlaying = false;
let playInterval = null;
let playSpeed = 800; // ms

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
  initClientGrid();
  updateRoundView(0);
  setupEventListeners();
  setupArchTabs();
});

// Render 50-client matrix
function initClientGrid() {
  const grid = document.getElementById('clientGrid');
  if (!grid) return;
  grid.innerHTML = '';

  CLIENT_PROFILES.forEach(client => {
    const node = document.createElement('div');
    node.className = 'client-node';
    node.id = `client-node-${client.id}`;
    node.dataset.id = client.id;

    // Quality color class
    let colorClass = '#34d399';
    if (client.quality >= 0.8) colorClass = '#047857';
    else if (client.quality >= 0.6) colorClass = '#059669';
    else if (client.quality >= 0.4) colorClass = '#f59e0b';
    else if (client.quality >= 0.3) colorClass = '#f97316';
    else colorClass = '#ef4444';

    node.style.borderTop = `3px solid ${colorClass}`;

    node.innerHTML = `
      <div class="node-id">#${client.id < 10 ? '0' + client.id : client.id}</div>
      <div class="node-score">${client.quality.toFixed(2)}</div>
    `;

    node.addEventListener('click', () => showClientModal(client));
    grid.appendChild(node);
  });
}

// Update simulation view
function updateRoundView(idx) {
  currentRoundIdx = Math.max(0, Math.min(idx, ROUNDS_DATA.length - 1));
  const data = ROUNDS_DATA[currentRoundIdx];

  // Update text stats
  document.getElementById('simCurrentRound').textContent = data.round;
  document.getElementById('simAccuracy').textContent = `${data.acc.toFixed(2)}%`;
  document.getElementById('simLoss').textContent = data.loss.toFixed(4);
  document.getElementById('simSelectedCount').textContent = `${data.selected.length} / 50`;
  document.getElementById('simAvgQuality').textContent = data.avgQ.toFixed(3);

  // Update client grid selection highlights
  const allNodes = document.querySelectorAll('.client-node');
  allNodes.forEach(node => {
    node.classList.remove('selected');
    node.style.background = 'var(--bg-secondary)';
  });

  data.selected.forEach(clientId => {
    const node = document.getElementById(`client-node-${clientId}`);
    if (node) {
      node.classList.add('selected');
      node.style.background = 'rgba(56, 189, 248, 0.2)';
    }
  });

  // Update progress bar
  const progressPercent = ((data.round) / ROUNDS_DATA.length) * 100;
  const bar = document.getElementById('simProgressBar');
  if (bar) bar.style.width = `${progressPercent}%`;
}

// Controls
function setupEventListeners() {
  const btnPlay = document.getElementById('btnPlay');
  const btnNext = document.getElementById('btnNext');
  const btnPrev = document.getElementById('btnPrev');
  const btnReset = document.getElementById('btnReset');
  const speedSlider = document.getElementById('speedSlider');

  if (btnPlay) {
    btnPlay.addEventListener('click', () => {
      if (isPlaying) {
        pauseSimulation();
      } else {
        startSimulation();
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      pauseSimulation();
      if (currentRoundIdx < ROUNDS_DATA.length - 1) {
        updateRoundView(currentRoundIdx + 1);
      }
    });
  }

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      pauseSimulation();
      if (currentRoundIdx > 0) {
        updateRoundView(currentRoundIdx - 1);
      }
    });
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      pauseSimulation();
      updateRoundView(0);
    });
  }

  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      playSpeed = 1600 - parseInt(e.target.value);
      if (isPlaying) {
        pauseSimulation();
        startSimulation();
      }
    });
  }
}

function startSimulation() {
  isPlaying = true;
  const btnPlay = document.getElementById('btnPlay');
  if (btnPlay) {
    btnPlay.classList.add('active');
    btnPlay.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="4" height="16"/>
        <rect x="14" y="4" width="4" height="16"/>
      </svg>
      Pause
    `;
  }

  playInterval = setInterval(() => {
    if (currentRoundIdx < ROUNDS_DATA.length - 1) {
      updateRoundView(currentRoundIdx + 1);
    } else {
      pauseSimulation();
    }
  }, playSpeed);
}

function pauseSimulation() {
  isPlaying = false;
  clearInterval(playInterval);
  const btnPlay = document.getElementById('btnPlay');
  if (btnPlay) {
    btnPlay.classList.remove('active');
    btnPlay.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3"/>
      </svg>
      Auto Play
    `;
  }
}

// Architecture Tabs logic
function setupArchTabs() {
  const tabBtns = document.querySelectorAll('.arch-tab-btn');
  const contents = {
    monitor: {
      title: "1. Pseudo-Gradient Extraction & L2 Norm Tracking",
      formula: "Δw_k^(t) = w_global^(t) - w_local,k^(t)\nCV_k = σ(||Δw_k||) / (μ(||Δw_k||) + ε)\nStability S_k = 1 / (1 + CV_k)",
      desc: "Rather than sharing private training batches, clients send back parameter updates. The server calculates the exact shift vector. The GradientMonitor tracks historical norm variability across communication rounds over window W=5 to identify unstable or drifting updates."
    },
    cosine: {
      title: "2. Directional Alignment & Magnitude Bounds",
      formula: "Cosine Consistency C_k = (cos(Δw_k^(t), Δw_k^(t-1)) + 1) / 2\nMagnitude Score M_k = 2 / (1 + exp(||Δw_k|| / 10))\nQuality Q_k = 0.5·S_k + 0.3·M_k + 0.2·C_k",
      desc: "Measures whether the client is moving cohesively towards the global minimum or oscillating randomly due to label corruption. A sigmoidal scaler penalizes exploding or vanishing gradients, yielding an online composite quality score Q_k ∈ [0, 1]."
    },
    ppo: {
      title: "3. PPO Policy with Adaptive ε-Greedy Exploration",
      formula: "State s_t = [Acc_t, Loss_t, t/T, Q_avg, N_noisy/K, σ_Q]\nSelection Score_k = 0.7·P_ppo(k) + 0.3·Q_k\nExploration: ε(t) = max(0.1, 0.3 - 0.2·(t/T))",
      desc: "A discrete-action Proximal Policy Optimization network generates client selection probabilities. To prevent straggler starvation, an ε-greedy exploration pool periodically probes low-quality clients to detect if their models stabilize, decaying exploration as the global model converges."
    },
    aggregator: {
      title: "4. Dual-Balanced Quality Aggregator",
      formula: "w_global^(t+1) = Σ [ 0.6·(|D_k| / Σ|D|) + 0.4·(Q_k / Σ Q) ] · w_local,k^(t)",
      desc: "Conventional FedAvg is vulnerable to corrupted gradients from high-data clients. GradFed balances data volume (60%) with verified gradient reliability (40%), completely filtering out corrupted client weights without requiring trusted benchmark test sets on the server."
    }
  };

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const tabKey = btn.dataset.tab;
      const data = contents[tabKey];
      if (data) {
        document.getElementById('archTabTitle').textContent = data.title;
        document.getElementById('archMathBlock').textContent = data.formula;
        document.getElementById('archTabDesc').textContent = data.desc;
      }
    });
  });
}

// Client inspection modal
function showClientModal(client) {
  const modal = document.getElementById('clientModal');
  if (!modal) return;

  document.getElementById('modalClientId').textContent = `#${client.id}`;
  document.getElementById('modalClientType').textContent = client.type;
  document.getElementById('modalClientQuality').textContent = client.quality.toFixed(3);
  document.getElementById('modalClientRounds').textContent = `${client.participation} / 30 rounds`;
  
  let rec = 'Included in primary aggregation';
  if (client.quality < 0.3) rec = 'Filtered out (Noisy Gradient Flagged)';
  else if (client.quality < 0.4) rec = 'Eligible for ε-Greedy Exploration only';
  document.getElementById('modalClientAction').textContent = rec;

  modal.style.display = 'flex';
}

function closeClientModal() {
  const modal = document.getElementById('clientModal');
  if (modal) modal.style.display = 'none';
}

// 1-Click BibTeX copy
function copyBibtex() {
  const code = document.getElementById('bibtexCode').innerText;
  navigator.clipboard.writeText(code).then(() => {
    showToast("BibTeX citation copied to clipboard!");
  }).catch(() => {
    showToast("Failed to copy BibTeX.");
  });
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 2500);
}
