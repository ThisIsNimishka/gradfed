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
let currentMatrixFilter = 'all';

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  initClientGrid();
  updateRoundView(0);
  setupEventListeners();
  setupArchTabs();
  setupQualityCalculator();
  setupCliGenerator();
  setupMatrixFilters();
  setupKeyboardShortcuts();
  window.addEventListener('resize', () => renderSimChart(currentRoundIdx));
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
    node.dataset.quality = client.quality;
    node.dataset.type = client.type;

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

  // Re-apply filter dimming
  applyMatrixFilter(currentMatrixFilter);

  // Update progress bar
  const progressPercent = ((data.round) / ROUNDS_DATA.length) * 100;
  const bar = document.getElementById('simProgressBar');
  if (bar) bar.style.width = `${progressPercent}%`;

  // Redraw dynamic canvas chart
  renderSimChart(currentRoundIdx);
}

// Live Dynamic Canvas Chart
function renderSimChart(activeIdx) {
  const canvas = document.getElementById('simChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Handle high-DPI displays
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const pad = { top: 25, right: 45, bottom: 35, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  // Draw background grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Accuracy labels (Left Y-axis)
    const accVal = Math.round(80 - i * 20);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`${accVal}%`, pad.left - 8, y + 3);

    // Loss labels (Right Y-axis)
    const lossVal = (2.5 - i * 0.625).toFixed(2);
    ctx.textAlign = 'left';
    ctx.fillText(lossVal, w - pad.right + 8, y + 3);
  }

  // Draw X-axis round ticks
  for (let r = 5; r <= 30; r += 5) {
    const x = pad.left + ((r - 1) / 29) * plotW;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText(`R${r}`, x, h - 12);
  }

  // Mapping functions
  const getX = (i) => pad.left + (i / 29) * plotW;
  const getYAcc = (acc) => pad.top + plotH - (acc / 80) * plotH;
  const getYLoss = (loss) => pad.top + plotH - (loss / 2.5) * plotH;

  const pointsCount = activeIdx + 1;

  // 1. Draw Accuracy Area Fill & Curve (Cyan)
  if (pointsCount > 0) {
    const gradAcc = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradAcc.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
    gradAcc.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    ctx.beginPath();
    ctx.moveTo(getX(0), getYAcc(ROUNDS_DATA[0].acc));
    for (let i = 1; i < pointsCount; i++) {
      ctx.lineTo(getX(i), getYAcc(ROUNDS_DATA[i].acc));
    }
    ctx.lineTo(getX(pointsCount - 1), pad.top + plotH);
    ctx.lineTo(getX(0), pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradAcc;
    ctx.fill();

    // Accuracy Stroke
    ctx.beginPath();
    ctx.moveTo(getX(0), getYAcc(ROUNDS_DATA[0].acc));
    for (let i = 1; i < pointsCount; i++) {
      ctx.lineTo(getX(i), getYAcc(ROUNDS_DATA[i].acc));
    }
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 2. Draw Loss Curve (Amber)
    ctx.beginPath();
    ctx.moveTo(getX(0), getYLoss(ROUNDS_DATA[0].loss));
    for (let i = 1; i < pointsCount; i++) {
      ctx.lineTo(getX(i), getYLoss(ROUNDS_DATA[i].loss));
    }
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Highlight Current Head Points
    const headX = getX(activeIdx);
    const headYAcc = getYAcc(ROUNDS_DATA[activeIdx].acc);
    const headYLoss = getYLoss(ROUNDS_DATA[activeIdx].loss);

    // Glowing head for Accuracy
    ctx.beginPath();
    ctx.arc(headX, headYAcc, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(headX, headYAcc, 10, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Head for Loss
    ctx.beginPath();
    ctx.arc(headX, headYLoss, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
  }
}

// Matrix Filter Pills
function setupMatrixFilters() {
  const container = document.getElementById('matrixFilterPills');
  if (!container) return;

  const pills = container.querySelectorAll('.filter-pill');
  pills.forEach(pill => {
    pill.addEventListener('click', () => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentMatrixFilter = pill.dataset.filter;
      applyMatrixFilter(currentMatrixFilter);
    });
  });
}

function applyMatrixFilter(filter) {
  const nodes = document.querySelectorAll('.client-node');
  const activeSelected = ROUNDS_DATA[currentRoundIdx].selected;

  nodes.forEach(node => {
    const id = parseInt(node.dataset.id);
    const q = parseFloat(node.dataset.quality);
    const type = node.dataset.type || '';
    let match = true;

    if (filter === 'selected') match = activeSelected.includes(id);
    else if (filter === 'optimal') match = q >= 0.8;
    else if (filter === 'stable') match = q >= 0.5 && q < 0.8;
    else if (filter === 'explore') match = type.includes('Exploration');
    else if (filter === 'noisy') match = q < 0.3;

    if (match) {
      node.classList.remove('dimmed');
    } else {
      node.classList.add('dimmed');
    }
  });
}

// Simulator Controls
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

// Global Keyboard Controls
function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    // Avoid triggering when focused on inputs or textareas
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    if (e.code === 'Space') {
      e.preventDefault();
      if (isPlaying) pauseSimulation();
      else startSimulation();
    } else if (e.code === 'ArrowRight') {
      e.preventDefault();
      pauseSimulation();
      if (currentRoundIdx < ROUNDS_DATA.length - 1) updateRoundView(currentRoundIdx + 1);
    } else if (e.code === 'ArrowLeft') {
      e.preventDefault();
      pauseSimulation();
      if (currentRoundIdx > 0) updateRoundView(currentRoundIdx - 1);
    } else if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      pauseSimulation();
      updateRoundView(0);
    }
  });
}

// Interactive Quality Calculator
function setupQualityCalculator() {
  const sSlider = document.getElementById('sliderStability');
  const mSlider = document.getElementById('sliderMagnitude');
  const cSlider = document.getElementById('sliderCosine');

  if (!sSlider || !mSlider || !cSlider) return;

  const updateCalc = () => {
    const s = parseInt(sSlider.value) / 100;
    const m = parseInt(mSlider.value) / 100;
    const c = parseInt(cSlider.value) / 100;

    document.getElementById('valStability').textContent = s.toFixed(2);
    document.getElementById('valMagnitude').textContent = m.toFixed(2);
    document.getElementById('valCosine').textContent = c.toFixed(2);

    const partS = 0.5 * s;
    const partM = 0.3 * m;
    const partC = 0.2 * c;
    const q = partS + partM + partC;

    document.getElementById('calcCompositeQ').textContent = q.toFixed(3);
    document.getElementById('calcPartS').textContent = partS.toFixed(3);
    document.getElementById('calcPartM').textContent = partM.toFixed(3);
    document.getElementById('calcPartC').textContent = partC.toFixed(3);

    const badge = document.getElementById('calcStatusBadge');
    const desc = document.getElementById('calcActionDesc');
    badge.className = 'calc-status-badge';

    if (q >= 0.75) {
      badge.classList.add('badge-optimal');
      badge.textContent = 'Optimal Participant (Prioritized in Aggregation)';
      desc.textContent = 'Client gradients exhibit stable norm progression and consistent trajectory alignment. Full aggregation contribution (40% quality weight + data volume) granted.';
    } else if (q >= 0.45) {
      badge.classList.add('badge-standard');
      badge.textContent = 'Standard Contributor (Acceptable Stability)';
      desc.textContent = 'Client displays moderate class variance or acceptable drift. Admitted into aggregation with standard quality-proportional weighting.';
    } else if (q >= 0.30) {
      badge.classList.add('badge-explore');
      badge.textContent = 'Candidate for ε-Greedy Probe (Recovering Pool)';
      desc.textContent = 'Client stability is sub-threshold. Excluded from primary round selection but eligible for periodic ε-greedy probing to detect gradient recovery.';
    } else {
      badge.classList.add('badge-noisy');
      badge.textContent = 'Flagged as Noisy Outlier (Filtered / Downweighted)';
      desc.textContent = 'Severe gradient norm oscillation or sign conflict detected. Client updates are aggressively attenuated by up to 85% to protect global model convergence.';
    }
  };

  [sSlider, mSlider, cSlider].forEach(slider => slider.addEventListener('input', updateCalc));
  updateCalc();
}

// Interactive CLI Generator
function setupCliGenerator() {
  const fields = ['cliDataset', 'cliModel', 'cliNumUsers', 'cliFrac', 'cliAlpha', 'cliEpochs', 'cliLocalEp', 'cliGpu'];
  fields.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateCliOutput);
  });
  updateCliOutput();
}

function updateCliOutput() {
  const dataset = document.getElementById('cliDataset')?.value || 'cifar10';
  const model = document.getElementById('cliModel')?.value || 'resnet10';
  const users = document.getElementById('cliNumUsers')?.value || '50';
  const frac = document.getElementById('cliFrac')?.value || '0.2';
  const alpha = document.getElementById('cliAlpha')?.value || '10.0';
  const epochs = document.getElementById('cliEpochs')?.value || '30';
  const localEp = document.getElementById('cliLocalEp')?.value || '10';
  const gpu = document.getElementById('cliGpu')?.value || '0';

  const cmd = `python src/gradfed_main.py \\
    --dataset ${dataset} \\
    --model ${model} \\
    --num_users ${users} \\
    --frac ${frac} \\
    --local_ep ${localEp} \\
    --local_bs 10 \\
    --epochs ${epochs} \\
    --alpha_dirichlet ${alpha} \\
    --gpu ${gpu} \\
    --iid 0`;

  const pre = document.getElementById('cliCommandText');
  if (pre) pre.textContent = cmd;
}

function copyCliCommand() {
  const cmd = document.getElementById('cliCommandText')?.textContent || '';
  navigator.clipboard.writeText(cmd).then(() => {
    showToast("Terminal command copied to clipboard!");
  }).catch(() => {
    showToast("Failed to copy command.");
  });
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
