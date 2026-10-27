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
  initTheme();
  initClientGrid();
  updateRoundView(0);
  setupEventListeners();
  setupArchSubsystems();
  setupArchPipeline();
  setupArchTabs();
  setupArchSandbox();
  setupArchModal();
  setupCliGenerator();
  setupMatrixFilters();
  setupKeyboardShortcuts();
  window.addEventListener('resize', () => renderSimChart(currentRoundIdx));
  
  // Mathematical Typesetting initialization
  renderAllMath();
  setTimeout(() => {
    renderAllMath();
    renderSimChart(currentRoundIdx);
  }, 250);
});

window.addEventListener('load', () => {
  renderAllMath();
  renderSimChart(currentRoundIdx);
});

// Render KaTeX mathematical formulations
function renderAllMath(target) {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(target || document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      throwOnError: false
    });
  }
}

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
    node.style.background = '';
  });

  data.selected.forEach(clientId => {
    const node = document.getElementById(`client-node-${clientId}`);
    if (node) {
      node.classList.add('selected');
      node.style.background = '';
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
  const w = rect.width > 0 ? rect.width : (canvas.clientWidth > 0 ? canvas.clientWidth : 1050);
  const h = rect.height > 0 ? rect.height : (canvas.clientHeight > 0 ? canvas.clientHeight : 230);
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  const pad = { top: 25, right: 45, bottom: 35, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);

  // Resolve active theme palette for chart styling
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  const themeCfg = THEMES[currentTheme] || THEMES.light;
  const isLight = currentTheme === 'light';
  const gridColor = themeCfg.gridColor || (isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.05)');
  const axisColor = themeCfg.axisColor || (isLight ? '#475569' : '#64748b');
  const accColor = themeCfg.chartAccColor || '#0284c7';
  const accGlow = themeCfg.chartAccGlow || 'rgba(2, 132, 199, 0.2)';
  const accStroke = themeCfg.chartAccStroke || 'rgba(2, 132, 199, 0.45)';
  const lossColor = themeCfg.lossColor || (isLight ? '#d97706' : '#f59e0b');

  // Draw background grid lines
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    // Accuracy labels (Left Y-axis)
    const accVal = Math.round(80 - i * 20);
    ctx.fillStyle = axisColor;
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
    ctx.strokeStyle = isLight ? 'rgba(15, 23, 42, 0.08)' : 'rgba(255, 255, 255, 0.08)';
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + plotH);
    ctx.stroke();

    ctx.fillStyle = axisColor;
    ctx.textAlign = 'center';
    ctx.fillText(`R${r}`, x, h - 12);
  }

  // Mapping functions
  const getX = (i) => pad.left + (i / 29) * plotW;
  const getYAcc = (acc) => pad.top + plotH - (acc / 80) * plotH;
  const getYLoss = (loss) => pad.top + plotH - (loss / 2.5) * plotH;

  const pointsCount = activeIdx + 1;

  // 1. Draw Accuracy Area Fill & Curve (Theme Accent)
  if (pointsCount > 0) {
    const gradAcc = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradAcc.addColorStop(0, accGlow);
    gradAcc.addColorStop(1, 'transparent');

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
    ctx.strokeStyle = accColor;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // 2. Draw Loss Curve (Amber / Warning)
    ctx.beginPath();
    ctx.moveTo(getX(0), getYLoss(ROUNDS_DATA[0].loss));
    for (let i = 1; i < pointsCount; i++) {
      ctx.lineTo(getX(i), getYLoss(ROUNDS_DATA[i].loss));
    }
    ctx.strokeStyle = lossColor;
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
    ctx.fillStyle = accColor;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(headX, headYAcc, 10, 0, Math.PI * 2);
    ctx.strokeStyle = accStroke;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Head for Loss
    ctx.beginPath();
    ctx.arc(headX, headYLoss, 5, 0, Math.PI * 2);
    ctx.fillStyle = lossColor;
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

// ============================================================================
// Architecture Hub: Subsystems, Pipeline Stepper, Tabs, Sandbox & Modal
// ============================================================================

// 1. Subsystem Explorer Navigator
function setupArchSubsystems() {
  const subBtns = document.querySelectorAll('.arch-sub-btn');
  const badgeEl = document.getElementById('archSubBadge');
  const titleEl = document.getElementById('archSubTitle');
  const fileEl = document.getElementById('archSubFile');
  const descEl = document.getElementById('archSubDesc');
  const eqEl = document.getElementById('archSubEq');

  if (!subBtns.length || !titleEl) return;

  const SUBSYSTEMS = {
    all: {
      badge: "System Topology",
      title: "End-to-End Privacy-Preserving Coordination",
      file: "src/gradfed_main.py",
      desc: "GradFed separates the training loop into two synchronized tiers: the central server orchestrates reinforcement learning client selection and gradient quality aggregation, while edge clients train on private, non-IID local data without transmitting raw samples.",
      eq: "$$\\Delta \\mathbf{w}_k^{(t)} = \\mathbf{w}^{(t)} - \\mathbf{w}_{k}^{(t)} \\quad \\Longrightarrow \\quad Q_k^{(t)} \\quad \\Longrightarrow \\quad \\mathbf{w}^{(t+1)} = \\sum_{k=1}^K \\alpha_k \\mathbf{w}_{k}^{(t)}$$"
    },
    model: {
      badge: "Server Core",
      title: "1. Global Model State & Parameter Broadcast",
      file: "src/models.py",
      desc: "Maintains the central deep learning architecture (ResNet-10 / CNN) and tracks the canonical parameter tensor w_t across communication rounds. At each round, w_t is broadcast to selected edge clients.",
      eq: "$$\\mathbf{w}_t \\in \\mathbb{R}^d, \\qquad \\mathbf{w}^{(t+1)} \\leftarrow \\mathbf{w}^{(t)} - \\eta_g \\sum_{k \\in \\mathcal{S}_t} \\alpha_k \\Delta \\mathbf{w}_k^{(t)}$$"
    },
    clients: {
      badge: "Edge Mesh",
      title: "2. Distributed Non-IID Edge Nodes (K=50)",
      file: "src/update.py",
      desc: "50 heterogeneous edge clients partitioned according to Dirichlet distribution Dir(α=10.0). Clients perform 10 local SGD epochs on local mini-batches. No raw user data leaves the edge device.",
      eq: "$$\\mathcal{D}_k \\sim \\text{Dir}(\\alpha=10.0), \\quad |\\mathcal{B}|=10, \\quad \\mathbf{w}_{k,\\tau+1} = \\mathbf{w}_{k,\\tau} - \\eta_l \\nabla \\mathcal{F}_k(\\mathbf{w}_{k,\\tau}; \\mathcal{B}_{k,\\tau})$$"
    },
    monitor: {
      badge: "Online Profiling",
      title: "3. GradientMonitor Quality Profiling Engine",
      file: "src/gradient_monitor.py",
      desc: "Extracts pseudo-gradient vectors Δw_k = w_t - w_{k,t}. Calculates coefficient of variation CV over rolling window W=5, directional cosine similarity, and magnitude bounds without server validation sets.",
      eq: "$$Q_k = 0.5 \\cdot S_k + 0.3 \\cdot M_k + 0.2 \\cdot C_k \\in [0, 1], \\quad S_k = \\frac{1}{1 + \\text{CV}_k}$$"
    },
    ppo: {
      badge: "RL Controller",
      title: "4. PPO Selection Policy & Adaptive Exploration",
      file: "src/ppo_server/ppo.py",
      desc: "An Actor-Critic neural network that maps global convergence states to client selection probabilities. Employs adaptive ε-greedy exploration (20% budget) to probe dormant or recovering clients.",
      eq: "$$\\mathbf{s}_t = [\\text{Acc}_t, \\text{Loss}_t, t/T, \\bar{Q}, N_{\\text{noisy}}/K, \\sigma_Q], \\quad \\text{Score}_k = 0.7 \\cdot \\pi_\\theta(k \\mid \\mathbf{s}_t) + 0.3 \\cdot Q_k$$"
    },
    aggregator: {
      badge: "Aggregation",
      title: "5. Robust Dual-Balanced Model Aggregator",
      file: "src/ppo_server/gradfed_ppo_update.py",
      desc: "Replaces naive sample-count weighting with dual-balanced weighting: 60% sample volume and 40% verified gradient quality. Penalizes corrupted gradients by up to 85% to protect the global model trajectory.",
      eq: "$$\\alpha_k = 0.6 \\cdot \\frac{|\\mathcal{D}_k|}{\\sum_j |\\mathcal{D}_j|} + 0.4 \\cdot \\frac{Q_k}{\\sum_j Q_j}, \\qquad \\mathbf{w}_{t+1} = \\sum_{k \\in \\mathcal{S}_t} \\alpha_k \\mathbf{w}_{k,t}$$"
    },
    reward: {
      badge: "RL Optimization",
      title: "6. Global Evaluation & Multi-Objective Reward",
      file: "src/gradfed_main.py",
      desc: "Computes global accuracy improvement ΔAcc_t, loss reduction, and cohort quality reward. Populates the PPO rollout buffer to trigger clipped surrogate policy updates.",
      eq: "$$R_t = \\Delta \\text{Acc}_t + \\lambda_1 \\bar{Q}_t - \\lambda_2 \\text{Loss}_t, \\quad \\mathcal{L}^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t [\\min(r_t \\hat{A}_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon)\\hat{A}_t)]$$"
    }
  };

  subBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      subBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const key = btn.dataset.sub;
      const data = SUBSYSTEMS[key];
      if (data) {
        if (badgeEl) badgeEl.textContent = data.badge;
        if (titleEl) titleEl.textContent = data.title;
        if (fileEl) fileEl.textContent = data.file;
        if (descEl) descEl.textContent = data.desc;
        if (eqEl) {
          eqEl.innerHTML = data.eq;
          renderAllMath(eqEl);
        }
      }
    });
  });
}

// 2. 6-Stage Round Lifecycle Pipeline Stepper
let currentStageIdx = 0;
let stageAutoPlayTimer = null;
let isStageAutoPlaying = false;

function setupArchPipeline() {
  const trackEl = document.getElementById('archStepperTrack');
  const displayEl = document.getElementById('archStageDisplay');
  const prevBtn = document.getElementById('btnPrevStage');
  const nextBtn = document.getElementById('btnNextStage');
  const autoPlayBtn = document.getElementById('btnAutoPlayStage');
  const autoPlayIcon = document.getElementById('autoPlayIcon');
  const autoPlayLabel = document.getElementById('autoPlayLabel');

  if (!trackEl || !displayEl) return;

  const PIPELINE_STAGES = [
    {
      num: "01",
      shortTitle: "Global Dispatch",
      title: "Phase 1: Global Model Serialization & Broadcast",
      subsystem: "Server Dispatcher",
      file: "src/gradfed_main.py",
      desc: "At the start of communication round t, the server serializes global parameter weights w_t and broadcasts them to the selected edge client cohort S_t. Client communication payloads are optimized via state serialization.",
      math: "$$\\text{Broadcast: } \\mathbf{w}_t \\in \\mathbb{R}^d \\longrightarrow \\text{Edge Cohort } \\mathcal{S}_t \\subset \\{1, 2, \\dots, K\\}$$",
      props: ["ResNet-10 Backbone", "Round t State", "Zero Leakage"]
    },
    {
      num: "02",
      shortTitle: "Edge Local SGD",
      title: "Phase 2: Edge Client Local Training on Non-IID Data",
      subsystem: "Edge Client Mesh",
      file: "src/update.py",
      desc: "Selected edge devices perform E=10 local SGD epochs using mini-batches of size B=10 on their private Dirichlet-partitioned data. Unbeknownst to the server, certain corrupted clients execute label flips or noisy updates.",
      math: "$$\\mathbf{w}_{k,t} = \\mathbf{w}_t - \\eta \\sum_{e=1}^E \\nabla \\mathcal{L}(\\mathbf{w}; \\mathcal{B}_{k,e}) \\quad \\text{where } \\mathcal{D}_k \\sim \\text{Dir}(\\alpha=10.0)$$",
      props: ["Private Local Data", "Non-IID Partitions", "Local Mini-Batches"]
    },
    {
      num: "03",
      shortTitle: "Pseudo-Gradients",
      title: "Phase 3: Privacy-Preserving Shift Vector Extraction",
      subsystem: "Delta Extractor",
      file: "src/gradient_monitor.py",
      desc: "Rather than uploading private gradients or training batches, clients transmit updated model weights w_{k,t}. The central server computes the exact parameter displacement without inspecting private user features.",
      math: "$$\\Delta \\mathbf{w}_k^{(t)} = \\mathbf{w}_t - \\mathbf{w}_{k,t}, \\qquad \\|\\Delta \\mathbf{w}_k^{(t)}\\|_2 = \\sqrt{\\sum_{i=1}^d (\\Delta w_{k,i})^2}$$",
      props: ["Differential Privacy", "Exact Shift Vector", "100% Privacy Compliant"]
    },
    {
      num: "04",
      shortTitle: "Quality Profiling",
      title: "Phase 4: GradientMonitor Online Reliability Profiling",
      subsystem: "GradientMonitor Engine",
      file: "src/gradient_monitor.py",
      desc: "Tracks rolling window statistics (W=5) for each client. Evaluates norm stability (inverse CV), directional cosine consistency with previous rounds, and sigmoidal magnitude bounds to output composite quality score Q_k ∈ [0, 1].",
      math: "$$S_k = \\frac{1}{1 + \\text{CV}_k}, \\quad M_k = \\frac{2}{1 + e^{\\|\\Delta \\mathbf{w}_k\\|/10}}, \\quad C_k = \\frac{\\cos \\theta + 1}{2} \\implies Q_k = 0.5 S_k + 0.3 M_k + 0.2 C_k$$",
      props: ["Sliding Window W=5", "Zero Server Validation", "Outlier Suppression"]
    },
    {
      num: "05",
      shortTitle: "PPO Selection",
      title: "Phase 5: PPO Policy Selection & Adaptive ε-Exploration",
      subsystem: "PPO Server Agent",
      file: "src/ppo_server/gradfed_ppo_update.py",
      desc: "A Proximal Policy Optimization agent observes the global environment state and produces client selection probabilities. Combined with historical Q_k, an adaptive ε-greedy scheduler probes candidate clients to prevent starvation.",
      math: "$$\\text{Score}_k = 0.7 \\cdot \\pi_\\theta(k \\mid \\mathbf{s}_t) + 0.3 \\cdot Q_k, \\qquad \\epsilon(t) = \\max\\left(0.1, \\, 0.3 - 0.2 \\frac{t}{T}\\right)$$",
      props: ["Actor-Critic MLP", "Hybrid Scoring", "Starvation Prevention"]
    },
    {
      num: "06",
      shortTitle: "Robust Aggregation",
      title: "Phase 6: Robust Dual-Balanced Model Aggregation",
      subsystem: "Dual-Balanced Aggregator",
      file: "src/ppo_server/gradfed_ppo_update.py",
      desc: "The server calculates normalized aggregation weights combining empirical client data size (60%) and verified gradient reliability (40%). Updates with Q_k < 0.3 are attenuated or discarded to safeguard global convergence.",
      math: "$$\\alpha_k = 0.6 \\frac{|\\mathcal{D}_k|}{\\sum_j |\\mathcal{D}_j|} + 0.4 \\frac{Q_k}{\\sum_j Q_j}, \\qquad \\mathbf{w}_{t+1} = \\sum_{k \\in \\mathcal{S}_t} \\alpha_k \\mathbf{w}_{k,t}$$",
      props: ["Dual-Balanced", "Noise Filtered (Q<0.3)", "Global Model Convergence"]
    }
  ];

  // Render stepper buttons
  trackEl.innerHTML = '';
  PIPELINE_STAGES.forEach((stage, idx) => {
    const node = document.createElement('div');
    node.className = `arch-step-node ${idx === 0 ? 'active' : ''}`;
    node.id = `archStepNode-${idx}`;
    node.innerHTML = `
      <div class="arch-step-node-top">
        <span class="arch-step-num">${stage.num}</span>
        <span class="arch-step-indicator"></span>
      </div>
      <div class="arch-step-title">${stage.shortTitle}</div>
    `;
    node.addEventListener('click', () => {
      stopStageAutoPlay();
      renderStage(idx);
    });
    trackEl.appendChild(node);
  });

  function renderStage(idx) {
    currentStageIdx = (idx + PIPELINE_STAGES.length) % PIPELINE_STAGES.length;
    const stage = PIPELINE_STAGES[currentStageIdx];

    document.querySelectorAll('.arch-step-node').forEach((node, i) => {
      node.classList.toggle('active', i === currentStageIdx);
    });

    displayEl.innerHTML = `
      <div class="stage-display-header">
        <div class="stage-display-tag-group">
          <span class="badge">${stage.subsystem}</span>
          <span class="arch-toolbar-badge">${stage.num} / 06</span>
        </div>
        <div class="arch-code-link">
          <span class="arch-code-lbl">Module:</span>
          <code>${stage.file}</code>
        </div>
      </div>
      <h4 class="stage-display-title">${stage.title}</h4>
      <p class="stage-display-desc">${stage.desc}</p>
      <div class="stage-display-math">${stage.math}</div>
      <div class="stage-display-footer">
        <div class="stage-display-props">
          ${stage.props.map(p => `<span class="pill-spec">✓ ${p}</span>`).join('')}
        </div>
      </div>
    `;
    renderAllMath(displayEl);
  }

  function startStageAutoPlay() {
    isStageAutoPlaying = true;
    if (autoPlayIcon) autoPlayIcon.innerHTML = '&#10074;&#10074;';
    if (autoPlayLabel) autoPlayLabel.textContent = 'Pause Loop';
    if (autoPlayBtn) autoPlayBtn.classList.add('active');

    stageAutoPlayTimer = setInterval(() => {
      renderStage(currentStageIdx + 1);
    }, 3800);
  }

  function stopStageAutoPlay() {
    isStageAutoPlaying = false;
    clearInterval(stageAutoPlayTimer);
    if (autoPlayIcon) autoPlayIcon.innerHTML = '&#9654;';
    if (autoPlayLabel) autoPlayLabel.textContent = 'Auto-Play Loop';
    if (autoPlayBtn) autoPlayBtn.classList.remove('active');
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      stopStageAutoPlay();
      renderStage(currentStageIdx - 1);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      stopStageAutoPlay();
      renderStage(currentStageIdx + 1);
    });
  }

  if (autoPlayBtn) {
    autoPlayBtn.addEventListener('click', () => {
      if (isStageAutoPlaying) stopStageAutoPlay();
      else startStageAutoPlay();
    });
  }

  renderStage(0);
}

// 3. Algorithmic Component Deep-Dive Tabs
function setupArchTabs() {
  const tabBtns = document.querySelectorAll('.arch-tab-btn');
  const titleEl = document.getElementById('archTabTitle');
  const mathEl = document.getElementById('archMathBlock');
  const descEl = document.getElementById('archTabDesc');
  const badgeEl = document.getElementById('archTabBadge');
  const fileEl = document.getElementById('archTabFile');
  const propsEl = document.getElementById('archTabProps');

  const ARCH_TABS = {
    monitor: {
      title: "1. Pseudo-Gradient Extraction & L2 Norm Tracking",
      badge: "Privacy-Preserving",
      formula: "$$\\Delta \\mathbf{w}_k^{(t)} = \\mathbf{w}_{\\text{global}}^{(t)} - \\mathbf{w}_{\\text{local},k}^{(t)}$$\n$$\\text{CV}_k = \\frac{\\sigma(\\|\\Delta \\mathbf{w}_k\\|_W)}{\\mu(\\|\\Delta \\mathbf{w}_k\\|_W) + \\epsilon}, \\qquad S_k = \\frac{1}{1 + \\text{CV}_k}$$",
      desc: "Rather than sharing private training batches, clients send back parameter updates. The server calculates the exact shift vector. The GradientMonitor tracks historical norm variability across communication rounds over window W=5 to identify unstable or drifting updates.",
      file: "src/gradient_monitor.py",
      props: ["Zero Data Leakage", "Sliding Window W=5", "L2 Drift Detection"]
    },
    cosine: {
      title: "2. Directional Alignment & Magnitude Bounds",
      badge: "Robust Dynamics",
      formula: "$$C_k = \\frac{1}{2}\\left(1 + \\frac{\\langle \\Delta \\mathbf{w}_k^{(t)}, \\, \\Delta \\mathbf{w}_k^{(t-1)} \\rangle}{\\|\\Delta \\mathbf{w}_k^{(t)}\\|_2 \\cdot \\|\\Delta \\mathbf{w}_k^{(t-1)}\\|_2 + \\epsilon}\\right)$$\n$$M_k = \\frac{2}{1 + \\exp\\left(\\frac{\\|\\Delta \\mathbf{w}_k\\|_2}{10}\\right)}, \\qquad Q_k = 0.5 \\cdot S_k + 0.3 \\cdot M_k + 0.2 \\cdot C_k$$",
      desc: "Measures whether the client is moving cohesively towards the global minimum or oscillating randomly due to label corruption. A sigmoidal scaler penalizes exploding or vanishing gradients, yielding an online composite quality score Q_k ∈ [0, 1].",
      file: "src/gradient_monitor.py",
      props: ["Cosine Angle Tracking", "Sigmoid Magnitude Bounds", "Bounded Metric [0, 1]"]
    },
    ppo: {
      title: "3. PPO Policy with Adaptive ε-Greedy Exploration",
      badge: "Actor-Critic RL",
      formula: "$$\\mathbf{s}_t = \\left[\\text{Acc}_t, \\, \\text{Loss}_t, \\, \\frac{t}{T}, \\, \\bar{Q}_t, \\, \\frac{N_{\\text{noisy}}}{K}, \\, \\sigma_Q\\right] \\in \\mathbb{R}^6$$\n$$\\text{Score}_k = 0.7 \\cdot \\pi_\\theta(k \\mid \\mathbf{s}_t) + 0.3 \\cdot Q_k, \\qquad \\epsilon(t) = \\max\\left(0.1, \\, 0.3 - 0.2 \\frac{t}{T}\\right)$$",
      desc: "A discrete-action Proximal Policy Optimization network generates client selection probabilities. To prevent straggler starvation, an ε-greedy exploration pool periodically probes low-quality clients to detect if their models stabilize, decaying exploration as the global model converges.",
      file: "src/ppo_server/ppo.py",
      props: ["6D State Representation", "Clipped Surrogate Objective", "Adaptive ε-Decay"]
    },
    aggregator: {
      title: "4. Dual-Balanced Quality Aggregator",
      badge: "Convergence Protection",
      formula: "$$\\alpha_k = 0.6 \\cdot \\frac{|\\mathcal{D}_k|}{\\sum_{j \\in \\mathcal{S}_t} |\\mathcal{D}_j|} + 0.4 \\cdot \\frac{Q_k}{\\sum_{j \\in \\mathcal{S}_t} Q_j}$$\n$$\\mathbf{w}_{\\text{global}}^{(t+1)} = \\sum_{k \\in \\mathcal{S}_t} \\alpha_k \\mathbf{w}_{\\text{local},k}^{(t)}$$",
      desc: "Conventional FedAvg is vulnerable to corrupted gradients from high-data clients. GradFed balances data volume (60%) with verified gradient reliability (40%), completely filtering out corrupted client weights without requiring trusted benchmark test sets on the server.",
      file: "src/ppo_server/gradfed_ppo_update.py",
      props: ["60/40 Ratio Balance", "Noise Attenuation ≤ 85%", "Dirichlet Robustness"]
    }
  };

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const tabKey = btn.dataset.tab;
      const data = ARCH_TABS[tabKey];
      if (data) {
        if (titleEl) titleEl.textContent = data.title;
        if (mathEl) {
          mathEl.innerHTML = data.formula;
          renderAllMath(mathEl);
        }
        if (descEl) descEl.textContent = data.desc;
        if (badgeEl) badgeEl.textContent = data.badge;
        if (fileEl) fileEl.textContent = data.file;
        if (propsEl) {
          propsEl.innerHTML = data.props.map(p => `<span class="pill-spec">✓ ${p}</span>`).join('');
        }
      }
    });
  });
}

// 4. Interactive Live Gradient Quality & Aggregation Sandbox
function setupArchSandbox() {
  const normCvSlider = document.getElementById('sbNormCv');
  const cosineSlider = document.getElementById('sbCosine');
  const magSlider = document.getElementById('sbMagnitude');
  const samplesSlider = document.getElementById('sbSamples');
  const resetBtn = document.getElementById('btnResetSandbox');

  if (!normCvSlider || !cosineSlider || !magSlider || !samplesSlider) return;

  function updateSandbox() {
    const cv = parseFloat(normCvSlider.value);
    const cos = parseFloat(cosineSlider.value);
    const mag = parseFloat(magSlider.value);
    const samples = parseInt(samplesSlider.value, 10);

    const normCvVal = document.getElementById('sbNormCvVal');
    const cosineVal = document.getElementById('sbCosineVal');
    const magVal = document.getElementById('sbMagnitudeVal');
    const samplesVal = document.getElementById('sbSamplesVal');

    if (normCvVal) normCvVal.textContent = cv.toFixed(2);
    if (cosineVal) cosineVal.textContent = (cos >= 0 ? '+' : '') + cos.toFixed(2);
    if (magVal) magVal.textContent = mag.toFixed(1);
    if (samplesVal) samplesVal.textContent = samples.toLocaleString();

    // Stability score S_k = 1 / (1 + CV)
    const s = 1.0 / (1.0 + cv);
    // Magnitude score M_k = 2 / (1 + exp(mag / 10))
    const m = 2.0 / (1.0 + Math.exp(mag / 10.0));
    // Consistency score C_k = (cos + 1) / 2
    const c = (cos + 1.0) / 2.0;

    const weightedS = 0.5 * s;
    const weightedM = 0.3 * m;
    const weightedC = 0.2 * c;
    const q = Math.max(0, Math.min(1, weightedS + weightedM + weightedC));

    const qScoreEl = document.getElementById('sbQualityScore');
    const qBarEl = document.getElementById('sbQualityBar');
    const scoreSEl = document.getElementById('sbScoreS');
    const scoreMEl = document.getElementById('sbScoreM');
    const scoreCEl = document.getElementById('sbScoreC');

    if (qScoreEl) qScoreEl.textContent = q.toFixed(3);
    if (qBarEl) {
      qBarEl.style.width = `${(q * 100).toFixed(1)}%`;
      if (q >= 0.55) {
        qBarEl.style.background = 'var(--accent-emerald)';
      } else if (q >= 0.35) {
        qBarEl.style.background = 'var(--accent-amber)';
      } else {
        qBarEl.style.background = 'var(--accent-rose)';
      }
    }

    if (scoreSEl) scoreSEl.textContent = weightedS.toFixed(3);
    if (scoreMEl) scoreMEl.textContent = weightedM.toFixed(3);
    if (scoreCEl) scoreCEl.textContent = weightedC.toFixed(3);

    // Update Verdict Display
    const badge = document.getElementById('sbVerdictBadge');
    const desc = document.getElementById('sbVerdictDesc');
    if (badge && desc) {
      if (q >= 0.55) {
        badge.className = 'verdict-badge approved';
        badge.innerHTML = '&#10003; Approved for Primary Aggregation';
        desc.textContent = 'Gradient trajectory is stable and aligned with global descent. Client is prioritized with full dual-balanced data and quality weighting.';
      } else if (q >= 0.35) {
        badge.className = 'verdict-badge warning';
        badge.innerHTML = '&#9888; Marginal Quality — Routed to &epsilon;-Exploration';
        desc.textContent = 'Moderate variance or drift detected. Client is held in the candidate pool and probed periodically via ε-greedy exploration.';
      } else {
        badge.className = 'verdict-badge danger';
        badge.innerHTML = '&#10007; Flagged as Noisy / Corrupted — Weight Suppressed';
        desc.textContent = 'Severe gradient oscillation or conflicting direction detected (Q < 0.35). Contribution is attenuated by up to 85% to protect the global model.';
      }
    }

    // Weight Calculation Simulation
    const totalData = 4000 + samples;
    const totalQ = 3.0 + q;

    const fedavgPct = (samples / totalData) * 100;
    const dataPart = samples / totalData;
    const qPart = q / totalQ;
    const gradfedPct = (0.6 * dataPart + 0.4 * qPart) * 100;

    const fedAvgBar = document.getElementById('sbFedAvgWeightBar');
    const fedAvgVal = document.getElementById('sbFedAvgWeightVal');
    const gradFedBar = document.getElementById('sbGradFedWeightBar');
    const gradFedVal = document.getElementById('sbGradFedWeightVal');
    const weightNote = document.getElementById('sbWeightNote');

    if (fedAvgBar && fedAvgVal && gradFedBar && gradFedVal) {
      fedAvgBar.style.width = `${Math.min(100, fedavgPct * 2.5).toFixed(1)}%`;
      fedAvgVal.textContent = `${fedavgPct.toFixed(1)}%`;
      gradFedBar.style.width = `${Math.min(100, gradfedPct * 2.5).toFixed(1)}%`;
      gradFedVal.textContent = `${gradfedPct.toFixed(1)}%`;

      if (weightNote) {
        const diff = gradfedPct - fedavgPct;
        if (diff >= 0) {
          weightNote.textContent = `GradFed enhances weight by +${diff.toFixed(1)}% relative to naive FedAvg due to high verified reliability.`;
        } else {
          weightNote.textContent = `GradFed safely dampens weight by ${Math.abs(diff).toFixed(1)}% relative to FedAvg to guard against unreliable gradients.`;
        }
      }
    }
  }

  [normCvSlider, cosineSlider, magSlider, samplesSlider].forEach(slider => {
    slider.addEventListener('input', updateSandbox);
  });

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      normCvSlider.value = 0.15;
      cosineSlider.value = 0.85;
      magSlider.value = 7.5;
      samplesSlider.value = 1000;
      updateSandbox();
    });
  }

  updateSandbox();
}

// 5. Architecture Fullscreen Vector Modal
let archZoomLevel = 1.0;

function setupArchModal() {
  const openBtn = document.getElementById('btnArchModalOpen');
  const modal = document.getElementById('archModal');
  const zoomIn = document.getElementById('btnZoomIn');
  const zoomOut = document.getElementById('btnZoomOut');
  const zoomReset = document.getElementById('btnZoomReset');
  const modalImg = document.getElementById('archModalImg');

  if (openBtn && modal) {
    openBtn.addEventListener('click', () => {
      archZoomLevel = 1.0;
      if (modalImg) modalImg.style.transform = `scale(${archZoomLevel})`;
      modal.style.display = 'flex';
    });
  }

  if (zoomIn && modalImg) {
    zoomIn.addEventListener('click', () => {
      archZoomLevel = Math.min(2.5, archZoomLevel + 0.25);
      modalImg.style.transform = `scale(${archZoomLevel})`;
    });
  }

  if (zoomOut && modalImg) {
    zoomOut.addEventListener('click', () => {
      archZoomLevel = Math.max(0.6, archZoomLevel - 0.25);
      modalImg.style.transform = `scale(${archZoomLevel})`;
    });
  }

  if (zoomReset && modalImg) {
    zoomReset.addEventListener('click', () => {
      archZoomLevel = 1.0;
      modalImg.style.transform = 'scale(1.0)';
    });
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeArchModal();
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeArchModal();
  });
}

function closeArchModal() {
  const modal = document.getElementById('archModal');
  if (modal) modal.style.display = 'none';
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

// ============================================================================
// Multi-Theme Controller Engine
// ============================================================================
const THEMES = {
  light: {
    name: 'Titanium Light',
    dotColor: '#0284c7',
    chartAccColor: '#0284c7',
    chartAccGlow: 'rgba(2, 132, 199, 0.22)',
    chartAccStroke: 'rgba(2, 132, 199, 0.45)',
    gridColor: 'rgba(15, 23, 42, 0.08)',
    axisColor: '#475569',
    lossColor: '#d97706'
  },
  emerald: {
    name: 'Emerald Aurora',
    dotColor: '#10b981',
    chartAccColor: '#10b981',
    chartAccGlow: 'rgba(16, 185, 129, 0.25)',
    chartAccStroke: 'rgba(16, 185, 129, 0.5)',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    axisColor: '#64748b',
    lossColor: '#f59e0b'
  },
  violet: {
    name: 'Cosmic Violet',
    dotColor: '#c084fc',
    chartAccColor: '#c084fc',
    chartAccGlow: 'rgba(192, 132, 252, 0.25)',
    chartAccStroke: 'rgba(192, 132, 252, 0.5)',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    axisColor: '#7c729e',
    lossColor: '#f59e0b'
  },
  cyan: {
    name: 'Quantum Cyan',
    dotColor: '#38bdf8',
    chartAccColor: '#38bdf8',
    chartAccGlow: 'rgba(56, 189, 248, 0.25)',
    chartAccStroke: 'rgba(56, 189, 248, 0.5)',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    axisColor: '#64748b',
    lossColor: '#fbbf24'
  },
  solar: {
    name: 'Solar Flare',
    dotColor: '#fbbf24',
    chartAccColor: '#fbbf24',
    chartAccGlow: 'rgba(251, 191, 36, 0.25)',
    chartAccStroke: 'rgba(251, 191, 36, 0.5)',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    axisColor: '#8c7b70',
    lossColor: '#f97316'
  }
};

function initTheme() {
  const savedTheme = localStorage.getItem('gradfed_theme') || 'light';
  setTheme(savedTheme, false);

  const dropdown = document.getElementById('themeDropdown');
  const toggleBtn = document.getElementById('themeToggleBtn');

  if (toggleBtn && dropdown) {
    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = dropdown.classList.toggle('open');
      toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });

    document.addEventListener('click', (e) => {
      if (!dropdown.contains(e.target)) {
        dropdown.classList.remove('open');
        toggleBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  const options = document.querySelectorAll('.theme-option-item');
  options.forEach(opt => {
    opt.addEventListener('click', () => {
      const selected = opt.dataset.themeVal;
      setTheme(selected, true);
      if (dropdown) dropdown.classList.remove('open');
      if (toggleBtn) toggleBtn.setAttribute('aria-expanded', 'false');
    });
  });
}

function setTheme(themeKey, shouldToast = false) {
  if (!THEMES[themeKey]) themeKey = 'light';
  document.documentElement.setAttribute('data-theme', themeKey);
  localStorage.setItem('gradfed_theme', themeKey);

  const config = THEMES[themeKey];
  const dot = document.getElementById('themeActiveDot');
  const label = document.getElementById('themeActiveName');

  if (dot) dot.style.backgroundColor = config.dotColor;
  if (label) label.textContent = config.name;

  document.querySelectorAll('.theme-option-item').forEach(item => {
    item.classList.toggle('active', item.dataset.themeVal === themeKey);
  });

  if (typeof renderSimChart === 'function') {
    renderSimChart(currentRoundIdx);
  }

  if (shouldToast && typeof showToast === 'function') {
    showToast(`Active Palette: ${config.name}`);
  }
}

