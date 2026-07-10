/**
 * StockSense — Main JavaScript
 * API client, chart rendering, score animations, and UI utilities.
 */

// ── API Client ────────────────────────────────────────────────────────────────
const API = {
  base: '/api',

  async get(path) {
    const r = await fetch(this.base + path);
    if (!r.ok) throw new Error(`API ${path} → ${r.status}`);
    return r.json();
  },

  async post(path, body) {
    const r = await fetch(this.base + path, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`API POST ${path} → ${r.status}`);
    return r.json();
  },

  pipeline:  (ticker, cache=false) => API.get(`/pipeline/${ticker}?cache=${cache}`),
  ta:        (ticker) => API.get(`/ta/${ticker}`),
  sentiment: (ticker) => API.get(`/sentiment/${ticker}`),
  volume:    (ticker) => API.get(`/volume/${ticker}`),
  chart:     (ticker, period='6mo', interval='1d') =>
                API.get(`/chart/${ticker}?period=${period}&interval=${interval}`),
  screener:  (body)   => API.post('/screener', body),
  indices:   ()       => API.get('/indices'),
};

// ── Signal colour helpers ─────────────────────────────────────────────────────
const SIGNAL_COLORS = {
  'Strong Buy':  '#00d4aa',
  'Buy':         '#68d391',
  'Hold':        '#f6ad55',
  'Sell':        '#fc8181',
  'Strong Sell': '#fc4a4a',
};

const SIGNAL_CLASS = {
  'Strong Buy':  'strong-buy',
  'Buy':         'buy',
  'Hold':        'hold',
  'Sell':        'sell',
  'Strong Sell': 'strong-sell',
};

function signalBadge(label) {
  const cls = SIGNAL_CLASS[label] || 'hold';
  return `<span class="signal-badge ${cls}">${label || '—'}</span>`;
}

function verdictBadge(verdict) {
  const cls = (verdict || 'PASS').toLowerCase();
  return `<span class="verdict-badge ${cls}">${verdict || 'PASS'}</span>`;
}

function scoreColor(score, max=100) {
  const pct = score / max;
  if (pct >= 0.8) return '#00d4aa';
  if (pct >= 0.6) return '#68d391';
  if (pct >= 0.4) return '#f6ad55';
  if (pct >= 0.2) return '#fc8181';
  return '#fc4a4a';
}

// Format sentiment score (–1..+1) to display
function fmtSentiment(score) {
  if (score === null || score === undefined) return '—';
  const pct = ((score + 1) / 2 * 100).toFixed(1);
  const color = score > 0.1 ? '#00d4aa' : score < -0.1 ? '#fc4a4a' : '#f6ad55';
  return `<span style="color:${color};font-family:var(--font-data)">${score >= 0 ? '+' : ''}${score.toFixed(3)}</span>`;
}

function fmtChange(pct) {
  if (pct === null || pct === undefined) return '—';
  const color = pct >= 0 ? '#00d4aa' : '#fc4a4a';
  const sign  = pct >= 0 ? '+' : '';
  return `<span style="color:${color}">${sign}${pct.toFixed(2)}%</span>`;
}

function fmtPrice(p) {
  return p ? `₹${Number(p).toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : '—';
}

// ── Score bar renderer ────────────────────────────────────────────────────────
function renderScoreBar(score, max=100, el) {
  const pct = Math.min(100, (score / max) * 100);
  const color = scoreColor(score, max);
  if (!el) return;
  el.innerHTML = `
    <div class="score-bar">
      <div class="score-bar-fill" style="width:0%;background:${color}" 
           data-target="${pct}"></div>
    </div>`;
  // Animate after paint
  setTimeout(() => {
    el.querySelector('.score-bar-fill').style.width = pct + '%';
  }, 50);
}

// ── Toast notifications ───────────────────────────────────────────────────────
const Toast = {
  container: null,

  init() {
    this.container = document.createElement('div');
    this.container.className = 'toast-container';
    document.body.appendChild(this.container);
  },

  show(message, type='success', duration=4000) {
    if (!this.container) this.init();
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    this.container.appendChild(el);
    setTimeout(() => el.remove(), duration);
  },

  success: (msg) => Toast.show(msg, 'success'),
  error:   (msg) => Toast.show(msg, 'error', 6000),
  warning: (msg) => Toast.show(msg, 'warning'),
};

// ── Loading state helpers ─────────────────────────────────────────────────────
function showLoading(el, message='Analysing…') {
  if (!el) return;
  el.innerHTML = `
    <div class="loading-overlay">
      <div class="spinner"></div>
      <div class="text-secondary text-sm">${message}</div>
    </div>`;
}

function showError(el, message) {
  if (!el) return;
  el.innerHTML = `
    <div class="loading-overlay">
      <div style="font-size:24px">⚠️</div>
      <div class="text-secondary text-sm">${message}</div>
    </div>`;
}

// ── Market status ─────────────────────────────────────────────────────────────
function updateMarketStatus() {
  const badge = document.querySelector('.market-badge');
  if (!badge) return;

  // IST: UTC+5:30
  const now = new Date();
  const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
  const h   = ist.getUTCHours();
  const m   = ist.getUTCMinutes();
  const day = ist.getUTCDay();
  const mins = h * 60 + m;
  const isWeekday = day >= 1 && day <= 5;
  const isOpen    = isWeekday && mins >= 555 && mins <= 930; // 9:15 to 15:30

  badge.classList.toggle('open', isOpen);
  badge.querySelector('.dot').title = isOpen ? 'Market Open' : 'Market Closed';
  badge.querySelector('span:last-child').textContent =
    isOpen ? 'NSE Open' : 'NSE Closed';
}

// ── Ticker input autocomplete (from indices) ──────────────────────────────────
async function initTickerAutocomplete(inputEl) {
  let tickers = [];
  try {
    const data = await API.indices();
    tickers = data.indices.NIFTY50 || [];
  } catch (e) { /* ignore */ }

  inputEl.addEventListener('input', function() {
    const val = this.value.toUpperCase();
    const list = document.getElementById('ticker-suggestions');
    if (!list) return;
    if (!val) { list.innerHTML = ''; return; }
    const matches = tickers.filter(t => t.startsWith(val)).slice(0, 6);
    list.innerHTML = matches.map(t =>
      `<div class="suggestion-item" onclick="selectTicker('${t}')">${t}</div>`
    ).join('');
  });
}

function selectTicker(ticker) {
  const input = document.getElementById('ticker-input');
  if (input) input.value = ticker;
  const list = document.getElementById('ticker-suggestions');
  if (list) list.innerHTML = '';
}

// ── Candlestick chart via Plotly ──────────────────────────────────────────────
async function renderCandlestickChart(ticker, period='6mo', containerId='chart-container') {
  const container = document.getElementById(containerId);
  if (!container || typeof Plotly === 'undefined') return;

  showLoading(container, `Loading ${ticker} chart…`);

  try {
    const data = await API.chart(ticker, period);
    const candles = data.candles || [];

    const candlestick = {
      type: 'candlestick',
      x:    candles.map(c => c.time),
      open: candles.map(c => c.open),
      high: candles.map(c => c.high),
      low:  candles.map(c => c.low),
      close:candles.map(c => c.close),
      increasing: { line: { color: '#00d4aa', width: 1 }, fillcolor: 'rgba(0,212,170,0.3)' },
      decreasing: { line: { color: '#fc4a4a', width: 1 }, fillcolor: 'rgba(252,74,74,0.3)' },
      name: ticker,
    };

    const volume = {
      type:  'bar',
      x:     candles.map(c => c.time),
      y:     candles.map(c => c.volume),
      name:  'Volume',
      yaxis: 'y2',
      marker: {
        color: candles.map((c, i) =>
          i === 0 || c.close >= c.open ? 'rgba(0,212,170,0.25)' : 'rgba(252,74,74,0.25)'
        ),
      },
    };

    // EMA overlays
    const overlays = data.overlays || {};
    const ema20Line = {
      type:  'scatter',
      mode:  'lines',
      x:     (overlays.ema20 || []).map(p => p.time),
      y:     (overlays.ema20 || []).map(p => p.value),
      line:  { color: '#4f8ef7', width: 1.5 },
      name:  'EMA 20',
    };
    const ema50Line = {
      type:  'scatter',
      mode:  'lines',
      x:     (overlays.ema50 || []).map(p => p.time),
      y:     (overlays.ema50 || []).map(p => p.value),
      line:  { color: '#f6ad55', width: 1.5 },
      name:  'EMA 50',
    };

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      font:          { family: 'IBM Plex Mono', color: '#8896b0', size: 11 },
      xaxis: {
        rangeslider:   { visible: false },
        gridcolor:     'rgba(255,255,255,0.04)',
        showgrid:      true,
        type:          'date',
      },
      yaxis: {
        gridcolor:  'rgba(255,255,255,0.04)',
        showgrid:   true,
        side:       'right',
        tickprefix: '₹',
      },
      yaxis2: {
        overlaying: 'y',
        side:       'left',
        showgrid:   false,
        showticklabels: false,
        range: [0, Math.max(...candles.map(c => c.volume)) * 5],
      },
      legend: {
        x: 0, y: 1.05,
        orientation: 'h',
        bgcolor:     'transparent',
        font:        { size: 10 },
      },
      margin:   { l: 10, r: 60, t: 10, b: 30 },
      dragmode: 'pan',
    };

    Plotly.newPlot(containerId, [volume, candlestick, ema20Line, ema50Line], layout, {
      responsive:  true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
    });

  } catch (e) {
    showError(container, `Chart error: ${e.message}`);
  }
}

// ── RSI sub-chart ─────────────────────────────────────────────────────────────
async function renderRSIChart(ticker, period='6mo', containerId='rsi-container') {
  const container = document.getElementById(containerId);
  if (!container || typeof Plotly === 'undefined') return;

  try {
    const data    = await API.chart(ticker, period);
    const rsiData = (data.overlays || {}).rsi || [];

    Plotly.newPlot(containerId, [{
      type:  'scatter',
      mode:  'lines',
      x:     rsiData.map(p => p.time),
      y:     rsiData.map(p => p.value),
      line:  { color: '#9f7aea', width: 1.5 },
      name:  'RSI',
    }], {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      font:          { family: 'IBM Plex Mono', color: '#8896b0', size: 10 },
      xaxis:         { gridcolor: 'rgba(255,255,255,0.04)', showgrid: true },
      yaxis:         { gridcolor: 'rgba(255,255,255,0.04)', range: [0, 100], side: 'right' },
      shapes: [
        { type: 'line', y0: 70, y1: 70, x0: 0, x1: 1, xref: 'paper',
          line: { color: 'rgba(252,74,74,0.4)', dash: 'dot', width: 1 } },
        { type: 'line', y0: 30, y1: 30, x0: 0, x1: 1, xref: 'paper',
          line: { color: 'rgba(0,212,170,0.4)', dash: 'dot', width: 1 } },
      ],
      margin:  { l: 10, r: 50, t: 5, b: 20 },
      height:  120,
    }, { responsive: true, displaylogo: false, displayModeBar: false });

  } catch (e) { /* silent */ }
}

// ── SHAP waterfall chart ──────────────────────────────────────────────────────
function renderWordImportance(wordData, containerId='shap-container') {
  const container = document.getElementById(containerId);
  if (!container || !wordData || !wordData.length) {
    if (container) container.innerHTML = '<div class="text-muted text-sm" style="padding:16px">No word importance data</div>';
    return;
  }

  if (typeof Plotly === 'undefined') return;

  const top = wordData.slice(0, 12);
  const colors = top.map(d => d.importance > 0 ? '#00d4aa' : '#fc4a4a');

  Plotly.newPlot(containerId, [{
    type:        'bar',
    orientation: 'h',
    y:    top.map(d => d.word),
    x:    top.map(d => d.importance),
    marker: { color: colors },
    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
  }], {
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font:          { family: 'IBM Plex Mono', color: '#8896b0', size: 11 },
    xaxis:         { gridcolor: 'rgba(255,255,255,0.04)', zeroline: true, zerolinecolor: 'rgba(255,255,255,0.15)' },
    yaxis:         { autorange: 'reversed' },
    margin:        { l: 80, r: 20, t: 10, b: 30 },
    height:        280,
  }, { responsive: true, displaylogo: false, displayModeBar: false });
}

// ── Radar / spider chart for agent scores ─────────────────────────────────────
function renderRadarChart(scores, containerId='radar-container') {
  const container = document.getElementById(containerId);
  if (!container || typeof Plotly === 'undefined') return;

  const categories = ['TA Score', 'Volume', 'Sentiment', 'News', 'Overall'];
  const values     = [
    scores.ta || 50,
    scores.volume || 50,
    ((scores.sentiment || 0) + 1) / 2 * 100,
    scores.news || 50,
    scores.signal || 50,
  ];
  values.push(values[0]); // close the radar shape
  categories.push(categories[0]);

  Plotly.newPlot(containerId, [{
    type:  'scatterpolar',
    r:     values,
    theta: categories,
    fill:  'toself',
    fillcolor: 'rgba(0, 212, 170, 0.15)',
    line:  { color: '#00d4aa', width: 2 },
    name:  'Agent Scores',
  }], {
    polar: {
      bgcolor: 'transparent',
      radialaxis: { range: [0, 100], gridcolor: 'rgba(255,255,255,0.1)', tickfont: { size: 9 } },
      angularaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
    },
    paper_bgcolor: 'transparent',
    font:          { family: 'IBM Plex Mono', color: '#8896b0', size: 11 },
    margin:        { l: 40, r: 40, t: 40, b: 40 },
    showlegend:    false,
    height:        220,
  }, { responsive: true, displaylogo: false, displayModeBar: false });
}

// ── Tab switcher ──────────────────────────────────────────────────────────────
function initTabs(containerEl) {
  if (!containerEl) return;
  containerEl.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      containerEl.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      containerEl.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const pane = containerEl.querySelector(`.tab-pane[data-tab="${target}"]`);
      if (pane) pane.classList.add('active');
    });
  });
}

// ── On DOM ready ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  Toast.init();
  updateMarketStatus();
  setInterval(updateMarketStatus, 60000);

  // Init all tab containers on the page
  document.querySelectorAll('[data-tabs]').forEach(initTabs);

  // Ticker suggestions input
  const tickerInput = document.getElementById('ticker-input');
  if (tickerInput) initTickerAutocomplete(tickerInput);
});
