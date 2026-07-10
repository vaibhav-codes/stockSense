# 📈 StockSense — Multi-Agent AI Stock Analysis for Indian Markets

> **Dissertation Project** | Finance × NLP/LLMs | NSE/BSE Indian Equities

A multi-agent AI system that combines real-time news sentiment (FinBERT), technical analysis, volume signals, and explainability (SHAP) to generate reliable trading signals for Indian equity markets.

---

## 🏗️ Architecture

```
stocksense/
├── app.py                          ← Flask entry point (run this)
├── .env                            ← API keys (never commit)
├── requirements.txt
├── setup.sh                        ← First-time setup script
│
├── config/
│   └── settings.py                 ← Centralised config (reads .env)
│
├── backend/
│   ├── orchestrator.py             ← LangGraph-style pipeline orchestrator
│   ├── routes.py                   ← Flask REST API Blueprint
│   ├── agents/
│   │   ├── base_agent.py           ← BaseAgent, AgentOutput, Verdict enum
│   │   ├── ta_agent.py             ← Technical Analysis (pandas-ta, 12 signals)
│   │   ├── volume_agent.py         ← Volume Analysis (OBV, VWAP, z-score)
│   │   ├── sentiment_agent.py      ← FinBERT + NewsAPI + SHAP + Groq fallback
│   │   ├── news_agent.py           ← News freshness & credibility scoring
│   │   ├── signal_agent.py         ← LightGBM signal fusion (3 strategies)
│   │   └── reviewer_agent.py       ← Critic/validator (5 checks, cross-agent)
│   ├── data/
│   │   └── scheduler.py            ← APScheduler market-hours auto-refresh
│   └── models/
│       └── signal_model.pkl        ← (auto-generated) LightGBM model
│
└── frontend/
    ├── templates/
    │   ├── base.html               ← Shared nav, Plotly import
    │   ├── dashboard.html          ← Quick analyse, metrics, radar chart
    │   ├── chart.html              ← Candlestick terminal + RSI panel
    │   ├── screener.html           ← Nifty 50 screener table
    │   ├── sentiment.html          ← FinBERT scores + SHAP waterfall
    │   └── signals.html            ← Strategy signal cards
    └── static/
        ├── css/main.css            ← Dark terminal aesthetic
        └── js/main.js              ← API client, Plotly charts, utilities
```

---

## 🚀 Quick Start

### 1. Clone / unzip the project

```bash
unzip stocksense.zip
cd stocksense
```

### 2. Run the setup script

```bash
bash setup.sh
```

This creates a virtual environment and installs all dependencies.

### 3. Configure API keys

Edit the `.env` file — it's pre-created for you:

```env
GROQ_API_KEY=your-groq-api-key-here      # https://console.groq.com (free)
NEWS_API_KEY=your-newsapi-key-here        # https://newsapi.org      (free)
GNEWS_API_KEY=your-gnews-api-key-here     # https://gnews.io         (free)
```

> **The app works without any API keys** — TA and Volume analysis run entirely offline using yfinance data. Sentiment uses a rule-based fallback.

### 4. Start the application

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
python app.py
```

The browser will open automatically at `http://localhost:5000`.

---

## 🤖 Multi-Agent Pipeline

Each ticker run executes this graph:

```
[yfinance / NSE data]
        ↓
┌───────────────────────────────────────────────────────────┐
│  Stage 1 (parallel threads)                                │
│  TA Agent ──┐                                             │
│  Volume Agent ──┤→ raw outputs                            │
│  News Agent ──┘                                           │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│  Stage 2                                                   │
│  Sentiment Agent (FinBERT + NewsAPI + Groq fallback)       │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│  Stage 3 — Reviewer Agent (CRITIC)                        │
│  • Schema check    • Confidence gate                       │
│  • Staleness check • Outlier guard (>30pt jump)           │
│  • Coverage gate   • Cross-agent conflict detection       │
│  → PASS (w=1.0) / WARN (w=0.5) / FAIL (w=0.0)           │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│  Stage 4 — Signal Agent (LightGBM / rule fusion)          │
│  → Strong Buy / Buy / Hold / Sell / Strong Sell           │
│  → Long-term / Swing / Short-term breakdowns              │
└───────────────────────────────────────────────────────────┘
```

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/pipeline/<ticker>` | Full 4-stage pipeline |
| POST | `/api/screener` | Screen multiple tickers |
| GET | `/api/ta/<ticker>` | TA agent only |
| GET | `/api/sentiment/<ticker>` | Sentiment agent only |
| GET | `/api/volume/<ticker>` | Volume agent only |
| GET | `/api/chart/<ticker>` | OHLCV + indicator data |
| GET | `/api/indices` | Nifty 50 ticker list |

---

## 🧠 Research Novelties (Dissertation)

1. **Domain-adapted FinBERT** — base model tuned on Indian financial corpus (ET, Mint, Moneycontrol) with Indian-specific terms: *upper circuit, FII outflows, promoter pledging, Sensex*

2. **Reviewer Agent (critic-actor pattern)** — formal validation gate between specialist agents and final aggregation; novel contribution vs standard agent pipelines

3. **LangGraph state machine** — formally describable agent DAG enabling systematic ablation studies (disable one agent, observe signal degradation)

4. **3-strategy fusion** — same data, different weightings for long-term / swing / short-term signals

---

## 🔧 Configuration Reference (`.env`)

| Key | Default | Description |
|-----|---------|-------------|
| `GROQ_API_KEY` | — | Groq LLM API key (sentiment fallback) |
| `NEWS_API_KEY` | — | NewsAPI.org key |
| `GNEWS_API_KEY` | — | GNews.io key |
| `FLASK_PORT` | `5000` | HTTP port |
| `FINBERT_MODEL` | `ProsusAI/finbert` | HuggingFace model ID |
| `SCHEDULER_ENABLED` | `True` | Market-hours auto-refresh |
| `PIPELINE_INTERVAL_MINUTES` | `15` | Refresh frequency |
| `USE_REDIS` | `False` | Use Redis instead of in-memory cache |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `yfinance` | NSE/BSE OHLCV price data |
| `pandas-ta` | 130+ technical indicators |
| `transformers` | FinBERT sentiment model |
| `torch` | Neural network backend |
| `lightgbm` | Signal fusion model |
| `groq` | Groq LLM API client |
| `apscheduler` | Market-hours scheduler |
| `plotly.js` | Interactive candlestick charts |

---

## 🖥️ UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Quick analyse any ticker; radar chart; reviewer report |
| Chart Terminal | `/chart` | Plotly candlestick + RSI + EMA overlays |
| Stock Screener | `/screener` | Rank Nifty 50 by strategy |
| Sentiment | `/sentiment` | FinBERT scores + word importance |
| Trade Signals | `/signals` | Strategy signal cards |

---

## ⚠️ Disclaimer

This project is for **academic research only**. It does not constitute financial advice. Always consult a SEBI-registered financial advisor before making investment decisions.
