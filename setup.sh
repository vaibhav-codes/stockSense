#!/usr/bin/env bash
# =============================================================================
#  setup.sh — StockSense First-Time Setup
#  Usage: bash setup.sh
#  Creates a virtual environment, installs dependencies, and guides API key setup.
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║            StockSense — First-Time Setup                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check Python version ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✔ Python: $PY_VERSION"

# ── Create virtual environment ────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment…"
    $PYTHON -m venv venv
    echo "✔ Virtual environment created"
else
    echo "✔ Virtual environment already exists"
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || true
echo "✔ Virtual environment activated"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "→ Installing Python dependencies (this may take 2–5 minutes)…"
echo "  (Torch + Transformers are large — please be patient)"
echo ""
pip install -r requirements.txt

echo ""
echo "✔ Dependencies installed"

# ── Create model placeholder ──────────────────────────────────────────────────
mkdir -p backend/models
touch backend/models/.gitkeep

# ── Check .env file ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  API KEY SETUP (required for news and LLM features)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Edit the .env file and fill in your API keys:"
echo ""
echo "  1. GROQ_API_KEY   → https://console.groq.com   (free)"
echo "  2. NEWS_API_KEY   → https://newsapi.org         (free tier: 100 req/day)"
echo "  3. GNEWS_API_KEY  → https://gnews.io            (free tier)"
echo ""
echo "  The app works without these keys but sentiment analysis"
echo "  will use rule-based fallback only."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✔ Setup complete!"
echo ""
echo "  To start StockSense, run:"
echo "    source venv/bin/activate"
echo "    python app.py"
echo ""
echo "  The app will open in your browser automatically."
echo ""
