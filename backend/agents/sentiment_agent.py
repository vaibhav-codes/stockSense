# =============================================================================
#  backend/agents/sentiment_agent.py
#  Sentiment Analysis Agent.
#  Fetches live financial news for a ticker, runs FinBERT sentiment inference,
#  aggregates into a –1 to +1 score, and generates SHAP-style word importance.
#  Falls back to Groq LLM if FinBERT is unavailable.
# =============================================================================

import re
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests
import numpy as np

from .base_agent import AgentOutput, BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Lazy-load heavy ML libraries so the app starts even without GPU ────────────
_finbert_pipeline = None

def _get_finbert():
    """Load FinBERT pipeline once on first use (lazy singleton)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline
            _finbert_pipeline = pipeline(
                "text-classification",
                model    = settings.FINBERT_MODEL,
                top_k    = None,       # return all class probabilities
                device   = -1,         # CPU — change to 0 for GPU
                truncation = True,
                max_length = 128,
            )
            logger.info("FinBERT pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT unavailable ({e}). Will use Groq fallback.")
    return _finbert_pipeline


class SentimentAgent(BaseAgent):
    """
    Sentiment Analysis Agent.

    Pipeline:
      1. Fetch headlines via NewsAPI (primary) or GNews (fallback)
      2. Run FinBERT on each headline → positive/negative/neutral probabilities
      3. Aggregate into composite score –1 (very negative) to +1 (very positive)
      4. Build word-importance map (pseudo-SHAP) for explainability
      5. Fall back to Groq LLM if model inference fails
    """

    LABEL_MAP = {
        "positive": 1,
        "negative": -1,
        "neutral":   0,
    }

    def __init__(self):
        super().__init__("SentimentAnalysis")

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self, ticker: str, max_articles: int = 20) -> AgentOutput:

        # Strip .NS / .BO suffix to get clean company name for news search
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

        # 1. Fetch headlines
        articles = self._fetch_news(clean_ticker, max_articles)
        if not articles:
            # Try Groq fallback analysis
            return self._groq_fallback(ticker, clean_ticker)

        # 2. Run sentiment model
        results = self._run_sentiment(articles)

        # 3. Aggregate score
        score, label, breakdown = self._aggregate(results)

        # 4. Word importance (pseudo-SHAP from token scores)
        word_importance = self._compute_word_importance(results)

        return AgentOutput(
            agent_name = self.name,
            ticker     = ticker,
            score      = score,
            label      = label,
            confidence = self._compute_confidence(results),
            data = {
                "score_normalized":  round((score + 1) / 2 * 100, 2),  # 0–100 for UI
                "breakdown":         breakdown,
                "articles":          results[:10],     # top 10 for UI display
                "word_importance":   word_importance,
                "article_count":     len(articles),
                "model_used":        "finbert" if _finbert_pipeline else "groq",
            },
            metadata = {
                "source":     "newsapi",
                "fetched_at": datetime.utcnow().isoformat(),
                "ticker_clean": clean_ticker,
            },
        )

    # ── News fetching ─────────────────────────────────────────────────────────
    def _fetch_news(self, company: str, max_articles: int) -> list[dict]:
        """Fetch headlines from NewsAPI first, GNews as fallback."""
        articles = self._fetch_newsapi(company, max_articles)
        if not articles and settings.GNEWS_API_KEY:
            articles = self._fetch_gnews(company, max_articles)
        return articles

    def _fetch_newsapi(self, company: str, max_articles: int) -> list[dict]:
        """NewsAPI.org — free tier: 100 requests/day."""
        if not settings.NEWS_API_KEY:
            return []
        try:
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":          f"{company} stock OR shares OR NSE",
                    "from":       from_date,
                    "language":   "en",
                    "sortBy":     "publishedAt",
                    "pageSize":   max_articles,
                    "apiKey":     settings.NEWS_API_KEY,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == "ok":
                return [
                    {
                        "title":       a.get("title", ""),
                        "description": a.get("description", ""),
                        "source":      a.get("source", {}).get("name", ""),
                        "published":   a.get("publishedAt", ""),
                        "url":         a.get("url", ""),
                    }
                    for a in data.get("articles", [])
                    if a.get("title")
                ]
        except Exception as e:
            self.logger.warning(f"NewsAPI fetch failed: {e}")
        return []

    def _fetch_gnews(self, company: str, max_articles: int) -> list[dict]:
        """GNews API — fallback news source."""
        try:
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q":       f"{company} stock",
                    "lang":    "en",
                    "max":     max_articles,
                    "token":   settings.GNEWS_API_KEY,
                },
                timeout=10,
            )
            data = resp.json()
            return [
                {
                    "title":     a.get("title", ""),
                    "source":    a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", ""),
                    "url":       a.get("url", ""),
                }
                for a in data.get("articles", [])
                if a.get("title")
            ]
        except Exception as e:
            self.logger.warning(f"GNews fetch failed: {e}")
        return []

    # ── FinBERT inference ─────────────────────────────────────────────────────
    def _run_sentiment(self, articles: list[dict]) -> list[dict]:
        """Run FinBERT on each headline. Returns enriched article list."""
        pipe    = _get_finbert()
        results = []

        for article in articles:
            text = article.get("title", "") + " " + article.get("description", "")
            text = text.strip()[:512]
            if not text:
                continue

            sentiment_data = self._infer_single(pipe, text)
            results.append({
                **article,
                "sentiment":   sentiment_data,
                "text_used":   text[:200],
            })

        return results

    def _infer_single(self, pipe, text: str) -> dict:
        """
        Run FinBERT on one text. Falls back to Groq if pipe is None.
        Returns dict: {label, score, positive, negative, neutral}
        """
        if pipe is not None:
            try:
                output = pipe(text)[0]   # top_k=None returns list of all labels
                # Normalise into a clean dict
                prob_map = {item["label"].lower(): item["score"] for item in output}
                pos  = prob_map.get("positive", 0.33)
                neg  = prob_map.get("negative", 0.33)
                neu  = prob_map.get("neutral",  0.34)
                pred = max(prob_map, key=prob_map.get)
                return {
                    "label":    pred,
                    "score":    self.LABEL_MAP.get(pred, 0),
                    "positive": round(pos, 4),
                    "negative": round(neg, 4),
                    "neutral":  round(neu, 4),
                }
            except Exception as e:
                self.logger.warning(f"FinBERT inference error: {e}")

        # Rule-based fallback when model unavailable
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> dict:
        """
        Simple keyword-based sentiment for demo / fallback.
        Covers common Indian financial news terms.
        """
        text_lower = text.lower()
        positive_words = [
            "surge", "rally", "gain", "profit", "beat", "bullish", "buy",
            "upgrade", "record", "high", "growth", "rise", "up", "positive",
            "strong", "outperform", "fii buying", "upper circuit", "target raised",
        ]
        negative_words = [
            "fall", "drop", "loss", "miss", "bearish", "sell", "downgrade",
            "low", "decline", "crash", "weak", "underperform", "fii selling",
            "lower circuit", "cut", "slump", "plunge", "target cut",
        ]
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        total = pos + neg + 1
        pos_prob = pos / total
        neg_prob = neg / total
        neu_prob = 1 - pos_prob - neg_prob

        if pos > neg:   label = "positive"
        elif neg > pos: label = "negative"
        else:           label = "neutral"

        return {
            "label":    label,
            "score":    self.LABEL_MAP[label],
            "positive": round(pos_prob, 4),
            "negative": round(neg_prob, 4),
            "neutral":  round(max(0, neu_prob), 4),
        }

    # ── Aggregation ───────────────────────────────────────────────────────────
    def _aggregate(self, results: list[dict]) -> tuple[float, str, dict]:
        """
        Weighted average sentiment: confidence-weighted mean of scores.
        Returns (score –1..+1, label, breakdown dict).
        """
        if not results:
            return 0.0, "Neutral", {}

        weighted_sum = 0.0
        weight_total = 0.0
        counts = {"positive": 0, "negative": 0, "neutral": 0}

        for r in results:
            sent = r.get("sentiment", {})
            label = sent.get("label", "neutral")
            # Confidence = max probability (how certain the model is)
            confidence = max(
                sent.get("positive", 0.33),
                sent.get("negative", 0.33),
                sent.get("neutral", 0.34),
            )
            weighted_sum  += sent.get("score", 0) * confidence
            weight_total  += confidence
            counts[label] = counts.get(label, 0) + 1

        score = weighted_sum / weight_total if weight_total > 0 else 0.0

        if score > 0.3:    label = "Very Positive"
        elif score > 0.1:  label = "Positive"
        elif score > -0.1: label = "Neutral"
        elif score > -0.3: label = "Negative"
        else:              label = "Very Negative"

        return round(score, 4), label, {
            "positive_count": counts["positive"],
            "negative_count": counts["negative"],
            "neutral_count":  counts["neutral"],
            "total":          len(results),
        }

    def _compute_confidence(self, results: list[dict]) -> float:
        """Confidence based on article count and agreement ratio."""
        if not results:
            return 0.0
        labels   = [r.get("sentiment", {}).get("label", "neutral") for r in results]
        dominant = max(set(labels), key=labels.count)
        agreement = labels.count(dominant) / len(labels)
        count_score = min(1.0, len(results) / 10)
        return round((agreement * 0.6 + count_score * 0.4), 4)

    def _compute_word_importance(self, results: list[dict]) -> list[dict]:
        """
        Pseudo-SHAP: finds words that frequently co-occur with strong sentiment.
        Returns top 15 words with their average sentiment contribution.
        """
        word_scores = {}
        stop_words  = {
            "the", "a", "an", "is", "in", "on", "at", "to", "for",
            "of", "and", "or", "but", "it", "its", "as", "be", "with",
            "that", "this", "by", "from", "are", "was", "have", "has",
        }

        for r in results:
            text  = r.get("text_used", "")
            score = r.get("sentiment", {}).get("score", 0)
            if score == 0:
                continue
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            for w in words:
                if w not in stop_words:
                    if w not in word_scores:
                        word_scores[w] = []
                    word_scores[w].append(score)

        importance = []
        for word, scores in word_scores.items():
            avg = np.mean(scores)
            importance.append({
                "word":        word,
                "importance":  round(float(avg), 4),
                "count":       len(scores),
                "direction":   "positive" if avg > 0 else "negative" if avg < 0 else "neutral",
            })

        # Sort by absolute importance, return top 15
        importance.sort(key=lambda x: abs(x["importance"]) * x["count"], reverse=True)
        return importance[:15]

    # ── Groq LLM fallback ─────────────────────────────────────────────────────
    def _groq_fallback(self, ticker: str, company: str) -> AgentOutput:
        """
        When no news is fetched, use Groq to generate a sentiment assessment
        based on general market knowledge about the company.
        """
        if not settings.GROQ_API_KEY:
            return AgentOutput.failure(
                self.name, ticker,
                "No news data and no Groq API key configured"
            )

        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)

            prompt = f"""You are an Indian stock market analyst. Analyse the current market sentiment for {company} (Indian listed company, NSE).

Provide a JSON response with exactly this structure:
{{
  "sentiment_score": <float between -1.0 (very negative) and 1.0 (very positive)>,
  "label": "<Very Positive|Positive|Neutral|Negative|Very Negative>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<2-3 sentence analysis>",
  "key_factors": ["factor1", "factor2", "factor3"]
}}

Base your analysis on recent sector trends, company fundamentals, and Indian market conditions.
Respond ONLY with the JSON object."""

            resp = client.chat.completions.create(
                model    = settings.GROQ_MODEL,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.3,
                max_tokens  = 500,
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)

            return AgentOutput(
                agent_name = self.name,
                ticker     = ticker,
                score      = float(data.get("sentiment_score", 0)),
                label      = data.get("label", "Neutral"),
                confidence = float(data.get("confidence", 0.5)),
                data = {
                    "score_normalized": round((float(data.get("sentiment_score", 0)) + 1) / 2 * 100, 2),
                    "reasoning":        data.get("reasoning", ""),
                    "key_factors":      data.get("key_factors", []),
                    "article_count":    0,
                    "model_used":       "groq",
                },
                metadata={"source": "groq_llm"},
            )

        except Exception as e:
            return AgentOutput.failure(self.name, ticker, f"Groq fallback failed: {e}")
