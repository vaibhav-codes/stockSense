# =============================================================================
#  backend/agents/news_agent.py
#  News Freshness & Credibility Agent.
#  Evaluates the quality of news available for a ticker:
#  — article recency (freshness score)
#  — source credibility
#  — coverage volume
#  This output feeds the Reviewer Agent's confidence gate.
# =============================================================================

import re
import requests
from datetime import datetime, timedelta, timezone

from .base_agent import AgentOutput, BaseAgent
from config.settings import settings


# Known credible Indian financial news sources
CREDIBLE_SOURCES = {
    "Economic Times":    1.0,
    "Moneycontrol":      1.0,
    "LiveMint":          0.95,
    "Business Standard": 0.95,
    "Financial Express": 0.90,
    "NDTV Profit":       0.85,
    "Bloomberg":         1.0,
    "Reuters":           1.0,
    "CNBC":              0.90,
    "Hindu Business":    0.85,
}


class NewsAgent(BaseAgent):
    """
    Assesses the freshness and credibility of news coverage for a ticker.
    Does NOT do sentiment — that's the SentimentAgent's job.
    Outputs a quality score (0–100) used by the Reviewer Agent.
    """

    def __init__(self):
        super().__init__("NewsFreshness")

    def run(self, ticker: str, max_articles: int = 30) -> AgentOutput:
        clean = ticker.replace(".NS", "").replace(".BO", "")
        articles = self._fetch(clean, max_articles)

        if not articles:
            return AgentOutput(
                agent_name = self.name,
                ticker     = ticker,
                score      = 20.0,
                label      = "Low Coverage",
                confidence = 0.5,
                data       = {"article_count": 0, "freshness": "no_data"},
                metadata   = {},
            )

        metrics = self._evaluate(articles)
        score, label = self._score(metrics)

        return AgentOutput(
            agent_name = self.name,
            ticker     = ticker,
            score      = score,
            label      = label,
            confidence = 0.8,
            data       = {
                "metrics":           metrics,
                "article_count":     len(articles),
                "latest_articles":   articles[:5],
            },
            metadata   = {"ticker_clean": clean},
        )

    def _fetch(self, company: str, max_articles: int) -> list[dict]:
        if not settings.NEWS_API_KEY:
            return []
        try:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        company,
                    "from":     from_date,
                    "language": "en",
                    "pageSize": max_articles,
                    "apiKey":   settings.NEWS_API_KEY,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == "ok":
                return data.get("articles", [])
        except Exception as e:
            self.logger.warning(f"News fetch failed: {e}")
        return []

    def _evaluate(self, articles: list[dict]) -> dict:
        now = datetime.now(timezone.utc)
        ages_hours    = []
        source_scores = []

        for a in articles:
            # Age calculation
            pub = a.get("publishedAt", "")
            if pub:
                try:
                    pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    ages_hours.append((now - pub_dt).total_seconds() / 3600)
                except Exception:
                    pass

            # Source credibility
            source_name = a.get("source", {}).get("name", "")
            credibility = next(
                (v for k, v in CREDIBLE_SOURCES.items() if k.lower() in source_name.lower()),
                0.5,   # unknown source gets 0.5
            )
            source_scores.append(credibility)

        avg_age_hours = sum(ages_hours) / len(ages_hours) if ages_hours else 999
        avg_credibility = sum(source_scores) / len(source_scores) if source_scores else 0.5
        fresh_count = sum(1 for h in ages_hours if h < 24)

        return {
            "avg_age_hours":    round(avg_age_hours, 1),
            "fresh_count_24h":  fresh_count,
            "avg_credibility":  round(avg_credibility, 3),
            "total_articles":   len(articles),
            "is_stale":         avg_age_hours > settings.REVIEWER_MAX_NEWS_AGE_HOURS,
        }

    def _score(self, m: dict) -> tuple[float, str]:
        score = 50.0
        score += min(30, m["fresh_count_24h"] * 5)          # up to +30 for fresh articles
        score += (m["avg_credibility"] - 0.5) * 40          # credibility contribution
        if m["is_stale"]:                                    score -= 20
        if m["total_articles"] < 3:                          score -= 20
        score = max(0, min(100, score))

        if score >= 75:   label = "High Coverage"
        elif score >= 50: label = "Good Coverage"
        elif score >= 30: label = "Low Coverage"
        else:             label = "Very Low Coverage"

        return round(score, 2), label
