"""
Stage-1 Laya News and Sentiment Classifier Runtime.
Evaluates breaking news, tweets, and liquidation cascades in sub-15ms.
Acts as the early-warning protective radar and circuit-breaker against macro dumps.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from config.settings import settings

# Question schemas for Stage-1 News Evaluation
STAGE1_NEWS_QUESTIONS: Dict[str, Any] = {
    "is_market_moving": {
        "type": "noul",
        "instructions": (
            "Does this breaking crypto headline, tweet, or liquidation cascade "
            "represent a significant market-moving catalyst for BTC/ETH?"
        )
    },
    "impact_regime": {
        "type": "choice",
        "instructions": "What is the primary directional and volatility impact of this news event?",
        "options": [
            "bullish_vol_expansion",
            "bearish_dump",
            "chop_iv_crush",
            "neutral_noise"
        ]
    }
}


class NewsSentimentResult(BaseModel):
    """Structured evaluation output from Stage-1 Laya News Classifier."""
    headline: str
    is_market_moving: bool
    impact_regime: str
    confidence: float
    circuit_breaker_triggered: bool = Field(
        description="True if bearish dump catalyst detected -> HALT ALL CALL BUYING"
    )


class LayaNewsEngine:
    """In-process Stage-1 News Classifier using convaiinnovations/laya."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or (settings.models_dir / "laya_news_agent")
        self._model = None
        self._load_model()

    def _load_model(self):
        """Attempts to load trained Laya checkpoint if present locally."""
        if self.model_dir.exists() and (self.model_dir / "config.json").exists():
            try:
                from laya import load
                self._model = load(str(self.model_dir))
            except Exception:
                self._model = None
        else:
            self._model = None

    def evaluate_headline(self, headline: str) -> NewsSentimentResult:
        """
        Evaluates a headline or tweet text in sub-15ms.
        """
        # If Laya model is loaded in-process, run prediction
        if self._model is not None:
            try:
                res = self._model.predict(headline, STAGE1_NEWS_QUESTIONS)
                answers = res.get("answers", {})

                is_moving = answers.get("is_market_moving", {}).get("choice", "no") == "yes"
                regime_data = answers.get("impact_regime", {})
                impact_regime = regime_data.get("choice", "neutral_noise")
                confidence = float(regime_data.get("confidence", 0.50))

                halt = (impact_regime == "bearish_dump" and confidence >= 0.70)

                return NewsSentimentResult(
                    headline=headline,
                    is_market_moving=is_moving,
                    impact_regime=impact_regime,
                    confidence=confidence,
                    circuit_breaker_triggered=halt
                )
            except Exception:
                pass

        # Intelligent Heuristic Fallback (active when weights are not yet downloaded)
        headline_lower = headline.lower()

        # Bearish indicators
        bearish_keywords = ["exploit", "hack", "wells notice", "sec charges", "insolvent", "liquidation waterfall", "outflows hit", "dump", "crater"]
        # Bullish indicators
        bullish_keywords = ["sec approves", "etf approved", "inflows hit", "short liquidation", "rate cut", "breakout", "accumulating", "treasury"]

        is_bearish = any(k in headline_lower for k in bearish_keywords)
        is_bullish = any(k in headline_lower for k in bullish_keywords)

        if is_bearish:
            impact_regime = "bearish_dump"
            confidence = 0.85
            is_moving = True
            halt = True
        elif is_bullish:
            impact_regime = "bullish_vol_expansion"
            confidence = 0.85
            is_moving = True
            halt = False
        else:
            impact_regime = "neutral_noise"
            confidence = 0.60
            is_moving = False
            halt = False

        return NewsSentimentResult(
            headline=headline,
            is_market_moving=is_moving,
            impact_regime=impact_regime,
            confidence=confidence,
            circuit_breaker_triggered=halt
        )
