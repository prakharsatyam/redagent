"""
Unit tests for NewsStreamer (CryptoPanic / news feeds).
"""

import pytest
from unittest.mock import MagicMock, patch
from src.data.news_streamer import NewsStreamer, NewsArticle, KEY_VOLATILITY_TERMS


def test_news_article_model():
    article = NewsArticle(
        title="SEC Approves ETF Options",
        source="CoinDesk",
        url="https://coindesk.com/sec-etf",
        currencies=["BTC", "ETH"]
    )
    assert article.title == "SEC Approves ETF Options"
    assert "BTC" in article.currencies
    assert article.published_at is not None


def test_news_streamer_fallback():
    streamer = NewsStreamer(api_key="")
    articles = streamer.fetch_cryptopanic(currency="BTC")
    assert len(articles) > 0
    assert any("SEC" in a.title or "Federal Reserve" in a.title for a in articles)

    headlines = streamer.get_latest_headlines(currency="BTC", limit=3)
    assert len(headlines) <= 3
    assert all(isinstance(h, str) for h in headlines)


def test_news_streamer_api_mock():
    streamer = NewsStreamer(api_key="TEST_KEY_123")
    mock_payload = {
        "results": [
            {
                "title": "Bitcoin Surges Past Key Resistance Level",
                "source": {"title": "CoinTelegraph", "domain": "cointelegraph.com"},
                "url": "https://cointelegraph.com/news/btc-surge",
                "currencies": [{"code": "BTC"}],
                "votes": {"positive": 25}
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch.object(streamer.session, "get", return_value=mock_resp):
        articles = streamer.fetch_cryptopanic(currency="BTC")
        assert len(articles) == 1
        assert articles[0].title == "Bitcoin Surges Past Key Resistance Level"
        assert articles[0].is_hot is True
        assert articles[0].domain == "cointelegraph.com"


def test_filter_high_impact_headlines():
    streamer = NewsStreamer(api_key="")
    raw_headlines = [
        "Someone made a new NFT collection on Solana",
        "Breaking: Federal Reserve cuts rates by 50 basis points",
        "SEC files motion regarding crypto exchange",
        "A random developer published a blog post about python"
    ]
    impact = streamer.filter_high_impact_headlines(raw_headlines)
    assert len(impact) == 2
    assert "Federal Reserve" in impact[0]
    assert "SEC files" in impact[1]
