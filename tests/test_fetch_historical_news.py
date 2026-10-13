"""
Unit tests for scripts/fetch_historical_news.py.
"""

from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from scripts.fetch_historical_news import fetch_historical_news, HISTORICAL_MAJOR_EVENTS


def test_fetch_historical_news_fallback(tmp_path: Path):
    out_jsonl = tmp_path / "news_fallback.jsonl"
    news = fetch_historical_news(days=30, api_key="", output_path=out_jsonl)

    assert len(news) >= len(HISTORICAL_MAJOR_EVENTS)
    assert out_jsonl.exists()

    # Verify JSON lines
    with open(out_jsonl, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "timestamp" in first_line
        assert "title" in first_line
        assert "source" in first_line


def test_fetch_historical_news_api_mock(tmp_path: Path):
    out_jsonl = tmp_path / "news_mock.jsonl"
    mock_payload = {
        "results": [
            {
                "published_at": "2026-10-10T12:00:00Z",
                "title": "Crypto Inflows Reach Record Highs in Q4",
                "source": {"title": "Bloomberg"},
                "currencies": [{"code": "BTC"}]
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        news = fetch_historical_news(days=10, api_key="TEST_KEY", output_path=out_jsonl)
        assert any(n["title"] == "Crypto Inflows Reach Record Highs in Q4" for n in news)
