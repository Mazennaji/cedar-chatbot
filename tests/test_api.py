import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_chatbot():
    from src.chatbot import ChatResponse as BotResponse

    bot = MagicMock()
    bot.chat.return_value = BotResponse(
        response="Hello! How can I help?",
        session_id="test-session",
        message_id="msg-001",
        metadata={
            "detected_language": "english",
            "language_confidence": 0.95,
            "normalized_text": None,
            "intent": "greeting",
            "intent_confidence": 0.9,
            "sentiment": {"label": "positive", "score": 0.8},
            "model": "facebook/blenderbot-400M-distill",
            "response_time_ms": 120.5,
        },
    )
    bot.get_history.return_value = []
    bot.get_stats.return_value = {
        "model": "facebook/blenderbot-400M-distill",
        "device": "cpu",
        "active_sessions": 1,
        "total_messages": 2,
        "total_feedback": 0,
        "avg_turns_per_session": 2.0,
    }
    bot.memory = MagicMock()
    bot.memory.get_feedback.return_value = []
    bot.memory._sessions = {}
    return bot


@pytest.fixture
def client(mock_chatbot):
    from api.main import app
    from api.routes import chat, feedback, analytics

    chat.set_chatbot(mock_chatbot)
    feedback.set_chatbot(mock_chatbot)
    analytics.set_chatbot(mock_chatbot)

    with TestClient(app) as c:
        yield c


class TestRootEndpoints:

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Cedar" in data["name"]

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestChatEndpoint:

    def test_chat_success(self, client):
        response = client.post("/api/v1/chat", json={
            "message": "Hello!",
            "session_id": "test-123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "metadata" in data

    def test_chat_with_arabizi(self, client):
        response = client.post("/api/v1/chat", json={
            "message": "keefak ya zalame?",
        })
        assert response.status_code == 200

    def test_chat_empty_message(self, client):
        response = client.post("/api/v1/chat", json={
            "message": "",
        })
        assert response.status_code == 422

    def test_chat_auto_session(self, client):
        response = client.post("/api/v1/chat", json={
            "message": "Hello!",
        })
        assert response.status_code == 200
        assert "session_id" in response.json()


class TestFeedbackEndpoint:

    def test_submit_feedback(self, client):
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-123",
            "message_id": "msg-456",
            "rating": 1,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_feedback_with_comment(self, client):
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-123",
            "message_id": "msg-456",
            "rating": -1,
            "comment": "Response was not helpful",
        })
        assert response.status_code == 200

    def test_feedback_invalid_rating(self, client):
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-123",
            "message_id": "msg-456",
            "rating": 5,
        })
        assert response.status_code == 422

    def test_get_feedback(self, client):
        response = client.get("/api/v1/feedback/test-123")
        assert response.status_code == 200

    def test_get_all_feedback(self, client):
        response = client.get("/api/v1/feedback")
        assert response.status_code == 200


class TestAnalyticsEndpoint:

    def test_get_stats(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "active_sessions" in data

    def test_language_distribution(self, client):
        response = client.get("/api/v1/analytics/languages")
        assert response.status_code == 200

    def test_intent_distribution(self, client):
        response = client.get("/api/v1/analytics/intents")
        assert response.status_code == 200

    def test_sentiment_distribution(self, client):
        response = client.get("/api/v1/analytics/sentiment")
        assert response.status_code == 200


class TestAuthEndpoint:

    def test_generate_token(self, client):
        response = client.post("/api/v1/auth/token?user_id=testuser")
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert data["type"] == "Bearer"