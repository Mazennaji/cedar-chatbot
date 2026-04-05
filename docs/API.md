# 📡 Cedar Chatbot — API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Authentication is optional and disabled by default. Enable with `CEDAR_AUTH_ENABLED=true`.

### Get Token

```
POST /api/v1/auth/token?user_id=your_id
```

Use in subsequent requests:
```
Authorization: Bearer <token>
```

---

## Endpoints

### `POST /api/v1/chat`

Send a message and receive a chatbot response.

**Request Body:**
```json
{
  "message": "keefak ya zalame?",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "response": "أهلاً! أنا منيح، كيفك إنت؟",
  "session_id": "auto-generated-or-provided",
  "message_id": "uuid",
  "metadata": {
    "detected_language": "lebanese_arabizi",
    "language_confidence": 0.92,
    "normalized_text": "كيفك يا زلمة؟",
    "intent": "greeting",
    "intent_confidence": 0.85,
    "sentiment": {
      "label": "positive",
      "score": 0.87
    },
    "model": "facebook/blenderbot-400M-distill",
    "response_time_ms": 142.3
  }
}
```

### `GET /api/v1/history/{session_id}`

Retrieve conversation history.

### `POST /api/v1/feedback`

Submit RLHF feedback.

**Request Body:**
```json
{
  "session_id": "session-id",
  "message_id": "msg-id",
  "rating": 1,
  "comment": "Great response!"
}
```

### `GET /api/v1/stats`

Get engine statistics.

### `GET /api/v1/analytics/languages`

Language usage distribution.

### `GET /api/v1/analytics/intents`

Intent classification distribution.

### `GET /api/v1/analytics/sentiment`

Sentiment analysis distribution.

### `WebSocket /api/v1/ws/chat`

Real-time chat via WebSocket.

**Send:**
```json
{"message": "Hello!", "session_id": "optional"}
```

**Receive:**
```json
{"response": "...", "session_id": "...", "metadata": {...}}
```

### `GET /health`

Health check.

```json
{"status": "healthy", "version": "1.0.0", "model": "...", "uptime_seconds": 123.4}
```