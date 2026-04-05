from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn")
    user_id: Optional[str] = Field(None, description="Optional user identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "keefak ya zalame?",
                "session_id": "user-123",
            }
        }


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    message_id: str = Field(..., description="Message to rate")
    rating: int = Field(..., ge=-1, le=1, description="-1 (bad), 0 (neutral), 1 (good)")
    preferred_response: Optional[str] = Field(None, description="User's preferred response")
    comment: Optional[str] = Field(None, max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123",
                "message_id": "msg-456",
                "rating": 1,
            }
        }

class SentimentInfo(BaseModel):
    label: str
    score: float


class ChatMetadata(BaseModel):
    detected_language: str
    language_confidence: float
    normalized_text: Optional[str] = None
    intent: str
    intent_confidence: float
    sentiment: SentimentInfo
    model: str
    response_time_ms: float


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    metadata: ChatMetadata


class FeedbackResponse(BaseModel):
    status: str
    message: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: float
    message_id: str
    metadata: dict


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
    turn_count: int


class StatsResponse(BaseModel):
    model: str
    device: str
    active_sessions: int
    total_messages: int
    total_feedback: int
    avg_turns_per_session: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    uptime_seconds: float