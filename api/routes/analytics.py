from fastapi import APIRouter, HTTPException
from api.schemas import StatsResponse

router = APIRouter(prefix="/api/v1", tags=["analytics"])

_chatbot = None


def set_chatbot(bot):
    global _chatbot
    _chatbot = bot


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")
    return StatsResponse(**_chatbot.get_stats())


@router.get("/analytics/languages")
async def language_distribution():
    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    lang_counts = {}
    for sid in list(_chatbot.memory._sessions.keys()):
        for msg in _chatbot.memory._sessions[sid].messages:
            if msg.role == "user":
                lang = msg.metadata.get("language", "unknown")
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return {"distribution": lang_counts}


@router.get("/analytics/intents")
async def intent_distribution():
    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    intent_counts = {}
    for sid in list(_chatbot.memory._sessions.keys()):
        for msg in _chatbot.memory._sessions[sid].messages:
            if msg.role == "user":
                intent = msg.metadata.get("intent", "unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

    return {"distribution": intent_counts}


@router.get("/analytics/sentiment")
async def sentiment_distribution():

    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    sentiment_counts = {}
    for sid in list(_chatbot.memory._sessions.keys()):
        for msg in _chatbot.memory._sessions[sid].messages:
            if msg.role == "user":
                sent = msg.metadata.get("sentiment", "unknown")
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1

    return {"distribution": sentiment_counts}