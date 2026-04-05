from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from api.schemas import ChatRequest, ChatResponse, ChatMetadata, SentimentInfo, HistoryResponse, HistoryMessage
import json

router = APIRouter(prefix="/api/v1", tags=["chat"])

_chatbot = None


def set_chatbot(bot):
    global _chatbot
    _chatbot = bot


def get_chatbot():
    if _chatbot is None:
        raise HTTPException(503, "Chatbot engine not initialized")
    return _chatbot


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    bot = get_chatbot()

    try:
        result = bot.chat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )

        meta = result.metadata
        return ChatResponse(
            response=result.response,
            session_id=result.session_id,
            message_id=result.message_id,
            metadata=ChatMetadata(
                detected_language=meta["detected_language"],
                language_confidence=meta["language_confidence"],
                normalized_text=meta.get("normalized_text"),
                intent=meta["intent"],
                intent_confidence=meta["intent_confidence"],
                sentiment=SentimentInfo(**meta["sentiment"]),
                model=meta["model"],
                response_time_ms=meta["response_time_ms"],
            ),
        )
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    bot = get_chatbot()
    history = bot.get_history(session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in history],
        turn_count=len(history),
    )


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):

    await websocket.accept()
    bot = get_chatbot()

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            result = bot.chat(
                message=payload.get("message", ""),
                session_id=payload.get("session_id"),
            )

            await websocket.send_json({
                "response": result.response,
                "session_id": result.session_id,
                "message_id": result.message_id,
                "metadata": result.metadata,
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))