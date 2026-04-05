from fastapi import APIRouter, HTTPException
from api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/api/v1", tags=["feedback"])

_chatbot = None


def set_chatbot(bot):
    global _chatbot
    _chatbot = bot


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):

    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    try:
        _chatbot.submit_feedback(
            session_id=request.session_id,
            message_id=request.message_id,
            rating=request.rating,
            comment=request.comment or "",
        )
        return FeedbackResponse(
            status="success",
            message="Feedback recorded. Thank you!",
        )
    except Exception as e:
        raise HTTPException(500, f"Feedback error: {str(e)}")


@router.get("/feedback/{session_id}")
async def get_feedback(session_id: str):
    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    feedback = _chatbot.memory.get_feedback(session_id)
    return {"session_id": session_id, "feedback": feedback}


@router.get("/feedback")
async def get_all_feedback():
    if _chatbot is None:
        raise HTTPException(503, "Chatbot not initialized")

    feedback = _chatbot.memory.get_feedback()
    return {"total": len(feedback), "data": feedback}