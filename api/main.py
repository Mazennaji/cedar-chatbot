import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, feedback, analytics
from api.middleware.rate_limiter import RateLimiter
from api.middleware.auth import JWTAuthMiddleware, create_token
from api.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("cedar")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌲 Cedar Chatbot starting...")

    from src.chatbot import CedarChatbot

    device = os.getenv("CEDAR_DEVICE", "cpu")
    model = os.getenv("CEDAR_MODEL", "facebook/blenderbot-400M-distill")

    bot = CedarChatbot(model_name=model, device=device)

    chat.set_chatbot(bot)
    feedback.set_chatbot(bot)
    analytics.set_chatbot(bot)

    logger.info(f"🌲 Cedar Chatbot ready — model: {model}, device: {device}")
    yield
    logger.info("🌲 Cedar Chatbot shutting down")


app = FastAPI(
    title="🌲 Cedar Chatbot API",
    description=(
        "A trilingual (English, Arabic, Lebanese dialect) "
        "context-aware chatbot with RLHF, built with "
        "Hugging Face Transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CEDAR_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    RateLimiter,
    max_requests=int(os.getenv("CEDAR_RATE_LIMIT", "60")),
    window_seconds=60,
)

app.add_middleware(JWTAuthMiddleware)



app.include_router(chat.router)
app.include_router(feedback.router)
app.include_router(analytics.router)


@app.get("/", tags=["root"])
async def root():

    return {
        "name": "🌲 Cedar Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/v1/chat",
            "feedback": "/api/v1/feedback",
            "stats": "/api/v1/stats",
            "websocket": "/api/v1/ws/chat",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=os.getenv("CEDAR_MODEL", "facebook/blenderbot-400M-distill"),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/api/v1/auth/token", tags=["auth"])
async def generate_token(user_id: str = "anonymous"):

    token = create_token(user_id)
    return {"token": token, "type": "Bearer", "expires_in": 86400}