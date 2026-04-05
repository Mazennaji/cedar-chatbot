import os
import hashlib
import hmac
import json
import time
import base64
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


SECRET_KEY = os.getenv("CEDAR_JWT_SECRET", "cedar-dev-secret-change-in-production")

PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class JWTAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, enabled: bool = False):
        super().__init__(app)
        self.enabled = enabled or os.getenv("CEDAR_AUTH_ENABLED", "").lower() == "true"

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if request.url.path.startswith("/ws/"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing or invalid Authorization header")

        token = auth_header[7:]
        payload = verify_token(token)

        if payload is None:
            raise HTTPException(401, "Invalid or expired token")

        request.state.user_id = payload.get("sub", "anonymous")
        return await call_next(request)


def create_token(user_id: str, expires_in: int = 86400) -> str:

    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).decode().rstrip("=")

    payload_data = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in,
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_data).encode()
    ).decode().rstrip("=")

    signature = hmac.new(
        SECRET_KEY.encode(),
        f"{header}.{payload}".encode(),
        hashlib.sha256,
    ).hexdigest()

    return f"{header}.{payload}.{signature}"


def verify_token(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header, payload, signature = parts

        expected = hmac.new(
            SECRET_KEY.encode(),
            f"{header}.{payload}".encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return None

        padding = 4 - len(payload) % 4
        payload_data = json.loads(
            base64.urlsafe_b64decode(payload + "=" * padding)
        )

        if payload_data.get("exp", 0) < time.time():
            return None

        return payload_data

    except Exception:
        return None