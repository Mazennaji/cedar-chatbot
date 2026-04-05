import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter(BaseHTTPMiddleware):

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict = defaultdict(lambda: {"count": 0, "reset": time.time()})

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/docs", "/openapi.json", "/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        client = self._clients[client_ip]

        if now - client["reset"] > self.window_seconds:
            client["count"] = 0
            client["reset"] = now

        client["count"] += 1

        if client["count"] > self.max_requests:
            raise HTTPException(
                429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": int(client["reset"] + self.window_seconds - now),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - client["count"])
        )
        return response