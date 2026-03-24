from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, content_security_policy: str, permissions_policy: str) -> None:
        super().__init__(app)
        self._headers = {
            "Content-Security-Policy": content_security_policy,
            "Permissions-Policy": permissions_policy,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for key, value in self._headers.items():
            response.headers.setdefault(key, value)
        return response


class MaxRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_request_bytes: int) -> None:
        super().__init__(app)
        self._max_request_bytes = max_request_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    length = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."},
                    )
                if length > self._max_request_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": (
                                f"Request body too large. Max allowed is {self._max_request_bytes} bytes."
                            )
                        },
                    )
        return await call_next(request)
