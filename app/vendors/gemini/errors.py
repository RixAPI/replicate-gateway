"""Gemini-flavoured error type and exception handler.

Routes/auth in this vendor raise :class:`GeminiAPIError` instead of a
plain ``HTTPException`` so the registered handler can format the
response as Gemini's standard ``{error: {code, message, status}}``
envelope — without leaking that format to other vendors hosted in the
same process.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.vendors.gemini.models import ErrorDetail, ErrorResponse


_STATUS_MAP = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    422: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    502: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


class GeminiAPIError(HTTPException):
    """An HTTPException rendered in Gemini's error envelope."""

    def __init__(
        self,
        status_code: int,
        detail: str = "",
        status: str | None = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.status = status or _STATUS_MAP.get(status_code, "UNKNOWN")


def gemini_error_response(code: int, message: str, status: str | None = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            status=status or _STATUS_MAP.get(code, "UNKNOWN"),
        )
    )
    return JSONResponse(status_code=code, content=body.model_dump())


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(GeminiAPIError)
    async def _handler(_request: Request, exc: GeminiAPIError):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return gemini_error_response(exc.status_code, message, exc.status)
