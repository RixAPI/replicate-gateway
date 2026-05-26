"""BFL API request/response models.

Most BFL endpoints take a free-form ``input`` payload that varies per
model, so we keep request parsing intentionally loose (``dict``) and
let the param converter enforce model-specific rules.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class BFLErrorResponse(BaseModel):
    """BFL keeps errors small — caller projects expect this shape."""

    model_config = ConfigDict(populate_by_name=True)

    error: str
    details: Optional[str] = None


class BFLTaskCreated(BaseModel):
    """Response from POST /v1/{model}: ``{id, [cost, input_mp, output_mp]}``."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    cost: Optional[float] = None
    input_mp: Optional[float] = None
    output_mp: Optional[float] = None


class BFLTaskResultPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: Optional[str | dict | list] = None
    sample: Optional[str] = None
    duration: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class BFLTaskResult(BaseModel):
    """Response from GET /v1/get_result?id=...

    ``status`` is BFL-flavoured: ``Pending`` / ``Ready`` / ``Error``.
    ``result`` is only present when status == ``Ready``.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    status: str
    result: Optional[BFLTaskResultPayload] = None
