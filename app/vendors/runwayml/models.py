"""Pydantic models matching the RunwayML API request/response formats."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RunwayModel(str, Enum):
    GEN4_5 = "gen4.5"
    GEN4_TURBO = "gen4_turbo"
    GEN4_ALEPH = "gen4_aleph"
    ACT_TWO = "act_two"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ImageToVideoRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. gen4.5, gen4_turbo")
    promptText: Optional[str] = Field(None, max_length=512, description="Text prompt")
    promptImage: Optional[str] = Field(
        None, description="Image URL or base64 data URI"
    )
    ratio: Optional[str] = Field("1280:720", description="Output aspect ratio")
    duration: Optional[int] = Field(5, description="Duration in seconds")
    seed: Optional[int] = Field(None, description="Random seed")
    watermark: Optional[bool] = Field(False, description="Add watermark")


class TextToVideoRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. gen4.5")
    promptText: str = Field(..., max_length=512, description="Text prompt")
    ratio: Optional[str] = Field("1280:720", description="Output aspect ratio")
    duration: Optional[int] = Field(5, description="Duration in seconds")
    seed: Optional[int] = Field(None, description="Random seed")
    watermark: Optional[bool] = Field(False, description="Add watermark")


class ReferenceImage(BaseModel):
    uri: str = Field(..., description="Image URL or base64 data URI")
    tag: Optional[str] = Field(None, description="Tag for referencing in prompt")


class VideoToVideoRequest(BaseModel):
    model: str = Field("gen4_aleph", description="Model name")
    promptText: Optional[str] = Field(None, max_length=1000, description="Text prompt")
    videoUri: str = Field(..., description="Input video URL")
    referenceImages: Optional[list[ReferenceImage]] = Field(
        None, description="Up to 3 reference images"
    )
    ratio: Optional[str] = Field("1280:720", description="Output aspect ratio")
    duration: Optional[int] = Field(5, description="Duration in seconds")
    seed: Optional[int] = Field(None, description="Random seed")


class CharacterInput(BaseModel):
    type: str = Field(..., description="'image' or 'video'")
    url: str = Field(..., description="Character asset URL")


class CharacterPerformanceRequest(BaseModel):
    model: str = Field("act_two", description="Model name")
    character: CharacterInput
    referenceVideo: str = Field(..., description="Driving performance video URL")
    bodyControl: Optional[bool] = Field(True)
    expressionIntensity: Optional[int] = Field(3, ge=1, le=5)
    ratio: Optional[str] = Field("1280:720")
    duration: Optional[int] = Field(None)
    seed: Optional[int] = Field(None)


class TaskCreatedResponse(BaseModel):
    id: str = Field(..., description="Task UUID")


class TaskDetailResponse(BaseModel):
    id: str
    status: TaskStatus
    createdAt: Optional[str] = None
    output: Optional[list[str]] = None
    failure: Optional[str] = None
    failureCode: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
