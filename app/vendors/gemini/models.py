"""Pydantic models mirroring the Gemini generateContent API.

Accepts both snake_case (REST wire format) and camelCase (SDK format)
thanks to ``populate_by_name=True`` + aliases. Responses serialise with
``by_alias=True`` so they go out in camelCase.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InlineData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mime_type: str = Field(alias="mimeType")
    data: str


class Part(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str | None = None
    inline_data: InlineData | None = Field(default=None, alias="inlineData")


class Content(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    parts: list[Part]
    role: str | None = None


class ImageConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    aspect_ratio: str | None = Field(default=None, alias="aspectRatio")
    image_size: str | None = Field(default=None, alias="imageSize")


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thinking_level: str | None = Field(default=None, alias="thinkingLevel")
    include_thoughts: bool | None = Field(default=None, alias="includeThoughts")


class SearchTypes(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    web_search: dict | None = Field(default=None, alias="webSearch")
    image_search: dict | None = Field(default=None, alias="imageSearch")


class GoogleSearchTool(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    search_types: SearchTypes | None = Field(default=None, alias="searchTypes")


class Tool(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    google_search: GoogleSearchTool | None = Field(
        default=None, alias="googleSearch"
    )


class SafetySetting(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    category: str | None = None
    threshold: str | None = None


class GenerationConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    response_modalities: list[str] | None = Field(
        default=None, alias="responseModalities"
    )
    image_config: ImageConfig | None = Field(default=None, alias="imageConfig")
    thinking_config: ThinkingConfig | None = Field(
        default=None, alias="thinkingConfig"
    )


class GenerateContentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    contents: list[Content]
    generation_config: GenerationConfig | None = Field(
        default=None, alias="generationConfig"
    )
    tools: list[Tool] | None = None
    safety_settings: list[SafetySetting] | None = Field(
        default=None, alias="safetySettings"
    )


class ModalityTokenCount(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    modality: str
    token_count: int = Field(alias="tokenCount")


class Candidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: Content
    finish_reason: str = Field(default="STOP", alias="finishReason")
    index: int = 0
    token_count: int | None = Field(default=None, alias="tokenCount")


class UsageMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt_token_count: int = Field(alias="promptTokenCount")
    candidates_token_count: int = Field(alias="candidatesTokenCount")
    total_token_count: int = Field(alias="totalTokenCount")
    prompt_tokens_details: list[ModalityTokenCount] | None = Field(
        default=None, alias="promptTokensDetails"
    )
    candidates_tokens_details: list[ModalityTokenCount] | None = Field(
        default=None, alias="candidatesTokensDetails"
    )


class GenerateContentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    candidates: list[Candidate]
    model_version: str | None = Field(default=None, alias="modelVersion")
    usage_metadata: UsageMetadata | None = Field(
        default=None, alias="usageMetadata"
    )
    response_id: str | None = Field(default=None, alias="responseId")


class ErrorDetail(BaseModel):
    code: int
    message: str
    status: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ── Veo (predictLongRunning) ────────────────────────────────────────


class VeoInlineDataInstance(BaseModel):
    """Wrapper for ``instances[].image`` / ``.lastFrame`` / ``.video``.

    Accepts both shapes Google uses across its two Veo APIs:

    * **Gemini Developer API**: ``{"inlineData": {"mimeType": "...", "data": "<b64>"}}``
    * **Vertex AI** (inline base64): ``{"bytesBase64Encoded": "<b64>", "mimeType": "..."}``
    * **Vertex AI** (GCS URI): ``{"gcsUri": "gs://...", "mimeType": "..."}``

    The Vertex AI form is normalised into the Gemini Developer form before
    downstream code runs, so the rest of the pipeline only deals with one shape.

    .. note::
       ``gcsUri`` cannot be forwarded to Replicate as-is (Replicate doesn't
       fetch ``gs://``). It is preserved on the model so the converter can
       reject it with a clear error rather than silently dropping data.
    """

    model_config = ConfigDict(populate_by_name=True)

    inline_data: InlineData = Field(alias="inlineData")
    gcs_uri: str | None = Field(default=None, alias="gcsUri")

    @model_validator(mode="before")
    @classmethod
    def _accept_vertex_ai_shape(cls, data):
        if not isinstance(data, dict):
            return data
        if "inlineData" in data or "inline_data" in data:
            return data
        # Vertex AI inline base64 form
        b64 = data.get("bytesBase64Encoded") or data.get("bytes_base64_encoded")
        if b64:
            mime = data.get("mimeType") or data.get("mime_type") or "image/png"
            out = {"inlineData": {"mimeType": mime, "data": b64}}
            # Preserve gcsUri if also present (rare but possible)
            if data.get("gcsUri") or data.get("gcs_uri"):
                out["gcsUri"] = data.get("gcsUri") or data.get("gcs_uri")
            return out
        # Vertex AI GCS URI form (no inline data) — provide a placeholder
        # InlineData so the model validates; converter will reject downstream.
        if data.get("gcsUri") or data.get("gcs_uri"):
            return {
                "inlineData": {"mimeType": "application/x-gcs-uri", "data": ""},
                "gcsUri": data.get("gcsUri") or data.get("gcs_uri"),
            }
        return data


class VeoReferenceImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image: VeoInlineDataInstance
    reference_type: str | None = Field(default=None, alias="referenceType")


class VeoInstance(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str | None = None
    image: VeoInlineDataInstance | None = None
    last_frame: VeoInlineDataInstance | None = Field(default=None, alias="lastFrame")
    reference_images: list[VeoReferenceImage] | None = Field(
        default=None, alias="referenceImages"
    )
    video: VeoInlineDataInstance | None = None  # for extension flows


class VeoParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    aspect_ratio: str | None = Field(default=None, alias="aspectRatio")
    # Gemini Developer API docs it as a string ("4"/"6"/"8") but Vertex AI
    # passes an integer; accept either and let the converter normalise.
    duration_seconds: str | int | None = Field(default=None, alias="durationSeconds")
    person_generation: str | None = Field(default=None, alias="personGeneration")
    resolution: str | None = None
    # Gemini Developer API uses ``numberOfVideos``; Vertex AI uses
    # ``sampleCount``. Both populate the same field via the alias machinery.
    number_of_videos: int | None = Field(default=None, alias="numberOfVideos")
    seed: int | None = None
    negative_prompt: str | None = Field(default=None, alias="negativePrompt")
    # Vertex-AI-only extras — accepted but not forwarded (Replicate has no
    # equivalent knob today). We surface them on the model so they don't get
    # silently dropped during parsing.
    enhance_prompt: bool | None = Field(default=None, alias="enhancePrompt")
    generate_audio: bool | None = Field(default=None, alias="generateAudio")
    storage_uri: str | None = Field(default=None, alias="storageUri")

    @model_validator(mode="before")
    @classmethod
    def _accept_vertex_ai_sample_count(cls, data):
        """Vertex AI's ``sampleCount`` is semantically equivalent to
        Gemini Developer API's ``numberOfVideos``. Map it before validation."""
        if not isinstance(data, dict):
            return data
        if "sampleCount" in data and "numberOfVideos" not in data and "number_of_videos" not in data:
            data = dict(data)
            data["numberOfVideos"] = data.pop("sampleCount")
        return data


class PredictLongRunningRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    instances: list[VeoInstance]
    parameters: VeoParameters | None = None


class VeoFileRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uri: str
    mime_type: str | None = Field(default=None, alias="mimeType")


class VeoGeneratedSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    video: VeoFileRef


class GenerateVideoResponseBody(BaseModel):
    """The ``generateVideoResponse`` field inside an Operation response."""

    model_config = ConfigDict(populate_by_name=True)

    generated_samples: list[VeoGeneratedSample] = Field(alias="generatedSamples")


class PredictLongRunningResponseBody(BaseModel):
    """Wrapper matching Google's ``response`` envelope for Veo."""

    model_config = ConfigDict(populate_by_name=True)

    type_url: str = Field(
        default="type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningResponse",
        alias="@type",
    )
    generate_video_response: GenerateVideoResponseBody = Field(
        alias="generateVideoResponse"
    )


class OperationError(BaseModel):
    code: int
    message: str


class Operation(BaseModel):
    """Google long-running Operation envelope."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    metadata: dict | None = None
    done: bool = False
    response: PredictLongRunningResponseBody | None = None
    error: OperationError | None = None

