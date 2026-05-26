# Replicate Gateway

A multi-vendor FastAPI proxy that accepts requests in each vendor's **native API
format** and forwards them to the corresponding model on **Replicate**, returning
responses in the vendor's native format.

Adding a new vendor is one folder + one line in [app/main.py](app/main.py).

## Supported vendors

| Vendor | Native paths | Models |
|---|---|---|
| **RunwayML** | `POST /runwayml/v1/image_to_video` · `POST /runwayml/v1/text_to_video` · `POST /runwayml/v1/video_to_video` · `GET/DELETE /runwayml/v1/tasks/{id}` | `gen4.5`, `gen4_turbo`, `gen4_aleph` |
| **Gemini (image)** | `POST /gemini/v1beta/models/{model}:generateContent` | `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview`, `gemini-2.5-flash-image` |
| **Gemini (Veo video)** | `POST /gemini/v1beta/models/{model}:predictLongRunning` · `GET /gemini/v1beta/models/{model}/operations/{id}` | `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `veo-3.1-lite-generate-preview`, `veo-3.0-generate-001`, `veo-3.0-fast-generate-001`, `veo-2.0-generate-001` |
| **BFL (Black Forest Labs)** | `POST /bfl/v1/{model}` · `GET /bfl/v1/get_result?id=...` | `flux-2-max`, `flux-2-pro`, `flux-2-flex`, `flux-2-klein-9b`, `flux-2-klein-4b`, plus legacy `flux-pro-1.1`, `flux-pro-1.1-ultra`, `flux-pro-1.0-fill` / `expand` / `canny` / `depth` |

Each vendor follows its own native conventions:

- **RunwayML** is async (long-running video tasks). Returns an opaque task id; the caller polls `/runwayml/v1/tasks/{id}`. Stores task→prediction mapping in SQLite.
- **Gemini image generation** (`:generateContent`) is sync. Waits for the prediction in-process and returns the image inlined as base64.
- **Gemini Veo video generation** (`:predictLongRunning`) follows Google's official long-running operation pattern. POST returns `{name, done: false}`; the caller polls `GET /gemini/v1beta/{name}` until `done: true`, then reads `response.generateVideoResponse.generatedSamples[0].video.uri`.
- **BFL** is async. POST returns `{id, [cost, input_mp, output_mp]}` immediately; the caller polls `GET /bfl/v1/get_result?id=...` which returns `{status: Pending|Ready|Error, [result]}` in the official BFL shape. Auth uses `x-key: <token>` (BFL convention) or `Authorization: Bearer <token>`. Base64 `input_image*` fields are auto-uploaded to Cloudflare R2 (or any S3-compatible bucket) — configure `STORAGE_*` env vars to enable.

## Quick Start

```bash
cd replicate-gateway

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env if you want to lock the proxy to a single API_SECRET.

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Authentication

Every request must include a Bearer token (or `x-goog-api-key` for Gemini SDKs). That token is your **Replicate API token** — it's forwarded to Replicate as-is so each end-user authenticates with their own account.

If you set `API_SECRET` in `.env`, the proxy additionally requires the token to equal that value before forwarding — useful for locking down the proxy to a single known key.

## Project structure

```
app/
  core/                  # Shared infrastructure
    config.py            # Unified Settings (all vendors share env keys)
    replicate_client.py  # One Replicate client (async + sync patterns)
    task_store.py        # SQLite task store (async vendors)
    auth.py              # API_SECRET enforcement
    registry.py          # Vendor registration helper
  vendors/
    runwayml/            # RunwayML vendor module
      models.py
      param_converter.py
      auth.py
      routes.py
    gemini/              # Gemini vendor module
      models.py
      param_converter.py
      auth.py
      errors.py          # Custom error envelope, scoped via GeminiAPIError
      routes.py
  main.py                # App factory; lists VENDORS to register
```

## Adding a new vendor

1. Create `app/vendors/<name>/`.
2. Implement four modules:
   - `models.py` — Pydantic models matching the vendor's request/response shape.
   - `param_converter.py` — Convert vendor format ↔ Replicate format.
   - `auth.py` — Extract the token from whichever headers the vendor SDK sends.
   - `routes.py` — APIRouter + `register(app)` entry point.
3. If the vendor wants a custom error envelope, add `errors.py` with its own `HTTPException` subclass and register a handler in `register()`.
4. Add `register_<name>` to the `VENDORS` tuple in [app/main.py](app/main.py).

Shared building blocks live in [app/core/](app/core/) — vendors compose them rather than reimplementing them.

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Configuration

| Env var | Description | Default |
|---|---|---|
| `API_SECRET` | Lock proxy to a single known key (optional) | — |
| `HOST` / `PORT` | Server bind | `0.0.0.0` / `8000` |
| `DB_PATH` | SQLite file for async task store | `tasks.db` |
| `PREDICT_TIMEOUT` | Max seconds for sync predictions | `300` |
| `POLL_INTERVAL` | Seconds between status polls | `2` |
| `MAX_CONCURRENT` | In-flight predictions per worker | `2000` |
| `API_POOL_SIZE` / `CDN_POOL_SIZE` | httpx connection pool sizes | `500` / `200` |
