"""FastAPI app factory.

Vendors are registered as a list of ``register(app)`` callables. Adding
a new one is a one-line change: drop a module under ``app/vendors/`` and
add it to the ``VENDORS`` tuple below.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.registry import register_vendors
from app.core.replicate_client import ReplicateClient
from app.core.task_store import TaskStore
from app.vendors.bfl.routes import register as register_bfl
from app.vendors.gemini.routes import register as register_gemini
from app.vendors.runwayml.routes import register as register_runwayml

logging.basicConfig(level=logging.INFO)

VENDORS = (
    register_runwayml,
    register_gemini,
    register_bfl,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = TaskStore(settings.db_path)
    await store.init()
    app.state.task_store = store
    app.state.replicate_client = ReplicateClient()
    yield
    await app.state.replicate_client.close()
    await store.close()


app = FastAPI(
    title="Replicate Gateway",
    description=(
        "Multi-vendor proxy that accepts requests in each vendor's native "
        "API format and forwards them to the corresponding model on Replicate."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

register_vendors(app, list(VENDORS))


@app.get("/health")
async def health():
    return {"status": "ok"}
