"""Vendor registration interface.

A vendor module exposes a ``register(app)`` function that mounts its
routes onto the FastAPI app. ``app/main.py`` calls each registered
vendor in turn so adding a new one is a one-line change.
"""

from __future__ import annotations

from typing import Callable, Protocol

from fastapi import FastAPI


class VendorRegister(Protocol):
    def __call__(self, app: FastAPI) -> None: ...


def register_vendors(app: FastAPI, vendors: list[Callable[[FastAPI], None]]) -> None:
    for register in vendors:
        register(app)
