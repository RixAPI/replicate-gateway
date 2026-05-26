"""SQLite-backed store mapping vendor-side task IDs to Replicate prediction IDs.

Used by async vendors (e.g. RunwayML) whose API surface exposes opaque
task IDs to the caller while we hold the actual Replicate prediction id
behind the scenes.

The ``vendor`` column lets multiple vendors share one table.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import aiosqlite


class TaskStore:
    def __init__(self, db_path: str = "tasks.db"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id         TEXT PRIMARY KEY,
                prediction_id   TEXT NOT NULL,
                vendor          TEXT NOT NULL,
                model           TEXT NOT NULL,
                endpoint        TEXT NOT NULL,
                created_at      TEXT NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_id ON tasks(prediction_id)"
        )
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def create(
        self,
        prediction_id: str,
        vendor: str,
        model: str,
        endpoint: str,
    ) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO tasks (task_id, prediction_id, vendor, model, endpoint, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, prediction_id, vendor, model, endpoint, now),
        )
        await self._db.commit()
        return task_id

    async def get(self, task_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT task_id, prediction_id, vendor, model, endpoint, created_at "
            "FROM tasks WHERE task_id = ?",
            (task_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "task_id": row[0],
            "prediction_id": row[1],
            "vendor": row[2],
            "model": row[3],
            "endpoint": row[4],
            "created_at": row[5],
        }

    async def delete(self, task_id: str) -> bool:
        cursor = await self._db.execute(
            "DELETE FROM tasks WHERE task_id = ?", (task_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
