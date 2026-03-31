"""PostgreSQL BYTEA-backed media store.

Stores binary data in a **separate** ``media_blobs`` table — never inside
the ``states`` or ``messages`` JSONB columns.

Requires ``asyncpg`` (already a dependency of the ``pg_checkpoint`` extra).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from .base import BaseMediaStore


logger = logging.getLogger("agentflow.media.storage.pg")


class PgBlobStore(BaseMediaStore):
    """Store binary media in a dedicated PostgreSQL ``media_blobs`` table.

    The table schema is::

        CREATE TABLE IF NOT EXISTS media_blobs (
            storage_key VARCHAR(255) PRIMARY KEY,
            data        BYTEA       NOT NULL,
            mime_type   VARCHAR(100) NOT NULL,
            size_bytes  BIGINT,
            thread_id   VARCHAR(255),
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            metadata    JSONB       DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_media_blobs_thread
            ON media_blobs(thread_id);

    Call :meth:`initialize` once to create the table.
    """

    def __init__(self, pool: Any, thread_id: str | None = None) -> None:
        """
        Args:
            pool: An ``asyncpg.Pool`` instance.
            thread_id: Optional thread-level scope for cleanup queries.
        """
        self._pool = pool
        self._thread_id = thread_id

    async def initialize(self) -> None:
        """Create the ``media_blobs`` table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS media_blobs (
                    storage_key VARCHAR(255) PRIMARY KEY,
                    data        BYTEA        NOT NULL,
                    mime_type   VARCHAR(100)  NOT NULL,
                    size_bytes  BIGINT,
                    thread_id   VARCHAR(255),
                    created_at  TIMESTAMPTZ   DEFAULT NOW(),
                    metadata    JSONB         DEFAULT '{}'
                );
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_media_blobs_thread
                    ON media_blobs(thread_id);
            """)

    async def store(
        self,
        data: bytes,
        mime_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        key = uuid4().hex
        import json

        meta_json = json.dumps(metadata or {})

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO media_blobs (storage_key, data, mime_type, size_bytes, thread_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                key,
                data,
                mime_type,
                len(data),
                self._thread_id,
                meta_json,
            )
        logger.debug("Stored %d bytes as %s in PG media_blobs", len(data), key)
        return key

    async def retrieve(self, storage_key: str) -> tuple[bytes, str]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data, mime_type FROM media_blobs WHERE storage_key = $1",
                storage_key,
            )
        if row is None:
            raise KeyError(f"Media not found: {storage_key}")
        return bytes(row["data"]), row["mime_type"]

    async def delete(self, storage_key: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM media_blobs WHERE storage_key = $1",
                storage_key,
            )
        return result == "DELETE 1"

    async def exists(self, storage_key: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT 1 FROM media_blobs WHERE storage_key = $1",
                storage_key,
            )
        return row is not None

    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all media associated with a thread.  Returns count deleted."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM media_blobs WHERE thread_id = $1",
                thread_id,
            )
        # result is like "DELETE 5"
        count_str = result.split()[-1] if result else "0"
        return int(count_str)
