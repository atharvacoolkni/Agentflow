"""Resolve ``MediaRef`` objects to provider-specific formats at LLM call time.

The resolver translates lightweight ``MediaRef`` references (which is all
that lives in messages/state) back into actual binary data or URLs that
the OpenAI / Google APIs understand.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from agentflow.state.message_block import MediaRef

from .storage.base import BaseMediaStore

logger = logging.getLogger("agentflow.media.resolver")

_AGENTFLOW_SCHEME = "agentflow://media/"


class MediaRefResolver:
    """Resolve ``MediaRef`` objects to provider-specific content parts.

    Args:
        media_store: Optional media store for resolving internal
            ``agentflow://media/{key}`` references.  If ``None``,
            internal references will raise.
    """

    def __init__(self, media_store: BaseMediaStore | None = None) -> None:
        self.media_store = media_store

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    async def resolve_for_openai(self, ref: MediaRef) -> dict[str, Any]:
        """Convert a ``MediaRef`` to an OpenAI-style content part dict.

        Returns:
            A dict like ``{"type": "image_url", "image_url": {"url": ...}}``.
        """
        if ref.kind == "url" and ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            data, mime = await self._retrieve(ref.url)
            b64 = base64.b64encode(data).decode()
            return _openai_image_url(f"data:{mime};base64,{b64}")

        if ref.kind == "url" and ref.url:
            return _openai_image_url(ref.url)

        if ref.kind == "data" and ref.data_base64:
            mime = ref.mime_type or "application/octet-stream"
            return _openai_image_url(f"data:{mime};base64,{ref.data_base64}")

        if ref.kind == "file_id":
            url = ref.url or ref.file_id or ""
            return _openai_image_url(url)

        return _openai_image_url("")

    # ------------------------------------------------------------------
    # Google GenAI
    # ------------------------------------------------------------------

    async def resolve_for_google(self, ref: MediaRef) -> Any:
        """Convert a ``MediaRef`` to a ``google.genai.types.Part``.

        Requires ``google-genai`` to be installed.
        """
        from google.genai import types

        if ref.kind == "url" and ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            data, mime = await self._retrieve(ref.url)
            return types.Part.from_bytes(data=data, mime_type=mime)

        if ref.kind == "url" and ref.url:
            mime = ref.mime_type or "image/jpeg"
            return types.Part.from_uri(file_uri=ref.url, mime_type=mime)

        if ref.kind == "data" and ref.data_base64:
            data = base64.b64decode(ref.data_base64)
            mime = ref.mime_type or "application/octet-stream"
            return types.Part.from_bytes(data=data, mime_type=mime)

        if ref.kind == "file_id":
            mime = ref.mime_type or "application/octet-stream"
            return types.Part(
                file_data=types.FileData(
                    file_uri=ref.file_id or "",
                    mime_type=mime,
                )
            )

        return types.Part(text="[Unresolvable media reference]")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _retrieve(self, agentflow_url: str) -> tuple[bytes, str]:
        """Retrieve bytes from the media store for an internal URL."""
        if self.media_store is None:
            raise RuntimeError(
                f"Cannot resolve internal media URL {agentflow_url!r} — "
                "no MediaStore configured. Pass a media_store to the resolver."
            )
        key = agentflow_url.removeprefix(_AGENTFLOW_SCHEME)
        return await self.media_store.retrieve(key)


def _openai_image_url(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}
