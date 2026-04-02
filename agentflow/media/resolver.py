"""Resolve ``MediaRef`` objects to provider-specific formats at LLM call time.

The resolver translates lightweight ``MediaRef`` references (which is all
that lives in messages/state) back into actual binary data or URLs that
the OpenAI / Google APIs understand.
"""

from __future__ import annotations

import base64
import logging
import time
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

    def __init__(
        self,
        media_store: BaseMediaStore | None = None,
        cache_backend: Any | None = None,
        direct_url_expiration_seconds: int = 3600,
        direct_url_refresh_buffer_seconds: int = 60,
    ) -> None:
        self.media_store = media_store
        self.cache_backend = cache_backend
        self.direct_url_expiration_seconds = direct_url_expiration_seconds
        self.direct_url_refresh_buffer_seconds = direct_url_refresh_buffer_seconds

    def with_cache(
        self,
        cache_backend: Any,
        expiration_seconds: int = 3600,
        refresh_buffer_seconds: int = 60,
    ) -> MediaRefResolver:
        """Attach a shared cache backend for signed URLs."""
        self.cache_backend = cache_backend
        self.direct_url_expiration_seconds = expiration_seconds
        self.direct_url_refresh_buffer_seconds = refresh_buffer_seconds
        return self

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    async def resolve_for_openai(self, ref: MediaRef) -> dict[str, Any]:
        """Convert a ``MediaRef`` to an OpenAI-style content part dict.

        Returns:
            A dict like ``{"type": "image_url", "image_url": {"url": ...}}``.
        """
        if ref.kind == "url" and ref.url and ref.url.startswith(_AGENTFLOW_SCHEME):
            direct_url = await self._get_direct_url(ref)
            if direct_url:
                return _openai_image_url(direct_url)

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
            direct_url = await self._get_direct_url(ref)
            if direct_url:
                mime = ref.mime_type or "application/octet-stream"
                return types.Part.from_uri(file_uri=direct_url, mime_type=mime)

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

    async def _get_direct_url(self, ref: MediaRef) -> str | None:
        """Return a direct URL for internal media when the store supports it."""
        if self.media_store is None or not ref.url:
            return None

        key = ref.url.removeprefix(_AGENTFLOW_SCHEME)
        mime_type = ref.mime_type or "application/octet-stream"
        cache_key = f"{key}:{mime_type}:{self.direct_url_expiration_seconds}"
        cached_url = await self._get_cached_signed_url(cache_key)
        if cached_url:
            return cached_url

        direct_url_kwargs: dict[str, Any] = {"mime_type": ref.mime_type}
        # Only force the expiration when the resolver is coordinating a shared
        # signed-URL cache. Otherwise the store's own default lifetime is fine.
        if self.cache_backend is not None:
            direct_url_kwargs["expiration"] = self.direct_url_expiration_seconds

        direct_url = await self.media_store.get_direct_url(key, **direct_url_kwargs)
        if direct_url:
            await self._cache_signed_url(cache_key, direct_url)
        return direct_url

    async def _get_cached_signed_url(self, cache_key: str) -> str | None:
        if self.cache_backend is None:
            return None

        payload = await self.cache_backend.aget_cache_value("media:signed-url", cache_key)
        if not isinstance(payload, dict):
            return None

        url = payload.get("url")
        expires_at = payload.get("expires_at")
        if not isinstance(url, str) or not isinstance(expires_at, int | float):
            return None

        if expires_at <= time.time() + self.direct_url_refresh_buffer_seconds:
            return None
        return url

    async def _cache_signed_url(self, cache_key: str, url: str) -> None:
        if self.cache_backend is None:
            return

        expires_at = int(time.time() + self.direct_url_expiration_seconds)
        await self.cache_backend.aput_cache_value(
            "media:signed-url",
            cache_key,
            {
                "url": url,
                "expires_at": expires_at,
            },
            ttl_seconds=self.direct_url_expiration_seconds,
        )


def _openai_image_url(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}
