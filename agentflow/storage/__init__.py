"""Storage components for Agentflow.

This package provides all persistence and media-handling infrastructure:

- ``agentflow.storage.checkpointer`` — agent state persistence (in-memory, Postgres)
- ``agentflow.storage.store``        — vector/long-term memory stores (Qdrant, Mem0, ...)
- ``agentflow.storage.media``        — multimodal media processing, storage, and resolution
"""

from __future__ import annotations

from . import checkpointer, media, store

# --- Checkpointer ---
from .checkpointer import (
    BaseCheckpointer,
    InMemoryCheckpointer,
    PgCheckpointer,
)

# --- Media ---
from .media import (
    GOOGLE_INLINE_THRESHOLD,
    BaseMediaStore,
    CloudMediaStore,
    DocumentHandling,
    ImageHandling,
    InMemoryMediaStore,
    LocalFileMediaStore,
    MediaOffloadPolicy,
    MediaProcessor,
    MediaRefResolver,
    MultimodalConfig,
    ProviderMediaCache,
    create_openai_file_attachment,
    create_openai_file_search_tool,
    enforce_file_size,
    ensure_media_offloaded,
    sanitize_filename,
    should_use_google_file_api,
    validate_magic_bytes,
)

# --- Store (vector / long-term memory) ---
from .store import (
    DEFAULT_COLLECTION,
    BaseEmbedding,
    BaseStore,
    DistanceMetric,
    GoogleEmbedding,
    Mem0Store,
    MemoryIntegration,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    OpenAIEmbedding,
    QdrantStore,
    ReadMode,
    create_cloud_qdrant_store,
    create_local_qdrant_store,
    create_mem0_store,
    create_mem0_store_with_qdrant,
    create_memory_preload_node,
    create_remote_qdrant_store,
    get_memory_system_prompt,
    memory_tool,
)


__all__ = [
    # Submodules
    "checkpointer",
    "media",
    "store",
    # Checkpointer
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "PgCheckpointer",
    # Store
    "DEFAULT_COLLECTION",
    "BaseEmbedding",
    "BaseStore",
    "DistanceMetric",
    "GoogleEmbedding",
    "Mem0Store",
    "MemoryIntegration",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "OpenAIEmbedding",
    "QdrantStore",
    "ReadMode",
    "create_cloud_qdrant_store",
    "create_local_qdrant_store",
    "create_mem0_store",
    "create_mem0_store_with_qdrant",
    "create_memory_preload_node",
    "create_remote_qdrant_store",
    "get_memory_system_prompt",
    "memory_tool",
    # Media
    "GOOGLE_INLINE_THRESHOLD",
    "BaseMediaStore",
    "CloudMediaStore",
    "DocumentHandling",
    "ImageHandling",
    "InMemoryMediaStore",
    "LocalFileMediaStore",
    "MediaOffloadPolicy",
    "MediaProcessor",
    "MediaRefResolver",
    "MultimodalConfig",
    "ProviderMediaCache",
    "create_openai_file_attachment",
    "create_openai_file_search_tool",
    "enforce_file_size",
    "ensure_media_offloaded",
    "sanitize_filename",
    "should_use_google_file_api",
    "validate_magic_bytes",
]
