"""Media processing module for multimodal support.

Document extraction (textxtract) lives in pyagenity-api, not here.
The core library provides config, media processing (images), storage,
resolution, and offloading.
"""

from .config import DocumentHandling, ImageHandling, MultimodalConfig
from .offload import MediaOffloadPolicy, ensure_media_offloaded
from .processor import MediaProcessor
from .resolver import MediaRefResolver
from .storage import BaseMediaStore, CloudMediaStore, InMemoryMediaStore, LocalFileMediaStore

__all__ = [
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
    "ensure_media_offloaded",
]
