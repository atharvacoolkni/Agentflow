"""Image validation and processing utilities.

``MediaProcessor`` handles only **images**: validate MIME type, enforce
file-size limits, and optionally resize.  It does NOT perform document
text extraction — that is the API layer's responsibility (see
``agentflow_cli.media.pipeline``).
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from agentflow.media.config import MultimodalConfig
from agentflow.state.message_block import ImageBlock, MediaRef

logger = logging.getLogger("agentflow.media.processor")


class MediaProcessor:
    """Validate and optionally resize images before they enter the pipeline.

    Pillow is an *optional* dependency (``pip install 10xscale-agentflow[images]``).
    Without Pillow, validation still works but resizing is skipped.
    """

    def __init__(self, config: MultimodalConfig | None = None):
        self.config = config or MultimodalConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_image(self, block: ImageBlock) -> None:
        """Validate an ImageBlock against the configured limits.

        Raises:
            ValueError: If MIME type is unsupported or size exceeds limit.
        """
        media = block.media
        mime = media.mime_type or "image/png"

        if mime not in self.config.supported_image_types:
            raise ValueError(
                f"Unsupported image type '{mime}'. "
                f"Allowed: {', '.join(sorted(self.config.supported_image_types))}"
            )

        if media.size_bytes is not None:
            max_bytes = int(self.config.max_image_size_mb * 1_048_576)
            if media.size_bytes > max_bytes:
                raise ValueError(
                    f"Image too large ({media.size_bytes} bytes). "
                    f"Max allowed: {max_bytes} bytes ({self.config.max_image_size_mb} MB)"
                )

        # Check decoded size for inline base64
        if media.kind == "data" and media.data_base64:
            approx_bytes = len(media.data_base64) * 3 // 4
            max_bytes = int(self.config.max_image_size_mb * 1_048_576)
            if approx_bytes > max_bytes:
                raise ValueError(
                    f"Image data too large (~{approx_bytes} bytes). "
                    f"Max allowed: {max_bytes} bytes ({self.config.max_image_size_mb} MB)"
                )

    def resize_image(self, block: ImageBlock) -> ImageBlock:
        """Resize an inline base64 image if it exceeds ``max_image_dimension``.

        Returns the original block unchanged if:
        - Pillow is not installed
        - The image is a URL/file_id reference (cannot resize remotely)
        - Both dimensions are within limits

        Returns:
            A new ImageBlock with resized data, or the original.
        """
        if block.media.kind != "data" or not block.media.data_base64:
            return block  # Can only resize inline data

        try:
            from PIL import Image
        except ImportError:
            logger.debug("Pillow not installed; skipping image resize")
            return block

        raw = base64.b64decode(block.media.data_base64)
        img = Image.open(io.BytesIO(raw))
        max_dim = self.config.max_image_dimension

        if img.width <= max_dim and img.height <= max_dim:
            return block  # Already within limits

        # Resize maintaining aspect ratio
        img.thumbnail((max_dim, max_dim))

        buf = io.BytesIO()
        fmt = _pil_format(block.media.mime_type)
        img.save(buf, format=fmt)
        new_b64 = base64.b64encode(buf.getvalue()).decode()

        new_media = block.media.model_copy(
            update={
                "data_base64": new_b64,
                "width": img.width,
                "height": img.height,
                "size_bytes": len(buf.getvalue()),
            }
        )
        return ImageBlock(media=new_media, alt_text=block.alt_text, bbox=block.bbox)

    def process(self, block: ImageBlock) -> ImageBlock:
        """Validate and optionally resize an image block.

        Convenience method that calls ``validate_image`` then ``resize_image``.
        """
        self.validate_image(block)
        return self.resize_image(block)


def _pil_format(mime_type: str | None) -> str:
    """Map MIME type to Pillow save format string."""
    mapping: dict[str, str] = {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
    }
    return mapping.get(mime_type or "", "PNG")
