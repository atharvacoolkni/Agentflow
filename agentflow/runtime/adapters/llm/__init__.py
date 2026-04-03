"""Integration adapters for optional third-party LLM SDKs.

This package exposes a small, stable surface for response converters without
eagerly importing every concrete implementation during package import. The
lazy behavior avoids import cycles with graph/runtime modules that reference
converter types during test collection.
"""

from __future__ import annotations

from .base_converter import BaseConverter, ConverterType


__all__ = [
    "BaseConverter",
    "ConverterType",
    "GoogleGenAIConverter",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
]


def __getattr__(name: str):
    if name == "GoogleGenAIConverter":
        from .google_genai_converter import GoogleGenAIConverter

        return GoogleGenAIConverter
    if name == "OpenAIConverter":
        from .openai_converter import OpenAIConverter

        return OpenAIConverter
    if name == "OpenAIResponsesConverter":
        from .openai_responses_converter import OpenAIResponsesConverter

        return OpenAIResponsesConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
