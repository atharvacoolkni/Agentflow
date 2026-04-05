"""
Integration adapters for optional third-party SDKs.

This package provides unified wrappers and converters for integrating external
tool registries, LLM SDKs, and other third-party services with agentflow agent graphs.
Adapters expose registry-based discovery, function-calling schemas, and normalized
execution for supported providers.
"""

from . import llm, tools
from .llm import (
    BaseConverter,
    ConverterType,
    GoogleGenAIConverter,
    OpenAIConverter,
    OpenAIResponsesConverter,
)
from .tools import ComposioAdapter, LangChainAdapter


__all__ = [
    "BaseConverter",
    "ComposioAdapter",
    "ConverterType",
    "GoogleGenAIConverter",
    "LangChainAdapter",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
    "llm",
    "tools",
]
