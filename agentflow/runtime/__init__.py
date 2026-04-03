"""Runtime components for Agentflow.

This package provides the runtime infrastructure for agent execution:

- ``agentflow.runtime.adapters``   — LLM response converters and third-party tool adapters
- ``agentflow.runtime.publisher``  — event publishers (console, Redis, Kafka, RabbitMQ)
- ``agentflow.runtime.protocols``  — agent communication protocols (ACP, A2A)
"""

from __future__ import annotations

from . import adapters, protocols, publisher

# --- Adapters: LLM converters ---
from .adapters.llm import (
    BaseConverter,
    ConverterType,
    GoogleGenAIConverter,
    OpenAIConverter,
    OpenAIResponsesConverter,
)

# --- Adapters: Tool integrations ---
from .adapters.tools import (
    ComposioAdapter,
    LangChainAdapter,
)

# --- Publisher ---
from .publisher import (
    BasePublisher,
    ConsolePublisher,
    ContentType,
    Event,
    EventModel,
    EventType,
    KafkaPublisher,
    RabbitMQPublisher,
    RedisPublisher,
    publish_event,
)


# --- Protocols: ACP (experimental) ---
try:
    from .protocols import (
        ACPMessage,
        ACPMessageType,
        ACPProtocol,
        MessageContent,
        MessageContext,
    )

    _ACP_EXPORTS = [
        "ACPMessage",
        "ACPMessageType",
        "ACPProtocol",
        "MessageContent",
        "MessageContext",
    ]
except ImportError:
    _ACP_EXPORTS = []

_BASE_EXPORTS = [
    # Submodules
    "adapters",
    "protocols",
    "publisher",
    # Adapters: LLM
    "BaseConverter",
    "ConverterType",
    "GoogleGenAIConverter",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
    # Adapters: Tools
    "ComposioAdapter",
    "LangChainAdapter",
    # Publisher
    "BasePublisher",
    "ConsolePublisher",
    "ContentType",
    "Event",
    "EventModel",
    "EventType",
    "KafkaPublisher",
    "RabbitMQPublisher",
    "RedisPublisher",
    "publish_event",
]

__all__ = _BASE_EXPORTS + _ACP_EXPORTS
