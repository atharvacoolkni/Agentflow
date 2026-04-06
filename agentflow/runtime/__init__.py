"""Runtime components for Agentflow.

This package provides the runtime infrastructure for agent execution:

- ``agentflow.runtime.adapters``   - LLM response converters and third-party tool adapters
- ``agentflow.runtime.publisher``  - event publishers (console, Redis, Kafka, RabbitMQ)
- ``agentflow.runtime.protocols``  - agent communication protocol packages
"""

from . import adapters, publisher
from .adapters.llm import (
    BaseConverter,
    ConverterType,
    GoogleGenAIConverter,
    OpenAIConverter,
    OpenAIResponsesConverter,
)
from .adapters.tools import ComposioAdapter, LangChainAdapter
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


def __getattr__(name: str):
    if name == "protocols":
        import importlib

        return importlib.import_module("agentflow.runtime.protocols")
    raise AttributeError(name)


__all__ = [
    "BaseConverter",
    "BasePublisher",
    "ComposioAdapter",
    "ConsolePublisher",
    "ContentType",
    "ConverterType",
    "Event",
    "EventModel",
    "EventType",
    "GoogleGenAIConverter",
    "KafkaPublisher",
    "LangChainAdapter",
    "OpenAIConverter",
    "OpenAIResponsesConverter",
    "RabbitMQPublisher",
    "RedisPublisher",
    "adapters",
    "protocols",
    "publish_event",
    "publisher",
]
