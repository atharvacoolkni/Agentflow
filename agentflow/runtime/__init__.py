"""Runtime components for Agentflow.

This package provides the runtime infrastructure for agent execution:

- ``agentflow.runtime.adapters``   - LLM response converters and third-party tool adapters
- ``agentflow.runtime.publisher``  - event publishers (console, Redis, Kafka, RabbitMQ)
- ``agentflow.runtime.protocols``  - agent communication protocols (ACP, A2A)
"""

from __future__ import annotations

import importlib


_MODULE_EXPORTS = {
    "adapters": ".adapters",
    "protocols": ".protocols",
    "publisher": ".publisher",
}

_SYMBOL_EXPORTS = {
    # Adapters: LLM
    "BaseConverter": ".adapters.llm",
    "ConverterType": ".adapters.llm",
    "GoogleGenAIConverter": ".adapters.llm",
    "OpenAIConverter": ".adapters.llm",
    "OpenAIResponsesConverter": ".adapters.llm",
    # Adapters: Tools
    "ComposioAdapter": ".adapters.tools",
    "LangChainAdapter": ".adapters.tools",
    # Publisher
    "BasePublisher": ".publisher",
    "ConsolePublisher": ".publisher",
    "ContentType": ".publisher",
    "Event": ".publisher",
    "EventModel": ".publisher",
    "EventType": ".publisher",
    "KafkaPublisher": ".publisher",
    "RabbitMQPublisher": ".publisher",
    "RedisPublisher": ".publisher",
    "publish_event": ".publisher",
    # Protocols: ACP
    "ACPMessage": ".protocols.acp",
    "ACPMessageType": ".protocols.acp",
    "ACPProtocol": ".protocols.acp",
    "MessageContent": ".protocols.acp",
    "MessageContext": ".protocols.acp",
}

__all__ = [* _MODULE_EXPORTS, * _SYMBOL_EXPORTS]


def __getattr__(name: str):
    """Lazily load runtime exports so optional extras stay optional."""
    if name in _MODULE_EXPORTS:
        module = importlib.import_module(_MODULE_EXPORTS[name], __name__)
        globals()[name] = module
        return module

    module_name = _SYMBOL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package exports for interactive discovery."""
    return sorted(__all__)
