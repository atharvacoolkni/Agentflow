"""Publisher module for TAF events.

This package exposes publishers that handle event delivery to various outputs,
such as console, Redis, Kafka, and RabbitMQ.
"""

from __future__ import annotations

import importlib


_SYMBOL_EXPORTS = {
    "BasePublisher": ".base_publisher",
    "ConsolePublisher": ".console_publisher",
    "ContentType": ".events",
    "Event": ".events",
    "EventModel": ".events",
    "EventType": ".events",
    "KafkaPublisher": ".kafka_publisher",
    "RabbitMQPublisher": ".rabbitmq_publisher",
    "RedisPublisher": ".redis_publisher",
    "publish_event": ".publish",
}

__all__ = list(_SYMBOL_EXPORTS)


def __getattr__(name: str):
    """Lazily load publisher exports so optional dependencies stay optional."""
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
