"""Agent communication protocols for Agentflow.

Protocols:
- ACP (Agent Communication Protocol) - standardized agent-to-agent messaging
- A2A - Google A2A SDK bridge (requires ``pip install 10xscale-agentflow[a2a_sdk]``)
"""

from __future__ import annotations

import importlib


_SYMBOL_EXPORTS = {
    "ACPMessage": ".acp",
    "ACPMessageType": ".acp",
    "ACPProtocol": ".acp",
    "MessageContent": ".acp",
    "MessageContext": ".acp",
    "a2a": ".a2a",
}

__all__ = list(_SYMBOL_EXPORTS)


def __getattr__(name: str):
    """Lazily load protocol implementations."""
    module_name = _SYMBOL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = module if name == "a2a" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package exports for interactive discovery."""
    return sorted(__all__)
