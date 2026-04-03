"""Agent communication protocols for Agentflow.

Protocols:
- ACP (Agent Communication Protocol) — standardized agent-to-agent messaging
- A2A — Google A2A SDK bridge (requires ``pip install agentflow[a2a_sdk]``)
"""

from __future__ import annotations


# ACP is experimental and may not be fully implemented yet.
try:
    from agentflow.runtime.protocols.acp import (
        ACPMessage,
        ACPMessageType,
        ACPProtocol,
        MessageContent,
        MessageContext,
    )

    __all__ = [
        "ACPMessage",
        "ACPMessageType",
        "ACPProtocol",
        "MessageContent",
        "MessageContext",
    ]
except ImportError:
    __all__ = []
