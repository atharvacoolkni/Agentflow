"""Prebuilt tools and agent packages for Agentflow.

Import concrete agent implementations from ``agentflow.prebuilt.agent`` and
tool helpers from ``agentflow.prebuilt.tools``.
"""

from __future__ import annotations

import importlib


def __getattr__(name: str):
    if name == "agent":
        return importlib.import_module("agentflow.prebuilt.agent")
    if name == "tools":
        return importlib.import_module("agentflow.prebuilt.tools")
    if name in {"RAGAgent", "ReactAgent", "RouterAgent"}:
        module = importlib.import_module("agentflow.prebuilt.agent")
        return getattr(module, name)
    if name in {
        "create_handoff_tool",
        "is_handoff_tool",
        "make_agent_memory_tool",
        "make_user_memory_tool",
        "memory_tool",
    }:
        module = importlib.import_module("agentflow.prebuilt.tools")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "RAGAgent",
    "ReactAgent",
    "RouterAgent",
    # Agents
    "agent",
    # Tools
    "create_handoff_tool",
    "is_handoff_tool",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
    "tools",
]
