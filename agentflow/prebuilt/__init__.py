"""Prebuilt agents and tools for Agentflow.

This package provides ready-to-use agent patterns and utility tools:

- ``agentflow.prebuilt.agent`` — prebuilt agent implementations (ReactAgent, RAGAgent, ...)
- ``agentflow.prebuilt.tools`` — prebuilt tools (handoff, ...)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from . import agent, tools
    from .agent import RAGAgent, ReactAgent, RouterAgent
    from .tools import create_handoff_tool, is_handoff_tool

__all__ = [
    # Submodules
    "agent",
    "tools",
    # Agents
    "RAGAgent",
    "ReactAgent",
    "RouterAgent",
    # Tools
    "create_handoff_tool",
    "is_handoff_tool",
]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "agent": (".agent", None),
    "tools": (".tools", None),
    "RAGAgent": (".agent", "RAGAgent"),
    "ReactAgent": (".agent", "ReactAgent"),
    "RouterAgent": (".agent", "RouterAgent"),
    "create_handoff_tool": (".tools", "create_handoff_tool"),
    "is_handoff_tool": (".tools", "is_handoff_tool"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose prebuilt agents and tools."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
