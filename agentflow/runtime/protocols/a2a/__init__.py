"""Optional A2A protocol bridge for Agentflow.

This package exposes any agentflow ``CompiledGraph`` as a standard A2A
agent using the official ``a2a-sdk`` package, and also provides client
helpers to call remote A2A agents from within a graph.

Install the extra:

    pip install 10xscale-agentflow[a2a_sdk]

Quick start - server:

    from agentflow.runtime.protocols.a2a import (
        AgentFlowExecutor,
        create_a2a_server,
        make_agent_card,
    )

Quick start - client:

    from agentflow.runtime.protocols.a2a import delegate_to_a2a_agent
"""

from __future__ import annotations

import importlib


_SYMBOL_EXPORTS = {
    "AgentFlowExecutor": ".executor",
    "build_a2a_app": ".server",
    "create_a2a_client_node": ".client",
    "create_a2a_server": ".server",
    "delegate_to_a2a_agent": ".client",
    "make_agent_card": ".server",
}

__all__ = list(_SYMBOL_EXPORTS)


def _raise_missing_a2a_dependency(exc: BaseException) -> None:
    raise ImportError(
        "agentflow.runtime.protocols.a2a requires the 'a2a-sdk' package. "
        "Install it with: pip install 10xscale-agentflow[a2a_sdk]"
    ) from exc


def _is_missing_a2a_dependency(exc: BaseException) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        missing_name = exc.name or ""
        return missing_name == "a2a" or missing_name.startswith("a2a.")

    if isinstance(exc, ImportError):
        return "a2a" in str(exc)

    return False


def __getattr__(name: str):
    """Lazily load A2A helpers so the SDK remains an optional extra."""
    module_name = _SYMBOL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = importlib.import_module(module_name, __name__)
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_missing_a2a_dependency(exc):
            _raise_missing_a2a_dependency(exc)
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the package exports for interactive discovery."""
    return sorted(__all__)
