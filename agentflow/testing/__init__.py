"""Backward-compatible testing import paths.

The testing helpers now live under ``agentflow.qa.testing``. This module keeps
``agentflow.testing`` and its common submodules pointing at the same live
implementations for compatibility.
"""

from __future__ import annotations

import importlib
import sys


_ALIASES = {
    "agentflow.testing": "agentflow.qa.testing",
    "agentflow.testing.in_memory_store": "agentflow.qa.testing.in_memory_store",
    "agentflow.testing.mock_mcp": "agentflow.qa.testing.mock_mcp",
    "agentflow.testing.mock_tools": "agentflow.qa.testing.mock_tools",
    "agentflow.testing.quick_test": "agentflow.qa.testing.quick_test",
    "agentflow.testing.test_agent": "agentflow.qa.testing.test_agent",
}


def _alias(name: str):
    module = importlib.import_module(_ALIASES[name])
    sys.modules[name] = module
    return module


for alias in _ALIASES:
    _alias(alias)

__all__ = list(getattr(sys.modules["agentflow.testing"], "__all__", []))
