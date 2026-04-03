"""Quality assurance utilities for Agentflow.

This package groups Agentflow's testing and evaluation tooling under a
single namespace:

- ``agentflow.qa.testing`` for test doubles, quick tests, and mock helpers
- ``agentflow.qa.evaluation`` for eval sets, criteria, runners, and reports

The ``qa`` package stays intentionally thin. Detailed exports continue to live
in the ``testing`` and ``evaluation`` subpackages, while this module provides
a few convenient entry points for common workflows.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from . import evaluation, testing
    from .evaluation import AgentEvaluator, EvalConfig, EvalSetBuilder, QuickEval
    from .testing import QuickTest, TestAgent, TestContext


__all__ = [
    "AgentEvaluator",
    "EvalConfig",
    "EvalSetBuilder",
    "QuickEval",
    "QuickTest",
    "TestAgent",
    "TestContext",
    "evaluation",
    "testing",
]

_LAZY_EXPORTS = {
    "evaluation": (".evaluation", None),
    "testing": (".testing", None),
    "AgentEvaluator": (".evaluation", "AgentEvaluator"),
    "EvalConfig": (".evaluation", "EvalConfig"),
    "EvalSetBuilder": (".evaluation", "EvalSetBuilder"),
    "QuickEval": (".evaluation", "QuickEval"),
    "QuickTest": (".testing", "QuickTest"),
    "TestAgent": (".testing", "TestAgent"),
    "TestContext": (".testing", "TestContext"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose QA subpackages and common entry points."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes plus lazy exports for discovery."""
    return sorted(set(globals()) | set(__all__))
