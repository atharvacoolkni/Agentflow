"""Quality assurance utilities for Agentflow.

This package groups Agentflow's testing and evaluation tooling under a
single namespace:

- ``agentflow.qa.testing``    — test doubles, mocks, quick tests, and helpers
- ``agentflow.qa.evaluation`` — eval sets, criteria, runners, reporters, and results
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from . import evaluation, testing

    # Evaluation
    from .evaluation import (
        AgentEvaluator,
        BaseCriterion,
        BatchSimulator,
        Colors,
        CompositeCriterion,
        ConsoleReporter,
        ContainsKeywordsCriterion,
        ConversationScenario,
        CriterionConfig,
        CriterionResult,
        EvalCase,
        EvalCaseResult,
        EvalConfig,
        EvalFixtures,
        EvalPlugin,
        EvalPresets,
        EvalReport,
        EvalSet,
        EvalSetBuilder,
        EvalSummary,
        EvalTestCase,
        EvaluationRunner,
        EventCollector,
        ExactMatchCriterion,
        ExecutionResult,
        FactualAccuracyCriterion,
        HallucinationCriterion,
        HTMLReporter,
        Invocation,
        JSONReporter,
        JUnitXMLReporter,
        LLMCallerMixin,
        LLMJudgeCriterion,
        MatchType,
        MessageContent,
        NodeOrderMatchCriterion,
        NodeResponseData,
        PublisherCallback,
        QuickEval,
        ReporterConfig,
        ReporterManager,
        ReporterOutput,
        ResponseMatchCriterion,
        RougeMatchCriterion,
        Rubric,
        RubricBasedCriterion,
        SafetyCriterion,
        SessionInput,
        SimulationGoalsCriterion,
        SimulationResult,
        StepType,
        SyncCriterion,
        ToolCall,
        ToolNameMatchCriterion,
        TrajectoryCollector,
        TrajectoryMatchCriterion,
        TrajectoryStep,
        UserSimulator,
        UserSimulatorConfig,
        WeightedCriterion,
        assert_criterion_passed,
        assert_eval_passed,
        create_eval_app,
        create_simple_eval_set,
        eval_test,
        make_trajectory_callback,
        parametrize_eval_cases,
        print_report,
        run_eval,
    )

    # Testing
    from .testing import (
        InMemoryStore,
        MockComposioAdapter,
        MockLangChainAdapter,
        MockMCPClient,
        MockToolRegistry,
        QuickTest,
        TestAgent,
        TestContext,
        TestResult,
    )


__all__ = [
    # Submodules
    "evaluation",
    "testing",
    # --- Testing ---
    "InMemoryStore",
    "MockComposioAdapter",
    "MockLangChainAdapter",
    "MockMCPClient",
    "MockToolRegistry",
    "QuickTest",
    "TestAgent",
    "TestContext",
    "TestResult",
    # --- Evaluation: Dataset ---
    "EvalCase",
    "EvalSet",
    "EvalSetBuilder",
    "Invocation",
    "MessageContent",
    "SessionInput",
    "StepType",
    "ToolCall",
    "TrajectoryStep",
    # --- Evaluation: Execution ---
    "ExecutionResult",
    "NodeResponseData",
    # --- Evaluation: Collectors ---
    "EventCollector",
    "PublisherCallback",
    "TrajectoryCollector",
    "make_trajectory_callback",
    # --- Evaluation: Criteria (base) ---
    "BaseCriterion",
    "CompositeCriterion",
    "LLMCallerMixin",
    "SyncCriterion",
    "WeightedCriterion",
    # --- Evaluation: Criteria (trajectory) ---
    "NodeOrderMatchCriterion",
    "ToolNameMatchCriterion",
    "TrajectoryMatchCriterion",
    # --- Evaluation: Criteria (response) ---
    "ContainsKeywordsCriterion",
    "ExactMatchCriterion",
    "ResponseMatchCriterion",
    "RougeMatchCriterion",
    # --- Evaluation: Criteria (LLM-as-judge) ---
    "LLMJudgeCriterion",
    "RubricBasedCriterion",
    "SimulationGoalsCriterion",
    # --- Evaluation: Criteria (advanced) ---
    "FactualAccuracyCriterion",
    "HallucinationCriterion",
    "SafetyCriterion",
    # --- Evaluation: Config ---
    "CriterionConfig",
    "EvalConfig",
    "EvalPresets",
    "MatchType",
    "ReporterConfig",
    "Rubric",
    "UserSimulatorConfig",
    # --- Evaluation: Results ---
    "CriterionResult",
    "EvalCaseResult",
    "EvalReport",
    "EvalSummary",
    # --- Evaluation: Reporters ---
    "Colors",
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
    "JUnitXMLReporter",
    "ReporterManager",
    "ReporterOutput",
    "print_report",
    # --- Evaluation: Runner ---
    "AgentEvaluator",
    "EvaluationRunner",
    "QuickEval",
    # --- Evaluation: Simulators ---
    "BatchSimulator",
    "ConversationScenario",
    "SimulationResult",
    "UserSimulator",
    # --- Evaluation: Testing helpers ---
    "EvalFixtures",
    "EvalPlugin",
    "EvalTestCase",
    "assert_criterion_passed",
    "assert_eval_passed",
    "create_eval_app",
    "create_simple_eval_set",
    "eval_test",
    "parametrize_eval_cases",
    "run_eval",
]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    # Submodules
    "evaluation": (".evaluation", None),
    "testing": (".testing", None),
    # Testing
    "InMemoryStore": (".testing", "InMemoryStore"),
    "MockComposioAdapter": (".testing", "MockComposioAdapter"),
    "MockLangChainAdapter": (".testing", "MockLangChainAdapter"),
    "MockMCPClient": (".testing", "MockMCPClient"),
    "MockToolRegistry": (".testing", "MockToolRegistry"),
    "QuickTest": (".testing", "QuickTest"),
    "TestAgent": (".testing", "TestAgent"),
    "TestContext": (".testing", "TestContext"),
    "TestResult": (".testing", "TestResult"),
    # Evaluation
    "AgentEvaluator": (".evaluation", "AgentEvaluator"),
    "BaseCriterion": (".evaluation", "BaseCriterion"),
    "BatchSimulator": (".evaluation", "BatchSimulator"),
    "Colors": (".evaluation", "Colors"),
    "CompositeCriterion": (".evaluation", "CompositeCriterion"),
    "ConsoleReporter": (".evaluation", "ConsoleReporter"),
    "ContainsKeywordsCriterion": (".evaluation", "ContainsKeywordsCriterion"),
    "ConversationScenario": (".evaluation", "ConversationScenario"),
    "CriterionConfig": (".evaluation", "CriterionConfig"),
    "CriterionResult": (".evaluation", "CriterionResult"),
    "EvalCase": (".evaluation", "EvalCase"),
    "EvalCaseResult": (".evaluation", "EvalCaseResult"),
    "EvalConfig": (".evaluation", "EvalConfig"),
    "EvalFixtures": (".evaluation", "EvalFixtures"),
    "EvalPlugin": (".evaluation", "EvalPlugin"),
    "EvalPresets": (".evaluation", "EvalPresets"),
    "EvalReport": (".evaluation", "EvalReport"),
    "EvalSet": (".evaluation", "EvalSet"),
    "EvalSetBuilder": (".evaluation", "EvalSetBuilder"),
    "EvalSummary": (".evaluation", "EvalSummary"),
    "EvalTestCase": (".evaluation", "EvalTestCase"),
    "EvaluationRunner": (".evaluation", "EvaluationRunner"),
    "EventCollector": (".evaluation", "EventCollector"),
    "ExactMatchCriterion": (".evaluation", "ExactMatchCriterion"),
    "ExecutionResult": (".evaluation", "ExecutionResult"),
    "FactualAccuracyCriterion": (".evaluation", "FactualAccuracyCriterion"),
    "HallucinationCriterion": (".evaluation", "HallucinationCriterion"),
    "HTMLReporter": (".evaluation", "HTMLReporter"),
    "Invocation": (".evaluation", "Invocation"),
    "JSONReporter": (".evaluation", "JSONReporter"),
    "JUnitXMLReporter": (".evaluation", "JUnitXMLReporter"),
    "LLMCallerMixin": (".evaluation", "LLMCallerMixin"),
    "LLMJudgeCriterion": (".evaluation", "LLMJudgeCriterion"),
    "MatchType": (".evaluation", "MatchType"),
    "MessageContent": (".evaluation", "MessageContent"),
    "NodeOrderMatchCriterion": (".evaluation", "NodeOrderMatchCriterion"),
    "NodeResponseData": (".evaluation", "NodeResponseData"),
    "PublisherCallback": (".evaluation", "PublisherCallback"),
    "QuickEval": (".evaluation", "QuickEval"),
    "ReporterConfig": (".evaluation", "ReporterConfig"),
    "ReporterManager": (".evaluation", "ReporterManager"),
    "ReporterOutput": (".evaluation", "ReporterOutput"),
    "ResponseMatchCriterion": (".evaluation", "ResponseMatchCriterion"),
    "RougeMatchCriterion": (".evaluation", "RougeMatchCriterion"),
    "Rubric": (".evaluation", "Rubric"),
    "RubricBasedCriterion": (".evaluation", "RubricBasedCriterion"),
    "SafetyCriterion": (".evaluation", "SafetyCriterion"),
    "SessionInput": (".evaluation", "SessionInput"),
    "SimulationGoalsCriterion": (".evaluation", "SimulationGoalsCriterion"),
    "SimulationResult": (".evaluation", "SimulationResult"),
    "StepType": (".evaluation", "StepType"),
    "SyncCriterion": (".evaluation", "SyncCriterion"),
    "ToolCall": (".evaluation", "ToolCall"),
    "ToolNameMatchCriterion": (".evaluation", "ToolNameMatchCriterion"),
    "TrajectoryCollector": (".evaluation", "TrajectoryCollector"),
    "TrajectoryMatchCriterion": (".evaluation", "TrajectoryMatchCriterion"),
    "TrajectoryStep": (".evaluation", "TrajectoryStep"),
    "UserSimulator": (".evaluation", "UserSimulator"),
    "UserSimulatorConfig": (".evaluation", "UserSimulatorConfig"),
    "WeightedCriterion": (".evaluation", "WeightedCriterion"),
    "assert_criterion_passed": (".evaluation", "assert_criterion_passed"),
    "assert_eval_passed": (".evaluation", "assert_eval_passed"),
    "create_eval_app": (".evaluation", "create_eval_app"),
    "create_simple_eval_set": (".evaluation", "create_simple_eval_set"),
    "eval_test": (".evaluation", "eval_test"),
    "make_trajectory_callback": (".evaluation", "make_trajectory_callback"),
    "parametrize_eval_cases": (".evaluation", "parametrize_eval_cases"),
    "print_report": (".evaluation", "print_report"),
    "run_eval": (".evaluation", "run_eval"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose QA subpackages and all entry points."""
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
