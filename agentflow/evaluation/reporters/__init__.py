"""
Evaluation result reporters.

This module provides various output formats for evaluation results:
    - ConsoleReporter: Pretty-print results to console
    - JSONReporter: Export results to JSON file
    - HTMLReporter: Generate HTML report
    - JUnitXMLReporter: Export results to JUnit XML
    - ReporterManager: Orchestrates all enabled reporters
    - BaseReporter: Abstract base class for custom reporters
"""

from agentflow.evaluation.reporters.base import BaseReporter
from agentflow.evaluation.reporters.console import (
    Colors,
    ConsoleReporter,
    print_report,
)
from agentflow.evaluation.reporters.html import (
    HTMLReporter,
)
from agentflow.evaluation.reporters.json import (
    JSONReporter,
    JUnitXMLReporter,
)
from agentflow.evaluation.reporters.manager import (
    ReporterManager,
    ReporterOutput,
)


__all__ = [
    # Base
    "BaseReporter",
    # Console
    "ConsoleReporter",
    "Colors",
    "print_report",
    # JSON
    "JSONReporter",
    "JUnitXMLReporter",
    # HTML
    "HTMLReporter",
    # Manager
    "ReporterManager",
    "ReporterOutput",
]
