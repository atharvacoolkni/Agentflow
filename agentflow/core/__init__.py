"""Core components for Agentflow.

This package provides the foundational building blocks for agent workflows:

- ``agentflow.core.graph``      — graph-based workflow engine (StateGraph, Agent, ...)
- ``agentflow.core.exceptions`` — custom exception hierarchy
- ``agentflow.core.skills``     — dynamic skill injection for agents
- ``agentflow.core.state``      — state management, messages, and reducers
"""

from __future__ import annotations

from . import exceptions, graph, skills, state

# --- Exceptions ---
from .exceptions import (
    GraphError,
    GraphRecursionError,
    MetricsError,
    NodeError,
    ResourceNotFoundError,
    SchemaVersionError,
    SerializationError,
    StorageError,
    TransientStorageError,
)

# --- Graph ---
from .graph import (
    Agent,
    BaseAgent,
    CompiledGraph,
    Edge,
    Node,
    RetryConfig,
    StateGraph,
    ToolNode,
)

# --- Skills ---
from .skills import SkillConfig, SkillMeta, SkillsRegistry

# --- State ---
from .state import (
    AgentState,
    AnnotationBlock,
    AnnotationRef,
    AudioBlock,
    BaseContextManager,
    ContentBlock,
    DataBlock,
    DocumentBlock,
    ErrorBlock,
    ExecutionState,
    ExecutionStatus,
    ImageBlock,
    MediaRef,
    Message,
    MessageContextManager,
    ReasoningBlock,
    StreamChunk,
    StreamEvent,
    TextBlock,
    TokenUsages,
    ToolCallBlock,
    ToolResult,
    ToolResultBlock,
    VideoBlock,
    add_messages,
    append_items,
    remove_tool_messages,
    replace_messages,
    replace_value,
)


__all__ = [
    # Submodules
    "exceptions",
    "graph",
    "skills",
    "state",
    # Graph
    "Agent",
    "BaseAgent",
    "CompiledGraph",
    "Edge",
    "Node",
    "RetryConfig",
    "StateGraph",
    "ToolNode",
    # Exceptions
    "GraphError",
    "GraphRecursionError",
    "MetricsError",
    "NodeError",
    "ResourceNotFoundError",
    "SchemaVersionError",
    "SerializationError",
    "StorageError",
    "TransientStorageError",
    # Skills
    "SkillConfig",
    "SkillMeta",
    "SkillsRegistry",
    # State
    "AgentState",
    "AnnotationBlock",
    "AnnotationRef",
    "AudioBlock",
    "BaseContextManager",
    "ContentBlock",
    "DataBlock",
    "DocumentBlock",
    "ErrorBlock",
    "ExecutionState",
    "ExecutionStatus",
    "ImageBlock",
    "MediaRef",
    "Message",
    "MessageContextManager",
    "ReasoningBlock",
    "StreamChunk",
    "StreamEvent",
    "TextBlock",
    "TokenUsages",
    "ToolCallBlock",
    "ToolResult",
    "ToolResultBlock",
    "VideoBlock",
    "add_messages",
    "append_items",
    "remove_tool_messages",
    "replace_messages",
    "replace_value",
]
