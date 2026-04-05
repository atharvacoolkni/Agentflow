"""
10xScale Agentflow: A lightweight Python framework for building intelligent
agents and multi-agent workflows.

Quick start::

    from agentflow import StateGraph, Agent, Message, START, END

    graph = StateGraph(AgentState)
    graph.add_node("agent", Agent(model="gpt-4o"))
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    app = graph.compile()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sub-packages (namespace imports for deep access)
# ---------------------------------------------------------------------------
from . import core, prebuilt, qa, runtime, utils
from .core import exceptions, graph, skills, state

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
from .core.exceptions import (
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

# ---------------------------------------------------------------------------
# Graph / Workflow Engine
# ---------------------------------------------------------------------------
from .core.graph import (
    Agent,
    BaseAgent,
    CompiledGraph,
    Edge,
    Node,
    RetryConfig,
    StateGraph,
    ToolNode,
)

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
from .core.skills import (
    SkillConfig,
    SkillMeta,
    SkillsRegistry,
)

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
from .core.state import (
    AgentState,
    AnnotationBlock,
    AnnotationRef,
    AudioBlock,
    # Context managers
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
    # Content blocks
    TextBlock,
    TokenUsages,
    ToolCallBlock,
    # Tool results
    ToolResult,
    ToolResultBlock,
    VideoBlock,
    # Reducers
    add_messages,
    append_items,
    remove_tool_messages,
    replace_messages,
    replace_value,
)

# ---------------------------------------------------------------------------
# Prebuilt Tools
# ---------------------------------------------------------------------------
from .prebuilt import create_handoff_tool
from .storage import checkpointer, media, store

# ---------------------------------------------------------------------------
# Storage (common classes re-exported for convenience)
# ---------------------------------------------------------------------------
from .storage.checkpointer import (
    BaseCheckpointer,
    InMemoryCheckpointer,
    PgCheckpointer,
)
from .storage.media import (
    BaseMediaStore,
    CloudMediaStore,
    DocumentHandling,
    ImageHandling,
    InMemoryMediaStore,
    LocalFileMediaStore,
    MediaOffloadPolicy,
    MediaProcessor,
    MediaRefResolver,
    MultimodalConfig,
    ProviderMediaCache,
    enforce_file_size,
    sanitize_filename,
    validate_magic_bytes,
)
from .storage.store import (
    BaseEmbedding,
    BaseStore,
    GoogleEmbedding,
    Mem0Store,
    MemoryIntegration,
    MemoryRecord,
    MemorySearchResult,
    OpenAIEmbedding,
    QdrantStore,
    ReadMode,
    create_cloud_qdrant_store,
    create_local_qdrant_store,
    create_mem0_store,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Utilities (most commonly used)
# ---------------------------------------------------------------------------
from .utils import (
    END,
    START,
    CallbackContext,
    CallbackManager,
    Command,
    ResponseGranularity,
    ThreadInfo,
    convert_messages,
    get_tool_metadata,
    tool,
)


__all__ = [
    # Sub-packages
    "core",
    "exceptions",
    "graph",
    "prebuilt",
    "qa",
    "runtime",
    "skills",
    "state",
    "storage",
    "utils",
    "checkpointer",
    "store",
    "media",
    # Graph
    "Agent",
    "BaseAgent",
    "CompiledGraph",
    "Edge",
    "Node",
    "RetryConfig",
    "StateGraph",
    "ToolNode",
    # State
    "AgentState",
    "ExecutionState",
    "ExecutionStatus",
    "Message",
    "StreamChunk",
    "StreamEvent",
    "TokenUsages",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "VideoBlock",
    "DocumentBlock",
    "DataBlock",
    "ErrorBlock",
    "ReasoningBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "AnnotationBlock",
    "ContentBlock",
    "AnnotationRef",
    "MediaRef",
    "BaseContextManager",
    "MessageContextManager",
    "ToolResult",
    # Reducers
    "add_messages",
    "append_items",
    "replace_messages",
    "replace_value",
    "remove_tool_messages",
    # Constants
    "START",
    "END",
    # Utilities
    "Command",
    "CallbackManager",
    "CallbackContext",
    "ResponseGranularity",
    "ThreadInfo",
    "tool",
    "get_tool_metadata",
    "convert_messages",
    # Exceptions
    "GraphError",
    "NodeError",
    "GraphRecursionError",
    "StorageError",
    "TransientStorageError",
    "ResourceNotFoundError",
    "SerializationError",
    "SchemaVersionError",
    "MetricsError",
    # Prebuilt
    "create_handoff_tool",
    # Skills
    "SkillConfig",
    "SkillMeta",
    "SkillsRegistry",
    # Checkpointer
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "PgCheckpointer",
    # Store
    "BaseStore",
    "QdrantStore",
    "Mem0Store",
    "BaseEmbedding",
    "OpenAIEmbedding",
    "GoogleEmbedding",
    "MemoryIntegration",
    "ReadMode",
    "MemoryRecord",
    "MemorySearchResult",
    "create_local_qdrant_store",
    "create_cloud_qdrant_store",
    "create_mem0_store",
    # Media
    "BaseMediaStore",
    "InMemoryMediaStore",
    "LocalFileMediaStore",
    "CloudMediaStore",
    "MediaRefResolver",
    "MediaProcessor",
    "MultimodalConfig",
    "DocumentHandling",
    "ImageHandling",
    "MediaOffloadPolicy",
    "ProviderMediaCache",
    "enforce_file_size",
    "sanitize_filename",
    "validate_magic_bytes",
]
