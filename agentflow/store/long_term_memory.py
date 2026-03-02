"""
Long-term memory integration for AgentFlow graphs.

Provides:
- memory_tool: an LLM-callable tool for search/store/update/delete on BaseStore
- create_memory_preload_node: factory that returns a node injecting memory into state
- get_memory_system_prompt: prompt fragments for each read mode
- MemoryWriteTracker: tracks pending async writes for guaranteed completion on shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from injectq import Inject

from agentflow.state import AgentState, Message
from agentflow.store.base_store import BaseStore
from agentflow.store.store_schema import MemorySearchResult, MemoryType
from agentflow.utils.background_task_manager import BackgroundTaskManager
from agentflow.utils.decorators import tool


logger = logging.getLogger("agentflow.store.long_term_memory")


_VALID_MEMORY_TYPES = {m.value for m in MemoryType}


class ReadMode(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    PRELOAD = "preload"
    POSTLOAD = "postload"


DEFAULT_READ_MODE = ReadMode.NO_RETRIEVAL


# ---------------------------------------------------------------------------
# Write tracker - guarantees all pending writes finish before shutdown
# ---------------------------------------------------------------------------


class MemoryWriteTracker:
    """Tracks pending async memory writes so shutdown can await them."""

    def __init__(self) -> None:
        self._pending: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    async def track(self, task: asyncio.Task) -> None:
        async with self._lock:
            self._pending.add(task)
            task.add_done_callback(lambda t: self._pending.discard(t))

    async def wait_for_pending(self, timeout: float | None = None) -> dict[str, Any]:
        """Wait for all pending writes. Returns stats dict."""
        tasks = list(self._pending)
        if not tasks:
            return {"status": "completed", "pending_writes": 0}

        count = len(tasks)
        logger.info("Waiting for %d pending memory writes to complete...", count)
        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout,
                )
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All %d pending memory writes completed.", count)
            return {"status": "completed", "pending_writes": 0, "completed": count}
        except TimeoutError:
            remaining = len(self._pending)
            logger.warning(
                "Timeout waiting for memory writes: %d/%d still pending", remaining, count
            )
            return {
                "status": "timeout",
                "pending_writes": remaining,
                "completed": count - remaining,
            }

    @property
    def pending_count(self) -> int:
        return len(self._pending)


_write_tracker = MemoryWriteTracker()


def get_write_tracker() -> MemoryWriteTracker:
    """Returns the global write-tracker instance."""
    return _write_tracker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_memory_type(value: str) -> MemoryType:
    if value in _VALID_MEMORY_TYPES:
        return MemoryType(value)
    return MemoryType.EPISODIC


def _format_search_results(results: list[MemorySearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "id": r.id,
            "content": r.content,
            "score": round(r.score, 4),
            "memory_type": r.memory_type.value if r.memory_type else "episodic",
            "metadata": r.metadata or {},
        }
        for r in results
    ]


async def _do_write(
    store: BaseStore,
    config: dict[str, Any],
    action: str,
    content: str,
    memory_id: str,
    mem_type: MemoryType,
    category: str,
    metadata: dict[str, Any] | None,
    write_mode: str,
) -> dict[str, Any]:
    """Execute a write operation against the store."""
    if action == "store":
        mid = await store.astore(
            config, content, memory_type=mem_type, category=category, metadata=metadata
        )
        return {"status": "stored", "memory_id": str(mid)}

    if action == "update":
        if write_mode == "merge":
            existing = await store.aget(config, memory_id)
            merged = {
                **(existing.metadata if existing and existing.metadata else {}),
                **(metadata or {}),
            }
            await store.aupdate(config, memory_id, content, metadata=merged)
        else:
            await store.aupdate(config, memory_id, content, metadata=metadata)
        return {"status": "updated", "memory_id": memory_id}

    if action == "delete":
        await store.adelete(config, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    return {"error": f"unknown write action: {action}"}


# ---------------------------------------------------------------------------
# memory_tool - the LLM-callable tool
# ---------------------------------------------------------------------------


@tool(
    name="memory_tool",
    description=(
        "Search, store, update or delete long-term memories. "
        "Use action='search' with a query to recall relevant memories. "
        "Use action='store' with content to save new memories. "
        "Use action='update' with memory_id and content to modify existing memories. "
        "Use action='delete' with memory_id to remove memories."
    ),
    tags=["memory", "long_term_memory"],
)
async def memory_tool(  # noqa: PLR0913, PLR0911
    action: Literal["search", "store", "update", "delete"],
    content: str = "",
    memory_id: str = "",
    query: str = "",
    memory_type: str = "episodic",
    category: str = "general",
    metadata: dict[str, Any] | None = None,
    limit: int = 5,
    score_threshold: float = 0.0,
    write_mode: Literal["merge", "replace"] = "merge",
    async_write: bool = True,
    # Injectable params (excluded from LLM schema automatically)
    tool_call_id: str = "",
    state: AgentState | None = None,
    config: dict[str, Any] | None = None,
    store: BaseStore | None = Inject[BaseStore],
    task_manager: BackgroundTaskManager = Inject[BackgroundTaskManager],
) -> str:
    """Search, store, update, or delete long-term memories."""
    if store is None:
        return json.dumps({"error": "no memory store configured"})

    cfg = config or {}
    mem_type = _validate_memory_type(memory_type)

    # --- Validation ---
    if action == "search" and not query:
        return json.dumps({"error": "query is required for search"})
    if action == "store" and not content:
        return json.dumps({"error": "content is required for store"})
    if action == "update" and not memory_id:
        return json.dumps({"error": "memory_id is required for update"})
    if action == "update" and not content:
        return json.dumps({"error": "content is required for update"})
    if action == "delete" and not memory_id:
        return json.dumps({"error": "memory_id is required for delete"})

    try:
        # --- Read ---
        if action == "search":
            results = await store.asearch(
                cfg,
                query,
                memory_type=mem_type,
                limit=limit,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )
            return json.dumps(_format_search_results(results))

        # --- Write (sync or async) ---
        if async_write:
            task = task_manager.create_task(
                _do_write(
                    store, cfg, action, content, memory_id, mem_type, category, metadata, write_mode
                ),
                name=f"memory_{action}_{memory_id or 'new'}",
            )
            await _write_tracker.track(task)
            return json.dumps({"status": "scheduled", "action": action})

        result = await _do_write(
            store, cfg, action, content, memory_id, mem_type, category, metadata, write_mode
        )
        return json.dumps(result)

    except Exception as e:
        logger.exception("memory_tool error (action=%s): %s", action, e)
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Preload node factory
# ---------------------------------------------------------------------------

_DEFAULT_MEMORY_PROMPT = (
    "[Long-term Memory Context]\n"
    "The following memories were retrieved for the current conversation:\n"
    "{memories}\n"
    "Use this context to inform your response when relevant."
)


def _default_query_builder(state: AgentState) -> str:
    """Extract the latest user message text as the search query."""
    for msg in reversed(state.context):
        if msg.role == "user":
            return msg.text()
    return ""


def create_memory_preload_node(
    store: BaseStore,
    query_builder: Callable[[AgentState], str] | None = None,
    limit: int = 5,
    score_threshold: float = 0.0,
    memory_types: list[MemoryType] | None = None,
    system_prompt_template: str = _DEFAULT_MEMORY_PROMPT,
    max_tokens: int | None = None,
) -> Callable:
    """Factory returning a node function that preloads memory into state context.

    Wire into a graph before the LLM node:
        preload = create_memory_preload_node(store)
        graph.add_node("memory_preload", preload)
        graph.add_edge("memory_preload", "main")
        graph.set_entry_point("memory_preload")
    """
    _builder = query_builder or _default_query_builder

    async def _preload_node(state: AgentState, config: dict[str, Any]) -> list[Message]:
        query = _builder(state)
        if not query:
            return []

        search_kwargs: dict[str, Any] = {
            "limit": limit,
        }
        if score_threshold > 0:
            search_kwargs["score_threshold"] = score_threshold
        if memory_types:
            search_kwargs["memory_type"] = memory_types[0]
        if max_tokens is not None:
            search_kwargs["max_tokens"] = max_tokens

        try:
            results = await store.asearch(config, query, **search_kwargs)
        except Exception:
            logger.exception("Memory preload search failed")
            return []

        if not results:
            return []

        lines = []
        for r in results:
            score_str = f" (relevance: {r.score:.2f})" if r.score else ""
            lines.append(f"- {r.content}{score_str}")
        memory_text = system_prompt_template.format(memories="\n".join(lines))

        return [Message.text_message(memory_text, role="system")]

    _preload_node.__name__ = "memory_preload"
    _preload_node.__qualname__ = "memory_preload"
    return _preload_node


# ---------------------------------------------------------------------------
# System prompt helpers
# ---------------------------------------------------------------------------


# Shared write instructions appended to every mode's prompt.
_WRITE_INSTRUCTIONS = (
    "\n\nYou have access to a memory_tool for writing to long-term memory.\n"
    "After processing each user message, decide whether any new information "
    "(facts, preferences, names, decisions) should be persisted.\n"
    "- To save important facts or preferences, call memory_tool with "
    "action='store' and the content to remember.\n"
    "- To modify existing memories, use action='update' with the memory_id.\n"
    "- To remove outdated information, use action='delete' with the memory_id.\n"
    "Writing is asynchronous — it will not slow down your response."
)


def get_memory_system_prompt(
    mode: Literal["no_retrieval", "preload", "postload"] = "no_retrieval",
) -> str:
    """Returns a system prompt fragment for the given read mode.

    All modes include write instructions because writing is always enabled
    and independent of the retrieval mode.
    Default mode is no_retrieval.
    """
    if mode == "no_retrieval":
        return (
            "You do NOT have access to read or search long-term memories. "
            "Do not attempt to recall information from previous sessions." + _WRITE_INSTRUCTIONS
        )

    if mode == "preload":
        return (
            "You have been provided with long-term memory context from previous "
            "interactions. Use it to personalize your responses when relevant. "
            "The memory context appears as system messages labeled "
            "'[Long-term Memory Context]'." + _WRITE_INSTRUCTIONS
        )

    if mode == "postload":
        return (
            "You have access to a memory_tool that can search, store, update, "
            "and delete long-term memories.\n"
            "- To recall relevant information, call memory_tool with action='search' "
            "and a descriptive query.\n"
            "- To save important facts or preferences, call memory_tool with "
            "action='store' and the content to remember.\n"
            "- To modify existing memories, use action='update' with the memory_id.\n"
            "- To remove outdated information, use action='delete' with the memory_id.\n"
            "Only search memory when prior context would genuinely improve your response.\n"
            "Writing is asynchronous — it will not slow down your response."
        )

    return ""
