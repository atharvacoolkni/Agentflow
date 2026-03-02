"""Tests for agentflow.store.long_term_memory module."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentflow.state import AgentState, Message
from agentflow.store.long_term_memory import (
    DEFAULT_READ_MODE,
    MemoryWriteTracker,
    ReadMode,
    _do_write,
    _format_search_results,
    _validate_memory_type,
    create_memory_preload_node,
    get_memory_system_prompt,
    get_write_tracker,
    memory_tool,
)
from agentflow.store.store_schema import MemorySearchResult, MemoryType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_store():
    store = AsyncMock()
    store.astore = AsyncMock(return_value="mem-001")
    store.asearch = AsyncMock(return_value=[])
    store.aget = AsyncMock(return_value=None)
    store.aupdate = AsyncMock()
    store.adelete = AsyncMock()
    return store


@pytest.fixture()
def mock_task_manager():
    mgr = MagicMock()
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    task.add_done_callback = MagicMock()
    mgr.create_task = MagicMock(return_value=task)
    return mgr


@pytest.fixture()
def sample_search_results():
    return [
        MemorySearchResult(
            id="r1",
            content="User prefers dark mode",
            score=0.92,
            memory_type=MemoryType.SEMANTIC,
            metadata={"source": "chat"},
        ),
        MemorySearchResult(
            id="r2",
            content="User works with Python 3.12",
            score=0.85,
            memory_type=MemoryType.EPISODIC,
            metadata={},
        ),
    ]


@pytest.fixture()
def sample_state():
    return AgentState(
        context=[Message.text_message("What is my preferred editor?", role="user")]
    )


@pytest.fixture()
def config():
    return {"user_id": "u1", "thread_id": "t1"}


# ---------------------------------------------------------------------------
# ReadMode & defaults
# ---------------------------------------------------------------------------


class TestReadModeDefaults:
    def test_default_read_mode_is_no_retrieval(self):
        assert DEFAULT_READ_MODE == ReadMode.NO_RETRIEVAL

    def test_read_mode_values(self):
        assert ReadMode.NO_RETRIEVAL.value == "no_retrieval"
        assert ReadMode.PRELOAD.value == "preload"
        assert ReadMode.POSTLOAD.value == "postload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestValidateMemoryType:
    def test_valid_type(self):
        assert _validate_memory_type("semantic") == MemoryType.SEMANTIC

    def test_invalid_type_returns_episodic(self):
        assert _validate_memory_type("invalid") == MemoryType.EPISODIC

    def test_all_valid_types(self):
        for mt in MemoryType:
            assert _validate_memory_type(mt.value) == mt


class TestFormatSearchResults:
    def test_empty(self):
        assert _format_search_results([]) == []

    def test_formats_correctly(self, sample_search_results):
        formatted = _format_search_results(sample_search_results)
        assert len(formatted) == 2
        assert formatted[0]["id"] == "r1"
        assert formatted[0]["content"] == "User prefers dark mode"
        assert formatted[0]["score"] == 0.92
        assert formatted[0]["memory_type"] == "semantic"
        assert formatted[1]["id"] == "r2"


# ---------------------------------------------------------------------------
# _do_write
# ---------------------------------------------------------------------------


class TestDoWrite:
    @pytest.mark.asyncio
    async def test_store(self, mock_store, config):
        result = await _do_write(
            mock_store, config, "store", "hello", "", MemoryType.EPISODIC, "general", None, "merge"
        )
        assert result == {"status": "stored", "memory_id": "mem-001"}
        mock_store.astore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_replace(self, mock_store, config):
        result = await _do_write(
            mock_store, config, "update", "new text", "m1", MemoryType.EPISODIC, "general", {"x": 1}, "replace"
        )
        assert result == {"status": "updated", "memory_id": "m1"}
        mock_store.aupdate.assert_awaited_once_with(config, "m1", "new text", metadata={"x": 1})

    @pytest.mark.asyncio
    async def test_update_merge(self, mock_store, config):
        mock_store.aget.return_value = MemorySearchResult(
            id="m1", content="old", metadata={"a": 1}
        )
        result = await _do_write(
            mock_store, config, "update", "new", "m1", MemoryType.EPISODIC, "general", {"b": 2}, "merge"
        )
        assert result["status"] == "updated"
        mock_store.aupdate.assert_awaited_once_with(
            config, "m1", "new", metadata={"a": 1, "b": 2}
        )

    @pytest.mark.asyncio
    async def test_update_merge_no_existing(self, mock_store, config):
        mock_store.aget.return_value = None
        result = await _do_write(
            mock_store, config, "update", "new", "m1", MemoryType.EPISODIC, "general", {"b": 2}, "merge"
        )
        assert result["status"] == "updated"
        mock_store.aupdate.assert_awaited_once_with(config, "m1", "new", metadata={"b": 2})

    @pytest.mark.asyncio
    async def test_delete(self, mock_store, config):
        result = await _do_write(
            mock_store, config, "delete", "", "m1", MemoryType.EPISODIC, "general", None, "merge"
        )
        assert result == {"status": "deleted", "memory_id": "m1"}
        mock_store.adelete.assert_awaited_once_with(config, "m1")


# ---------------------------------------------------------------------------
# memory_tool
# ---------------------------------------------------------------------------


class TestMemoryToolSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_store, mock_task_manager, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        result = json.loads(
            await memory_tool(
                action="search",
                query="preferences",
                store=mock_store,
                task_manager=mock_task_manager,
                config=config,
            )
        )
        assert len(result) == 2
        assert result[0]["id"] == "r1"
        mock_store.asearch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_empty_query_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="search", query="", store=mock_store,
                task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "query" in result["error"]

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, mock_store, mock_task_manager, config):
        mock_store.asearch.return_value = []
        await memory_tool(
            action="search", query="test", score_threshold=0.5,
            store=mock_store, task_manager=mock_task_manager, config=config,
        )
        call_kwargs = mock_store.asearch.call_args
        assert call_kwargs.kwargs.get("score_threshold") == 0.5


class TestMemoryToolStore:
    @pytest.mark.asyncio
    async def test_store_success(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="user likes Python", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "stored"
        assert result["memory_id"] == "mem-001"

    @pytest.mark.asyncio
    async def test_store_empty_content_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolUpdate:
    @pytest.mark.asyncio
    async def test_update_merge(self, mock_store, mock_task_manager, config):
        mock_store.aget.return_value = MemorySearchResult(
            id="m1", content="old", metadata={"a": 1}
        )
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="new",
                metadata={"b": 2}, write_mode="merge", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_replace(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="new",
                metadata={"b": 2}, write_mode="replace", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_no_memory_id_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="", content="new", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_no_content_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="update", memory_id="m1", content="", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolDelete:
    @pytest.mark.asyncio
    async def test_delete_success(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="m1", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_no_memory_id_error(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


class TestMemoryToolNoStore:
    @pytest.mark.asyncio
    async def test_no_store_returns_error(self, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="search", query="test",
                store=None, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "no memory store" in result["error"]


class TestMemoryToolAsyncWrite:
    @pytest.mark.asyncio
    async def test_async_store_schedules_task(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="store", content="data", async_write=True,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()
        mock_store.astore.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_async_delete_schedules_task(self, mock_store, mock_task_manager, config):
        result = json.loads(
            await memory_tool(
                action="delete", memory_id="m1", async_write=True,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert result["status"] == "scheduled"
        mock_task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_search_not_scheduled(self, mock_store, mock_task_manager, config):
        """Search is always synchronous, async_write is ignored."""
        mock_store.asearch.return_value = []
        result = json.loads(
            await memory_tool(
                action="search", query="test", async_write=True,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert isinstance(result, list)
        mock_task_manager.create_task.assert_not_called()


class TestMemoryToolExceptionHandling:
    @pytest.mark.asyncio
    async def test_store_exception_returns_error(self, mock_store, mock_task_manager, config):
        mock_store.astore.side_effect = RuntimeError("connection failed")
        result = json.loads(
            await memory_tool(
                action="store", content="data", async_write=False,
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result
        assert "connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_exception_returns_error(self, mock_store, mock_task_manager, config):
        mock_store.asearch.side_effect = RuntimeError("timeout")
        result = json.loads(
            await memory_tool(
                action="search", query="test",
                store=mock_store, task_manager=mock_task_manager, config=config,
            )
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# MemoryWriteTracker
# ---------------------------------------------------------------------------


class TestMemoryWriteTracker:
    @pytest.mark.asyncio
    async def test_track_and_wait(self):
        tracker = MemoryWriteTracker()
        completed = False

        async def dummy_write():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        task = asyncio.create_task(dummy_write())
        await tracker.track(task)
        assert tracker.pending_count == 1

        stats = await tracker.wait_for_pending(timeout=5.0)
        assert stats["status"] == "completed"
        assert completed is True

    @pytest.mark.asyncio
    async def test_empty_wait(self):
        tracker = MemoryWriteTracker()
        stats = await tracker.wait_for_pending()
        assert stats["status"] == "completed"
        assert stats["pending_writes"] == 0

    @pytest.mark.asyncio
    async def test_task_auto_discards_on_done(self):
        tracker = MemoryWriteTracker()

        async def quick():
            return

        task = asyncio.create_task(quick())
        await tracker.track(task)
        await asyncio.sleep(0.05)
        assert tracker.pending_count == 0

    @pytest.mark.asyncio
    async def test_timeout_returns_stats(self):
        tracker = MemoryWriteTracker()

        async def slow_write():
            await asyncio.sleep(10)

        task = asyncio.create_task(slow_write())
        await tracker.track(task)

        stats = await tracker.wait_for_pending(timeout=0.05)
        assert stats["status"] == "timeout"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestGetWriteTracker:
    def test_returns_same_instance(self):
        a = get_write_tracker()
        b = get_write_tracker()
        assert a is b

    def test_is_memory_write_tracker(self):
        assert isinstance(get_write_tracker(), MemoryWriteTracker)


# ---------------------------------------------------------------------------
# create_memory_preload_node
# ---------------------------------------------------------------------------


class TestPreloadNode:
    @pytest.mark.asyncio
    async def test_basic_preload(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results

        node_fn = create_memory_preload_node(mock_store, limit=5)
        state = AgentState(
            context=[Message.text_message("Tell me about my setup", role="user")]
        )
        result = await node_fn(state, config)

        assert len(result) == 1
        assert result[0].role == "system"
        text = result[0].text()
        assert "User prefers dark mode" in text
        assert "Long-term Memory Context" in text
        mock_store.asearch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_preload_no_results(self, mock_store, config):
        mock_store.asearch.return_value = []
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_no_user_message(self, mock_store, config):
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(context=[Message.text_message("I am an assistant", role="assistant")])
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_custom_query_builder(self, mock_store, config):
        mock_store.asearch.return_value = []
        custom_query = lambda state: "custom query"
        node_fn = create_memory_preload_node(mock_store, query_builder=custom_query)
        state = AgentState(
            context=[Message.text_message("irrelevant", role="user")]
        )
        await node_fn(state, config)
        call_args = mock_store.asearch.call_args
        assert call_args[0][1] == "custom query"

    @pytest.mark.asyncio
    async def test_preload_store_exception(self, mock_store, config):
        mock_store.asearch.side_effect = RuntimeError("connection lost")
        node_fn = create_memory_preload_node(mock_store)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        assert result == []

    @pytest.mark.asyncio
    async def test_preload_custom_template(self, mock_store, sample_search_results, config):
        mock_store.asearch.return_value = sample_search_results
        template = "MEMORIES:\n{memories}"
        node_fn = create_memory_preload_node(mock_store, system_prompt_template=template)
        state = AgentState(
            context=[Message.text_message("hello", role="user")]
        )
        result = await node_fn(state, config)
        text = result[0].text()
        assert text.startswith("MEMORIES:")


# ---------------------------------------------------------------------------
# get_memory_system_prompt
# ---------------------------------------------------------------------------


class TestGetMemorySystemPrompt:
    def test_no_retrieval_includes_write_instructions(self):
        prompt = get_memory_system_prompt("no_retrieval")
        assert len(prompt) > 0
        assert "do not" in prompt.lower() or "do not have access" in prompt.lower()
        assert "memory_tool" in prompt
        assert "store" in prompt.lower()

    def test_preload_includes_read_and_write(self):
        prompt = get_memory_system_prompt("preload")
        assert "memory context" in prompt.lower()
        assert "memory_tool" in prompt
        assert "store" in prompt.lower()
        assert len(prompt) > 0

    def test_postload_returns_text(self):
        prompt = get_memory_system_prompt("postload")
        assert "memory_tool" in prompt
        assert "search" in prompt
        assert "store" in prompt

    def test_default_mode_includes_write(self):
        prompt = get_memory_system_prompt()
        assert "memory_tool" in prompt

    def test_unknown_mode_returns_empty(self):
        assert get_memory_system_prompt("unknown_mode") == ""


# ---------------------------------------------------------------------------
# Tool schema generation
# ---------------------------------------------------------------------------


class TestMemoryToolSchema:
    def test_tool_decorator_metadata(self):
        assert memory_tool._py_tool_name == "memory_tool"
        assert "memory" in memory_tool._py_tool_tags

    def test_tool_description(self):
        assert "Search" in memory_tool._py_tool_description
        assert "store" in memory_tool._py_tool_description.lower()
