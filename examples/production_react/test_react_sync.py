"""Unit tests for the production_react/react_sync.py agent.

NOTE: react_sync.py is a runnable script (it calls app.invoke at module level),
so we cannot import from it directly without triggering a real LLM call.
Instead, these tests define their own copies of the two functions under test
(get_weather and should_use_tools). This also illustrates the best practice:
extract your tool functions and routing logic into a separate module so they
can be imported and tested cleanly.

Test coverage:
    1. get_weather()     — the tool function (pure business logic)
    2. should_use_tools() — the routing / conditional-edge function
    3. Graph flow        — end-to-end wiring using TestAgent (no real LLM)
"""

import pytest

from agentflow.state import AgentState, Message
from agentflow.utils.constants import END


# ---------------------------------------------------------------------------
# Reproduced from react_sync.py — functions under test
# ---------------------------------------------------------------------------


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
) -> str:
    """
    Get the current weather for a specific location.
    Injectable parameters (tool_call_id, state) are provided automatically
    by the framework at runtime; they don't affect the return value.
    """
    return f"The weather in {location} is sunny"


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return "MAIN"

    return END


# ===========================================================================
# 1. Tool function tests
# ===========================================================================


class TestGetWeather:
    """Tests for the get_weather tool function."""

    def test_returns_location_in_response(self):
        result = get_weather("New York City")
        assert "New York City" in result

    def test_returns_weather_description(self):
        result = get_weather("London")
        assert "sunny" in result.lower()

    def test_different_locations_give_different_responses(self):
        assert get_weather("Paris") != get_weather("Tokyo")

    def test_injectable_tool_call_id_does_not_change_output(self):
        without = get_weather("Berlin")
        with_id = get_weather("Berlin", tool_call_id="call_abc123")
        assert without == with_id

    def test_injectable_state_does_not_change_output(self):
        state = AgentState()
        state.context = [Message.text_message("hello", role="user")]
        without = get_weather("Sydney")
        with_state = get_weather("Sydney", state=state)
        assert without == with_state

    def test_all_injectable_params_together(self):
        state = AgentState()
        result = get_weather("Chicago", tool_call_id="call_xyz", state=state)
        assert "Chicago" in result


# ===========================================================================
# 2. Routing logic tests
# ===========================================================================


class TestShouldUseTools:
    """Tests for the should_use_tools conditional routing function."""

    def test_empty_context_routes_to_tool(self):
        """No messages yet — default to checking for tools."""
        state = AgentState()
        assert should_use_tools(state) == "TOOL"

    def test_empty_list_routes_to_tool(self):
        state = AgentState()
        state.context = []
        assert should_use_tools(state) == "TOOL"

    def test_assistant_with_tool_calls_routes_to_tool(self):
        """Agent decided to call a tool — send to TOOL node."""
        state = AgentState()
        msg = Message.text_message("I'll check the weather for you", role="assistant")
        msg.tools_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather"}}
        ]
        state.context = [msg]
        assert should_use_tools(state) == "TOOL"

    def test_assistant_without_tool_calls_ends(self):
        """Agent gave a final answer — conversation is done."""
        state = AgentState()
        msg = Message.text_message("The weather in NYC is sunny.", role="assistant")
        msg.tools_calls = []
        state.context = [msg]
        assert should_use_tools(state) == END

    def test_tool_result_routes_back_to_main(self):
        """Tool ran and returned a result — go back to MAIN for final response."""
        state = AgentState()
        msg = Message.text_message("The weather in NYC is sunny", role="tool")
        state.context = [msg]
        assert should_use_tools(state) == "MAIN"

    def test_user_message_ends(self):
        """Bare user message with no prior context defaults to END."""
        state = AgentState()
        msg = Message.text_message("What's the weather in Paris?", role="user")
        state.context = [msg]
        assert should_use_tools(state) == END

    def test_system_message_ends(self):
        state = AgentState()
        msg = Message.text_message("You are a helpful assistant.", role="system")
        state.context = [msg]
        assert should_use_tools(state) == END

    def test_full_turn_last_is_tool_result(self):
        """Complete turn: user → assistant(tool call) → tool result → MAIN."""
        state = AgentState()
        user_msg = Message.text_message("Weather in NYC?", role="user")
        assistant_msg = Message.text_message("Let me check", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_1", "function": {"name": "get_weather"}}]
        tool_msg = Message.text_message("The weather in NYC is sunny", role="tool")
        state.context = [user_msg, assistant_msg, tool_msg]
        assert should_use_tools(state) == "MAIN"

    def test_full_turn_last_is_assistant_with_tools(self):
        """After user sends message, assistant decides to call tool → TOOL."""
        state = AgentState()
        user_msg = Message.text_message("Weather in NYC?", role="user")
        assistant_msg = Message.text_message("I'll check", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_1", "function": {"name": "get_weather"}}]
        state.context = [user_msg, assistant_msg]
        assert should_use_tools(state) == "TOOL"

    def test_final_answer_after_tool_ends(self):
        """After tool result, assistant gives final answer → END."""
        state = AgentState()
        user_msg = Message.text_message("Weather in NYC?", role="user")
        assistant_msg = Message.text_message("I'll check", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_1", "function": {"name": "get_weather"}}]
        tool_msg = Message.text_message("Sunny", role="tool")
        final_msg = Message.text_message("The weather is sunny!", role="assistant")
        final_msg.tools_calls = []
        state.context = [user_msg, assistant_msg, tool_msg, final_msg]
        assert should_use_tools(state) == END


# ===========================================================================
# 3. Graph flow tests  (no real LLM — uses TestAgent)
# ===========================================================================


class TestGraphFlow:
    """
    End-to-end graph tests using TestAgent and MockToolRegistry.

    TestAgent replaces the real Agent and returns predefined responses,
    so tests run instantly with no API keys required.
    """

    @pytest.mark.asyncio
    async def test_graph_executes_and_returns_messages(self):
        """Graph compiles, runs, and returns a messages dict."""
        from agentflow.checkpointer import InMemoryCheckpointer
        from agentflow.graph import StateGraph, ToolNode
        from agentflow.testing import MockToolRegistry, TestAgent

        mock_tools = MockToolRegistry()
        mock_tools.register(
            "get_weather",
            lambda location, **_: f"The weather in {location} is sunny",
        )
        tool_node = ToolNode(mock_tools.get_tool_list())

        # TestAgent returns a plain text response (no tool calls),
        # so should_use_tools will route to END after the first turn.
        test_agent = TestAgent(responses=["The weather in New York City is sunny."])

        graph = StateGraph()
        graph.add_node("MAIN", test_agent)
        graph.add_node("TOOL", tool_node)
        graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
        graph.add_edge("TOOL", "MAIN")
        graph.set_entry_point("MAIN")

        app = graph.compile(checkpointer=InMemoryCheckpointer())
        result = await app.ainvoke(
            {"messages": [Message.text_message("What is the weather in New York City?")]},
            config={"thread_id": "test-001"},
        )

        assert "messages" in result
        assert len(result["messages"]) > 0
        test_agent.assert_called()

    @pytest.mark.asyncio
    async def test_agent_called_exactly_once_for_simple_query(self):
        """Without tool calls in the response, the agent runs once then ends."""
        from agentflow.graph import StateGraph, ToolNode
        from agentflow.testing import MockToolRegistry, TestAgent

        mock_tools = MockToolRegistry()
        mock_tools.register("get_weather", lambda location, **_: f"Sunny in {location}")
        tool_node = ToolNode(mock_tools.get_tool_list())

        test_agent = TestAgent(responses=["It is sunny in Paris."])

        graph = StateGraph()
        graph.add_node("MAIN", test_agent)
        graph.add_node("TOOL", tool_node)
        graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
        graph.add_edge("TOOL", "MAIN")
        graph.set_entry_point("MAIN")

        app = graph.compile()
        await app.ainvoke(
            {"messages": [Message.text_message("Weather in Paris?")]},
            config={"thread_id": "test-002"},
        )

        test_agent.assert_called_times(1)

    @pytest.mark.asyncio
    async def test_graph_result_contains_assistant_message(self):
        """The final result must include the agent's response message."""
        from agentflow.graph import StateGraph, ToolNode
        from agentflow.testing import MockToolRegistry, TestAgent

        mock_tools = MockToolRegistry()
        mock_tools.register("get_weather", lambda location, **_: f"Sunny in {location}")
        tool_node = ToolNode(mock_tools.get_tool_list())

        expected_response = "The weather in Tokyo is sunny!"
        test_agent = TestAgent(responses=[expected_response])

        graph = StateGraph()
        graph.add_node("MAIN", test_agent)
        graph.add_node("TOOL", tool_node)
        graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
        graph.add_edge("TOOL", "MAIN")
        graph.set_entry_point("MAIN")

        app = graph.compile()
        result = await app.ainvoke(
            {"messages": [Message.text_message("Weather in Tokyo?")]},
            config={"thread_id": "test-003"},
        )

        messages = result["messages"]
        assistant_messages = [m for m in messages if hasattr(m, "role") and m.role == "assistant"]
        assert len(assistant_messages) > 0

    @pytest.mark.asyncio
    async def test_get_weather_tool_callable_via_tool_node(self):
        """Verify the tool function is correctly registered and callable through ToolNode."""
        from agentflow.graph import ToolNode
        from agentflow.testing import MockToolRegistry

        mock_tools = MockToolRegistry()
        mock_tools.register(
            "get_weather",
            lambda location, **_: get_weather(location),
        )

        tool_node = ToolNode(mock_tools.get_tool_list())
        result = await tool_node.invoke(
            name="get_weather",
            args={"location": "New York City"},
            tool_call_id="call_test_123",
            config={},
            state=AgentState(),
        )

        mock_tools.assert_called("get_weather")
        mock_tools.assert_called_with("get_weather", location="New York City")
        assert "New York City" in result.text()
        assert "sunny" in result.text().lower()
