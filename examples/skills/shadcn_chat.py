"""Shadcn Skills Chat - terminal REPL using a normal Agent node.

This version uses the standard Agentflow graph pattern:
1. Agent node (LLM)
2. Tool node (includes set_skill from SkillConfig)
3. Conditional routing between MAIN and TOOL

Run:
    python examples/skills/shadcn_chat.py
"""

import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from agentflow.graph import Agent, StateGraph
from agentflow.skills import SkillConfig
from agentflow.skills.registry import SkillsRegistry
from agentflow.state import AgentState, Message
from agentflow.state.message_context_manager import MessageContextManager
from agentflow.utils.constants import END

load_dotenv()

SKILLS_DIR = str(Path(__file__).parent / "skills")
MODEL = os.getenv("MODEL", "google/gemini-2.5-flash")


def discover_skill_names(skills_dir: str) -> list[str]:
    """Read skill names for banner output."""
    registry = SkillsRegistry()
    registry.discover(skills_dir)
    return registry.names()


agent = Agent(
    model=MODEL,
    system_prompt=[
        {
            "role": "system",
            "content": (
                "You are a helpful UI assistant. "
                "When a request matches an available skill, call set_skill() first "
                "and use the loaded instructions to answer with concrete code and steps."
            ),
        }
    ],
    skills=SkillConfig(
        skills_dir=SKILLS_DIR,
        inject_trigger_table=True,
        hot_reload=True,
    ),
    trim_context=True,
)

tool_node = agent.get_tool_node()
if tool_node is None:
    raise RuntimeError("Tool node was not created. Check skills configuration.")


def should_use_tools(state: AgentState) -> str:
    """Route MAIN -> TOOL when assistant returns tool calls."""
    if not state.context:
        return END

    last = state.context[-1]

    if last.role == "assistant" and getattr(last, "tools_calls", None):
        return "TOOL"

    if last.role == "tool":
        return "MAIN"

    return END


graph = StateGraph(context_manager=MessageContextManager(max_messages=20))
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()


def print_banner(thread_id: str) -> None:
    """Print startup information."""
    skills = discover_skill_names(SKILLS_DIR)
    width = 64
    print("\n" + "=" * width)
    print("  Shadcn Skills Chat - Agent + Skill Tool + Terminal REPL")
    print("=" * width)
    print(f"  Model : {MODEL}")
    print(f"  Thread: {thread_id}")
    print(f"  Skills: {', '.join(skills) if skills else '(none found)'}")
    print()
    print("Try prompts like:")
    print("  - create a shadcn login form")
    print("  - style this dialog with proper shadcn patterns")
    print("  - how to add a sidebar in shadcn")
    print("Exit: type 'quit' or 'exit' or Ctrl-C")
    print("=" * width + "\n")


def print_loaded_skills(messages: list[Message]) -> None:
    """Print which skill/resource was loaded in the current turn."""
    for msg in messages:
        if msg.role != "tool":
            continue
        text = msg.text() or ""
        if text.startswith("## SKILL:"):
            skill_line = text.splitlines()[0].replace("## SKILL:", "").strip()
            print(f"  [skill loaded] {skill_line}")
        elif text.startswith("## Resource:"):
            resource_line = text.splitlines()[0].replace("## Resource:", "").strip()
            print(f"  [resource loaded] {resource_line}")


def main() -> None:
    """Run terminal chat loop for testing skills."""
    if not any(os.getenv(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")):
        print("Warning: no LLM API key found (GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY).")

    thread_id = f"shadcn-skills-{uuid4().hex[:8]}"
    print_banner(thread_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            result = app.invoke(
                {"messages": [Message.text_message(user_input)]},
                config={"thread_id": thread_id, "recursion_limit": 20},
            )
        except Exception as exc:
            print(f"\n[error] {exc}\n")
            continue

        print_loaded_skills(result["messages"])

        for msg in reversed(result["messages"]):
            if msg.role == "assistant" and msg.text():
                print(f"\nAssistant: {msg.text()}\n")
                break


if __name__ == "__main__":
    main()
