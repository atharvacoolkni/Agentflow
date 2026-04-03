"""Agentflow Skills — dynamic skill injection for agents.

Usage::

    from agentflow.skills import SkillConfig

    agent = Agent(
        model="gpt-4o",
        system_prompt=[{"role": "system", "content": "You are helpful."}],
        skills=SkillConfig(skills_dir="./skills/"),
    )
"""

from .models import SkillConfig, SkillMeta
from .registry import SkillsRegistry


__all__ = [
    "SkillConfig",
    "SkillMeta",
    "SkillsRegistry",
]
