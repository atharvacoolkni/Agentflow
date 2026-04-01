"""Quick test to show trigger table format."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentflow.skills.registry import SkillsRegistry

reg = SkillsRegistry()
reg.discover("examples/skills/skills")
print(reg.build_trigger_table())
