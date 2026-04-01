"""Quick validation script for shadcn skill loading."""
import sys
import os
from pathlib import Path
import tempfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from agentflow.skills.loader import discover_skills, load_skill_content, load_resource
from agentflow.skills.registry import SkillsRegistry

# Minimal shadcn skill
SHADCN_SKILL_MD = '''---
name: shadcn
description: Manages shadcn components and projects - adding, searching, fixing UI.
metadata:
  triggers:
    - shadcn init
    - add component
  resources:
    - cli.md
    - rules/forms.md
  tags:
    - ui
    - react
  priority: 10
---

# shadcn/ui

A framework for building ui components.
'''

CLI_MD = '# CLI Reference\n\n`npx shadcn@latest init`'
FORMS_MD = '# Forms\n\nUse FieldGroup + Field.'

def main():
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "shadcn"
        skill_dir.mkdir()
        rules_dir = skill_dir / "rules"
        rules_dir.mkdir()
        
        (skill_dir / "SKILL.md").write_text(SHADCN_SKILL_MD)
        (skill_dir / "cli.md").write_text(CLI_MD)
        (rules_dir / "forms.md").write_text(FORMS_MD)
        
        # Test discovery
        skills = discover_skills(tmp)
        assert len(skills) == 1, f"Expected 1 skill, got {len(skills)}"
        print(f"✅ Discovered {len(skills)} skill(s)")
        
        skill = skills[0]
        assert skill.name == "shadcn"
        print(f"✅ Name: {skill.name}")
        
        assert "Manages shadcn" in skill.description
        print(f"✅ Description parsed correctly")
        
        assert len(skill.triggers) == 2
        print(f"✅ Triggers: {skill.triggers}")
        
        assert len(skill.resources) == 2
        assert "cli.md" in skill.resources
        assert "rules/forms.md" in skill.resources
        print(f"✅ Resources: {skill.resources}")
        
        assert skill.tags == {"ui", "react"}
        print(f"✅ Tags: {skill.tags}")
        
        assert skill.priority == 10
        print(f"✅ Priority: {skill.priority}")
        
        # Test content loading
        content = load_skill_content(skill)
        assert "# shadcn/ui" in content
        assert "---" not in content[:10]  # Frontmatter stripped
        print(f"✅ Content loaded: {len(content)} chars (frontmatter stripped)")
        
        # Test resource loading
        cli = load_resource(skill, "cli.md")
        assert cli is not None
        assert "CLI Reference" in cli
        print(f"✅ cli.md loaded: {len(cli)} chars")
        
        forms = load_resource(skill, "rules/forms.md")
        assert forms is not None
        assert "FieldGroup" in forms
        print(f"✅ rules/forms.md (nested) loaded: {len(forms)} chars")
        
        # Test path traversal blocked
        bad = load_resource(skill, "../outside.md")
        assert bad is None
        print(f"✅ Path traversal blocked")
        
        # Test registry
        registry = SkillsRegistry()
        registry.register(skill)
        found = registry.get("shadcn")
        assert found is not None
        assert found.name == "shadcn"
        print(f"✅ Registry lookup works")
        
        # Test tag filtering (using get_all with tags parameter)
        ui_skills = registry.get_all(tags={"ui"})
        assert len(ui_skills) == 1
        print(f"✅ Tag filter works (get_all with tags)")
        
        # Test trigger table (returns markdown string, not dict)
        trigger_table = registry.build_trigger_table()
        assert "shadcn" in trigger_table
        assert "shadcn init" in trigger_table
        print(f"✅ Trigger table generated: {len(trigger_table)} chars")

    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)
    print("\nYour Agentflow skills system can handle real-world")
    print("skills like shadcn/ui with:")
    print("  - Multi-file resources")
    print("  - Nested directories (rules/)")
    print("  - YAML frontmatter parsing")
    print("  - Content loading")
    print("  - Registry integration")
    print("  - Trigger table generation")

if __name__ == "__main__":
    main()
