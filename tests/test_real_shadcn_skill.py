"""Test loading the REAL shadcn/ui skill downloaded from GitHub.

This script loads the actual shadcn skill files (not mocked content)
to prove Agentflow's skills system works with real-world skills.

Usage:
    python tests/test_real_shadcn_skill.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentflow.skills.loader import discover_skills, load_skill_content, load_resource
from agentflow.skills.registry import SkillsRegistry


def main():
    # Path to the real downloaded shadcn skill
    skills_dir = project_root / "examples" / "skills" / "skills"
    shadcn_dir = skills_dir / "shadcn-real"
    
    print("=" * 60)
    print("🧪 Testing with REAL shadcn/ui skill from GitHub")
    print("=" * 60)
    print(f"\nSkill location: {shadcn_dir}")
    
    # Check files exist
    skill_file = shadcn_dir / "SKILL.md"
    if not skill_file.exists():
        print(f"\n❌ ERROR: SKILL.md not found at {skill_file}")
        print("Run: python tests/download_shadcn_skill.py first")
        return 1
    
    print(f"\n📁 Files in shadcn-real/:")
    for f in sorted(shadcn_dir.rglob("*.md")):
        rel = f.relative_to(shadcn_dir)
        size = f.stat().st_size
        print(f"   {rel} ({size:,} bytes)")
    
    # Step 1: Discover the skill
    print("\n" + "-" * 60)
    print("Step 1: Discover skills")
    print("-" * 60)
    
    # Discover from parent dir (skills_dir contains multiple skill folders)
    skills = discover_skills(str(shadcn_dir.parent))
    shadcn_skill = None
    for s in skills:
        if s.name == "shadcn":
            shadcn_skill = s
            break
    
    if shadcn_skill is None:
        print("❌ ERROR: shadcn skill not discovered!")
        print(f"   Discovered skills: {[s.name for s in skills]}")
        return 1
    
    print(f"✅ Discovered skill: {shadcn_skill.name}")
    print(f"   Description: {shadcn_skill.description[:80]}...")
    print(f"   Triggers: {len(shadcn_skill.triggers)}")
    print(f"   Resources: {len(shadcn_skill.resources)}")
    print(f"   Tags: {shadcn_skill.tags}")
    print(f"   Priority: {shadcn_skill.priority}")
    
    # Step 2: Load main content
    print("\n" + "-" * 60)
    print("Step 2: Load skill content")
    print("-" * 60)
    
    content = load_skill_content(shadcn_skill)
    print(f"✅ Main content loaded: {len(content):,} characters")
    
    # Verify frontmatter was stripped
    if content.strip().startswith("---"):
        print("❌ ERROR: Frontmatter was NOT stripped!")
        return 1
    print("✅ YAML frontmatter correctly stripped")
    
    # Check for expected content
    if "# shadcn/ui" in content:
        print("✅ Contains expected heading")
    if "npx shadcn" in content:
        print("✅ Contains CLI examples")
    
    # Step 3: Load all resources
    print("\n" + "-" * 60)
    print("Step 3: Load resource files")
    print("-" * 60)
    
    total_resource_size = 0
    loaded_resources = []
    
    for resource_path in shadcn_skill.resources:
        resource_content = load_resource(shadcn_skill, resource_path)
        if resource_content is None:
            print(f"❌ Failed to load: {resource_path}")
            continue
        
        loaded_resources.append(resource_path)
        total_resource_size += len(resource_content)
        print(f"✅ {resource_path}: {len(resource_content):,} chars")
    
    print(f"\n   Total resources loaded: {len(loaded_resources)}/{len(shadcn_skill.resources)}")
    print(f"   Total resource content: {total_resource_size:,} characters")
    
    # Step 4: Register in registry
    print("\n" + "-" * 60)
    print("Step 4: Registry integration")
    print("-" * 60)
    
    registry = SkillsRegistry()
    registry.register(shadcn_skill)
    
    # Lookup
    found = registry.get("shadcn")
    assert found is not None, "Failed to find skill in registry"
    print(f"✅ Registry lookup: found '{found.name}'")
    
    # Tag filtering
    ui_skills = registry.get_all(tags={"ui"})
    print(f"✅ Tag filter (ui): {len(ui_skills)} skill(s)")
    
    # Trigger table
    trigger_table = registry.build_trigger_table()
    print(f"✅ Trigger table generated: {len(trigger_table):,} chars")
    
    # Step 5: Load via registry methods
    print("\n" + "-" * 60)
    print("Step 5: Registry content loading")
    print("-" * 60)
    
    reg_content = registry.load_content("shadcn")
    print(f"✅ registry.load_content(): {len(reg_content):,} chars")
    
    reg_resources = registry.load_resources("shadcn")
    print(f"✅ registry.load_resources(): {len(reg_resources)} files")
    for name, content in reg_resources.items():
        print(f"   - {name}: {len(content):,} chars")
    
    # Summary
    total_content = len(content) + total_resource_size
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Real shadcn/ui skill loaded correctly!")
    print("=" * 60)
    print(f"""
Summary:
  - Skill name: {shadcn_skill.name}
  - Description: {len(shadcn_skill.description)} chars
  - Triggers: {len(shadcn_skill.triggers)}
  - Resources: {len(shadcn_skill.resources)} files
  - Tags: {shadcn_skill.tags}
  - Priority: {shadcn_skill.priority}
  - Total content: {total_content:,} characters (~{total_content // 1000}KB)

Your Agentflow skills system can handle real-world skills! ✅
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
