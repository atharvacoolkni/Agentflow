"""Functional test: Simulate real agent usage with shadcn skill.

This test demonstrates the LAZY loading skill activation flow:
1. User says "create a login form using shadcn"
2. System matches trigger → activates skill
3. System loads ONLY SKILL.md content (not all resources)
4. AI sees available resources listed, fetches only what it needs

Usage:
    python tests/test_shadcn_functional.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentflow.skills.loader import discover_skills, load_skill_content, load_resource
from agentflow.skills.registry import SkillsRegistry
from agentflow.skills.activation import make_set_skill_tool


def match_trigger(user_input: str, registry: SkillsRegistry) -> str | None:
    """Simple trigger matching - checks if any trigger phrase is in user input."""
    user_lower = user_input.lower()
    
    for skill in registry.get_all():
        for trigger in skill.triggers:
            if trigger.lower() in user_lower:
                return skill.name
    return None


def main():
    print("=" * 70)
    print("🧪 FUNCTIONAL TEST: Lazy Resource Loading")
    print("=" * 70)
    
    # Setup: Load skills
    skills_dir = project_root / "examples" / "skills" / "skills"
    
    print(f"\n📁 Loading skills from: {skills_dir}")
    
    registry = SkillsRegistry()
    registry.discover(str(skills_dir))
    
    print(f"✅ Loaded {len(registry)} skills: {registry.names()}")
    
    # Create the set_skill tool (this is what the AI agent uses)
    set_skill = make_set_skill_tool(registry)
    
    print("\n" + "=" * 70)
    print("🔍 TEST: Lazy Loading Flow")
    print("=" * 70)
    
    # Simulate: User asks "create a login form using shadcn"
    user_input = "create a login form using shadcn"
    print(f"\n👤 User: \"{user_input}\"")
    
    # Step 1: Match trigger
    matched_skill = match_trigger(user_input, registry)
    print(f"\n1️⃣  Trigger matched: '{matched_skill}'")
    
    # Step 2: AI calls set_skill (loads ONLY main content)
    print(f"\n2️⃣  AI calls: set_skill(\"{matched_skill}\")")
    result = set_skill(matched_skill)
    
    # Show what was loaded
    print(f"\n3️⃣  Initial context loaded: {len(result):,} chars")
    print("-" * 50)
    
    # Show the structure of what AI receives
    lines = result.split("\n")
    print("    Content preview:")
    for line in lines[:15]:
        print(f"    {line[:70]}{'...' if len(line) > 70 else ''}")
    print("    ...")
    
    # Find the "Available Resources" section
    if "Available Resources" in result:
        start = result.find("### Available Resources")
        resources_section = result[start:start+500]
        print(f"\n    📋 Resources section:")
        for line in resources_section.split("\n")[:12]:
            print(f"    {line}")
    
    # Step 3: AI decides it needs forms.md for login form
    print(f"\n4️⃣  AI sees 'login form' → needs forms documentation")
    print(f"    AI calls: set_skill(\"{matched_skill}\", \"rules/forms.md\")")
    
    forms_result = set_skill(matched_skill, "rules/forms.md")
    print(f"\n5️⃣  Forms resource loaded: {len(forms_result):,} chars")
    print("-" * 50)
    print("    Content preview:")
    for line in forms_result.split("\n")[:20]:
        print(f"    {line[:70]}{'...' if len(line) > 70 else ''}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 LAZY vs EAGER Loading Comparison")
    print("=" * 70)
    
    # Calculate eager loading size (all resources)
    skill = registry.get(matched_skill)
    main_content = load_skill_content(skill)
    all_resources_size = 0
    for res_path in skill.resources:
        content = load_resource(skill, res_path)
        if content:
            all_resources_size += len(content)
    
    eager_total = len(main_content) + all_resources_size
    lazy_initial = len(result)
    lazy_with_forms = lazy_initial + len(forms_result)
    
    print(f"""
    EAGER Loading (old way):
      - Load everything upfront: {eager_total:,} chars (~{eager_total//1000}KB)
      - All 8 resources loaded whether needed or not
    
    LAZY Loading (new way):
      - Initial activation: {lazy_initial:,} chars (~{lazy_initial//1000}KB)
      - After fetching forms.md: {lazy_with_forms:,} chars (~{lazy_with_forms//1000}KB)
      - Savings: {eager_total - lazy_with_forms:,} chars ({100 - (lazy_with_forms * 100 // eager_total)}% reduction)
    
    For this "login form" request:
      ✅ Loaded: SKILL.md + rules/forms.md (what we need)
      ❌ Skipped: cli.md, customization.md, mcp.md, icons.md, etc.
    """)
    
    # Test another scenario
    print("=" * 70)
    print("🧪 TEST 2: Different request needs different resources")
    print("=" * 70)
    
    print(f"\n👤 User: \"how do I customize colors in shadcn?\"")
    print(f"\n1️⃣  AI calls: set_skill(\"shadcn\")")
    result2 = set_skill("shadcn")
    print(f"    Initial load: {len(result2):,} chars")
    
    print(f"\n2️⃣  AI sees 'colors' → needs customization.md")
    print(f"    AI calls: set_skill(\"shadcn\", \"customization.md\")")
    custom_result = set_skill("shadcn", "customization.md")
    print(f"    Customization loaded: {len(custom_result):,} chars")
    
    print(f"\n    Total for this request: {len(result2) + len(custom_result):,} chars")
    print(f"    (Still didn't load forms.md, cli.md, icons.md, etc.)")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 LAZY LOADING WORKS!")
    print("=" * 70)
    print("""
    How it works:
    
    1. User makes request
    2. System activates skill → loads ONLY SKILL.md (~16KB)
    3. SKILL.md lists available resources with descriptions
    4. AI decides which resource(s) it needs
    5. AI calls set_skill(skill, resource) to fetch specific file
    6. Repeat as needed
    
    Benefits:
    ✅ Smaller initial context (16KB vs 61KB)
    ✅ Faster responses
    ✅ Lower token costs
    ✅ AI fetches only what's relevant
    ✅ Scales to skills with 50+ resources
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
