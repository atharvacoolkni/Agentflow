"""Real-world test: Load the shadcn/ui skill (14 files) to validate the skills system.

This test validates that Agentflow's skills system can handle production-quality
skills like shadcn/ui's skill (from skills.sh), which has:
- 1 SKILL.md with YAML frontmatter
- 8 resource files (cli.md, customization.md, mcp.md, rules/*.md)

The skill content is embedded directly from:
https://github.com/shadcn-ui/ui/tree/main/skills/shadcn
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from agentflow.skills.loader import (
    discover_skills,
    load_resource,
    load_skill_content,
)
from agentflow.skills.registry import SkillsRegistry


# ────────────────────────────────────────────────────────────────────────────
# Shadcn Skill Content (from https://github.com/shadcn-ui/ui/tree/main/skills/shadcn)
# ────────────────────────────────────────────────────────────────────────────

SHADCN_SKILL_MD = '''\
---
name: shadcn
description: Manages shadcn components and projects — adding, searching, fixing, debugging, styling, and composing UI. Provides project context, component docs, and usage examples. Applies when working with shadcn/ui, component registries, presets, --preset codes, or any project with a components.json file. Also triggers for "shadcn init", "create an app with --preset", or "switch to --preset".
metadata:
  triggers:
    - shadcn init
    - add component
    - create an app with --preset
    - switch to --preset
    - components.json
    - shadcn/ui
  resources:
    - cli.md
    - customization.md
    - mcp.md
    - rules/forms.md
    - rules/composition.md
    - rules/styling.md
    - rules/icons.md
    - rules/base-vs-radix.md
  tags:
    - ui
    - components
    - tailwind
    - react
  priority: 10
---

# shadcn/ui

A framework for building ui, components and design systems. Components are added as source code to the user's project via the CLI.

> **IMPORTANT:** Run all CLI commands using the project's package runner: `npx shadcn@latest`, `pnpm dlx shadcn@latest`, or `bunx --bun shadcn@latest`.

## Principles

1. **Use existing components first.** Use `npx shadcn@latest search` to check registries before writing custom UI.
2. **Compose, don't reinvent.** Settings page = Tabs + Card + form controls.
3. **Use built-in variants before custom styles.** `variant="outline"`, `size="sm"`, etc.
4. **Use semantic colors.** `bg-primary`, `text-muted-foreground` — never raw values like `bg-blue-500`.

## Quick Reference

```bash
# Create a new project.
npx shadcn@latest init --name my-app --preset base-nova

# Add components.
npx shadcn@latest add button card dialog

# Search registries.
npx shadcn@latest search @shadcn -q "sidebar"
```
'''

CLI_MD = '''\
# shadcn CLI Reference

Configuration is read from `components.json`.

## Commands

### `init` — Initialize or create a project

```bash
npx shadcn@latest init [components...] [options]
```

| Flag                    | Description                               |
| ----------------------- | ----------------------------------------- |
| `--template <template>` | Template (next, start, vite)              |
| `--preset [name]`       | Preset configuration                      |
| `--yes`                 | Skip confirmation prompt                  |

### `add` — Add components

```bash
npx shadcn@latest add [components...] [options]
```

| Flag            | Description                              |
| --------------- | ---------------------------------------- |
| `--overwrite`   | Overwrite existing files                 |
| `--dry-run`     | Preview changes without writing files    |
| `--diff [path]` | Show diffs                               |
'''

CUSTOMIZATION_MD = '''\
# Customization & Theming

Components reference semantic CSS variable tokens. Change the variables to change every component.

## Color Variables

| Variable                         | Purpose                          |
| -------------------------------- | -------------------------------- |
| `--background` / `--foreground`  | Page background and default text |
| `--primary` / `--primary-foreground` | Primary buttons and actions  |
| `--muted` / `--muted-foreground` | Muted/disabled states            |

Colors use OKLCH: `--primary: oklch(0.205 0 0)`.

## Dark Mode

Class-based toggle via `.dark` on the root element.

```tsx
import { ThemeProvider } from "next-themes"

<ThemeProvider attribute="class" defaultTheme="system" enableSystem>
  {children}
</ThemeProvider>
```
'''

MCP_MD = '''\
# shadcn MCP Server

The CLI includes an MCP server that lets AI assistants search, browse, view, and install components.

## Setup

```bash
shadcn mcp        # start the MCP server (stdio)
shadcn mcp init   # write config for your editor
```

## Tools

### `shadcn:search_items_in_registries`

Fuzzy search across registries.

**Input:** `registries` (string[]), `query` (string)

### `shadcn:view_items_in_registries`

View item details including full file contents.

**Input:** `items` (string[])
'''

FORMS_MD = '''\
# Forms & Inputs

## Forms use FieldGroup + Field

Always use `FieldGroup` + `Field` — never raw `div` with `space-y-*`:

```tsx
<FieldGroup>
  <Field>
    <FieldLabel htmlFor="email">Email</FieldLabel>
    <Input id="email" type="email" />
  </Field>
</FieldGroup>
```

## Option sets (2–7 choices) use ToggleGroup

Don't manually loop `Button` components with active state.

**Correct:**

```tsx
<ToggleGroup spacing={2}>
  <ToggleGroupItem value="daily">Daily</ToggleGroupItem>
  <ToggleGroupItem value="weekly">Weekly</ToggleGroupItem>
</ToggleGroup>
```
'''

COMPOSITION_MD = '''\
# Component Composition

## Items always inside their Group component

Never render items directly inside the content container.

**Correct:**

```tsx
<SelectContent>
  <SelectGroup>
    <SelectItem value="apple">Apple</SelectItem>
  </SelectGroup>
</SelectContent>
```

## Dialog, Sheet, and Drawer always need a Title

`DialogTitle`, `SheetTitle`, `DrawerTitle` are required for accessibility.

```tsx
<DialogContent>
  <DialogHeader>
    <DialogTitle>Edit Profile</DialogTitle>
  </DialogHeader>
</DialogContent>
```
'''

STYLING_MD = '''\
# Styling & Customization

## Semantic colors

**Incorrect:**

```tsx
<div className="bg-blue-500 text-white">
```

**Correct:**

```tsx
<div className="bg-primary text-primary-foreground">
```

## No space-x-* / space-y-*

Use `gap-*` instead. `space-y-4` → `flex flex-col gap-4`.

## Use cn() for conditional classes

```tsx
import { cn } from "@/lib/utils"

<div className={cn("flex", isActive && "bg-primary")}>
```
'''

ICONS_MD = '''\
# Icons

## Icons in Button use data-icon attribute

Add `data-icon="inline-start"` or `data-icon="inline-end"` to the icon.

**Correct:**

```tsx
<Button>
  <SearchIcon data-icon="inline-start"/>
  Search
</Button>
```

## No sizing classes on icons inside components

Components handle icon sizing via CSS. Don't add `size-4` or `w-4 h-4`.
'''

BASE_VS_RADIX_MD = '''\
# Base vs Radix

API differences between `base` and `radix`. Check the `base` field from `npx shadcn@latest info`.

## Composition: asChild (radix) vs render (base)

**Correct (radix):**

```tsx
<DialogTrigger asChild>
  <Button>Open</Button>
</DialogTrigger>
```

**Correct (base):**

```tsx
<DialogTrigger render={<Button />}>Open</DialogTrigger>
```

## ToggleGroup

Base uses a `multiple` boolean prop. Radix uses `type="single"` or `type="multiple"`.

## Slider

Base accepts a plain number. Radix always requires an array.
'''


# ────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def shadcn_skill_dir(tmp_path: Path) -> Path:
    """Create the full shadcn skill directory structure with 9 files."""
    skill_dir = tmp_path / "shadcn"
    skill_dir.mkdir()

    # Create rules subdirectory
    rules_dir = skill_dir / "rules"
    rules_dir.mkdir()

    # Write all files
    (skill_dir / "SKILL.md").write_text(SHADCN_SKILL_MD, encoding="utf-8")
    (skill_dir / "cli.md").write_text(CLI_MD, encoding="utf-8")
    (skill_dir / "customization.md").write_text(CUSTOMIZATION_MD, encoding="utf-8")
    (skill_dir / "mcp.md").write_text(MCP_MD, encoding="utf-8")
    (rules_dir / "forms.md").write_text(FORMS_MD, encoding="utf-8")
    (rules_dir / "composition.md").write_text(COMPOSITION_MD, encoding="utf-8")
    (rules_dir / "styling.md").write_text(STYLING_MD, encoding="utf-8")
    (rules_dir / "icons.md").write_text(ICONS_MD, encoding="utf-8")
    (rules_dir / "base-vs-radix.md").write_text(BASE_VS_RADIX_MD, encoding="utf-8")

    return tmp_path  # Return parent dir (skills_dir)


# ────────────────────────────────────────────────────────────────────────────
# Tests: Skill Discovery
# ────────────────────────────────────────────────────────────────────────────


class TestShadcnSkillDiscovery:
    """Test that discover_skills() correctly parses the shadcn skill."""

    def test_discovers_shadcn_skill(self, shadcn_skill_dir: Path) -> None:
        """Skill is discovered from the directory."""
        skills = discover_skills(str(shadcn_skill_dir))
        assert len(skills) == 1
        assert skills[0].name == "shadcn"

    def test_parses_long_description(self, shadcn_skill_dir: Path) -> None:
        """Long multi-purpose description is parsed correctly."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        assert "Manages shadcn components" in skill.description
        assert "component registries" in skill.description
        assert "components.json" in skill.description

    def test_parses_all_triggers(self, shadcn_skill_dir: Path) -> None:
        """All 6 triggers are parsed from metadata block."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        assert len(skill.triggers) == 6
        assert "shadcn init" in skill.triggers
        assert "add component" in skill.triggers
        assert "components.json" in skill.triggers

    def test_parses_all_8_resources(self, shadcn_skill_dir: Path) -> None:
        """All 8 resource files are registered."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        assert len(skill.resources) == 8
        assert "cli.md" in skill.resources
        assert "customization.md" in skill.resources
        assert "mcp.md" in skill.resources
        assert "rules/forms.md" in skill.resources
        assert "rules/composition.md" in skill.resources
        assert "rules/styling.md" in skill.resources
        assert "rules/icons.md" in skill.resources
        assert "rules/base-vs-radix.md" in skill.resources

    def test_parses_all_tags(self, shadcn_skill_dir: Path) -> None:
        """All 4 tags are parsed."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        assert skill.tags == {"ui", "components", "tailwind", "react"}

    def test_parses_priority(self, shadcn_skill_dir: Path) -> None:
        """Priority 10 is parsed correctly."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        assert skill.priority == 10


# ────────────────────────────────────────────────────────────────────────────
# Tests: Content Loading
# ────────────────────────────────────────────────────────────────────────────


class TestShadcnContentLoading:
    """Test loading skill content and resources."""

    def test_load_skill_content_strips_frontmatter(self, shadcn_skill_dir: Path) -> None:
        """load_skill_content() returns body without YAML frontmatter."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        content = load_skill_content(skill)

        # Should NOT contain frontmatter
        assert "---" not in content[:10]  # No YAML delimiter at start
        assert "metadata:" not in content

        # Should contain body content
        assert "# shadcn/ui" in content
        assert "A framework for building ui" in content
        assert "npx shadcn@latest init" in content

    def test_load_top_level_resources(self, shadcn_skill_dir: Path) -> None:
        """Top-level resource files (cli.md, etc.) are loadable."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        cli_content = load_resource(skill, "cli.md")
        assert cli_content is not None
        assert "# shadcn CLI Reference" in cli_content
        assert "npx shadcn@latest init" in cli_content

        custom_content = load_resource(skill, "customization.md")
        assert custom_content is not None
        assert "# Customization & Theming" in custom_content
        assert "OKLCH" in custom_content

        mcp_content = load_resource(skill, "mcp.md")
        assert mcp_content is not None
        assert "# shadcn MCP Server" in mcp_content

    def test_load_nested_resources_in_rules_dir(self, shadcn_skill_dir: Path) -> None:
        """Nested resources in rules/ subdirectory are loadable."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        forms = load_resource(skill, "rules/forms.md")
        assert forms is not None
        assert "# Forms & Inputs" in forms
        assert "FieldGroup" in forms

        composition = load_resource(skill, "rules/composition.md")
        assert composition is not None
        assert "# Component Composition" in composition
        assert "SelectGroup" in composition

        styling = load_resource(skill, "rules/styling.md")
        assert styling is not None
        assert "# Styling & Customization" in styling
        assert "cn()" in styling

        icons = load_resource(skill, "rules/icons.md")
        assert icons is not None
        assert "data-icon" in icons

        base_radix = load_resource(skill, "rules/base-vs-radix.md")
        assert base_radix is not None
        assert "asChild" in base_radix
        assert "render" in base_radix

    def test_load_all_resources_successfully(self, shadcn_skill_dir: Path) -> None:
        """All 8 registered resources can be loaded without errors."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        loaded_count = 0
        for resource_path in skill.resources:
            content = load_resource(skill, resource_path)
            assert content is not None, f"Failed to load resource: {resource_path}"
            assert len(content) > 50, f"Resource too short: {resource_path}"
            loaded_count += 1

        assert loaded_count == 8, f"Expected 8 resources, loaded {loaded_count}"


# ────────────────────────────────────────────────────────────────────────────
# Tests: Registry Integration
# ────────────────────────────────────────────────────────────────────────────


class TestShadcnRegistryIntegration:
    """Test that SkillsRegistry can manage the shadcn skill."""

    def test_register_shadcn_skill(self, shadcn_skill_dir: Path) -> None:
        """Shadcn skill can be registered in SkillsRegistry."""
        skills = discover_skills(str(shadcn_skill_dir))
        registry = SkillsRegistry()

        for skill in skills:
            registry.register(skill)

        assert registry.get("shadcn") is not None
        assert registry.get("shadcn").name == "shadcn"

    def test_filter_by_tags(self, shadcn_skill_dir: Path) -> None:
        """Skill can be filtered by its tags using get_all(tags=...)."""
        skills = discover_skills(str(shadcn_skill_dir))
        registry = SkillsRegistry()

        for skill in skills:
            registry.register(skill)

        ui_skills = registry.get_all(tags={"ui"})
        assert len(ui_skills) == 1
        assert ui_skills[0].name == "shadcn"

        react_skills = registry.get_all(tags={"react"})
        assert len(react_skills) == 1

        # No match
        python_skills = registry.get_all(tags={"python"})
        assert len(python_skills) == 0

    def test_trigger_table_generation(self, shadcn_skill_dir: Path) -> None:
        """Trigger table correctly includes skill triggers in markdown format."""
        skills = discover_skills(str(shadcn_skill_dir))
        registry = SkillsRegistry()

        for skill in skills:
            registry.register(skill)

        trigger_table = registry.build_trigger_table()
        # build_trigger_table returns markdown string
        assert "shadcn" in trigger_table
        assert "shadcn init" in trigger_table
        assert "|" in trigger_table  # markdown table format


# ────────────────────────────────────────────────────────────────────────────
# Tests: Edge Cases & Robustness
# ────────────────────────────────────────────────────────────────────────────


class TestShadcnEdgeCases:
    """Test edge cases and robustness with real-world content."""

    def test_handles_special_characters_in_content(self, shadcn_skill_dir: Path) -> None:
        """Special characters (code blocks, pipes, etc.) are handled correctly."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        # CLI.md has lots of pipes in tables
        cli_content = load_resource(skill, "cli.md")
        assert "|" in cli_content

        # Content has code blocks with backticks
        assert "```" in cli_content

    def test_handles_long_description_under_limit(self, shadcn_skill_dir: Path) -> None:
        """Long description stays under the 2000 char limit."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]
        # Shadcn's description is ~350 chars, well under 2000
        assert len(skill.description) < 2000
        assert len(skill.description) > 100  # But substantial

    def test_path_traversal_blocked(self, shadcn_skill_dir: Path) -> None:
        """Attempting to load ../outside resources is blocked."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        # These should return None, not raise
        result = load_resource(skill, "../outside.md")
        assert result is None

        result = load_resource(skill, "rules/../../etc/passwd")
        assert result is None

    def test_nonexistent_resource_returns_none(self, shadcn_skill_dir: Path) -> None:
        """Requesting a nonexistent resource returns None."""
        skills = discover_skills(str(shadcn_skill_dir))
        skill = skills[0]

        result = load_resource(skill, "nonexistent.md")
        assert result is None


# ────────────────────────────────────────────────────────────────────────────
# Tests: Multiple Skills Directory (shadcn + custom)
# ────────────────────────────────────────────────────────────────────────────


class TestMultipleSkillsWithShadcn:
    """Test shadcn skill alongside other skills."""

    def test_shadcn_with_custom_skill(self, shadcn_skill_dir: Path) -> None:
        """Shadcn skill works alongside a simpler custom skill."""
        # Add another skill to the same directory
        custom_dir = shadcn_skill_dir / "my-custom-skill"
        custom_dir.mkdir()
        (custom_dir / "SKILL.md").write_text(
            dedent("""\
            ---
            name: my-custom-skill
            description: A simple custom skill
            metadata:
              triggers:
                - custom action
              priority: 5
            ---
            # My Custom Skill

            This is a simple skill.
            """),
            encoding="utf-8",
        )

        skills = discover_skills(str(shadcn_skill_dir))
        assert len(skills) == 2

        names = {s.name for s in skills}
        assert "shadcn" in names
        assert "my-custom-skill" in names

        # Shadcn should have higher priority
        shadcn = next(s for s in skills if s.name == "shadcn")
        custom = next(s for s in skills if s.name == "my-custom-skill")
        assert shadcn.priority > custom.priority

    def test_registry_with_multiple_skills(self, shadcn_skill_dir: Path) -> None:
        """Registry correctly handles multiple skills with different resources."""
        # Add another skill
        react_dir = shadcn_skill_dir / "react-patterns"
        react_dir.mkdir()
        (react_dir / "SKILL.md").write_text(
            dedent("""\
            ---
            name: react-patterns
            description: React component patterns
            metadata:
              tags:
                - react
                - patterns
              priority: 8
            ---
            # React Patterns
            """),
            encoding="utf-8",
        )

        skills = discover_skills(str(shadcn_skill_dir))
        registry = SkillsRegistry()

        for skill in skills:
            registry.register(skill)

        # Both skills tagged with 'react' should be found
        react_skills = registry.get_all(tags={"react"})
        assert len(react_skills) == 2

        # Only shadcn tagged with 'ui'
        ui_skills = registry.get_all(tags={"ui"})
        assert len(ui_skills) == 1
        assert ui_skills[0].name == "shadcn"


# ────────────────────────────────────────────────────────────────────────────
# Summary Test: Full Integration
# ────────────────────────────────────────────────────────────────────────────


class TestFullShadcnIntegration:
    """End-to-end test simulating real usage."""

    def test_full_workflow(self, shadcn_skill_dir: Path) -> None:
        """Complete workflow: discover → register → load → use."""
        # 1. Discover skills
        skills = discover_skills(str(shadcn_skill_dir))
        assert len(skills) == 1

        # 2. Register in registry
        registry = SkillsRegistry()
        for skill in skills:
            registry.register(skill)

        # 3. Look up by name
        shadcn = registry.get("shadcn")
        assert shadcn is not None

        # 4. Load main content
        content = load_skill_content(shadcn)
        assert "# shadcn/ui" in content

        # 5. Load all resources and verify total content size
        total_content_size = len(content)
        for resource_path in shadcn.resources:
            resource_content = load_resource(shadcn, resource_path)
            assert resource_content is not None
            total_content_size += len(resource_content)

        # Total content should be substantial (> 5KB)
        assert total_content_size > 5000, f"Total content only {total_content_size} bytes"

        # 6. Verify trigger table for activation (returns markdown string)
        trigger_table = registry.build_trigger_table()
        assert "shadcn" in trigger_table
        assert len(trigger_table) > 100  # Should have substantial content

        print(f"\n✅ Shadcn skill loaded successfully!")
        print(f"   - Name: {shadcn.name}")
        print(f"   - Description: {shadcn.description[:60]}...")
        print(f"   - Resources: {len(shadcn.resources)} files")
        print(f"   - Triggers: {len(shadcn.triggers)}")
        print(f"   - Tags: {shadcn.tags}")
        print(f"   - Total content: {total_content_size:,} bytes")
