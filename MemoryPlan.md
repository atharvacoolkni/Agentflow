# Agent-Level Memory Config

## Summary
- Remove `MemoryIntegration.add_agent`; memory should be configured directly on `Agent`, similar to skills.
- Introduce a single `MemoryConfig` passed to `Agent(...)`, with two sub-configs:
  - `user_memory`: model can read and write
  - `agent_memory`: model can read only
- Keep `MemoryIntegration` as the internal orchestration layer, but make `Agent(memory=...)` the primary public API.

## Key Changes
- Add `memory: MemoryConfig | None` to [`agent.py`](/Users/shudipto/Projects/agentflow/agentflow/agentflow/core/graph/agent.py).
- Add new config models in the storage/memory package:
  - `MemoryConfig`
  - `UserMemoryConfig`
  - `AgentMemoryConfig`
- `MemoryConfig` owns:
  - retrieval mode defaults,
  - preload/search limits,
  - prompt behavior,
  - optional per-scope store/backend config,
  - enable/disable flags for `user_memory` and `agent_memory`.

- During `Agent` initialization:
  - build memory integration from `MemoryConfig`,
  - append memory system prompts automatically,
  - auto-register memory tools on the agent’s internal `ToolNode`,
  - auto-create any preload behavior needed for the agent path.
- Follow the skills pattern:
  - config object passed once,
  - agent setup performs tool/prompt registration internally,
  - no manual `ToolNode([..., *memory.tools])` step in normal usage.

- Split the model-facing tools into two explicit tools:
  - `user_memory_tool`: `search`, `remember`
  - `agent_memory_tool`: `search` only
- `user_memory_tool` default schema should be simple:
  - `action`: `search | remember`
  - `text` for search or remember content, depending on action
  - optional `memory_type`, `category`, `limit`
- `agent_memory_tool` should support only retrieval inputs:
  - `query`
  - optional `memory_type`, `category`, `limit`
- Do not expose `memory_id`, `memory_key`, `update`, or `delete` to the model in the default agent flow.

- Treat user-vs-agent memory as separate policy layers, not just prompt wording:
  - user memory allows store + retrieve,
  - agent memory allows retrieve only,
  - agent-memory writes are reserved for non-LLM flows such as optimizer/manual pipelines.
- Keep admin/maintenance APIs internal or explicitly advanced-only; they should not be added to the agent by default.

- Refine store handling so runtime config supplies identity/scoping:
  - user memory uses `user_id` from config for retrieval and writes,
  - agent memory uses agent/app identity from config or memory config,
  - the model never generates identifiers.
- Fix `Mem0Store` so long-term memory does not require `thread_id` for normal cross-thread retrieval/write behavior.
- Preserve cross-thread retrieval semantics for long-term memory in both Mem0 and Qdrant-backed flows.

- Update prompts/docs/examples to show the new primary usage:
  - `agent = Agent(..., memory=MemoryConfig(...))`
- Remove the current manual setup pattern from primary docs:
  - no manual `memory.wire(...)`,
  - no manual `tool_node = ToolNode([my_tool, *memory.tools])`,
  - no model instruction telling the LLM to invent memory keys.

## Public Interfaces
- New Agent API:
  - `Agent(..., memory=MemoryConfig(...))`
- New config types:
  - `MemoryConfig`
  - `UserMemoryConfig`
  - `AgentMemoryConfig`
- Default model-facing tools:
  - `user_memory_tool`
  - `agent_memory_tool`
- `MemoryIntegration` remains available internally/advanced, but is no longer the main onboarding API.

## Test Plan
- Add agent integration tests covering:
  - `Agent(memory=...)` auto-registers prompts and tools,
  - user-memory tool can search and remember,
  - agent-memory tool can search but rejects writes by schema and runtime policy,
  - no duplicate tool registration when agent already has other tools.
- Add store behavior tests covering:
  - Mem0 operations without `thread_id`,
  - user-memory cross-thread retrieval,
  - agent-memory read-only enforcement in the agent layer.
- Update long-term-memory tests to validate the new default tool schemas and removal of model-facing `memory_id` / `memory_key`.

## Sprint Plan

### Sprint 1: Agent-Level API Foundation
- Add public config models:
  - `MemoryConfig`
  - `UserMemoryConfig`
  - `AgentMemoryConfig`
- Add `memory: MemoryConfig | None` to `Agent(...)`.
- Auto-register the default model-facing memory tools on the agent-owned `ToolNode`.
- Auto-append the memory system prompt when `MemoryConfig.inject_system_prompt=True`.
- Add explicit scoped tools:
  - `user_memory_tool`: `search`, `remember`
  - `agent_memory_tool`: `search` only
- Preserve the existing `MemoryIntegration` and `memory_tool` path for advanced/backward-compatible graph wiring.
- Fix the Mem0 long-term-memory path so normal writes/retrieval do not require `thread_id`.
- Add focused tests for config export, agent initialization, tool schemas, tool runtime policy, and Mem0 no-thread behavior.

### Sprint 2: Retrieval And Preload Behavior
- Implement agent-level preload behavior for `MemoryConfig(retrieval_mode="preload")`.
- Validate cross-thread retrieval for user memory in Qdrant-backed and Mem0-backed stores.
- Add stricter agent-memory scoping around agent/app identity.
- Add integration tests for graph execution using `Agent(memory=...)`.

### Sprint 3: Docs, Examples, And Migration
- Update primary docs and examples to use `Agent(..., memory=MemoryConfig(...))`.
- Move manual `MemoryIntegration.wire(...)` and `ToolNode([..., *memory.tools])` guidance to advanced docs.
- Add migration notes for the older `memory_tool` schema.
- Remove model-facing guidance that asks the LLM to invent memory keys.

## Assumptions
- `MemoryConfig` is the single user-facing entry point for memory on `Agent`.
- User memory and agent memory may use different stores/configs, but both are configured under one `MemoryConfig`.
- Agent-memory mutation is out of scope for the LLM path and handled elsewhere by optimizer/human workflows.
- Manual graph-level memory wiring remains supported only as an advanced path, not the recommended one.
