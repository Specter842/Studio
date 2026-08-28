# Skills

Markdown playbooks for the MCP tools in `src/mcp_server.py`. Not code —
judgment calls a tool's schema can't carry on its own (which cut density
suits which energy section, which effects suit which brief), meant to be
read by whichever LLM client is driving the tools before it starts calling
them for a task that matches.

The pattern is deliberately borrowed from Kaestral
(github.com/prabindersinghh/Kaestral-pro) — expose editing primitives as MCP
tools, keep the domain knowledge in read-on-demand docs rather than a
hardcoded rule engine — reimplemented from scratch rather than adapted from
its GPL-3.0 source, and pointed at a different underlying engine. See
`beat-sync-cutting/SKILL.md` for why that distinction matters in practice,
not just licensing: Kaestral's own equivalent skill has its agent compute
cut placement by hand, tool call by tool call, against a plain tempo grid.
Every skill here instead delegates cut timing to `plan_edit`/`render_edit`,
which call the same deterministic, render-verified assembler the CLI does —
the skill's job is choosing *parameters*, never re-deriving frame math.

## Format

```markdown
---
name: Human-readable name
description: One line — what it's for and when to use it.
---

# Title

Body: concrete guidance, concrete tool calls, concrete parameter tables.
Written for an LLM client to follow, not a human reading documentation —
be specific about *why*, since that's what lets it generalize past the
exact scenario written down.
```

## Available

- [`beat-sync-cutting`](beat-sync-cutting/SKILL.md) — cut footage to a
  track's rhythm, choosing cut density from the beat grid's energy sections.

More get added the same way Kaestral's 25 do: one skill per recurring
editorial goal, not one per tool.
