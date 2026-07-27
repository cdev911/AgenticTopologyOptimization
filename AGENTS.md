# Agent Orientation — AgenticTopologyOptimization

This is a learning project: an LLM-driven agentic workflow (CrewAI) operating a
topology-optimization solver (`fenitop`) as a tool. Explicitly a
demonstration/learning repo, not a production system.

Read these two files before doing anything else:
- `README.md` — what the project is, how to build/run/test it.
- `docs/spec.md` — the living decision log: what's been decided and why, current
  status, and what's next. Start with its `## 0. Current Status` section at the top.

Do not assume anything about scope or architecture beyond what `docs/spec.md`
records. If something you need isn't covered there, ask the user rather than
guessing or re-deciding it yourself.

After reading, summarize your understanding of the current state and your proposed
next step back to the user before taking any action.

When you finish work in a session, update `docs/spec.md` yourself: check off the
relevant `§11 Implementation Checklist` items, add a dated `§9 Decision Log` entry
for any new decision, and refresh `§0 Current Status` at the top. That file—not
chat history—carries context into the next session.
