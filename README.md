# Guard Arch

**English** | [中文](README.zh-CN.md)

A Claude Code-style AI Agent framework: a self-hosted **Agent Runtime** (PydanticAI as the execution engine) with a production-grade harness around a custom model — ReAct agent loop, rich context management, tool interface, constraints, verification and error correction. Ships with a ready-to-use **CLI** and a **FastAPI service** (SSE streaming), and can be embedded as a library into any backend.

## Architecture

```text
CLI (rich) / FastAPI (SSE) / embedded library
        │
AgentRuntime ── EventBus (typed events: thinking / tool_call / tool_retry /
        │                tool_verified / message_delta / user_question / ...)
        ├── AgentRegistry     agents/*.yaml       config-driven agent definitions
        ├── SkillRegistry     skills/*/SKILL.md   frontmatter + markdown instructions
        ├── ToolRegistry      native / MCP tools, unified Tool abstraction
        ├── ModelRouter       config/models.yaml  role -> provider/model (+ extra_body)
        ├── ContextEngine     layered system prompt with token budget
        ├── MemoryManager     4-layer memory, sessions, compaction (SQLite)
        ├── PermissionEngine  allow/ask/deny + LOW/MID/HIGH risk levels
        ├── HistoryCompactor  compress at 90% of token budget, then continue
        ├── QuestionManager   ask_user_question: suspend run, resume on answer
        └── Workspace         sandboxed filesystem root
        │
PydanticAI (ReAct agent loop: model decides -> tool calls -> results fed back)
        │
OpenAI-compatible / Anthropic / Google / keyless TestModel
```

### The execution model (model + harness)

Every run flows through typed execution phases, all emitted as events:

| Phase | Event | What happens |
| --- | --- | --- |
| thinking | `thinking` | Harness makes the model analyze the request against capabilities & context, then guides execution (distinct from the model's internal reasoning) |
| tool_call | `tool_call` / `tool_result` | Tools execute through the dispatch chain: event -> permission gate -> timeout+silent retry -> verify -> result |
| text | `message_delta` | Incremental text output (streaming) |
| askUserQuestion | `user_question` / `user_answered` | Real interaction: the run **suspends** until the user answers, then resumes with the answer |

## Feature checklist

**Context management** — what goes into each model request, in order: base operating principles -> agent instructions -> capability list (tool definitions) -> skills -> project instructions (workspace `GUARD.md`/`AGENTS.md`/`CLAUDE.md` auto-injected) -> memory -> environment state (time/OS/workspace). Token budget truncation + history compaction at 90%.

**Tool interface** — filesystem (read/write/edit/list/search), terminal (run_command), web (`web_search` + `web_fetch`), memory (`remember`/`recall_memory`), MCP tools (`config/mcp.json`), plus per-run meta capabilities: `todo_write/read`, `dispatch_agent` (sub-agents in isolated context), `ask_user_question`, `list_capabilities`. Base read/search/web/memory tools are auto-included for every agent.

**Constraints** — permission rules with **risk levels**: LOW (read-only, auto-allow) / MID (bounded side effects, ask) / HIGH (irreversible, e.g. `rm -rf`, `git push --force` — always denied, auto-approve can never override). Resource limits: per-tool execution timeout, concurrent-run semaphore. Fail-safe defaults: unknown tools are treated as MID + ASK.

**Verification** — tool verifiers check the *result*, not the model's claim: e.g. `write_file`/`edit_file` re-read the file to confirm the write landed. `tool_verified` events; failures are annotated back into the tool output.

**Error correction** — transient failures (network jitter / rate limit / timeout / 5xx) are **silently retried** (`Tool.retry_attempts`, linear backoff) with `tool_retry` events for observability; intermediate failure states never surface to the model.

**Memory & sessions** — 4-layer memory (conversation/user/project/agent, SQLite at `workspace/.guard_arch/memory.db`), per-session model-history persistence, `list_sessions` with prefix scoping, long-term recall via `recall_memory`.

## Install

```powershell
uv sync
```

## Configure models

Models are driven by `config/models.yaml` (role -> provider/model/base_url/api_key_env):

| Role | Provider | Model | API key env |
| --- | --- | --- | --- |
| default | openai-compatible | deepseek-chat | `DEEPSEEK_API_KEY` |
| reasoning | openai-compatible | deepseek-reasoner | `DEEPSEEK_API_KEY` |
| cheap | openai-compatible | qwen-turbo | `DASHSCOPE_API_KEY` |
| coding | anthropic | claude-sonnet-4-5 | `ANTHROPIC_API_KEY` |
| test | built-in TestModel | — | — |

Provider-specific request params (e.g. a thinking on/off switch) can be passed via `extra_body` in the role config. Copy `.env.example` to `.env` and fill in keys, or use `--model test` for a keyless run.

## CLI

```powershell
uv run guard                                   # interactive chat (workspace = cwd)
uv run guard --message "hello" --model test    # single non-interactive turn
```

Options: `--workspace <dir>` (sandbox root), `--agent <id>`, `--model <role>`, `--session <id>` (persistent history), `--auto-approve`, `--message <text>`.
Slash commands: `/help` `/model [role]` `/skills` `/agents` `/clear` `/exit`.

Tool calls show status lines; dangerous commands are hard-denied; regular shell commands ask y/n.

## API service

```powershell
uv run guard-api    # 127.0.0.1:8100
```

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| GET | `/api/v1/agents` | Agent list |
| GET | `/api/v1/skills` | Skill list |
| POST | `/api/v1/sessions` | Create session |
| GET | `/api/v1/sessions/{id}/messages` | Session history |
| POST | `/api/v1/chat` | SSE streaming chat |

SSE events mirror the EventBus: `agent_started` / `thinking` / `message_delta` / `tool_call` / `tool_retry` / `tool_verified` / `tool_result` / `permission_required` / `user_question` / `user_answered` / `agent_finished` / `error`. In API mode ASK resolves to deny (event still emitted); `auto_approve: true` allows all except HIGH-risk deny rules.

## Extend

- **New agent**: add a YAML in `agents/` (id/name/model/skills/tools/instructions; optional `intake:` clarification gate, `thinking:` harness thinking phase)
- **New skill**: add `skills/<name>/SKILL.md` (YAML frontmatter + markdown instructions)
- **New tool**: wrap a typed function as `Tool(name, description, handler, timeout_seconds=, retry_attempts=, verifier=)` and register it; permission rules in `permissions/engine.py`
- **MCP**: add `config/mcp.json` (Claude Desktop format); load failures degrade gracefully

## Test

```powershell
uv run pytest -q     # keyless test models only
uv run ruff check .
```

## License

See [LICENSE](LICENSE).
