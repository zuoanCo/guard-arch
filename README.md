# Guard Arch

对标 Claude Code 的 AI Agent 工具：核心是一套自研 **Agent Runtime**（以 PydanticAI 作为底层执行引擎），对外提供可直接使用的 CLI 对话工具。

## 安装

```powershell
uv sync
```

## 配置模型

模型由 `config/models.yaml` 驱动，角色 → provider/model/base_url/api_key_env：

| 角色 | provider | 模型 | API key 环境变量 |
| --- | --- | --- | --- |
| default | openai 兼容 | deepseek-chat | `DEEPSEEK_API_KEY` |
| reasoning | openai 兼容 | deepseek-reasoner | `DEEPSEEK_API_KEY` |
| cheap | openai 兼容 | qwen-turbo | `DASHSCOPE_API_KEY` |
| coding | anthropic | claude-sonnet-4-5 | `ANTHROPIC_API_KEY` |
| test | 内置 TestModel | 无需 key | — |
| test-demo | 内置脚本模型（先调 read_file 再回复） | 无需 key | — |

复制 `.env.example` 为 `.env` 并填入 key，或直接设置环境变量。缺 key 时会得到指出具体环境变量名的友好报错。

## CLI 用法

```powershell
# 交互式对话（默认当前目录为 workspace）
uv run guard

# 非交互单轮
uv run guard --message "你好，介绍一下你自己" --model test

# 等价入口
uv run python -m guard_arch --message "..." --model test
```

参数：

| 参数 | 说明 |
| --- | --- |
| `--workspace <dir>` | 工作区根目录（默认 cwd），所有文件操作被沙箱限制在其中 |
| `--agent <id>` | agent id（默认 `assistant`，定义见 `agents/*.yaml`） |
| `--model <role>` | 覆盖模型角色（如 `default`/`test`/`test-demo`） |
| `--session <id>` | 会话 id，对话历史按会话持久化，同 session 连续对话 |
| `--auto-approve` | 自动允许所有权限请求（用于脚本） |
| `--message <text>` | 非交互模式：跑一轮退出 |

斜杠命令：`/help` `/model [role]` `/skills` `/agents` `/clear` `/exit`

工具调用时会显示状态行（如 `✓ read_file README.md`），危险命令被直接拒绝，普通 shell 命令会交互询问 y/n。

## API 服务（后台模式）

```powershell
uv run guard-api                                              # 127.0.0.1:8100
uv run uvicorn guard_arch.api.app:app --port 8100                  # 等效
```

端点：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查，返回 `{"ok": true}` |
| GET | `/api/v1/agents` | agent 列表（id/name/model/skills/tools） |
| GET | `/api/v1/skills` | skill 列表 |
| POST | `/api/v1/sessions` | 创建会话 `{ "workspace?": "..." }` → `{ session_id }` |
| GET | `/api/v1/sessions/{id}/messages` | 会话历史 `[{role, content}]` |
| POST | `/api/v1/chat` | SSE 流式对话 |

`/api/v1/chat` 请求体：`{ "session_id"?, "agent"?, "message", "workspace"?, "model"?, "auto_approve"? }`；省略 `session_id` 时自动新建并在首个事件（`agent_started`）的 data 里下发。SSE 事件与 EventBus 一一对应：`agent_started` / `message_delta` / `tool_call` / `tool_result` / `permission_required` / `agent_finished` / `error`，`agent_finished` 后流正常关闭。

权限：API 模式无交互回调，`ask` 默认拒绝（仍会发 `permission_required` 事件供前端展示）；传 `auto_approve: true` 全部允许（开发便利，生产应接确认回调）；**deny 规则（如 `rm -rf`）不受 auto_approve 影响**。

curl 示例：

```powershell
# 健康检查
curl http://127.0.0.1:8100/health

# SSE 流式对话（-N 关闭缓冲）
curl -N -X POST http://127.0.0.1:8100/api/v1/chat `
  -H "Content-Type: application/json" `
  --data-binary '{\"message\": \"你好\", \"model\": \"test\", \"workspace\": \".\"}'

# 创建会话并读取历史
curl -X POST http://127.0.0.1:8100/api/v1/sessions -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:8100/api/v1/sessions/<session_id>/messages
```

## 架构

```text
CLI (rich)
   │
AgentRuntime ── EventBus ── (message_delta / tool_call / tool_result / permission_required / ...)
   │
   ├── AgentRegistry     agents/*.yaml          配置驱动的 Agent 定义
   ├── SkillRegistry     skills/*/SKILL.md      frontmatter + Markdown instructions
   ├── ToolRegistry      native/MCP 统一 Tool
   ├── ModelRouter       config/models.yaml     角色 → pydantic-ai 模型
   ├── ContextEngine     system prompt 组装 + token 预算
   ├── MemoryManager     四层记忆，SQLite（workspace/.guard_arch/memory.db）
   ├── PermissionEngine  allow / ask / deny 规则表
   ├── SandboxManager    路径逃逸防护
   └── RunManager        每次执行的 Run 记录
   │
PydanticAI Agent (agent loop, tool calls, streaming events)
   │
OpenAI 兼容 / Anthropic / Google / TestModel
```

详见 [docs/architecture.md](docs/architecture.md)。

## 如何扩展

**新增 Agent**：在 `agents/` 加一个 YAML：

```yaml
id: reviewer
name: 代码审查员
model: coding
skills: [coding]
tools: [read_file, search_text]
instructions: 只读审查，不修改文件……
```

**新增 Skill**：在 `skills/<name>/SKILL.md` 写 YAML frontmatter（`name`/`description`/`tools`）+ Markdown instructions，然后在 agent YAML 的 `skills` 里引用。

**新增 Tool**：写一个带类型签名的函数，包成 `Tool(name, description, handler)` 注册进 `ToolRegistry`（参考 `src/guard_arch/tools/filesystem.py`），再加入 agent 的 `tools` 列表。权限规则在 `permissions/engine.py` 的 `default_rules()` 中配置。

**接入 MCP**：在 `config/mcp.json` 写 Claude Desktop 格式的 `mcpServers`；加载失败只记 warning，不影响启动。

## 测试

```powershell
uv run pytest -q     # 全部使用 keyless test 模型
uv run ruff check .
```
