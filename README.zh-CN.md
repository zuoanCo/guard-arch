# Guard Arch

[English](README.md) | **中文**

对标 Claude Code 的 AI Agent 框架：自研 **Agent Runtime**（PydanticAI 作为底层执行引擎），围绕自定义模型构建生产级 harness——ReAct agent loop、分层上下文管理、工具接口、约束、验证与纠正。开箱提供 **CLI** 与 **FastAPI 服务**（SSE 流式），也可作为库嵌入任意后端。

## 架构

```text
CLI (rich) / FastAPI (SSE) / 嵌入式库
        │
AgentRuntime ── EventBus（类型化事件：thinking / tool_call / tool_retry /
        │                tool_verified / message_delta / user_question / …）
        ├── AgentRegistry     agents/*.yaml       配置驱动的 Agent 定义
        ├── SkillRegistry     skills/*/SKILL.md   frontmatter + Markdown 指令
        ├── ToolRegistry      native / MCP 统一 Tool 抽象
        ├── ModelRouter       config/models.yaml  角色 → provider/模型（+ extra_body）
        ├── ContextEngine     分层 system prompt + token 预算
        ├── MemoryManager     四层记忆、会话管理、压缩（SQLite）
        ├── PermissionEngine  allow/ask/deny + 低/中/高风险评级
        ├── HistoryCompactor  上下文达 90% 先压缩再继续
        ├── QuestionManager   ask_user_question：挂起等待、回答后原 run 继续
        └── Workspace         文件系统沙箱
        │
PydanticAI（ReAct agent loop：模型决策 → 工具调用 → 结果回灌 → 再决策）
        │
OpenAI 兼容 / Anthropic / Google / 免 key TestModel
```

### 执行模型（model + harness）

每次 run 按类型化阶段执行，全程事件可观测：

| 阶段 | 事件 | 说明 |
| --- | --- | --- |
| thinking | `thinking` | 框架让模型先分析需求（结合能力面与上下文），得出的方向注入主执行（区别于模型内部思维链） |
| tool_call | `tool_call` / `tool_result` | 工具经派发链执行：事件 → 权限门控 → 超时+静默重试 → 验证 → 结果 |
| text | `message_delta` | 增量文本流式输出 |
| askUserQuestion | `user_question` / `user_answered` | 真交互：run **挂起等待**用户回答，回答注入后**原 run 继续执行** |

## 功能清单

**上下文管理**——每次发给模型的内容与顺序：base 工作方式 → agent 人设 → 能力清单（工具定义）→ skills → 项目指令（工作区 `GUARD.md`/`AGENTS.md`/`CLAUDE.md` 自动注入）→ memory → 环境状态（时间/系统/工作区）。token 预算截断 + 90% 历史压缩。

**工具接口**——文件（读/写/改/列/搜）、命令执行（run_command）、网络（`web_search` + `web_fetch`）、记忆（`remember`/`recall_memory`）、MCP（`config/mcp.json`），以及每 run 自动挂载的元能力：`todo_write/read`、`dispatch_agent`（隔离上下文子代理）、`ask_user_question`、`list_capabilities`。只读/搜索/web/记忆类基础工具每个 agent 自动拥有。

**约束**——权限规则带**风险评级**：LOW（只读，默认放行）/ MID（有边界副作用，默认询问）/ HIGH（不可逆，如 `rm -rf`、`git push --force`，**永远拒绝，auto_approve 不可覆盖**）。资源限制：工具执行超时、并发 run 信号量。故障安全默认：未知工具按 MID + ASK 对待。

**验证**——工具验证器检查**执行结果**而非模型自述：`write_file`/`edit_file` 写入后回读确认内容落盘。`tool_verified` 事件；验证失败附注回灌工具输出。

**纠正**——瞬时故障（网络抖动/限流/超时/5xx）**静默重试**（`Tool.retry_attempts` 次，线性退避），发 `tool_retry` 事件供观测；中间失败态永不暴露给模型。

**记忆与会话**——四层记忆（conversation/user/project/agent，SQLite 存于 `workspace/.guard_arch/memory.db`）、按 session 持久化模型历史、`list_sessions` 前缀归属过滤、`recall_memory` 按需召回。

## 安装

```powershell
uv sync
```

## 配置模型

模型由 `config/models.yaml` 驱动（角色 → provider/模型/base_url/api_key 环境变量）：

| 角色 | provider | 模型 | API key 环境变量 |
| --- | --- | --- | --- |
| default | openai 兼容 | deepseek-chat | `DEEPSEEK_API_KEY` |
| reasoning | openai 兼容 | deepseek-reasoner | `DEEPSEEK_API_KEY` |
| cheap | openai 兼容 | qwen-turbo | `DASHSCOPE_API_KEY` |
| coding | anthropic | claude-sonnet-4-5 | `ANTHROPIC_API_KEY` |
| test | 内置 TestModel | 无需 key | — |

provider 专有请求参数（如思考开关）可经角色配置里的 `extra_body` 透传。复制 `.env.example` 为 `.env` 填入 key，或用 `--model test` 免 key 体验。

## CLI 用法

```powershell
uv run guard                                   # 交互式对话（默认当前目录为工作区）
uv run guard --message "你好" --model test      # 非交互单轮
```

参数：`--workspace <dir>`（沙箱根目录）、`--agent <id>`、`--model <role>`、`--session <id>`（同 session 连续对话）、`--auto-approve`、`--message <text>`。

斜杠命令：

| 命令 | 说明 |
| --- | --- |
| `/help` | 显示帮助 |
| `/model [role]` `/agent [id]` | 查看或切换模型角色 / agent |
| `/workspace` | 显示工作区根目录 |
| `/agents` `/skills` `/tools` | 列出 agents / skills / 工具 |
| `/sessions` `/session <id>` `/new` | 会话列表 / 切换 / 新建 |
| `/todos` `/runs` | 会话任务清单 / 最近 run 记录 |
| `/memory [layer]` `/remember <层> <键> <值>` `/forget <层> <键>` | 查看 / 写入 / 删除长期记忆 |
| `/clear` `/exit` | 清空当前会话历史 / 退出 |

工具调用显示状态行；危险命令直接拒绝；普通 shell 命令交互询问 y/n。

## API 服务

```powershell
uv run guard-api    # 127.0.0.1:8100
```

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/api/v1/agents` | agent 列表 |
| GET | `/api/v1/skills` | skill 列表 |
| GET | `/api/v1/tools` | 工具清单（名称/描述/来源） |
| GET | `/api/v1/models` | 可用模型角色 |
| POST | `/api/v1/sessions` | 创建会话 |
| GET | `/api/v1/sessions` | 会话列表（`like` 前缀过滤归属） |
| GET | `/api/v1/sessions/{id}/messages` | 会话历史 |
| GET | `/api/v1/sessions/{id}/todos` | 会话任务清单 |
| DELETE | `/api/v1/sessions/{id}` | 删除会话 |
| GET/POST/DELETE | `/api/v1/memory[/{layer}/{key}]` | 读取 / 写入 / 删除长期记忆 |
| GET | `/api/v1/runs` `/api/v1/runs/{id}` | 最近 run 记录 / run 详情（含事件） |
| POST | `/api/v1/chat` | SSE 流式对话 |
| POST | `/api/v1/chat/answer` | 提交挂起提问的回答 |

SSE 事件与 EventBus 一一对应：`agent_started` / `thinking` / `message_delta` / `tool_call` / `tool_retry` / `tool_verified` / `tool_result` / `permission_required` / `user_question` / `user_answered` / `agent_finished` / `error`。API 模式 ASK 默认拒绝（仍发事件供前端展示）；`auto_approve: true` 放行除 HIGH 高危外的全部操作。

## 如何扩展

- **新增 Agent**：`agents/` 加 YAML（id/name/model/skills/tools/instructions；可选 `intake:` 澄清门禁、`thinking:` 框架思考阶段）
- **新增 Skill**：`skills/<name>/SKILL.md`（YAML frontmatter + Markdown 指令）
- **新增 Tool**：带类型签名的函数包成 `Tool(name, description, handler, timeout_seconds=, retry_attempts=, verifier=)` 注册；权限规则在 `permissions/engine.py` 配置
- **接入 MCP**：`config/mcp.json` 写 Claude Desktop 格式；加载失败优雅降级不影响启动

## 测试

```powershell
uv run pytest -q     # 全部使用免 key 测试模型
uv run ruff check .
```

## 许可

见 [LICENSE](LICENSE)。
