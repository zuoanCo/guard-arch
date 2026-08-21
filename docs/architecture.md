# Guard Arch 落地架构

本文档描述 Phase 1 + Phase 2（设计文档第二十四节）落地后的架构，以及各模块与设计文档的映射关系。

## 总览

```text
┌──────────────────────────────────────────────────────────────┐
│ CLI (guard_arch.cli, rich 渲染)                               │
│   交互循环 / 斜杠命令 / 权限询问 / 流式输出                  │
└───────────────────────┬──────────────────────────────────────┘
                        │ runtime.run(agent, session, message)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ AgentRuntime (guard_arch.runtime)                             │
│                                                              │
│  AgentRegistry ─ SkillRegistry ─ ToolRegistry                │
│  ModelRouter ─ ContextEngine ─ MemoryManager                 │
│  PermissionEngine ─ SandboxManager ─ RunManager              │
│                        │                                     │
│                   EventBus ──► CLI 订阅（message_delta /      │
│                    tool_call / tool_result / error …）       │
└───────────────────────┬──────────────────────────────────────┘
                        │ Agent + tools + toolsets + event_stream_handler
                        ▼
┌──────────────────────────────────────────────────────────────┐
│ PydanticAI (执行引擎: agent loop / tool 调度 / 流式事件)     │
└───────┬──────────────────────┬───────────────────────────────┘
        ▼                      ▼
  Model 层               Tool 层
  openai 兼容            Native tools (filesystem / terminal / remember)
  anthropic              MCP toolsets (config/mcp.json)
  google                 全部经 PermissionEngine，文件类再过 Sandbox
  test (TestModel /
  FunctionModel，免 key)
```

## 与设计文档的映射

| 设计文档 | 落地模块 |
| --- | --- |
| 三、Agent Runtime 层 | `src/guard_arch/runtime.py`（UI 不直接碰 PydanticAI） |
| 四、Agent 配置驱动 | `core/agent.py` + `agents/*.yaml`（AgentDefinition/AgentRegistry） |
| 五、Skill | `core/skill.py` + `skills/*/SKILL.md`（frontmatter + instructions，注入 system prompt 并按声明启用工具） |
| 六、Tool 系统 | `core/tool.py`（统一 Tool 抽象），Native 在 `tools/`，MCP 在 `mcp/client.py` |
| 七、Terminal | `tools/terminal.py`：CommandPolicy=PermissionEngine，超时/输出截断，cwd 锁定 workspace |
| 八、Sandbox | `core/workspace.py`：Workspace.resolve 强制路径在 workspace 根内（Phase 1 为本地沙箱，Run 级快照留待后续） |
| 九、Permission | `permissions/engine.py`：规则表 allow/ask/deny，glob 匹配工具名 + 正则匹配参数；`--auto-approve` |
| 十、Memory 四层 | `core/memory.py`：conversation（pydantic-ai 消息序列化 + 可读消息表）/ user / project / agent（kv），SQLite 持久化到 `workspace/.guard_arch/memory.db`；`remember` 工具供 agent 主动写入 |
| 十一/十二、Context Engine | `core/context.py`：system prompt = base + agent instructions + skills + memory + environment，按 token 预算截断可选段 |
| 十三/十四、Model Router | `core/model.py` + `config/models.yaml`：角色 → provider；openai 兼容 / anthropic / google / test；缺 key 报友好错误 |
| 十五、Agent Loop | PydanticAI 负责底层 loop；Runtime 管 Task/Run/Context/Permission/Skill/Memory/Tool/事件/持久化 |
| 十六、Event Bus | `events/bus.py`：同步+异步订阅，事件 agent_started / message_delta / tool_call / tool_result / permission_required / agent_finished / error |

## 关键机制

- **工具派发链**：pydantic-ai 工具调用 → Runtime 包装器（`functools.wraps` 保留原始签名以生成 schema）→ 发 `tool_call` 事件 → PermissionEngine（ask 时发 `permission_required` 并回调 CLI 询问）→ 执行 handler（文件类工具内部再过 Workspace 沙箱）→ 发 `tool_result` 事件。工具异常以 `Error: ...` 字符串返回给模型，不会中断 run。
- **流式**：`agent.run(event_stream_handler=...)` 接收 pydantic-ai 的 `PartDeltaEvent(TextPartDelta)` → `message_delta` 事件 → CLI 实时打印；结尾打印未流式输出的剩余文本。
- **会话持久化**：`result.all_messages()` 经 `ModelMessagesTypeAdapter` 序列化存入 `session_state` 表，同 session 下次运行恢复。
- **MCP 优雅降级**：`config/mcp.json` 缺失 → 跳过；配置错误或 server 构建失败 → warning 且不影响启动；工具调用失败经 `process_tool_call` 包装成错误文本返回模型，MCP 工具同样受 PermissionEngine 管控（`mcp:<tool>` 规则）。

## API 服务层

`src/guard_arch/api/app.py` 在 AgentRuntime 之上包了一层 FastAPI（对应设计文档二节的 Application API 层），让任何前端（Web/移动端）可以通过 HTTP 调用：

```text
Web / Mobile / 任意前端
        │ HTTP + SSE
        ▼
FastAPI (guard_arch.api.app, title="Guard Arch API")
        │  APIServer: session→workspace 映射 + (workspace, auto_approve)→Runtime 缓存
        ▼
AgentRuntime（与 CLI 完全共用同一套运行时）
```

端点：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | `{"ok": true}` |
| GET | `/api/v1/agents` | AgentRegistry 列表 |
| GET | `/api/v1/skills` | SkillRegistry 列表 |
| POST | `/api/v1/sessions` | 创建会话，返回 `{session_id, workspace}` |
| GET | `/api/v1/sessions/{id}/messages` | 从 Memory conversation 层读历史 |
| POST | `/api/v1/chat` | SSE 流式对话 |

关键机制：

- **SSE 翻译**：订阅 runtime EventBus（`*`），按 `session_id` 过滤后写入 `asyncio.Queue`，生成器逐条产出 `event: <类型>\ndata: <json>\n\n`；`agent_finished` / `error` 为终止事件，之后关闭流并取消订阅。
- **线程模型**：全异步。runtime.run 以 `asyncio.create_task` 跑在与请求相同的 uvicorn 事件循环上，事件经队列流向 SSE 生成器；多个并发 SSE 会话共享 per-workspace runtime，靠事件 data 里的 `session_id` 隔离，互不干扰。
- **权限**：API 模式无交互批准回调（`approval_handler=None`），ASK 一律拒绝但仍发 `permission_required` 事件供前端展示；`auto_approve=true` 为开发便利全部放行（生产应接确认回调）；DENY 规则（rm -rf 等）为绝对规则，auto_approve 不豁免。
- **会话下发**：`session_id` 省略时服务端新建，客户端从首个 `agent_started` 事件的 data 中取得。
- **CORS**：开发期全开，生产应收紧 origins。

## 已知限制 / 后续阶段

- 向量检索记忆（设计文档十二节的 Context Router）未实现，v1 按层全量 + 每层最多 10 条注入。
- Run 级文件快照/回滚（八节）留待 Phase 3，当前沙箱仅做路径隔离。
- MCP 工具在 ToolRegistry 中无静态条目（经 toolsets 直接挂到 pydantic-ai Agent），但权限与事件均生效。
- agents/skills/config 默认目录按源码仓库根解析；打包为独立 wheel 安装时需要指定路径。
- API 的 session→workspace 映射在内存中，进程重启丢失（conversation 历史本身在 SQLite 中持久化）。
- API 暂无鉴权，CORS 全开，仅限开发环境。
