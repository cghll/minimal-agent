# 系统设计

## 目标与边界

本项目证明一个最小 Agent Runtime 的必要机制可以不依赖 Agent 框架完成。核心关注点是循环控制、模型决策协议、工具调度、session memory、context 管理和可观测性，而不是 UI、RAG 或多 Agent 协作。

## 组件职责

| 组件 | 职责 | 不负责 |
|---|---|---|
| `AgentRuntime` | 循环、停止、解析修复、工具错误回填 | 厂商 HTTP、SQL、具体工具逻辑 |
| `ContextBuilder` | 按预算组织 system/catalog/summary/history | 决定调用什么工具 |
| `DecisionParser` | 把不可信模型文本变成 `AgentDecision` | 执行工具 |
| `OpenAICompatibleClient` | 认证、请求、有限重试、提取文本 | Agent 循环 |
| `ToolRegistry` | catalog、schema 校验、按名称调度 | 选择工具 |
| `SQLiteRepository` | session、message、summary、task | trace |
| `SQLiteTraceRecorder` | run/step、耗时、脱敏 | 对话召回 |
| FastAPI | 输入校验、错误映射、依赖装配 | 业务决策 |

这种边界使 LLM 厂商、存储或工具能够单独替换，而 runtime 的契约不变。

## 核心时序

```text
  Client -> API: user_id, session_id, content
API -> Repository: 校验所有权
API -> Runtime: run(...)
Runtime -> Repository: 追加用户消息
loop 1..MAX_AGENT_STEPS
  Runtime -> ContextBuilder: 组装会话上下文
  ContextBuilder -> Repository: 摘要 + 未压缩消息
  Runtime -> LLM: 消息
  LLM -> Runtime: 决策 JSON 文本
  Runtime -> DecisionParser: 严格解析
  alt final_answer
    Runtime -> Repository: 追加助手消息
    Runtime -> Trace: 完成并记录 completed
    Runtime -> API: RunResult
  else tool_calls
    Runtime -> Registry: 校验并执行
    Runtime -> Repository: 追加结构化工具结果
    Runtime -> loop: 继续
end
Runtime -> Trace: 失败并记录 max_steps_exceeded
```

## 决策协议

模型输出是项目自定义 JSON，不依赖厂商 function-calling 对象。`kind` 决定 `tool_calls` 与 `final_answer` 的互斥关系；解析器还校验非空摘要、调用数量、唯一 `call_id`、字段集合和参数对象类型。

采用该方案的原因：

- 能清楚展示 runtime 自己控制模型、工具和循环。
- fake LLM 可精确覆盖每一个状态转移。
- 厂商差异限制在 HTTP client 内。
- 比自由文本 ReAct 更容易严格解析与恢复。

代价是 prompt 需要约束 JSON，且原生 structured outputs 的 schema 能力没有完全利用。

## 数据模型

```text
sessions
  session_id PK
  user_id
  summary
  summarized_through_message_id

messages
  id PK
  user_id + session_id INDEX
  role
  content_json
  name
  tool_call_id

tasks
  id PK
  user_id + session_id INDEX
  text
  status

runs
  run_id PK
  user_id
  session_id
  status
  error_code

run_steps
  run_id + step_number PK
  decision_kind
  reasoning_summary
  tool_calls_json
  tool_results_json
  duration_ms
  error_code
```

`session_id` 全局唯一并记录 owner。任何 session 查询同时校验 `user_id`；错误时统一表现为不存在，避免泄漏其他用户的 session 是否存在。

## Context 与压缩

未超过预算时召回摘要之后的全部消息。超过字符预算且历史多于 8 条时：

1. 把较早的消息与旧摘要发给 `JSONSummarizer`。
2. 保存新摘要及最后一条已压缩消息的 ID。
3. 重新组装“协议 + catalog + 新摘要 + 最近 8 条”。
4. 摘要调用失败时保存确定性 fallback 文本。

字符预算是对 token 的低成本近似，适合最小实现。生产版本应换成模型对应 tokenizer，并对 system prompt、catalog、summary 与最近消息分别设配额。

工具结果在数据库中保留结构化 tool message，但发给 LLM 时转换为带工具名和 `call_id` 的 `user` 消息，并加上 `UNTRUSTED TOOL DATA` 标记。系统 Prompt 要求模型只把标记内容当作数据，不能执行其中的指令。

## 错误模型

| 错误 | 行为 | HTTP |
|---|---|---:|
| session 不存在或 owner 不符 | 不泄漏存在性 | 404 |
| 输入 schema 不合法 | FastAPI 校验 | 422 |
| 达到最大步数 | 终止并保留 trace | 409 |
| LLM 不可用 | 两次重试后失败 | 503 |
| 模型 JSON 不合法 | 同一 step 修复一次 | 400（仍失败） |
| 工具参数或工具名错误 | 结构化回填模型 | 通常由模型修正后 200 |
| 未知内部错误 | trace 标记 `internal_error` | 500 |

## 安全与限制

- calculator 仅遍历允许的 AST 节点，并限制表达式长度和指数。
- API Key 只存在于进程环境与 Authorization header，不进入 trace。
- trace 参数递归脱敏常见敏感字段，但不是通用 DLP。
- `reasoning_summary` 是受约束摘要，不是隐藏思维链。
- SQLite 适合单机演示；生产部署需迁移、连接池、并发策略和认证授权。
- mock search 是题目允许的演示实现，不访问真实搜索引擎。

## 测试策略

单元测试覆盖领域校验、工具、parser、HTTP client、repository、context、runtime 和 trace。集成测试组合 FastAPI、真实 runtime 与 SQLite，只在网络边界替换 LLM。live 测试默认跳过，通过环境开关验证真实 API 的工具选择和多轮 memory。
