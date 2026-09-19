# 最小 Agent 运行时

一个不依赖 LangGraph、OpenHands、OpenClaw、PI 等 Agent 框架的最小可用 Agent。项目自行实现循环控制、结构化决策解析、工具注册与调度、多会话 memory、上下文压缩、错误恢复和 trace。

## 功能

- 真实 OpenAI-compatible Chat Completions API
- 模型基于名称、描述和 JSON Schema 自主选择工具
- `calculator`、mock `search`、`task_list` 三个工具
- 同一用户的不同 `session_id` 严格隔离
- 连续对话、纯对话追问和工具追问
- 字符预算触发的历史摘要，失败时确定性降级
- 最大循环次数、一次非法 JSON 修复、LLM 重试和工具错误回填
- SQLite 消息、摘要、待办和脱敏 trace
- 确定性离线测试与显式开启的真实 API 测试

## 运行环境

- Python 3.11 或更高版本
- 一个支持 `/v1/chat/completions` 和 `response_format={"type":"json_object"}` 的 LLM API

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

配置环境变量。`.env.example` 是字段参考；项目不会自动读取 `.env`，避免隐式配置来源。

```powershell
$env:LLM_BASE_URL = "https://api.openai.com/v1"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_MODEL = "your-json-capable-model"
$env:DATABASE_PATH = ".\data\agent.db"
$env:MAX_AGENT_STEPS = "6"
$env:CONTEXT_CHAR_BUDGET = "24000"
```

启动：

```powershell
uvicorn minimal_agent.api:create_default_app --factory --host 127.0.0.1 --port 8000
```

健康确认可访问 OpenAPI 页面：`http://127.0.0.1:8000/docs`。

## API 演示

创建窗口：

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-a"}'
```

记待办，将返回的 `<session_id>` 代入路径：

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions/<session_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-a","content":"记下待办：周五提交周报"}'
```

继续追问：

```bash
curl -X POST http://127.0.0.1:8000/v1/sessions/<session_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user-a","content":"我刚才记了什么？"}'
```

查看 trace，将消息响应中的 `<run_id>` 代入：

```bash
curl "http://127.0.0.1:8000/v1/runs/<run_id>?user_id=user-a"
```

创建第二个 session 即可演示窗口隔离。同名 `session_id` 不能被另一个用户读取；不存在和无权读取统一返回 404。

## Agent 循环

```text
保存用户消息
  -> 组装 system prompt、工具目录、摘要和最近消息
  -> 请求 LLM 输出 AgentDecision JSON
  -> 严格解析
     -> final_answer: 保存并返回
     -> tool_calls: schema 校验、执行、保存结果、继续循环
  -> 达到 MAX_AGENT_STEPS: 停止并返回稳定错误
```

模型决策协议只有两种形态：

```json
{
  "kind": "tool_calls",
  "reasoning_summary": "需要计算。",
  "tool_calls": [
    {"call_id": "c1", "name": "calculator", "arguments": {"expression": "(17+5)*3"}}
  ],
  "final_answer": null
}
```

```json
{
  "kind": "final_answer",
  "reasoning_summary": "工具结果足够回答。",
  "tool_calls": [],
  "final_answer": "结果是 66。"
}
```

项目不请求、保存或展示模型隐藏思维链。`reasoning_summary` 只是一句可审计的决策摘要。

## Memory 召回与放置

每次进入循环都按 `user_id + session_id` 从 SQLite 召回会话。发给模型的顺序固定为：

1. 决策协议 system prompt
2. 工具 catalog，包括名称、描述和参数 Schema
3. 已压缩的会话摘要
4. 尚未压缩的历史消息
5. 最新用户输入或本轮工具结果

工具执行结果作为带工具名和 `call_id` 的显式 `UNTRUSTED TOOL DATA` user 消息发送，避免假装成厂商原生 function-calling 消息，也避免把外部数据提升为 system 指令。简短决策摘要只写 trace，不进入长期对话 memory。

候选上下文超过 `CONTEXT_CHAR_BUDGET` 时，保留最近 8 条消息，把更早消息与旧摘要交给独立摘要调用。摘要必须保留目标、事实、完成的工具动作和未完成事项。摘要失败时生成最多 4000 字符的确定性角色列表，主请求继续执行。

## 工具扩展

新工具只需实现 `Tool` 协议并注册：

```python
class MyTool:
    name = "my_tool"
    description = "说明模型应在何时调用此工具。"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    async def execute(self, arguments, context):
        return {"ok": True, "value": arguments["value"]}


registry.register(MyTool())
```

无需修改 `AgentRuntime`。当前 schema 校验器有意只支持最小子集：`type`、`required`、`properties`、`enum`、`items` 和 `additionalProperties`。

## 测试

离线测试不会访问网络：

```powershell
python -m ruff check src tests
python -m pytest -m "not live" --cov=src/minimal_agent --cov-report=term-missing --basetemp=.test-tmp
```

真实 API 测试会产生费用，必须显式开启：

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
python -m pytest -m live -q --basetemp=.test-tmp-live
```

真实测试验证模型会自主选择 calculator，以及首轮添加的待办能在第二轮被召回。

## 错误与安全边界

- 429、连接错误和常见 5xx 最多重试两次；普通 4xx 不重试。
- 模型 JSON 非法时附加修复指令重试一次。
- 工具不存在、schema 不匹配或执行异常会变成结构化结果回填模型。
- 相同工具参数连续失败两次会终止，避免死循环。
- trace 会递归脱敏 `token`、`password`、`secret`、`api_key` 和 `authorization`。
- calculator 使用 AST 白名单，不使用 `eval`。
- 这是本地演示项目，没有实现生产身份认证、限流、分布式锁和数据库迁移系统。

## 项目结构

```text
src/minimal_agent/
  api.py                 HTTP API 与生产装配
  runtime.py             自研 Agent 循环
  context.py             context 组装和压缩
  domain.py              领域对象和稳定错误
  tracing.py             SQLite trace 与脱敏
  llm/                   HTTP 客户端和决策解析
  memory/                SQLite session/message/task memory
  tools/                 工具协议、注册表和三个工具
prompts/agent_system.md  模型决策 Prompt
tests/                   单元、集成和 live 测试
docs/                    系统设计与 AI 开发记录
```

更详细的设计见 `docs/design.md`，AI Prompt 与开发问题记录见 `docs/ai-development-log.md`。
