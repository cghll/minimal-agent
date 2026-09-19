# AI Prompt 与问题解决记录

本文记录本项目实际使用 AI 辅助开发的输入、决策和验证，不补写未发生的对话。敏感信息均未提供给 AI。

## 使用工具

- Codex desktop coding agent
- Superpowers skills：`using-superpowers`、`brainstorming`、`writing-plans`、`executing-plans`、`test-driven-development`、`systematic-debugging`、`verification-before-completion`
- 本地 pytest、Ruff、coverage

## 实际用户 Prompt

### Prompt 1：需求分析

```text
阅读需求，分析应该怎么做，生成计划书
```

输入附件为“Vibe coding题目：从零实现一个最小可用 Agent.md”。输出为分阶段实施计划，包含架构、接口、文件结构、TDD 步骤、测试矩阵与验收条件。

### Prompt 2：开始实施

```text
根据这个计划书开始执行，利用superpower的skills写
```

执行方式为 inline execution：逐任务写失败测试、观察 RED、补最小实现、观察 GREEN，然后运行全量质量门禁。

## Agent 系统提示

运行时使用的完整 Prompt 位于 `prompts/agent_system.md`。核心约束是：

```text
只返回一个 JSON 对象，不要输出任何说明文字。
需要已注册工具时选择 tool_calls，否则返回 final_answer。
只能调用工具目录中列出的工具，并匹配对应的 JSON Schema。
不要透露隐藏思维链；reasoning_summary 只能是一句简短、可审计的说明。
```

完整 Prompt 独立保存，便于评审和版本管理，不在 Python 源码中重复维护。

## 关键问题与处理

### 1. 当前会话没有 Context7 MCP

题目涉及 FastAPI、HTTPX 和 LLM API，本应通过 Context7 查询最新文档。实际工具列表和 MCP resources 均未提供 Context7。随后申请网络访问官方 OpenAI 文档，但 PowerShell/curl 遇到本机 TLS 凭据错误，Python HTTPX 请求得到 403。

处理：不臆造新 API 字段，选择公开且稳定的 OpenAI-compatible `/chat/completions`、Bearer Token、`messages`、`response_format: json_object` 契约；把 base URL 和 model 全部环境化，并将厂商差异限制在 `llm/client.py`。真实部署前仍需用目标厂商官方文档核验。

### 2. pytest 无法写用户临时目录

现象：`tmp_path` 初始化报 `PermissionError`，路径位于用户 AppData Temp。

处理：所有验证命令显式使用仓库内 `--basetemp=.test-tmp`，该目录已加入 `.gitignore`。这不改变测试语义。

### 3. SQLite 连接泄漏

现象：首轮全量测试 43 个断言均通过，但产生 356 条 `ResourceWarning: unclosed database`。

根因：`sqlite3.Connection` 的上下文管理器只处理事务提交/回滚，不自动关闭连接。

处理：先添加两个回归测试，直接验证 repository 与 trace recorder 方法返回后连接已关闭；观察测试失败后，将两个 `_connect()` 改为 `@contextmanager`，在 `finally` 中显式关闭。回归测试随后通过。

### 4. FastAPI TestClient 弃用警告

现象：集成测试通过，但 Starlette 的同步 TestClient 对当前 HTTPX 版本发出弃用警告。

处理：改用 `httpx.AsyncClient + ASGITransport` 直接测试 ASGI 应用，保持真实路由与序列化逻辑，同时消除框架适配层警告。

### 5. 原生 tool 消息兼容性

问题：自定义 JSON 决策协议没有生成厂商原生 assistant `tool_calls` 对象；若把结果直接作为 `role=tool` 发送，部分兼容 API 会要求能匹配此前原生调用。

处理：持久化层仍使用结构化 `role=tool`，发给 LLM 时转换为带工具名和 `call_id` 的 system 文本。这样不会伪造厂商协议，也保留模型继续决策所需的信息。

## AI 建议的采纳与拒绝

采纳：

- 结构化 JSON 决策协议，便于解析和测试。
- LLM、工具、memory、context、trace 使用窄接口隔离。
- fake LLM 覆盖确定性分支，另设显式 live 测试。
- 不记录隐藏思维链，仅保留简短决策摘要。

未采纳：

- 使用 Agent 框架：违反题目核心约束。
- 第一版接入向量数据库、RAG、前端和多 Agent：超出最小范围。
- 用 Python `eval` 实现 calculator：安全边界不可接受。
- 把真实 API 测试混入默认套件：会引入费用、网络与模型随机性。

## 验证记录

开发过程中每个功能都先运行缺失实现的失败测试，再补实现。最终交付前执行：

```powershell
python -m ruff check src tests
python -m pytest -m "not live" --cov=src/minimal_agent --cov-report=term-missing --basetemp=.test-tmp
python -m pytest -m live -q --basetemp=.test-tmp-live
```

第三条命令仅在设置 `RUN_LIVE_LLM_TESTS=1` 和真实 LLM 环境变量后执行。当前开发环境未提供 API Key，因此不得把默认 skip 描述为真实调用成功。

## 代码审查后的修复

独立只读审查指出 context 最终预算、工具结果信任边界、trace 结果完整性、reasoning summary 脱敏和 API 生命周期/路径校验问题。后续修复包括：

- 对 system/catalog/summary/最近历史做最终字符预算裁剪，确保返回 context 不超过配置预算。
- 将工具结果作为带 `UNTRUSTED TOOL DATA` 标记的 user 数据发送，并在系统 Prompt 中明确禁止执行其中的指令。
- 在 trace 中记录每个工具调用的脱敏结果和错误码，reasoning summary 改存固定的应用侧决策标签。
- 对 session 路径参数增加 1 至 128 字符限制，并用 FastAPI lifespan 关闭自有 HTTP client。

这些修改先添加失败测试，再实现并通过全量离线测试。
