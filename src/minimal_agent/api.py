from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from minimal_agent.config import Settings
from minimal_agent.context import ContextBuilder, JSONSummarizer
from minimal_agent.domain import (
    AgentError,
    LLMUnavailable,
    MaxStepsExceeded,
    SessionNotFoundError,
)
from minimal_agent.llm.client import OpenAICompatibleClient
from minimal_agent.llm.parser import DecisionParser
from minimal_agent.memory.sqlite_repository import SQLiteRepository
from minimal_agent.runtime import AgentRuntime
from minimal_agent.tools.calculator import CalculatorTool
from minimal_agent.tools.registry import ToolRegistry
from minimal_agent.tools.search import SearchItem, SearchTool
from minimal_agent.tools.task_list import TaskListTool
from minimal_agent.tracing import SQLiteTraceRecorder


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)


class CreateMessageRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=10_000)


def create_app(
    *,
    runtime: AgentRuntime,
    repository: SQLiteRepository,
    trace: SQLiteTraceRecorder,
    llm_client: OpenAICompatibleClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        yield
        if llm_client is not None:
            await llm_client.close()

    app = FastAPI(title="最小 Agent 运行时", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        request: Request, exc: SessionNotFoundError
    ) -> JSONResponse:
        del request
        return _error_response(404, exc.code, "会话不存在")

    @app.exception_handler(MaxStepsExceeded)
    async def max_steps_handler(request: Request, exc: MaxStepsExceeded) -> JSONResponse:
        del request
        return _error_response(409, exc.code, str(exc), exc.run_id)

    @app.exception_handler(LLMUnavailable)
    async def llm_unavailable_handler(request: Request, exc: LLMUnavailable) -> JSONResponse:
        del request
        return _error_response(503, exc.code, "LLM 服务不可用")

    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
        del request
        return _error_response(400, exc.code, str(exc))

    @app.post("/v1/sessions", status_code=201)
    async def create_session(body: CreateSessionRequest) -> dict[str, str]:
        session_id = str(uuid.uuid4())
        repository.create_session(body.user_id, session_id)
        return {"session_id": session_id, "user_id": body.user_id}

    @app.post("/v1/sessions/{session_id}/messages")
    async def create_message(
        session_id: str = ApiPath(min_length=1, max_length=128),
        body: CreateMessageRequest = ...,
    ) -> dict[str, object]:
        repository.get_session(body.user_id, session_id)
        result = await runtime.run(body.user_id, session_id, body.content)
        return {
            "session_id": session_id,
            "run_id": result.run_id,
            "answer": result.answer,
            "step_count": result.step_count,
        }

    @app.get("/v1/sessions/{session_id}/messages")
    async def list_messages(
        session_id: str = ApiPath(min_length=1, max_length=128),
        user_id: str = Query(min_length=1, max_length=128),
    ) -> dict[str, object]:
        messages = repository.list_messages(user_id, session_id)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                }
                for message in messages
            ],
        }

    @app.get("/v1/runs/{run_id}")
    async def get_run(
        run_id: str,
        user_id: str = Query(min_length=1, max_length=128),
    ) -> dict[str, object]:
        try:
            result = trace.get_trace(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="运行记录不存在") from exc
        if result["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="运行记录不存在")
        return result

    return app


def create_default_app() -> FastAPI:
    settings = Settings.from_env()
    repository = SQLiteRepository(settings.database_path)
    repository.initialize()
    trace = SQLiteTraceRecorder(settings.database_path)
    trace.initialize()
    llm = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    registry = ToolRegistry(
        [
            CalculatorTool(),
            SearchTool(
                {
                    "上海明天天气": [
                        SearchItem(
                            title="上海天气（演示数据）",
                            snippet="晴，最高 26C，最低 18C。",
                            url="mock://weather/shanghai/tomorrow",
                        )
                    ]
                }
            ),
            TaskListTool(),
        ]
    )
    prompt_path = Path(__file__).parents[2] / "prompts" / "agent_system.md"
    context_builder = ContextBuilder(
        repository=repository,
        registry=registry,
        summarizer=JSONSummarizer(llm),
        system_prompt=prompt_path.read_text(encoding="utf-8"),
        char_budget=settings.context_char_budget,
    )
    runtime = AgentRuntime(
        llm=llm,
        parser=DecisionParser(),
        registry=registry,
        repository=repository,
        context_builder=context_builder,
        trace=trace,
        max_steps=settings.max_agent_steps,
    )
    return create_app(runtime=runtime, repository=repository, trace=trace, llm_client=llm)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    run_id: str | None = None,
) -> JSONResponse:
    error: dict[str, str] = {"code": code, "message": message}
    if run_id:
        error["run_id"] = run_id
    return JSONResponse(status_code=status_code, content={"error": error})
