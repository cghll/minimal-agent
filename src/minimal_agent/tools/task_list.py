from __future__ import annotations

from minimal_agent.domain import JsonObject, ToolValidationError
from minimal_agent.tools.base import ToolContext


class TaskListTool:
    name = "task_list"
    description = "Add, list, or complete tasks in the current conversation session."
    parameters: JsonObject = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "complete"]},
            "text": {"type": "string"},
            "task_id": {"type": "integer"},
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: JsonObject, context: ToolContext) -> JsonObject:
        action = arguments["action"]
        if action == "list":
            return {
                "ok": True,
                "tasks": context.repository.list_tasks(context.user_id, context.session_id),
            }
        if action == "add":
            text = str(arguments.get("text", "")).strip()
            if not text:
                raise ToolValidationError("text is required when action is add")
            task = context.repository.add_task(context.user_id, context.session_id, text)
            return {"ok": True, "task": task}
        if "task_id" not in arguments:
            raise ToolValidationError("task_id is required when action is complete")
        try:
            task = context.repository.complete_task(
                context.user_id, context.session_id, int(arguments["task_id"])
            )
        except KeyError as exc:
            raise ToolValidationError(f"task not found: {arguments['task_id']}") from exc
        return {"ok": True, "task": task}
