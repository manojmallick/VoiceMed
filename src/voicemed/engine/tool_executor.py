"""Executes whitelisted triage tools."""

from __future__ import annotations

from typing import Any

from voicemed.engine.tools import TOOL_REGISTRY


class ToolExecutor:
    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_REGISTRY:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        func = TOOL_REGISTRY[name]
        try:
            payload = func(**arguments)
            return {"ok": True, "result": payload}
        except TypeError as exc:
            return {"ok": False, "error": f"Invalid args for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "error": f"Tool execution failed: {exc}"}
