"""Context manager for wrapping tool calls with automatic audit logging.

Usage::

    from mcp_audit import AuditEngine, traced_call

    engine = AuditEngine()
    session = engine.start_session(agent_id="my-agent")

    with traced_call(engine, session_id=session.id, tool_name="search") as tc:
        tc.set_tokens(input_tokens=500, output_tokens=200)
        tc.set_cost(0.003)
        result = do_something()
        tc.set_result(result)
"""
from __future__ import annotations

import time
from typing import Any

from .engine import AuditEngine
from .models import CallStatus, ToolCall


class traced_call:
    """Context manager that records a tool call automatically.

    The call is started on ``__enter__`` and finalised (with duration,
    status, result) on ``__exit__``.  If an exception occurs inside the
    ``with`` block the call is recorded as ``CallStatus.ERROR``.

    Example::

        with traced_call(engine, session_id=sid, tool_name="fetch_url") as tc:
            tc.set_cost(0.001)
            data = requests.get(url).json()
            tc.set_result(data)
    """

    def __init__(
        self,
        engine: AuditEngine,
        session_id: str,
        tool_name: str,
        *,
        agent_id: str | None = None,
        server_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._engine = engine
        self._session_id = session_id
        self._tool_name = tool_name
        self._agent_id = agent_id
        self._server_name = server_name
        self._arguments = arguments or {}
        self._tags = tags or []
        self._metadata = metadata or {}
        self._result: Any = None
        self._error: str | None = None
        self._status: CallStatus = CallStatus.SUCCESS
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_usd: float = 0.0
        self._start: float = 0.0
        self.call: ToolCall | None = None

    # ── setters for the user to populate inside the with-block ──

    def set_result(self, result: Any) -> None:
        self._result = result

    def set_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def set_cost(self, cost_usd: float) -> None:
        self._cost_usd = cost_usd

    def add_tag(self, tag: str) -> None:
        self._tags.append(tag)

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    # ── context manager protocol ─────────────────────────────────

    def __enter__(self) -> "traced_call":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        duration_ms = round((time.monotonic() - self._start) * 1000, 2)
        if exc_type is not None:
            self._status = CallStatus.ERROR
            self._error = f"{exc_type.__name__}: {exc_val}"
        self.call = self._engine.record_call(
            session_id=self._session_id,
            tool_name=self._tool_name,
            agent_id=self._agent_id,
            server_name=self._server_name,
            arguments=self._arguments,
            result=self._result,
            status=self._status,
            error=self._error,
            duration_ms=duration_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost_usd,
            tags=self._tags,
            metadata=self._metadata,
        )
        # Don't suppress the exception
        return False
