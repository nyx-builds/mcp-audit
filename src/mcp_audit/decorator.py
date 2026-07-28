"""Decorator for automatic tool-call instrumentation.

Usage::

    from mcp_audit import AuditEngine
    from mcp_audit.decorator import audit_call

    engine = AuditEngine()
    session = engine.start_session(agent_id="my-agent")

    @audit_call(engine, session_id=session.id)
    def search(query: str) -> list[dict]:
        return [{"title": "result", "url": "..."}]

    # Every call to search() is now automatically recorded:
    results = search("best MCP servers")
    # → creates a ToolCall with timing, status, result

The decorator automatically:
- Records the call duration (wall-clock)
- Captures the return value as the result
- Marks errors/exceptions as ``CallStatus.ERROR``
- Optionally estimates cost via a user-provided function
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from .engine import AuditEngine
from .models import CallStatus

F = TypeVar("F", bound=Callable[..., Any])


class _AuditContext:
    """Thread-local context for tracking the current audit session and engine.

    This enables nested ``audit_call`` decorators and global session
    management without passing session_id to every function.
    """

    def __init__(self) -> None:
        import threading

        self._local = threading.local()

    @property
    def engine(self) -> AuditEngine | None:
        return getattr(self._local, "engine", None)

    @property
    def session_id(self) -> str | None:
        return getattr(self._local, "session_id", None)

    def set(self, engine: AuditEngine | None, session_id: str | None) -> None:
        self._local.engine = engine
        self._local.session_id = session_id

    def reset(self) -> None:
        prev_engine = getattr(self._local, "prev_engine", None)
        prev_session = getattr(self._local, "prev_session", None)
        self._local.engine = prev_engine
        self._local.session_id = prev_session
        self._local.prev_engine = None
        self._local.prev_session = None


# Global context singleton
_audit_ctx = _AuditContext()


def get_audit_context() -> _AuditContext:
    """Get the global audit context for programmatic access."""
    return _audit_ctx


def audit_call(
    engine: AuditEngine | None = None,
    *,
    session_id: str | None = None,
    tool_name: str | None = None,
    server_name: str | None = None,
    cost_fn: Callable[..., float] | None = None,
    token_fn: Callable[..., tuple[int, int]] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    record_errors: bool = True,
) -> Callable[[F], F]:
    """Decorator that records every invocation of the wrapped function.

    Parameters
    ----------
    engine:
        The AuditEngine to record to.  If ``None``, uses the engine from
        the global audit context (set via ``bind_session``).
    session_id:
        Session to associate calls with.  If ``None``, uses the session
        from the global context.
    tool_name:
        Override the tool name (defaults to the function's ``__name__``).
    server_name:
        MCP server name (optional, for multi-server setups).
    cost_fn:
        Optional callable ``(func, args, kwargs, result_or_error) -> float``
        that computes the cost in USD for each call.
    token_fn:
        Optional callable ``(func, args, kwargs, result_or_error) -> (in, out)``
        that returns input/output token counts.
    tags:
        Static tags to attach to every call.
    metadata:
        Static metadata to attach to every call.
    record_errors:
        If ``True`` (default), exceptions are recorded as ``CallStatus.ERROR``
        and then re-raised.  If ``False``, exceptions propagate without recording.

    Example::

        @audit_call(engine, session_id=sid, cost_fn=lambda *_: 0.001)
        def fetch_weather(city: str) -> dict:
            return {"temp": 72}

        @audit_call(engine, session_id=sid, tool_name="llm_complete")
        def complete(prompt: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            eng = engine or _audit_ctx.engine
            sid = session_id or _audit_ctx.session_id

            if eng is None or sid is None:
                # No engine/session bound — just call the function
                return func(*args, **kwargs)

            start = time.monotonic()
            status = CallStatus.SUCCESS
            error_msg: str | None = None
            result: Any = None
            caught_exc: BaseException | None = None

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                status = CallStatus.ERROR
                error_msg = f"{type(exc).__name__}: {exc}"
                caught_exc = exc
            finally:
                duration_ms = round((time.monotonic() - start) * 1000, 2)

                # Only record if it's a success OR record_errors is enabled
                if status == CallStatus.SUCCESS or record_errors:
                    cost = 0.0
                    if cost_fn is not None:
                        try:
                            cost = cost_fn(func, args, kwargs, result)
                        except Exception:
                            pass

                    in_tok = out_tok = 0
                    if token_fn is not None:
                        try:
                            tok = token_fn(func, args, kwargs, result)
                            in_tok, out_tok = tok
                        except Exception:
                            pass

                    eng.record_call(
                        session_id=sid,
                        tool_name=name,
                        server_name=server_name,
                        arguments={"args": list(args), "kwargs": dict(kwargs)},
                        result=result,
                        status=status,
                        error=error_msg,
                        duration_ms=duration_ms,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cost_usd=cost,
                        tags=tags or [],
                        metadata=metadata or {},
                    )

            if caught_exc is not None:
                raise caught_exc
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def bind_session(engine: AuditEngine, session_id: str) -> _AuditContext:
    """Bind an engine and session to the current thread's audit context.

    After calling this, any ``@audit_call`` decorator without explicit
    engine/session will use these values.

    Returns the context so you can call ``reset()`` later.

    Example::

        ctx = bind_session(engine, session.id)

        @audit_call()  # uses bound engine + session
        def search(query):
            ...

        ctx.reset()  # unbind
    """
    # Save previous values for restoration
    _audit_ctx._local.prev_engine = _audit_ctx.engine
    _audit_ctx._local.prev_session = _audit_ctx.session_id
    _audit_ctx.set(engine, session_id)
    return _audit_ctx
