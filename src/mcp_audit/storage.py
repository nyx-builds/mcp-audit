"""Storage layer — pluggable backends for persisting audit records.

v0.1 ships an in-memory backend that keeps the last N records (configurable
ring buffer).  This is sufficient for local agents and testing.  Future
versions will add SQLite, Postgres, and OpenTelemetry exporters.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol

from .models import AlertRule, Session, ToolCall, TraceEvent


class AuditStore(Protocol):
    """Storage protocol for audit data."""

    def save_call(self, call: ToolCall) -> None: ...
    def save_session(self, session: Session) -> None: ...
    def save_event(self, event: TraceEvent) -> None: ...
    def save_rule(self, rule: AlertRule) -> None: ...
    def delete_rule(self, rule_id: str) -> bool: ...

    def get_call(self, call_id: str) -> ToolCall | None: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def get_rule(self, rule_id: str) -> AlertRule | None: ...

    def query_calls(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        server_name: str | None = None,
        status: str | None = None,
        min_cost: float | None = None,
        min_duration: float | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolCall]: ...

    def query_sessions(
        self,
        *,
        agent_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Session]: ...

    def query_events(
        self,
        *,
        trace_id: str | None = None,
        call_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[TraceEvent]: ...

    def all_rules(self) -> list[AlertRule]: ...
    def count_calls(self) -> int: ...


class MemoryStore:
    """In-memory ring-buffer store.

    Parameters
    ----------
    max_calls:
        Maximum number of tool calls to retain (oldest evicted first).
    """

    def __init__(self, max_calls: int = 10_000) -> None:
        self._calls: deque[ToolCall] = deque(maxlen=max_calls)
        self._call_index: dict[str, ToolCall] = {}
        self._sessions: dict[str, Session] = {}
        self._events: deque[TraceEvent] = deque(maxlen=max_calls * 5)
        self._rules: dict[str, AlertRule] = {}
        self._max_calls = max_calls

    # ── writes ──────────────────────────────────────────────────────

    def save_call(self, call: ToolCall) -> None:
        # Maintain index — evict oldest from index when ring wraps
        if len(self._calls) == self._calls.maxlen:
            oldest = self._calls[0]
            self._call_index.pop(oldest.id, None)
        self._calls.append(call)
        self._call_index[call.id] = call

    def save_session(self, session: Session) -> None:
        self._sessions[session.id] = session

    def save_event(self, event: TraceEvent) -> None:
        self._events.append(event)

    def save_rule(self, rule: AlertRule) -> None:
        self._rules[rule.id] = rule

    def delete_rule(self, rule_id: str) -> bool:
        return self._rules.pop(rule_id, None) is not None

    # ── single reads ────────────────────────────────────────────────

    def get_call(self, call_id: str) -> ToolCall | None:
        return self._call_index.get(call_id)

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_rule(self, rule_id: str) -> AlertRule | None:
        return self._rules.get(rule_id)

    # ── queries ─────────────────────────────────────────────────────

    def query_calls(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        server_name: str | None = None,
        status: str | None = None,
        min_cost: float | None = None,
        min_duration: float | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolCall]:
        results: list[ToolCall] = []
        # newest first
        for call in reversed(self._calls):
            if session_id and call.session_id != session_id:
                continue
            if agent_id and call.agent_id != agent_id:
                continue
            if tool_name and call.tool_name != tool_name:
                continue
            if server_name and call.server_name != server_name:
                continue
            if status and call.status.value != status:
                continue
            if min_cost is not None and call.cost_usd < min_cost:
                continue
            if min_duration is not None and (call.duration_ms or 0) < min_duration:
                continue
            if tag and tag not in call.tags:
                continue
            results.append(call)
        return results[offset : offset + limit]

    def query_sessions(
        self,
        *,
        agent_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Session]:
        results = list(self._sessions.values())
        if agent_id:
            results = [s for s in results if s.agent_id == agent_id]
        if active_only:
            results = [s for s in results if s.is_active]
        results.sort(key=lambda s: s.started_at, reverse=True)
        return results[:limit]

    def query_events(
        self,
        *,
        trace_id: str | None = None,
        call_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[TraceEvent]:
        results: list[TraceEvent] = []
        for event in reversed(self._events):
            if trace_id and event.trace_id != trace_id:
                continue
            if call_id and event.call_id != call_id:
                continue
            if severity and event.severity.value != severity:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results

    def all_rules(self) -> list[AlertRule]:
        return list(self._rules.values())

    def count_calls(self) -> int:
        return len(self._calls)

    # ── internals for testing ───────────────────────────────────────

    def clear(self) -> None:
        """Wipe all data (testing helper)."""
        self._calls.clear()
        self._call_index.clear()
        self._sessions.clear()
        self._events.clear()
        self._rules.clear()
