"""Core audit engine — the business logic layer.

The engine wraps the storage backend and provides high-level operations:
recording calls, computing aggregates, evaluating alert rules, and
generating reports.  The MCP server and CLI both delegate to this.
"""
from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import (
    AgentReport,
    AlertRule,
    CallStatus,
    Session,
    Severity,
    ToolCall,
    TraceEvent,
)
from .storage import AuditStore, MemoryStore


class AuditEngine:
    """High-level audit / observability operations."""

    def __init__(self, store: AuditStore | None = None) -> None:
        self.store: AuditStore = store or MemoryStore()

    # ── Sessions ────────────────────────────────────────────────────

    def start_session(
        self,
        agent_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create and persist a new agent session."""
        session = Session(
            agent_id=agent_id,
            name=name,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.store.save_session(session)
        return session

    def end_session(self, session_id: str) -> Session | None:
        """Mark a session as ended and update aggregates."""
        session = self.store.get_session(session_id)
        if session is None:
            return None
        calls = self.store.query_calls(session_id=session_id, limit=10_000)
        session.total_calls = len(calls)
        session.error_count = sum(1 for c in calls if c.is_error)
        session.total_cost_usd = round(sum(c.cost_usd for c in calls), 6)
        session.total_tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        session.end()
        self.store.save_session(session)
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.store.get_session(session_id)

    def list_sessions(
        self,
        agent_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Session]:
        return self.store.query_sessions(
            agent_id=agent_id, active_only=active_only, limit=limit
        )

    # ── Tool calls ──────────────────────────────────────────────────

    def record_call(
        self,
        session_id: str,
        tool_name: str,
        *,
        agent_id: str | None = None,
        server_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        status: CallStatus = CallStatus.SUCCESS,
        error: str | None = None,
        duration_ms: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCall:
        """Record a completed tool call.

        This is the primary ingestion method.  Pass the full result/error
        and timing data — the engine handles storage and aggregation.
        """
        call = ToolCall(
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments or {},
            result=result,
            status=status,
            error=error,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            tags=tags or [],
            metadata=metadata or {},
        )
        # If duration not provided, mark as completed now
        if call.duration_ms is None and call.completed_at is None:
            call.completed_at = call.started_at
            call.duration_ms = 0.0

        self.store.save_call(call)

        # Update live session aggregates
        session = self.store.get_session(session_id)
        if session is not None:
            session.total_calls += 1
            if call.is_error:
                session.error_count += 1
            session.total_cost_usd = round(session.total_cost_usd + call.cost_usd, 6)
            session.total_tokens += call.input_tokens + call.output_tokens
            self.store.save_session(session)

        return call

    def get_call(self, call_id: str) -> ToolCall | None:
        return self.store.get_call(call_id)

    def query_calls(self, **kwargs: Any) -> list[ToolCall]:
        """Query tool calls with flexible filters."""
        return self.store.query_calls(**kwargs)

    # ── Trace events ────────────────────────────────────────────────

    def log_event(
        self,
        trace_id: str,
        event_type: str,
        message: str = "",
        *,
        call_id: str | None = None,
        severity: Severity = Severity.INFO,
        data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> TraceEvent:
        """Log a structured trace event."""
        event = TraceEvent(
            trace_id=trace_id,
            call_id=call_id,
            event_type=event_type,
            message=message,
            severity=severity,
            data=data or {},
            duration_ms=duration_ms,
        )
        self.store.save_event(event)
        return event

    def query_events(self, **kwargs: Any) -> list[TraceEvent]:
        return self.store.query_events(**kwargs)

    # ── Aggregates / Stats ──────────────────────────────────────────

    def get_stats(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute aggregate statistics over matching calls."""
        calls = self.store.query_calls(
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=100_000,
        )
        if not calls:
            return self._empty_stats()

        total = len(calls)
        errors = sum(1 for c in calls if c.status == CallStatus.ERROR)
        timeouts = sum(1 for c in calls if c.status == CallStatus.TIMEOUT)
        blocked = sum(1 for c in calls if c.status == CallStatus.BLOCKED)
        successes = total - errors - timeouts - blocked
        durations = [c.duration_ms or 0.0 for c in calls]
        costs = [c.cost_usd for c in calls]
        tokens = [c.input_tokens + c.output_tokens for c in calls]

        # Tool frequency
        tool_counter = Counter(c.tool_name for c in calls)
        server_counter = Counter(c.server_name or "unknown" for c in calls)

        return {
            "total_calls": total,
            "success_count": successes,
            "error_count": errors,
            "timeout_count": timeouts,
            "blocked_count": blocked,
            "error_rate": round(errors / total * 100, 2) if total else 0.0,
            "timeout_rate": round(timeouts / total * 100, 2) if total else 0.0,
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_usd": round(sum(costs) / total, 6) if total else 0.0,
            "total_tokens": sum(tokens),
            "avg_latency_ms": round(statistics.mean(durations), 2) if durations else 0.0,
            "p50_latency_ms": round(statistics.median(durations), 2) if durations else 0.0,
            "p95_latency_ms": round(self._percentile(durations, 95), 2),
            "p99_latency_ms": round(self._percentile(durations, 99), 2),
            "min_latency_ms": round(min(durations), 2) if durations else 0.0,
            "max_latency_ms": round(max(durations), 2) if durations else 0.0,
            "top_tools": [
                {"tool": name, "count": count}
                for name, count in tool_counter.most_common(10)
            ],
            "top_servers": [
                {"server": name, "count": count}
                for name, count in server_counter.most_common(10)
            ],
            "unique_tools": len(tool_counter),
            "unique_sessions": len({c.session_id for c in calls}),
        }

    def get_agent_report(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
    ) -> AgentReport:
        """Generate a comprehensive performance report for an agent."""
        calls = self.store.query_calls(
            agent_id=agent_id, session_id=session_id, limit=100_000
        )
        if not calls:
            return AgentReport(agent_id=agent_id)

        total = len(calls)
        errors = sum(1 for c in calls if c.status == CallStatus.ERROR)
        timeouts = sum(1 for c in calls if c.status == CallStatus.TIMEOUT)
        blocked = sum(1 for c in calls if c.status == CallStatus.BLOCKED)
        durations = [c.duration_ms or 0.0 for c in calls]
        tool_counter = Counter(c.tool_name for c in calls)

        sessions = self.store.query_sessions(agent_id=agent_id, limit=100_000)

        return AgentReport(
            agent_id=agent_id,
            session_count=len(sessions),
            total_calls=total,
            success_count=total - errors - timeouts - blocked,
            error_count=errors,
            timeout_count=timeouts,
            blocked_count=blocked,
            error_rate=round(errors / total * 100, 2) if total else 0.0,
            total_cost_usd=round(sum(c.cost_usd for c in calls), 6),
            total_tokens=sum(c.input_tokens + c.output_tokens for c in calls),
            avg_latency_ms=round(statistics.mean(durations), 2),
            p95_latency_ms=round(self._percentile(durations, 95), 2),
            p99_latency_ms=round(self._percentile(durations, 99), 2),
            top_tools=[
                {"tool": name, "count": count, "pct": round(count / total * 100, 1)}
                for name, count in tool_counter.most_common(10)
            ],
            most_called_tool=tool_counter.most_common(1)[0][0] if tool_counter else None,
            time_range_start=min(c.started_at for c in calls),
            time_range_end=max(c.started_at for c in calls),
        )

    def get_cost_breakdown(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        group_by: str = "tool",
    ) -> dict[str, Any]:
        """Break down costs by tool, server, or session."""
        calls = self.store.query_calls(
            session_id=session_id, agent_id=agent_id, limit=100_000
        )
        if not calls:
            return {"total_cost_usd": 0.0, "breakdown": [], "group_by": group_by}

        groups: dict[str, list[float]] = {}
        for c in calls:
            if group_by == "tool":
                key = c.tool_name
            elif group_by == "server":
                key = c.server_name or "unknown"
            elif group_by == "session":
                key = c.session_id
            else:
                key = c.tool_name
            groups.setdefault(key, []).append(c.cost_usd)

        breakdown = sorted(
            (
                {
                    "name": name,
                    "total_cost": round(sum(costs), 6),
                    "avg_cost": round(sum(costs) / len(costs), 6),
                    "call_count": len(costs),
                    "pct_of_total": round(sum(costs) / sum(sum(v) for v in groups.values()) * 100, 2)
                    if groups
                    else 0.0,
                }
                for name, costs in groups.items()
            ),
            key=lambda x: x["total_cost"],
            reverse=True,
        )
        return {
            "total_cost_usd": round(sum(sum(v) for v in groups.values()), 6),
            "breakdown": breakdown,
            "group_by": group_by,
        }

    # ── Alert rules ─────────────────────────────────────────────────

    def create_rule(
        self,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        window: int = 100,
    ) -> AlertRule:
        """Create an alert rule."""
        # validate
        valid_metrics = {"error_rate", "p95_latency", "cost_per_call", "total_cost", "call_volume"}
        valid_ops = {">", ">=", "<", "<=", "=="}
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}")
        if operator not in valid_ops:
            raise ValueError(f"Invalid operator '{operator}'. Must be one of {valid_ops}")

        rule = AlertRule(
            name=name,
            metric=metric,
            operator=operator,
            threshold=threshold,
            window=window,
        )
        self.store.save_rule(rule)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        return self.store.delete_rule(rule_id)

    def list_rules(self) -> list[AlertRule]:
        return self.store.all_rules()

    def get_rule(self, rule_id: str) -> AlertRule | None:
        return self.store.get_rule(rule_id)

    def evaluate_rules(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all enabled alert rules against recent calls."""
        calls = self.store.query_calls(
            session_id=session_id, agent_id=agent_id, limit=100_000
        )
        calls = list(reversed(calls))  # newest first for windowing
        triggered = []
        for rule in self.store.all_rules():
            if not rule.enabled:
                continue
            is_breached = rule.evaluate(calls)
            if is_breached:
                rule.last_triggered = datetime.now(timezone.utc)
                rule.trigger_count += 1
                self.store.save_rule(rule)
                triggered.append(
                    {
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "metric": rule.metric,
                        "operator": rule.operator,
                        "threshold": rule.threshold,
                        "message": f"{rule.metric} {rule.operator} {rule.threshold}",
                    }
                )
        return triggered

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * pct / 100
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        if f == c:
            return sorted_data[f]
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "total_calls": 0,
            "success_count": 0,
            "error_count": 0,
            "timeout_count": 0,
            "blocked_count": 0,
            "error_rate": 0.0,
            "timeout_rate": 0.0,
            "total_cost_usd": 0.0,
            "avg_cost_usd": 0.0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "top_tools": [],
            "top_servers": [],
            "unique_tools": 0,
            "unique_sessions": 0,
        }
