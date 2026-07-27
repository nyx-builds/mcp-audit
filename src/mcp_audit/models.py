"""Data models for the MCP audit observability server."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class CallStatus(str, Enum):
    """Outcome of a tool call."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # blocked by policy / guardrail


class Severity(str, Enum):
    """Severity level for trace events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ToolCall(BaseModel):
    """A single recorded tool invocation.

    This is the core entity — every time an agent calls any tool (MCP or
    otherwise), a ``ToolCall`` record captures who/what/when/why/result
    plus optional cost and performance data.
    """

    id: str = Field(default_factory=_uuid)
    session_id: str = Field(description="Agent session identifier")
    agent_id: str | None = Field(default=None, description="Agent that made the call")
    tool_name: str = Field(description="Name of the tool invoked")
    server_name: str | None = Field(default=None, description="MCP server that hosts the tool")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Input arguments")
    result: Any = Field(default=None, description="Return value (truncated if large)")
    status: CallStatus = Field(default=CallStatus.SUCCESS)
    error: str | None = Field(default=None, description="Error message if status is error/timeout")

    # ── timing ──
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    duration_ms: float | None = Field(default=None, description="Wall-clock duration in milliseconds")

    # ── cost ──
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0, description="Monetary cost in USD")

    # ── metadata ──
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.status != CallStatus.SUCCESS

    def finish(
        self,
        result: Any = None,
        status: CallStatus = CallStatus.SUCCESS,
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Mark the call as complete and compute duration."""
        self.completed_at = _utcnow()
        self.duration_ms = round(
            (self.completed_at - self.started_at).total_seconds() * 1000, 2
        )
        if result is not None:
            self.result = result
        self.status = status
        self.error = error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = round(cost_usd, 6)


class TraceEvent(BaseModel):
    """A structured event within a trace (sub-step of a tool call or session).

    Trace events let agents record fine-grained steps inside a single tool
    call — e.g. "fetching data", "running validation", "writing to DB".
    """

    id: str = Field(default_factory=_uuid)
    trace_id: str = Field(description="Parent trace / session ID")
    call_id: str | None = Field(default=None, description="Parent tool call ID if nested")
    event_type: str = Field(description="E.g. 'http_request', 'db_query', 'llm_call'")
    message: str = Field(default="")
    severity: Severity = Field(default=Severity.INFO)
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
    duration_ms: float | None = None


class Session(BaseModel):
    """An agent session — groups related tool calls."""

    id: str = Field(default_factory=_uuid)
    agent_id: str | None = None
    name: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # aggregates (updated by engine)
    total_calls: int = Field(default=0)
    error_count: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    total_tokens: int = Field(default=0)

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    def end(self) -> None:
        self.ended_at = _utcnow()


class AlertRule(BaseModel):
    """A rule that fires when tool-call metrics cross a threshold.

    Example: "alert if error_rate > 50% in the last 100 calls" or
    "alert if any single call costs > $1.00".
    """

    id: str = Field(default_factory=_uuid)
    name: str
    metric: str = Field(description="One of: error_rate, p95_latency, cost_per_call, total_cost, call_volume")
    operator: str = Field(description="One of: >, >=, <, <=, ==")
    threshold: float
    window: int = Field(default=100, description="Number of recent calls to evaluate (0 = all)")
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_utcnow)
    last_triggered: datetime | None = None
    trigger_count: int = Field(default=0)

    def evaluate(self, calls: list[ToolCall]) -> bool:
        """Check whether *calls* (most recent first) breach this rule."""
        if not self.enabled:
            return False
        window = calls[-self.window :] if self.window > 0 else calls
        if not window:
            return False
        if self.metric == "error_rate":
            errors = sum(1 for c in window if c.is_error)
            value = errors / len(window) * 100
        elif self.metric == "p95_latency":
            latencies = sorted(c.duration_ms or 0 for c in window)
            idx = int(len(latencies) * 0.95)
            value = latencies[min(idx, len(latencies) - 1)]
        elif self.metric == "cost_per_call":
            value = max((c.cost_usd for c in window), default=0.0)
        elif self.metric == "total_cost":
            value = sum(c.cost_usd for c in window)
        elif self.metric == "call_volume":
            value = float(len(window))
        else:
            return False
        return self._compare(value)

    def _compare(self, value: float) -> bool:
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: abs(a - b) < 1e-9,
        }
        op = ops.get(self.operator)
        if op is None:
            return False
        return op(value, self.threshold)


class AgentReport(BaseModel):
    """Aggregated metrics for an agent over a time range."""

    agent_id: str
    session_count: int = 0
    total_calls: int = 0
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    blocked_count: int = 0
    error_rate: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    top_tools: list[dict[str, Any]] = Field(default_factory=list)
    most_called_tool: str | None = None
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
