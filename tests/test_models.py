"""Tests for data models."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from mcp_audit.models import (
    AgentReport,
    AlertRule,
    CallStatus,
    Severity,
    Session,
    ToolCall,
    TraceEvent,
)


class TestToolCall:
    def test_defaults(self):
        call = ToolCall(session_id="s1", tool_name="search")
        assert call.id  # auto-generated UUID
        assert call.status == CallStatus.SUCCESS
        assert call.session_id == "s1"
        assert call.tool_name == "search"
        assert call.arguments == {}
        assert call.tags == []
        assert call.duration_ms is None
        assert call.is_error is False

    def test_is_error(self):
        for status in [CallStatus.ERROR, CallStatus.TIMEOUT, CallStatus.BLOCKED]:
            call = ToolCall(session_id="s1", tool_name="t", status=status)
            assert call.is_error is True

    def test_success_is_not_error(self):
        call = ToolCall(session_id="s1", tool_name="t", status=CallStatus.SUCCESS)
        assert call.is_error is False

    def test_finish_success(self):
        call = ToolCall(session_id="s1", tool_name="t")
        call.finish(
            result={"answer": 42},
            status=CallStatus.SUCCESS,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
        )
        assert call.completed_at is not None
        assert call.duration_ms is not None
        assert call.duration_ms >= 0
        assert call.result == {"answer": 42}
        assert call.input_tokens == 100
        assert call.output_tokens == 50
        assert call.cost_usd == 0.005

    def test_finish_error(self):
        call = ToolCall(session_id="s1", tool_name="t")
        call.finish(status=CallStatus.ERROR, error="Connection refused")
        assert call.is_error is True
        assert call.error == "Connection refused"

    def test_finish_calculates_duration(self):
        call = ToolCall(session_id="s1", tool_name="t")
        time.sleep(0.01)
        call.finish()
        assert call.duration_ms is not None
        assert call.duration_ms >= 8  # at least ~10ms

    def test_cost_rounding(self):
        call = ToolCall(session_id="s1", tool_name="t")
        call.finish(cost_usd=0.123456789)
        assert call.cost_usd == 0.123457  # rounded to 6 decimal places


class TestSession:
    def test_defaults(self):
        session = Session()
        assert session.id
        assert session.is_active is True
        assert session.total_calls == 0

    def test_end(self):
        session = Session()
        session.end()
        assert session.is_active is False
        assert session.ended_at is not None

    def test_with_agent(self):
        session = Session(agent_id="agent-001", name="test-run")
        assert session.agent_id == "agent-001"
        assert session.name == "test-run"


class TestTraceEvent:
    def test_defaults(self):
        event = TraceEvent(trace_id="t1", event_type="http_request")
        assert event.id
        assert event.trace_id == "t1"
        assert event.event_type == "http_request"
        assert event.severity == Severity.INFO

    def test_with_severity(self):
        event = TraceEvent(
            trace_id="t1", event_type="error", severity=Severity.CRITICAL
        )
        assert event.severity == Severity.CRITICAL

    def test_nested_under_call(self):
        event = TraceEvent(
            trace_id="t1", event_type="sub_step", call_id="call-123"
        )
        assert event.call_id == "call-123"


class TestAlertRule:
    def test_error_rate_triggered(self):
        rule = AlertRule(
            name="high_error_rate",
            metric="error_rate",
            operator=">",
            threshold=50.0,
        )
        calls = [
            ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR),
            ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR),
            ToolCall(session_id="s", tool_name="t"),
            ToolCall(session_id="s", tool_name="t"),
        ]
        # 50% error rate, threshold is > 50, so NOT triggered
        assert rule.evaluate(calls) is False

        calls.append(ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR))
        # Now 60% error rate
        assert rule.evaluate(calls) is True

    def test_p95_latency_triggered(self):
        rule = AlertRule(
            name="slow_p95",
            metric="p95_latency",
            operator=">",
            threshold=1000.0,
        )
        calls = [
            ToolCall(session_id="s", tool_name="t", duration_ms=100),
            ToolCall(session_id="s", tool_name="t", duration_ms=200),
            ToolCall(session_id="s", tool_name="t", duration_ms=150),
            ToolCall(session_id="s", tool_name="t", duration_ms=5000),
            ToolCall(session_id="s", tool_name="t", duration_ms=3000),
            ToolCall(session_id="s", tool_name="t", duration_ms=1200),
        ]
        # p95 should be around 5000
        assert rule.evaluate(calls) is True

    def test_cost_per_call_triggered(self):
        rule = AlertRule(
            name="expensive_call",
            metric="cost_per_call",
            operator=">",
            threshold=1.0,
        )
        calls = [
            ToolCall(session_id="s", tool_name="t", cost_usd=0.01),
            ToolCall(session_id="s", tool_name="t", cost_usd=1.50),
        ]
        assert rule.evaluate(calls) is True

    def test_total_cost_triggered(self):
        rule = AlertRule(
            name="budget_exceeded",
            metric="total_cost",
            operator=">=",
            threshold=10.0,
        )
        calls = [ToolCall(session_id="s", tool_name="t", cost_usd=5.0) for _ in range(3)]
        assert rule.evaluate(calls) is True

    def test_call_volume_triggered(self):
        rule = AlertRule(
            name="too_many_calls",
            metric="call_volume",
            operator=">=",
            threshold=5,
        )
        calls = [ToolCall(session_id="s", tool_name="t") for _ in range(6)]
        assert rule.evaluate(calls) is True

    def test_disabled_rule_never_triggers(self):
        rule = AlertRule(
            name="disabled",
            metric="error_rate",
            operator=">",
            threshold=0.0,
            enabled=False,
        )
        calls = [ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR)]
        assert rule.evaluate(calls) is False

    def test_window_limits_evaluation(self):
        rule = AlertRule(
            name="recent_errors",
            metric="error_rate",
            operator=">",
            threshold=50.0,
            window=2,
        )
        # 5 calls: first 3 are success, last 2 are errors
        calls = [
            ToolCall(session_id="s", tool_name="t"),
            ToolCall(session_id="s", tool_name="t"),
            ToolCall(session_id="s", tool_name="t"),
            ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR),
            ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR),
        ]
        # With window=2, we look at last 2 calls: both errors → 100% error rate
        assert rule.evaluate(calls) is True

    def test_empty_calls_returns_false(self):
        rule = AlertRule(
            name="test", metric="error_rate", operator=">", threshold=0.0
        )
        assert rule.evaluate([]) is False

    def test_invalid_operator(self):
        rule = AlertRule(
            name="test", metric="error_rate", operator="!=", threshold=0.0
        )
        assert rule.evaluate([ToolCall(session_id="s", tool_name="t")]) is False

    def test_invalid_metric(self):
        rule = AlertRule(
            name="test", metric="unknown_metric", operator=">", threshold=0.0
        )
        assert rule.evaluate([ToolCall(session_id="s", tool_name="t")]) is False


class TestCallStatus:
    def test_values(self):
        assert CallStatus.SUCCESS.value == "success"
        assert CallStatus.ERROR.value == "error"
        assert CallStatus.TIMEOUT.value == "timeout"
        assert CallStatus.BLOCKED.value == "blocked"


class TestSeverity:
    def test_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"


class TestAgentReport:
    def test_defaults(self):
        report = AgentReport(agent_id="a1")
        assert report.agent_id == "a1"
        assert report.total_calls == 0
        assert report.error_rate == 0.0
