"""Tests for the audit engine."""
from __future__ import annotations

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import CallStatus, Severity
from mcp_audit.storage import MemoryStore


@pytest.fixture
def engine():
    return AuditEngine(store=MemoryStore())


class TestSessions:
    def test_start_session(self, engine):
        session = engine.start_session(agent_id="a1", name="test")
        assert session.agent_id == "a1"
        assert session.name == "test"
        assert session.is_active is True

    def test_end_session(self, engine):
        session = engine.start_session(agent_id="a1")
        engine.record_call(session.id, "tool1", cost_usd=0.5)
        engine.record_call(session.id, "tool2", cost_usd=0.3)
        ended = engine.end_session(session.id)
        assert ended is not None
        assert ended.is_active is False
        assert ended.total_calls == 2
        assert ended.total_cost_usd == 0.8

    def test_end_session_with_errors(self, engine):
        session = engine.start_session(agent_id="a1")
        engine.record_call(session.id, "tool1")
        engine.record_call(session.id, "tool2", status=CallStatus.ERROR)
        ended = engine.end_session(session.id)
        assert ended.error_count == 1
        assert ended.total_calls == 2

    def test_end_nonexistent_session(self, engine):
        assert engine.end_session("nonexistent") is None

    def test_get_session(self, engine):
        session = engine.start_session(agent_id="a1")
        assert engine.get_session(session.id) is not None
        assert engine.get_session("nonexistent") is None

    def test_list_sessions(self, engine):
        engine.start_session(agent_id="a1")
        engine.start_session(agent_id="a2")
        engine.start_session(agent_id="a1")
        assert len(engine.list_sessions()) == 3
        assert len(engine.list_sessions(agent_id="a1")) == 2

    def test_list_active_sessions(self, engine):
        s1 = engine.start_session(agent_id="a1")
        s2 = engine.start_session(agent_id="a1")
        engine.end_session(s1.id)
        active = engine.list_sessions(active_only=True)
        assert len(active) == 1
        assert active[0].id == s2.id


class TestRecordCall:
    def test_basic_record(self, engine):
        session = engine.start_session()
        call = engine.record_call(session.id, "search", duration_ms=150, cost_usd=0.01)
        assert call.tool_name == "search"
        assert call.session_id == session.id
        assert call.duration_ms == 150
        assert call.cost_usd == 0.01
        assert call.status == CallStatus.SUCCESS

    def test_record_updates_session_aggregates(self, engine):
        session = engine.start_session(agent_id="a1")
        engine.record_call(session.id, "tool1", cost_usd=0.5, input_tokens=100, output_tokens=50)
        engine.record_call(session.id, "tool2", cost_usd=0.3, input_tokens=200, output_tokens=100)

        updated = engine.get_session(session.id)
        assert updated.total_calls == 2
        assert updated.total_cost_usd == 0.8
        assert updated.total_tokens == 450

    def test_record_error_updates_error_count(self, engine):
        session = engine.start_session()
        engine.record_call(session.id, "tool1", status=CallStatus.ERROR)
        updated = engine.get_session(session.id)
        assert updated.error_count == 1

    def test_record_with_all_fields(self, engine):
        session = engine.start_session(agent_id="a1")
        call = engine.record_call(
            session.id,
            "expensive_tool",
            agent_id="a1",
            server_name="payments",
            arguments={"amount": 100},
            result={"status": "ok"},
            status=CallStatus.SUCCESS,
            duration_ms=250,
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.03,
            tags=["production", "api"],
            metadata={"request_id": "req-123"},
        )
        assert call.agent_id == "a1"
        assert call.server_name == "payments"
        assert call.arguments == {"amount": 100}
        assert call.result == {"status": "ok"}
        assert call.tags == ["production", "api"]
        assert call.metadata == {"request_id": "req-123"}

    def test_get_call(self, engine):
        session = engine.start_session()
        call = engine.record_call(session.id, "tool1")
        assert engine.get_call(call.id) is not None
        assert engine.get_call("nonexistent") is None

    def test_query_calls(self, engine):
        session = engine.start_session()
        engine.record_call(session.id, "search")
        engine.record_call(session.id, "fetch")
        engine.record_call(session.id, "search")
        results = engine.query_calls(tool_name="search")
        assert len(results) == 2


class TestTraceEvents:
    def test_log_event(self, engine):
        event = engine.log_event("trace1", "http_request", "GET /api")
        assert event.trace_id == "trace1"
        assert event.event_type == "http_request"
        assert event.message == "GET /api"

    def test_log_event_with_severity(self, engine):
        event = engine.log_event(
            "trace1", "error", "failed", severity=Severity.CRITICAL
        )
        assert event.severity == Severity.CRITICAL

    def test_query_events(self, engine):
        engine.log_event("t1", "step1")
        engine.log_event("t2", "step2")
        engine.log_event("t1", "step3")
        results = engine.query_events(trace_id="t1")
        assert len(results) == 2

    def test_query_events_by_severity(self, engine):
        engine.log_event("t", "info_event", severity=Severity.INFO)
        engine.log_event("t", "error_event", severity=Severity.ERROR)
        results = engine.query_events(severity="error")
        assert len(results) == 1


class TestStats:
    def test_empty_stats(self, engine):
        stats = engine.get_stats()
        assert stats["total_calls"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["total_cost_usd"] == 0.0

    def test_basic_stats(self, engine):
        session = engine.start_session(agent_id="a1")
        engine.record_call(session.id, "tool1", duration_ms=100, cost_usd=0.01)
        engine.record_call(session.id, "tool2", duration_ms=200, cost_usd=0.02)
        engine.record_call(session.id, "tool1", status=CallStatus.ERROR, duration_ms=50)

        stats = engine.get_stats()
        assert stats["total_calls"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert stats["error_rate"] == pytest.approx(33.33, abs=0.1)
        assert stats["total_cost_usd"] == 0.03
        assert stats["avg_latency_ms"] == pytest.approx(116.67, abs=0.5)
        assert stats["unique_tools"] == 2

    def test_stats_with_filters(self, engine):
        session = engine.start_session(agent_id="a1")
        engine.record_call(session.id, "tool1", duration_ms=100)
        engine.record_call(session.id, "tool2", duration_ms=200)

        stats = engine.get_stats(tool_name="tool1")
        assert stats["total_calls"] == 1

    def test_stats_top_tools(self, engine):
        session = engine.start_session()
        for _ in range(5):
            engine.record_call(session.id, "popular_tool")
        for _ in range(2):
            engine.record_call(session.id, "rare_tool")

        stats = engine.get_stats()
        assert stats["top_tools"][0]["tool"] == "popular_tool"
        assert stats["top_tools"][0]["count"] == 5

    def test_stats_percentiles(self, engine):
        session = engine.start_session()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            engine.record_call(session.id, "tool", duration_ms=float(ms))

        stats = engine.get_stats()
        assert stats["min_latency_ms"] == 10.0
        assert stats["max_latency_ms"] == 100.0
        assert stats["avg_latency_ms"] == 55.0
        assert 80 <= stats["p95_latency_ms"] <= 100
        assert 90 <= stats["p99_latency_ms"] <= 100

    def test_stats_top_servers(self, engine):
        session = engine.start_session()
        for _ in range(3):
            engine.record_call(session.id, "tool", server_name="payments")
        for _ in range(1):
            engine.record_call(session.id, "tool", server_name="search")

        stats = engine.get_stats()
        assert stats["top_servers"][0]["server"] == "payments"
        assert stats["top_servers"][0]["count"] == 3


class TestAgentReport:
    def test_empty_report(self, engine):
        report = engine.get_agent_report("nonexistent")
        assert report.total_calls == 0

    def test_report_with_data(self, engine):
        s1 = engine.start_session(agent_id="a1")
        s2 = engine.start_session(agent_id="a1")
        engine.record_call(s1.id, "tool1", agent_id="a1", duration_ms=100, cost_usd=0.01)
        engine.record_call(s2.id, "tool2", agent_id="a1", duration_ms=200, cost_usd=0.02)
        engine.record_call(s2.id, "tool1", agent_id="a1", status=CallStatus.ERROR, duration_ms=50)

        report = engine.get_agent_report("a1")
        assert report.agent_id == "a1"
        assert report.total_calls == 3
        assert report.error_count == 1
        assert report.session_count == 2
        assert report.most_called_tool == "tool1"
        assert report.total_cost_usd == 0.03


class TestCostBreakdown:
    def test_empty(self, engine):
        result = engine.get_cost_breakdown()
        assert result["total_cost_usd"] == 0.0

    def test_by_tool(self, engine):
        session = engine.start_session()
        engine.record_call(session.id, "cheap", cost_usd=0.001)
        engine.record_call(session.id, "expensive", cost_usd=0.50)
        engine.record_call(session.id, "expensive", cost_usd=0.30)

        result = engine.get_cost_breakdown(group_by="tool")
        assert result["total_cost_usd"] == 0.801
        assert result["breakdown"][0]["name"] == "expensive"
        assert result["breakdown"][0]["total_cost"] == 0.80

    def test_by_server(self, engine):
        session = engine.start_session()
        engine.record_call(session.id, "tool", server_name="srv1", cost_usd=0.10)
        engine.record_call(session.id, "tool", server_name="srv2", cost_usd=0.20)

        result = engine.get_cost_breakdown(group_by="server")
        assert result["breakdown"][0]["name"] == "srv2"


class TestAlertRules:
    def test_create_rule(self, engine):
        rule = engine.create_rule("test", "error_rate", ">", 50.0)
        assert rule.name == "test"
        assert rule.metric == "error_rate"
        assert engine.get_rule(rule.id) is not None

    def test_create_invalid_metric(self, engine):
        with pytest.raises(ValueError, match="Invalid metric"):
            engine.create_rule("test", "invalid_metric", ">", 1.0)

    def test_create_invalid_operator(self, engine):
        with pytest.raises(ValueError, match="Invalid operator"):
            engine.create_rule("test", "error_rate", "!=", 1.0)

    def test_delete_rule(self, engine):
        rule = engine.create_rule("test", "error_rate", ">", 50.0)
        assert engine.delete_rule(rule.id) is True
        assert engine.delete_rule("nonexistent") is False

    def test_list_rules(self, engine):
        engine.create_rule("r1", "error_rate", ">", 10.0)
        engine.create_rule("r2", "p95_latency", ">", 1000.0)
        assert len(engine.list_rules()) == 2

    def test_evaluate_no_rules(self, engine):
        result = engine.evaluate_rules()
        assert result == []

    def test_evaluate_triggered(self, engine):
        engine.create_rule("high_error", "error_rate", ">", 50.0, window=10)
        session = engine.start_session()
        # 3 errors out of 4 = 75%
        engine.record_call(session.id, "tool", status=CallStatus.ERROR)
        engine.record_call(session.id, "tool", status=CallStatus.ERROR)
        engine.record_call(session.id, "tool", status=CallStatus.ERROR)
        engine.record_call(session.id, "tool")

        triggered = engine.evaluate_rules()
        assert len(triggered) == 1
        assert triggered[0]["rule_name"] == "high_error"

    def test_evaluate_not_triggered(self, engine):
        engine.create_rule("high_error", "error_rate", ">", 50.0)
        session = engine.start_session()
        engine.record_call(session.id, "tool")  # 0% error rate
        triggered = engine.evaluate_rules()
        assert len(triggered) == 0

    def test_evaluate_updates_trigger_count(self, engine):
        rule = engine.create_rule("test", "error_rate", ">", 0.0)
        session = engine.start_session()
        engine.record_call(session.id, "tool", status=CallStatus.ERROR)
        engine.evaluate_rules()
        updated = engine.get_rule(rule.id)
        assert updated.trigger_count == 1
        assert updated.last_triggered is not None

    def test_disabled_rule_not_evaluated(self, engine):
        rule = engine.create_rule("test", "error_rate", ">", 0.0)
        rule.enabled = False
        engine.store.save_rule(rule)
        session = engine.start_session()
        engine.record_call(session.id, "tool", status=CallStatus.ERROR)
        triggered = engine.evaluate_rules()
        assert len(triggered) == 0

    def test_evaluate_with_filters(self, engine):
        engine.create_rule("test", "error_rate", ">", 0.0)
        s1 = engine.start_session()
        s2 = engine.start_session()
        engine.record_call(s1.id, "tool", status=CallStatus.ERROR)
        engine.record_call(s2.id, "tool")

        # evaluate only s2 (which has no errors)
        triggered = engine.evaluate_rules(session_id=s2.id)
        assert len(triggered) == 0


class TestPercentile:
    def test_single_value(self, engine):
        assert engine._percentile([42.0], 95) == 42.0

    def test_empty(self, engine):
        assert engine._percentile([], 95) == 0.0

    def test_known_values(self, engine):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        assert engine._percentile(data, 50) == pytest.approx(5.5, abs=0.1)
        assert engine._percentile(data, 0) == 1.0
        assert engine._percentile(data, 100) == 10.0
