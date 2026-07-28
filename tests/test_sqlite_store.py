"""Tests for SQLiteStore — the persistent storage backend."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import AlertRule, CallStatus, Session, Severity, ToolCall, TraceEvent
from mcp_audit.sqlite_store import SQLiteStore


@pytest.fixture
def store():
    """Fresh in-memory SQLite store for each test."""
    return SQLiteStore(":memory:")


@pytest.fixture
def engine(store):
    return AuditEngine(store=store)


class TestSQLiteStoreBasic:
    def test_save_and_get_call(self, store):
        call = ToolCall(
            session_id="sess1",
            tool_name="search",
            agent_id="agent-a",
            server_name="web-server",
            arguments={"query": "hello"},
            result={"items": [1, 2, 3]},
            status=CallStatus.SUCCESS,
            duration_ms=42.5,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
            tags=["production", "web"],
            metadata={"region": "us-east"},
        )
        store.save_call(call)

        retrieved = store.get_call(call.id)
        assert retrieved is not None
        assert retrieved.id == call.id
        assert retrieved.session_id == "sess1"
        assert retrieved.tool_name == "search"
        assert retrieved.agent_id == "agent-a"
        assert retrieved.server_name == "web-server"
        assert retrieved.status == CallStatus.SUCCESS
        assert retrieved.duration_ms == 42.5
        assert retrieved.input_tokens == 100
        assert retrieved.output_tokens == 50
        assert retrieved.cost_usd == 0.005
        assert retrieved.tags == ["production", "web"]
        assert retrieved.arguments == {"query": "hello"}
        assert retrieved.result == {"items": [1, 2, 3]}
        assert retrieved.metadata == {"region": "us-east"}

    def test_save_and_get_session(self, store):
        session = Session(
            agent_id="agent-a",
            name="test session",
            tags=["test"],
            metadata={"env": "prod"},
        )
        store.save_session(session)

        retrieved = store.get_session(session.id)
        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.agent_id == "agent-a"
        assert retrieved.name == "test session"
        assert retrieved.tags == ["test"]
        assert retrieved.metadata == {"env": "prod"}
        assert retrieved.is_active is True

    def test_save_and_get_event(self, store):
        event = TraceEvent(
            trace_id="trace1",
            call_id="call1",
            event_type="http_request",
            message="GET /api/data",
            severity=Severity.INFO,
            data={"status_code": 200},
            duration_ms=15.3,
        )
        store.save_event(event)

        events = store.query_events(trace_id="trace1")
        assert len(events) == 1
        assert events[0].id == event.id
        assert events[0].trace_id == "trace1"
        assert events[0].event_type == "http_request"
        assert events[0].severity == Severity.INFO
        assert events[0].data == {"status_code": 200}
        assert events[0].duration_ms == 15.3

    def test_save_and_get_rule(self, store):
        rule = AlertRule(
            name="high_error_rate",
            metric="error_rate",
            operator=">",
            threshold=50.0,
            window=100,
        )
        store.save_rule(rule)

        retrieved = store.get_rule(rule.id)
        assert retrieved is not None
        assert retrieved.name == "high_error_rate"
        assert retrieved.metric == "error_rate"
        assert retrieved.operator == ">"
        assert retrieved.threshold == 50.0
        assert retrieved.enabled is True


class TestSQLiteStoreQueries:
    def test_query_calls_by_session(self, store):
        for i in range(5):
            store.save_call(ToolCall(
                session_id="sess1",
                tool_name=f"tool_{i}",
            ))
        for i in range(3):
            store.save_call(ToolCall(
                session_id="sess2",
                tool_name=f"tool_{i}",
            ))

        s1_calls = store.query_calls(session_id="sess1")
        assert len(s1_calls) == 5

        s2_calls = store.query_calls(session_id="sess2")
        assert len(s2_calls) == 3

    def test_query_calls_by_agent(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="t", agent_id="a1"))
        store.save_call(ToolCall(session_id="s2", tool_name="t", agent_id="a2"))
        store.save_call(ToolCall(session_id="s3", tool_name="t", agent_id="a1"))

        results = store.query_calls(agent_id="a1")
        assert len(results) == 2
        assert all(c.agent_id == "a1" for c in results)

    def test_query_calls_by_tool(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="search"))
        store.save_call(ToolCall(session_id="s1", tool_name="fetch"))
        store.save_call(ToolCall(session_id="s1", tool_name="search"))

        results = store.query_calls(tool_name="search")
        assert len(results) == 2

    def test_query_calls_by_status(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="t", status=CallStatus.SUCCESS))
        store.save_call(ToolCall(session_id="s1", tool_name="t", status=CallStatus.ERROR))
        store.save_call(ToolCall(session_id="s1", tool_name="t", status=CallStatus.SUCCESS))

        errors = store.query_calls(status="error")
        assert len(errors) == 1
        assert errors[0].status == CallStatus.ERROR

    def test_query_calls_by_min_cost(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="t", cost_usd=0.001))
        store.save_call(ToolCall(session_id="s1", tool_name="t", cost_usd=0.05))
        store.save_call(ToolCall(session_id="s1", tool_name="t", cost_usd=0.01))

        expensive = store.query_calls(min_cost=0.01)
        assert len(expensive) == 2

    def test_query_calls_by_min_duration(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="t", duration_ms=10.0))
        store.save_call(ToolCall(session_id="s1", tool_name="t", duration_ms=500.0))
        store.save_call(ToolCall(session_id="s1", tool_name="t", duration_ms=50.0))

        slow = store.query_calls(min_duration=100.0)
        assert len(slow) == 1
        assert slow[0].duration_ms == 500.0

    def test_query_calls_by_tag(self, store):
        store.save_call(ToolCall(session_id="s1", tool_name="t", tags=["prod"]))
        store.save_call(ToolCall(session_id="s1", tool_name="t", tags=["dev"]))
        store.save_call(ToolCall(session_id="s1", tool_name="t", tags=["prod", "critical"]))

        prod = store.query_calls(tag="prod")
        assert len(prod) == 2

    def test_query_calls_limit_and_offset(self, store):
        for i in range(10):
            store.save_call(ToolCall(session_id="s1", tool_name=f"t{i}"))

        page1 = store.query_calls(session_id="s1", limit=5, offset=0)
        page2 = store.query_calls(session_id="s1", limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        # No overlap
        ids1 = {c.id for c in page1}
        ids2 = {c.id for c in page2}
        assert ids1.isdisjoint(ids2)

    def test_query_calls_order_newest_first(self, store):
        import time as _time
        for i in range(3):
            store.save_call(ToolCall(session_id="s1", tool_name=f"t{i}"))
            _time.sleep(0.001)  # ensure different timestamps

        calls = store.query_calls(session_id="s1")
        assert calls[0].tool_name == "t2"  # newest first

    def test_count_calls(self, store):
        for _ in range(5):
            store.save_call(ToolCall(session_id="s1", tool_name="t"))
        assert store.count_calls() == 5


class TestSQLiteStoreSessions:
    def test_query_sessions_by_agent(self, store):
        store.save_session(Session(agent_id="a1"))
        store.save_session(Session(agent_id="a2"))
        store.save_session(Session(agent_id="a1"))

        results = store.query_sessions(agent_id="a1")
        assert len(results) == 2

    def test_query_sessions_active_only(self, store):
        s1 = Session(agent_id="a1")
        store.save_session(s1)

        s2 = Session(agent_id="a1")
        s2.end()
        store.save_session(s2)

        active = store.query_sessions(active_only=True)
        assert len(active) == 1
        assert active[0].id == s1.id


class TestSQLiteStoreRules:
    def test_all_rules(self, store):
        r1 = AlertRule(name="r1", metric="error_rate", operator=">", threshold=50.0)
        r2 = AlertRule(name="r2", metric="total_cost", operator=">=", threshold=10.0)
        store.save_rule(r1)
        store.save_rule(r2)

        rules = store.all_rules()
        assert len(rules) == 2

    def test_delete_rule(self, store):
        rule = AlertRule(name="r", metric="error_rate", operator=">", threshold=50.0)
        store.save_rule(rule)
        assert store.delete_rule(rule.id) is True
        assert store.get_rule(rule.id) is None
        assert store.delete_rule("nonexistent") is False


class TestSQLiteStoreUpdateReplace:
    def test_update_session(self, store):
        session = Session(agent_id="a1", name="original")
        store.save_session(session)

        session.name = "updated"
        session.total_calls = 42
        store.save_session(session)  # INSERT OR REPLACE

        retrieved = store.get_session(session.id)
        assert retrieved is not None
        assert retrieved.name == "updated"
        assert retrieved.total_calls == 42

    def test_update_rule(self, store):
        rule = AlertRule(name="r", metric="error_rate", operator=">", threshold=50.0)
        store.save_rule(rule)

        rule.trigger_count = 5
        store.save_rule(rule)

        retrieved = store.get_rule(rule.id)
        assert retrieved.trigger_count == 5


class TestSQLiteStoreFilePersistence:
    def test_persistence_across_connections(self):
        """Data should survive close and reopen (file-based SQLite)."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Write data with one store instance
            store1 = SQLiteStore(db_path)
            session = Session(agent_id="a1", name="persisted")
            store1.save_session(session)
            store1.save_call(ToolCall(
                session_id=session.id,
                tool_name="search",
                cost_usd=0.01,
            ))
            del store1  # release connection

            # Read with a new store instance
            store2 = SQLiteStore(db_path)
            retrieved = store2.get_session(session.id)
            assert retrieved is not None
            assert retrieved.agent_id == "a1"
            assert retrieved.name == "persisted"

            calls = store2.query_calls(session_id=session.id)
            assert len(calls) == 1
            assert calls[0].tool_name == "search"
            assert calls[0].cost_usd == 0.01
        finally:
            os.unlink(db_path)


class TestSQLiteStoreWithEngine:
    def test_engine_with_sqlite_store(self, engine):
        session = engine.start_session(agent_id="test-agent", name="sqlite test")

        engine.record_call(
            session_id=session.id,
            tool_name="search",
            agent_id="test-agent",
            status=CallStatus.SUCCESS,
            duration_ms=100.0,
            cost_usd=0.002,
            input_tokens=500,
            output_tokens=200,
        )
        engine.record_call(
            session_id=session.id,
            tool_name="search",
            agent_id="test-agent",
            status=CallStatus.ERROR,
            duration_ms=200.0,
            cost_usd=0.003,
        )

        stats = engine.get_stats(session_id=session.id)
        assert stats["total_calls"] == 2
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 50.0
        assert stats["total_cost_usd"] == 0.005

    def test_engine_cost_breakdown_with_sqlite(self, engine):
        session = engine.start_session(agent_id="a1")

        engine.record_call(
            session_id=session.id, tool_name="llm_complete",
            cost_usd=0.10, agent_id="a1",
        )
        engine.record_call(
            session_id=session.id, tool_name="web_search",
            cost_usd=0.001, agent_id="a1",
        )

        breakdown = engine.get_cost_breakdown(session_id=session.id)
        assert breakdown["total_cost_usd"] == 0.101
        assert len(breakdown["breakdown"]) == 2

    def test_engine_alerts_with_sqlite(self, engine):
        session = engine.start_session(agent_id="a1")

        rule = engine.create_rule(
            name="cost_limit",
            metric="total_cost",
            operator=">=",
            threshold=0.05,
        )

        engine.record_call(
            session_id=session.id, tool_name="llm",
            cost_usd=0.06, agent_id="a1",
        )

        triggered = engine.evaluate_rules()
        assert len(triggered) == 1
        assert triggered[0]["rule_name"] == "cost_limit"


class TestSQLiteStoreClear:
    def test_clear_wipes_all(self, store):
        store.save_session(Session(agent_id="a1"))
        store.save_call(ToolCall(session_id="s1", tool_name="t"))
        store.save_event(TraceEvent(trace_id="t1", event_type="x"))
        store.save_rule(AlertRule(name="r", metric="error_rate", operator=">", threshold=1.0))

        store.clear()

        assert store.count_calls() == 0
        assert len(store.all_rules()) == 0
        assert len(store.query_sessions()) == 0
        assert len(store.query_events()) == 0


class TestSQLiteStoreEdgeCases:
    def test_get_nonexistent(self, store):
        assert store.get_call("nonexistent") is None
        assert store.get_session("nonexistent") is None
        assert store.get_rule("nonexistent") is None

    def test_null_result(self, store):
        call = ToolCall(
            session_id="s1",
            tool_name="void_tool",
            result=None,
        )
        store.save_call(call)
        retrieved = store.get_call(call.id)
        assert retrieved is not None
        assert retrieved.result is None

    def test_complex_result_serialization(self, store):
        call = ToolCall(
            session_id="s1",
            tool_name="search",
            result={
                "items": [
                    {"title": "a", "url": "http://a"},
                    {"title": "b", "url": "http://b"},
                ],
                "meta": {"total": 2, "page": 1},
            },
        )
        store.save_call(call)
        retrieved = store.get_call(call.id)
        assert retrieved.result["meta"]["total"] == 2
        assert len(retrieved.result["items"]) == 2

    def test_empty_tags_and_metadata(self, store):
        call = ToolCall(session_id="s1", tool_name="t")
        store.save_call(call)
        retrieved = store.get_call(call.id)
        assert retrieved.tags == []
        assert retrieved.metadata == {}
