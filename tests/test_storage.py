"""Tests for the storage layer."""
from __future__ import annotations

import pytest

from mcp_audit.models import (
    AlertRule,
    CallStatus,
    Session,
    Severity,
    ToolCall,
    TraceEvent,
)
from mcp_audit.storage import MemoryStore


class TestMemoryStoreCalls:
    def test_save_and_get_call(self):
        store = MemoryStore()
        call = ToolCall(session_id="s1", tool_name="search")
        store.save_call(call)
        assert store.get_call(call.id) is call

    def test_get_nonexistent_call(self):
        store = MemoryStore()
        assert store.get_call("nonexistent") is None

    def test_count(self):
        store = MemoryStore()
        for i in range(10):
            store.save_call(ToolCall(session_id="s1", tool_name=f"tool_{i}"))
        assert store.count_calls() == 10

    def test_ring_buffer_eviction(self):
        store = MemoryStore(max_calls=3)
        c1 = ToolCall(session_id="s", tool_name="t1")
        c2 = ToolCall(session_id="s", tool_name="t2")
        c3 = ToolCall(session_id="s", tool_name="t3")
        c4 = ToolCall(session_id="s", tool_name="t4")
        store.save_call(c1)
        store.save_call(c2)
        store.save_call(c3)
        store.save_call(c4)
        assert store.count_calls() == 3
        assert store.get_call(c1.id) is None  # evicted
        assert store.get_call(c4.id) is not None

    def test_query_by_session(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s1", tool_name="t"))
        store.save_call(ToolCall(session_id="s2", tool_name="t"))
        store.save_call(ToolCall(session_id="s1", tool_name="t"))
        results = store.query_calls(session_id="s1")
        assert len(results) == 2

    def test_query_by_tool_name(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="search"))
        store.save_call(ToolCall(session_id="s", tool_name="fetch"))
        store.save_call(ToolCall(session_id="s", tool_name="search"))
        results = store.query_calls(tool_name="search")
        assert len(results) == 2

    def test_query_by_agent_id(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", agent_id="a1"))
        store.save_call(ToolCall(session_id="s", tool_name="t", agent_id="a2"))
        results = store.query_calls(agent_id="a1")
        assert len(results) == 1

    def test_query_by_status(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", status=CallStatus.SUCCESS))
        store.save_call(ToolCall(session_id="s", tool_name="t", status=CallStatus.ERROR))
        results = store.query_calls(status="error")
        assert len(results) == 1

    def test_query_by_server_name(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", server_name="payments"))
        store.save_call(ToolCall(session_id="s", tool_name="t", server_name="search"))
        results = store.query_calls(server_name="payments")
        assert len(results) == 1

    def test_query_by_min_cost(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", cost_usd=0.001))
        store.save_call(ToolCall(session_id="s", tool_name="t", cost_usd=0.50))
        store.save_call(ToolCall(session_id="s", tool_name="t", cost_usd=0.01))
        results = store.query_calls(min_cost=0.02)
        assert len(results) == 1

    def test_query_by_min_duration(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", duration_ms=10))
        store.save_call(ToolCall(session_id="s", tool_name="t", duration_ms=500))
        results = store.query_calls(min_duration=100)
        assert len(results) == 1

    def test_query_by_tag(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t", tags=["production"]))
        store.save_call(ToolCall(session_id="s", tool_name="t", tags=["test"]))
        store.save_call(ToolCall(session_id="s", tool_name="t", tags=["production", "api"]))
        results = store.query_calls(tag="production")
        assert len(results) == 2

    def test_query_limit_and_offset(self):
        store = MemoryStore()
        for i in range(20):
            store.save_call(ToolCall(session_id="s", tool_name=f"t{i}"))
        page1 = store.query_calls(limit=5, offset=0)
        page2 = store.query_calls(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].tool_name != page2[0].tool_name

    def test_query_returns_newest_first(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="first"))
        store.save_call(ToolCall(session_id="s", tool_name="second"))
        store.save_call(ToolCall(session_id="s", tool_name="third"))
        results = store.query_calls()
        assert results[0].tool_name == "third"
        assert results[-1].tool_name == "first"


class TestMemoryStoreSessions:
    def test_save_and_get_session(self):
        store = MemoryStore()
        session = Session(agent_id="a1", name="test")
        store.save_session(session)
        assert store.get_session(session.id) is session

    def test_query_sessions_by_agent(self):
        store = MemoryStore()
        store.save_session(Session(agent_id="a1"))
        store.save_session(Session(agent_id="a2"))
        store.save_session(Session(agent_id="a1"))
        results = store.query_sessions(agent_id="a1")
        assert len(results) == 2

    def test_query_active_sessions(self):
        store = MemoryStore()
        active = Session(agent_id="a1")
        ended = Session(agent_id="a1")
        ended.end()
        store.save_session(active)
        store.save_session(ended)
        results = store.query_sessions(active_only=True)
        assert len(results) == 1
        assert results[0].is_active


class TestMemoryStoreEvents:
    def test_save_and_query_events(self):
        store = MemoryStore()
        event = TraceEvent(trace_id="t1", event_type="http_request")
        store.save_event(event)
        results = store.query_events(trace_id="t1")
        assert len(results) == 1

    def test_query_by_call_id(self):
        store = MemoryStore()
        store.save_event(TraceEvent(trace_id="t", event_type="a", call_id="c1"))
        store.save_event(TraceEvent(trace_id="t", event_type="b", call_id="c2"))
        results = store.query_events(call_id="c1")
        assert len(results) == 1

    def test_query_by_severity(self):
        store = MemoryStore()
        store.save_event(TraceEvent(trace_id="t", event_type="a", severity=Severity.INFO))
        store.save_event(TraceEvent(trace_id="t", event_type="b", severity=Severity.ERROR))
        results = store.query_events(severity="error")
        assert len(results) == 1

    def test_query_returns_newest_first(self):
        store = MemoryStore()
        store.save_event(TraceEvent(trace_id="t", event_type="first"))
        store.save_event(TraceEvent(trace_id="t", event_type="second"))
        results = store.query_events(trace_id="t")
        assert results[0].event_type == "second"


class TestMemoryStoreRules:
    def test_save_and_get_rule(self):
        store = MemoryStore()
        rule = AlertRule(name="test", metric="error_rate", operator=">", threshold=50.0)
        store.save_rule(rule)
        assert store.get_rule(rule.id) is rule

    def test_delete_rule(self):
        store = MemoryStore()
        rule = AlertRule(name="test", metric="error_rate", operator=">", threshold=50.0)
        store.save_rule(rule)
        assert store.delete_rule(rule.id) is True
        assert store.get_rule(rule.id) is None

    def test_delete_nonexistent_rule(self):
        store = MemoryStore()
        assert store.delete_rule("nonexistent") is False

    def test_all_rules(self):
        store = MemoryStore()
        r1 = AlertRule(name="r1", metric="error_rate", operator=">", threshold=10.0)
        r2 = AlertRule(name="r2", metric="p95_latency", operator=">", threshold=1000.0)
        store.save_rule(r1)
        store.save_rule(r2)
        rules = store.all_rules()
        assert len(rules) == 2


class TestMemoryStoreClear:
    def test_clear(self):
        store = MemoryStore()
        store.save_call(ToolCall(session_id="s", tool_name="t"))
        store.save_session(Session())
        store.save_event(TraceEvent(trace_id="t", event_type="e"))
        store.save_rule(AlertRule(name="r", metric="error_rate", operator=">", threshold=1.0))
        store.clear()
        assert store.count_calls() == 0
        assert len(store.all_rules()) == 0
        assert store.query_sessions() == []
