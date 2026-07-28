"""Tests for engine features added in v0.3.0: get_recent_calls, get_tool_health."""
from __future__ import annotations

import pytest

from mcp_audit import AuditEngine, CallStatus


@pytest.fixture
def engine():
    return AuditEngine()


@pytest.fixture
def populated_engine(engine):
    """Engine with varied calls for health testing."""
    session = engine.start_session(agent_id="health-agent")

    # search: 4 success, 1 error
    for i in range(5):
        engine.record_call(
            session_id=session.id,
            tool_name="search",
            agent_id="health-agent",
            status=CallStatus.SUCCESS if i < 4 else CallStatus.ERROR,
            duration_ms=100.0 + i * 50,
            cost_usd=0.001,
        )

    # fetch: 3 success
    for i in range(3):
        engine.record_call(
            session_id=session.id,
            tool_name="fetch",
            agent_id="health-agent",
            status=CallStatus.SUCCESS,
            duration_ms=50.0,
            cost_usd=0.0005,
        )

    # timeout: 1 timeout
    engine.record_call(
        session_id=session.id,
        tool_name="slow_api",
        agent_id="health-agent",
        status=CallStatus.TIMEOUT,
        duration_ms=5000.0,
        cost_usd=0.01,
    )

    return engine


class TestGetRecentCalls:
    def test_returns_n_most_recent(self, populated_engine):
        calls = populated_engine.get_recent_calls(n=3)
        assert len(calls) == 3

    def test_default_n_is_10(self, populated_engine):
        calls = populated_engine.get_recent_calls()
        assert len(calls) == 9  # we have 9 calls total

    def test_more_than_available(self, populated_engine):
        calls = populated_engine.get_recent_calls(n=100)
        assert len(calls) == 9

    def test_newest_first(self, populated_engine):
        calls = populated_engine.get_recent_calls(n=5)
        # The last recorded call was slow_api timeout
        assert calls[0].tool_name == "slow_api"

    def test_filter_by_session(self, populated_engine):
        sessions = populated_engine.list_sessions()
        session_id = sessions[0].id

        calls = populated_engine.get_recent_calls(n=100, session_id=session_id)
        assert len(calls) == 9

    def test_filter_by_agent(self, populated_engine):
        calls = populated_engine.get_recent_calls(n=100, agent_id="health-agent")
        assert len(calls) == 9

        calls = populated_engine.get_recent_calls(n=100, agent_id="no-such-agent")
        assert len(calls) == 0


class TestGetToolHealth:
    def test_returns_one_entry_per_tool(self, populated_engine):
        health = populated_engine.get_tool_health()
        tool_names = {h["tool_name"] for h in health}
        assert tool_names == {"search", "fetch", "slow_api"}

    def test_sorted_by_call_count(self, populated_engine):
        health = populated_engine.get_tool_health()
        assert health[0]["tool_name"] == "search"  # 5 calls
        assert health[1]["tool_name"] == "fetch"   # 3 calls
        assert health[2]["tool_name"] == "slow_api"  # 1 call

    def test_search_health_metrics(self, populated_engine):
        health = populated_engine.get_tool_health()
        search = next(h for h in health if h["tool_name"] == "search")

        assert search["call_count"] == 5
        assert search["error_count"] == 1
        assert search["error_rate"] == 20.0
        assert search["total_cost_usd"] == 0.005
        assert search["avg_cost_usd"] == 0.001

    def test_slow_api_health(self, populated_engine):
        health = populated_engine.get_tool_health()
        slow = next(h for h in health if h["tool_name"] == "slow_api")

        assert slow["call_count"] == 1
        assert slow["timeout_count"] == 1
        assert slow["avg_latency_ms"] == 5000.0
        assert slow["p95_latency_ms"] == 5000.0

    def test_empty_engine(self, engine):
        health = engine.get_tool_health()
        assert health == []

    def test_filter_by_agent(self, populated_engine):
        health = populated_engine.get_tool_health(agent_id="health-agent")
        assert len(health) == 3

        health = populated_engine.get_tool_health(agent_id="no-such-agent")
        assert health == []

    def test_has_p95_latency(self, populated_engine):
        health = populated_engine.get_tool_health()
        search = next(h for h in health if h["tool_name"] == "search")
        # search has durations [100, 150, 200, 250, 300] + one error at [300]
        # p95 should be near max
        assert search["p95_latency_ms"] >= 250.0
