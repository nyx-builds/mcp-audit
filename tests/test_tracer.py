"""Tests for the traced_call context manager."""
from __future__ import annotations

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import CallStatus
from mcp_audit.storage import MemoryStore
from mcp_audit.tracer import traced_call


@pytest.fixture
def engine():
    return AuditEngine(store=MemoryStore())


class TestTracedCall:
    def test_basic_usage(self, engine):
        session = engine.start_session(agent_id="a1")
        with traced_call(engine, session_id=session.id, tool_name="search") as tc:
            tc.set_result({"found": True})
            tc.set_cost(0.001)
            tc.set_tokens(input_tokens=100, output_tokens=50)

        assert tc.call is not None
        assert tc.call.tool_name == "search"
        assert tc.call.status == CallStatus.SUCCESS
        assert tc.call.result == {"found": True}
        assert tc.call.cost_usd == 0.001
        assert tc.call.input_tokens == 100
        assert tc.call.output_tokens == 50
        assert tc.call.duration_ms is not None
        assert tc.call.duration_ms >= 0

    def test_records_on_exception(self, engine):
        session = engine.start_session()
        with pytest.raises(ValueError, match="boom"):
            with traced_call(engine, session_id=session.id, tool_name="fail") as tc:
                raise ValueError("boom")

        assert tc.call is not None
        assert tc.call.status == CallStatus.ERROR
        assert "ValueError" in (tc.call.error or "")

    def test_exception_not_suppressed(self, engine):
        session = engine.start_session()

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            with traced_call(engine, session_id=session.id, tool_name="t"):
                raise CustomError("test")

    def test_call_recorded_in_engine(self, engine):
        session = engine.start_session()
        with traced_call(engine, session_id=session.id, tool_name="recorded"):
            pass

        calls = engine.query_calls(tool_name="recorded")
        assert len(calls) == 1

    def test_with_arguments(self, engine):
        session = engine.start_session()
        with traced_call(
            engine,
            session_id=session.id,
            tool_name="search",
            arguments={"query": "test"},
        ) as tc:
            tc.set_result([])

        assert tc.call.arguments == {"query": "test"}

    def test_with_tags(self, engine):
        session = engine.start_session()
        with traced_call(
            engine, session_id=session.id, tool_name="t", tags=["production"]
        ) as tc:
            tc.add_tag("api")

        assert "production" in tc.call.tags
        assert "api" in tc.call.tags

    def test_with_server_name(self, engine):
        session = engine.start_session()
        with traced_call(
            engine,
            session_id=session.id,
            tool_name="charge",
            server_name="mcp-payments",
        ) as tc:
            pass

        assert tc.call.server_name == "mcp-payments"

    def test_with_metadata(self, engine):
        session = engine.start_session()
        with traced_call(engine, session_id=session.id, tool_name="t") as tc:
            tc.set_metadata("request_id", "req-123")
            tc.set_metadata("user", "alice")

        assert tc.call.metadata == {"request_id": "req-123", "user": "alice"}

    def test_updates_session_aggregates(self, engine):
        session = engine.start_session(agent_id="a1")
        with traced_call(engine, session_id=session.id, tool_name="t") as tc:
            tc.set_cost(0.05)
            tc.set_tokens(input_tokens=100, output_tokens=50)

        updated = engine.get_session(session.id)
        assert updated.total_calls == 1
        assert updated.total_cost_usd == 0.05
        assert updated.total_tokens == 150

    def test_duration_measured(self, engine):
        import time

        session = engine.start_session()
        with traced_call(engine, session_id=session.id, tool_name="slow") as tc:
            time.sleep(0.05)

        assert tc.call.duration_ms >= 40  # at least ~50ms

    def test_multiple_calls_in_sequence(self, engine):
        session = engine.start_session()
        for i in range(5):
            with traced_call(engine, session_id=session.id, tool_name=f"tool_{i}"):
                pass

        stats = engine.get_stats(session_id=session.id)
        assert stats["total_calls"] == 5
