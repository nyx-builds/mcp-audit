"""Tests for the @audit_call decorator and bind_session."""
from __future__ import annotations

import pytest

from mcp_audit import AuditEngine, CallStatus
from mcp_audit.decorator import audit_call, bind_session, get_audit_context


@pytest.fixture
def engine():
    return AuditEngine()


@pytest.fixture
def session(engine):
    return engine.start_session(agent_id="test-agent")


class TestAuditCallDecorator:
    def test_basic_decoration(self, engine, session):
        @audit_call(engine, session_id=session.id)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5

        calls = engine.query_calls(session_id=session.id)
        assert len(calls) == 1
        assert calls[0].tool_name == "add"
        assert calls[0].status == CallStatus.SUCCESS
        assert calls[0].duration_ms is not None
        assert calls[0].duration_ms >= 0

    def test_custom_tool_name(self, engine, session):
        @audit_call(engine, session_id=session.id, tool_name="my_search")
        def search(query: str):
            return [f"result for {query}"]

        search("test")
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].tool_name == "my_search"

    def test_records_exception(self, engine, session):
        @audit_call(engine, session_id=session.id)
        def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()

        calls = engine.query_calls(session_id=session.id)
        assert len(calls) == 1
        assert calls[0].status == CallStatus.ERROR
        assert "ValueError" in (calls[0].error or "")
        assert "boom" in (calls[0].error or "")

    def test_no_record_on_error_when_disabled(self, engine, session):
        @audit_call(engine, session_id=session.id, record_errors=False)
        def fail():
            raise ValueError("silent")

        with pytest.raises(ValueError):
            fail()

        calls = engine.query_calls(session_id=session.id)
        assert len(calls) == 0

    def test_cost_fn(self, engine, session):
        @audit_call(engine, session_id=session.id, cost_fn=lambda *_: 0.005)
        def expensive_call():
            return "done"

        expensive_call()
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].cost_usd == 0.005

    def test_token_fn(self, engine, session):
        @audit_call(
            engine, session_id=session.id,
            token_fn=lambda *_: (500, 200),
        )
        def llm_call():
            return "response"

        llm_call()
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].input_tokens == 500
        assert calls[0].output_tokens == 200

    def test_tags(self, engine, session):
        @audit_call(engine, session_id=session.id, tags=["production", "v2"])
        def tagged():
            return None

        tagged()
        calls = engine.query_calls(session_id=session.id)
        assert "production" in calls[0].tags
        assert "v2" in calls[0].tags

    def test_metadata(self, engine, session):
        @audit_call(engine, session_id=session.id, metadata={"region": "us-east"})
        def meta_call():
            return None

        meta_call()
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].metadata["region"] == "us-east"

    def test_server_name(self, engine, session):
        @audit_call(engine, session_id=session.id, server_name="search-server")
        def search():
            return []

        search()
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].server_name == "search-server"

    def test_preserves_function_metadata(self, engine, session):
        @audit_call(engine, session_id=session.id)
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_multiple_calls(self, engine, session):
        @audit_call(engine, session_id=session.id)
        def square(x):
            return x * x

        square(2)
        square(3)
        square(4)

        calls = engine.query_calls(session_id=session.id)
        assert len(calls) == 3

    def test_arguments_recorded(self, engine, session):
        @audit_call(engine, session_id=session.id)
        def greet(name, greeting="hello"):
            return f"{greeting}, {name}"

        greet("world", greeting="hi")
        calls = engine.query_calls(session_id=session.id)
        assert calls[0].arguments is not None
        # Arguments include positional args and keyword args
        assert "args" in calls[0].arguments


class TestBindSession:
    def test_bind_and_auto_audit(self, engine, session):
        ctx = bind_session(engine, session.id)

        @audit_call()  # no engine/session specified — uses bound context
        def search(query):
            return f"results for {query}"

        search("hello")
        calls = engine.query_calls(session_id=session.id)
        assert len(calls) == 1
        assert calls[0].tool_name == "search"

        ctx.reset()

    def test_no_engine_bound_just_calls(self):
        # Without binding, decorator should just call the function
        @audit_call()
        def add(a, b):
            return a + b

        assert add(1, 2) == 3  # no recording, no error

    def test_reset_restores_previous(self, engine, session):
        ctx = bind_session(engine, session.id)

        @audit_call()
        def f():
            return 42

        f()
        assert len(engine.query_calls(session_id=session.id)) == 1

        ctx.reset()

        # After reset, function should not record
        f()
        assert len(engine.query_calls(session_id=session.id)) == 1  # still 1


class TestAuditCallWithoutEngine:
    def test_no_engine_just_runs_function(self):
        @audit_call()
        def compute(x):
            return x * 10

        assert compute(5) == 50

    def test_no_engine_with_exception(self):
        @audit_call()
        def fail():
            raise RuntimeError("error")

        with pytest.raises(RuntimeError, match="error"):
            fail()


class TestAuditCallWithStats:
    def test_stats_after_multiple_calls(self, engine, session):
        @audit_call(engine, session_id=session.id, cost_fn=lambda *_: 0.001)
        def llm():
            return "ok"

        for _ in range(10):
            llm()

        stats = engine.get_stats(session_id=session.id)
        assert stats["total_calls"] == 10
        assert stats["total_cost_usd"] == 0.01
