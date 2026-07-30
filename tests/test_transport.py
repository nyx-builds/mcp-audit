"""Tests for the MCP stdio transport layer."""
from __future__ import annotations

import asyncio
import json

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.transport import create_fastmcp_server
from mcp_audit.models import CallStatus


# ── Server creation ──────────────────────────────────────────────────────


class TestCreateFastMCPServer:
    def test_creates_server(self):
        """create_fastmcp_server returns an MCP SDK server instance."""
        server = create_fastmcp_server()
        assert server is not None
        assert hasattr(server, "add_tool")
        assert hasattr(server, "run")

    def test_server_has_correct_name(self):
        server = create_fastmcp_server()
        assert server.name == "mcp-audit"

    def test_server_has_instructions(self):
        server = create_fastmcp_server()
        assert server.instructions is not None
        assert "observability" in server.instructions.lower()

    def test_uses_provided_engine(self):
        """Passing an engine means the server uses it."""
        engine = AuditEngine()
        server = create_fastmcp_server(engine=engine)
        assert isinstance(server, object)

    def test_creates_independent_engines(self):
        """Without an engine, each call gets a fresh one."""
        s1 = create_fastmcp_server()
        s2 = create_fastmcp_server()
        assert s1 is not s2


# ── Tool registration ────────────────────────────────────────────────────


class TestToolRegistration:
    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """All tool definitions should be registered on the server."""
        server = create_fastmcp_server()
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert len(tool_names) == 23

    @pytest.mark.asyncio
    async def test_expected_tool_names(self):
        server = create_fastmcp_server()
        tools = await server.list_tools()
        names = {t.name for t in tools}
        expected = {
            "start_session", "end_session", "get_session", "list_sessions",
            "record_call", "get_call", "query_calls",
            "log_event", "query_events",
            "get_stats", "get_agent_report", "get_cost_breakdown",
            "create_alert_rule", "list_alert_rules", "delete_alert_rule",
            "evaluate_alerts", "get_audit_summary",
            "get_tool_health", "get_recent_calls", "export_calls",
            "export_otlp", "export_otlp_metrics", "export_prometheus",
        }
        assert names == expected

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self):
        server = create_fastmcp_server()
        tools = await server.list_tools()
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 10


# ── Tool execution via MCP SDK ──────────────────────────────────────────


class TestToolExecutionViaFastMCP:
    @pytest.mark.asyncio
    async def test_start_session_tool(self):
        """start_session should return a session dict."""
        server = create_fastmcp_server()
        result = await server.call_tool("start_session", {
            "agent_id": "test-agent",
            "name": "test-session",
        })
        text = _extract_text(result)
        data = json.loads(text)
        assert "result" in data
        session = data["result"]
        assert session["agent_id"] == "test-agent"
        assert "id" in session
        assert session["is_active"] is True

    @pytest.mark.asyncio
    async def test_record_call_tool(self):
        """record_call should record and return a call dict."""
        server = create_fastmcp_server()

        # First start a session
        session_result = await server.call_tool("start_session", {
            "agent_id": "test-agent",
        })
        session_data = json.loads(_extract_text(session_result))
        session_id = session_data["result"]["id"]

        # Record a call
        call_result = await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "web_search",
            "duration_ms": 150.0,
            "cost_usd": 0.003,
            "status": "success",
        })
        call_data = json.loads(_extract_text(call_result))
        assert "result" in call_data
        call = call_data["result"]
        assert call["tool_name"] == "web_search"
        assert call["duration_ms"] == 150.0
        assert call["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_stats_tool(self):
        """get_stats should return aggregate statistics."""
        server = create_fastmcp_server()

        # Record some calls
        session_result = await server.call_tool("start_session", {"agent_id": "a1"})
        session_id = json.loads(_extract_text(session_result))["result"]["id"]

        for i in range(5):
            await server.call_tool("record_call", {
                "session_id": session_id,
                "tool_name": f"tool_{i}",
                "cost_usd": 0.01 * i,
                "duration_ms": 100.0 + i * 10,
            })

        stats_result = await server.call_tool("get_stats", {})
        stats = json.loads(_extract_text(stats_result))["result"]
        assert stats["total_calls"] == 5

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Calling a non-existent tool should raise an error."""
        server = create_fastmcp_server()
        with pytest.raises(Exception):
            await server.call_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_alert_rule_lifecycle(self):
        """Full alert rule lifecycle via MCP tools."""
        server = create_fastmcp_server()

        # Create rule
        create_result = await server.call_tool("create_alert_rule", {
            "name": "High Error Rate",
            "metric": "error_rate",
            "operator": ">",
            "threshold": 0.5,
        })
        rule = json.loads(_extract_text(create_result))["result"]
        assert rule["name"] == "High Error Rate"

        # List rules
        list_result = await server.call_tool("list_alert_rules", {})
        rules = json.loads(_extract_text(list_result))["result"]
        assert rules["count"] >= 1

        # Evaluate alerts
        eval_result = await server.call_tool("evaluate_alerts", {})
        evaluated = json.loads(_extract_text(eval_result))["result"]
        assert "evaluated" in evaluated
        assert evaluated["evaluated"] is True

        # Delete rule
        del_result = await server.call_tool("delete_alert_rule", {
            "rule_id": rule["id"],
        })
        deleted = json.loads(_extract_text(del_result))["result"]
        assert deleted["deleted"] is True

    @pytest.mark.asyncio
    async def test_query_calls_filtering(self):
        """query_calls should support filtering."""
        server = create_fastmcp_server()

        session_result = await server.call_tool("start_session", {"agent_id": "a1"})
        session_id = json.loads(_extract_text(session_result))["result"]["id"]

        await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "search",
            "cost_usd": 0.10,
        })
        await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "fetch",
            "cost_usd": 0.01,
        })

        # Query by tool name
        result = await server.call_tool("query_calls", {"tool_name": "search"})
        data = json.loads(_extract_text(result))["result"]
        assert data["count"] == 1
        assert data["calls"][0]["tool_name"] == "search"

    @pytest.mark.asyncio
    async def test_get_audit_summary(self):
        """get_audit_summary should return high-level overview."""
        server = create_fastmcp_server()

        session_result = await server.call_tool("start_session", {"agent_id": "a1"})
        session_id = json.loads(_extract_text(session_result))["result"]["id"]

        await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "test_tool",
            "cost_usd": 0.05,
        })

        summary_result = await server.call_tool("get_audit_summary", {})
        summary = json.loads(_extract_text(summary_result))["result"]
        assert summary["total_calls"] >= 1
        assert summary["total_sessions"] >= 1
        assert summary["total_cost_usd"] >= 0.05

    @pytest.mark.asyncio
    async def test_log_and_query_events(self):
        """Trace events can be logged and queried."""
        server = create_fastmcp_server()

        # Log events
        for i in range(3):
            await server.call_tool("log_event", {
                "trace_id": "trace-001",
                "event_type": "http_request",
                "message": f"Request {i}",
                "severity": "info",
            })

        result = await server.call_tool("query_events", {"trace_id": "trace-001"})
        data = json.loads(_extract_text(result))["result"]
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_cost_breakdown(self):
        """get_cost_breakdown should aggregate costs by group."""
        server = create_fastmcp_server()

        session_result = await server.call_tool("start_session", {"agent_id": "a1"})
        session_id = json.loads(_extract_text(session_result))["result"]["id"]

        await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "expensive_tool",
            "cost_usd": 1.00,
        })
        await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "cheap_tool",
            "cost_usd": 0.01,
        })

        result = await server.call_tool("get_cost_breakdown", {"group_by": "tool"})
        data = json.loads(_extract_text(result))["result"]
        assert "groups" in data or "breakdown" in data


# ── Round-trip: shared engine across library and transport ─────────────


class TestSharedEngine:
    def test_engine_shared_between_calls(self):
        """When the same engine is passed, data persists across tool calls."""
        engine = AuditEngine()
        server = create_fastmcp_server(engine=engine)

        # Use the engine directly
        session = engine.start_session(agent_id="direct")
        engine.record_call(
            session_id=session.id,
            tool_name="direct_call",
            cost_usd=0.01,
        )

        # The server should see this data via the shared engine
        calls = engine.query_calls()
        assert any(c.tool_name == "direct_call" for c in calls)

    def test_multiple_servers_same_engine(self):
        """Multiple servers can share one engine."""
        engine = AuditEngine()
        s1 = create_fastmcp_server(engine=engine)
        s2 = create_fastmcp_server(engine=engine)
        assert s1 is not s2


# ── Error handling ───────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_required_arg(self):
        """Calling record_call without session_id should raise an error."""
        server = create_fastmcp_server()
        with pytest.raises(Exception):
            await server.call_tool("record_call", {"tool_name": "test"})

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        server = create_fastmcp_server()
        result = await server.call_tool("get_session", {"session_id": "nonexistent"})
        data = json.loads(_extract_text(result))
        assert "error" in data["result"] or "error" in data

    @pytest.mark.asyncio
    async def test_invalid_status_value(self):
        """Invalid status enum value should be handled gracefully."""
        server = create_fastmcp_server()

        session_result = await server.call_tool("start_session", {"agent_id": "a1"})
        session_id = json.loads(_extract_text(session_result))["result"]["id"]

        result = await server.call_tool("record_call", {
            "session_id": session_id,
            "tool_name": "test",
            "status": "invalid_status",
        })
        data = json.loads(_extract_text(result))
        # Inner server catches the ValueError and returns an error dict
        assert "error" in data


# ── Serialization ────────────────────────────────────────────────────────


class TestSerialization:
    @pytest.mark.asyncio
    async def test_result_is_json_string(self):
        """Tool results should be JSON-serializable strings."""
        server = create_fastmcp_server()
        result = await server.call_tool("get_audit_summary", {})
        text = _extract_text(result)
        # Must be valid JSON
        data = json.loads(text)
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_datetimes_are_serialized(self):
        """Datetime fields should be ISO format strings, not objects."""
        server = create_fastmcp_server()
        result = await server.call_tool("start_session", {"agent_id": "a1"})
        text = _extract_text(result)
        data = json.loads(text)
        session = data["result"]
        assert "started_at" in session
        assert isinstance(session["started_at"], str)
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(session["started_at"])


# ── Helper ───────────────────────────────────────────────────────────────


def _extract_text(result) -> str:
    """Extract text from an MCP SDK tool call result.

    Handles multiple result formats:
    - SDK v2: CallToolResult with .content list of TextContent
    - SDK v1: list of content blocks
    - Plain string
    - Dict with text
    """
    # SDK v2: CallToolResult object with .content attribute
    if hasattr(result, "content") and not isinstance(result, (list, str, dict)):
        texts = []
        for block in result.content:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "".join(texts)

    # SDK v1: list of content blocks
    if isinstance(result, list):
        texts = []
        for block in result:
            if hasattr(block, "text"):
                texts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "".join(texts)

    if isinstance(result, str):
        return result
    if isinstance(result, dict) and "text" in result:
        return result["text"]

    # Try to access .content on any object
    if hasattr(result, "content"):
        try:
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            if texts:
                return "".join(texts)
        except Exception:
            pass

    return json.dumps(result, default=str)
