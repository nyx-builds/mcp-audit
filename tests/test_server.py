"""Tests for the MCP server tool definitions and handlers."""
from __future__ import annotations

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import CallStatus
from mcp_audit.server import MCPServer, TOOL_DEFINITIONS
from mcp_audit.storage import MemoryStore


@pytest.fixture
def server():
    return MCPServer(engine=AuditEngine(store=MemoryStore()))


class TestToolDefinitions:
    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 20

    def test_all_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Tool missing name: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "inputSchema" in tool, f"Tool {tool.get('name')} missing inputSchema"
            assert tool["inputSchema"]["type"] == "object"

    def test_tool_names_unique(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    @pytest.mark.parametrize(
        "expected_tool",
        [
            "start_session",
            "end_session",
            "get_session",
            "list_sessions",
            "record_call",
            "get_call",
            "query_calls",
            "log_event",
            "query_events",
            "get_stats",
            "get_agent_report",
            "get_cost_breakdown",
            "create_alert_rule",
            "list_alert_rules",
            "delete_alert_rule",
            "evaluate_alerts",
            "get_audit_summary",
        ],
    )
    def test_expected_tools_exist(self, expected_tool):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert expected_tool in names


class TestServerBasic:
    def test_list_tools(self, server):
        tools = server.list_tools()
        assert len(tools) == 20
        assert all("name" in t for t in tools)

    def test_tool_count_property(self, server):
        assert server.tool_count == 20

    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert "error" in result


class TestSessionTools:
    def test_start_session(self, server):
        result = server.call_tool("start_session", {"agent_id": "a1", "name": "test"})
        assert "result" in result
        assert result["result"]["agent_id"] == "a1"
        assert result["result"]["name"] == "test"
        assert result["result"]["is_active"] is True

    def test_end_session(self, server):
        start = server.call_tool("start_session", {"agent_id": "a1"})
        sid = start["result"]["id"]
        result = server.call_tool("end_session", {"session_id": sid})
        assert result["result"]["is_active"] is False

    def test_end_nonexistent_session(self, server):
        result = server.call_tool("end_session", {"session_id": "nonexistent"})
        assert "error" in result["result"]

    def test_get_session(self, server):
        start = server.call_tool("start_session", {"agent_id": "a1"})
        sid = start["result"]["id"]
        result = server.call_tool("get_session", {"session_id": sid})
        assert result["result"]["agent_id"] == "a1"

    def test_get_nonexistent_session(self, server):
        result = server.call_tool("get_session", {"session_id": "nonexistent"})
        assert "error" in result["result"]

    def test_list_sessions(self, server):
        server.call_tool("start_session", {"agent_id": "a1"})
        server.call_tool("start_session", {"agent_id": "a2"})
        result = server.call_tool("list_sessions", {})
        assert result["result"]["count"] == 2

    def test_list_sessions_filtered(self, server):
        server.call_tool("start_session", {"agent_id": "a1"})
        server.call_tool("start_session", {"agent_id": "a2"})
        result = server.call_tool("list_sessions", {"agent_id": "a1"})
        assert result["result"]["count"] == 1


class TestRecordCallTool:
    def test_record_call(self, server):
        session = server.call_tool("start_session", {"agent_id": "a1"})
        sid = session["result"]["id"]
        result = server.call_tool("record_call", {
            "session_id": sid,
            "tool_name": "search",
            "duration_ms": 150,
            "cost_usd": 0.01,
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert result["result"]["tool_name"] == "search"
        assert result["result"]["duration_ms"] == 150
        assert result["result"]["cost_usd"] == 0.01

    def test_record_error_call(self, server):
        session = server.call_tool("start_session", {})
        sid = session["result"]["id"]
        result = server.call_tool("record_call", {
            "session_id": sid,
            "tool_name": "fail_tool",
            "status": "error",
            "error": "Timeout",
        })
        assert result["result"]["status"] == "error"
        assert result["result"]["error"] == "Timeout"

    def test_get_call(self, server):
        session = server.call_tool("start_session", {})
        sid = session["result"]["id"]
        recorded = server.call_tool("record_call", {"session_id": sid, "tool_name": "t"})
        cid = recorded["result"]["id"]
        result = server.call_tool("get_call", {"call_id": cid})
        assert result["result"]["tool_name"] == "t"

    def test_get_nonexistent_call(self, server):
        result = server.call_tool("get_call", {"call_id": "nonexistent"})
        assert "error" in result["result"]

    def test_query_calls(self, server):
        session = server.call_tool("start_session", {})
        sid = session["result"]["id"]
        server.call_tool("record_call", {"session_id": sid, "tool_name": "search"})
        server.call_tool("record_call", {"session_id": sid, "tool_name": "fetch"})
        result = server.call_tool("query_calls", {"tool_name": "search"})
        assert result["result"]["count"] == 1


class TestTraceEventTools:
    def test_log_event(self, server):
        result = server.call_tool("log_event", {
            "trace_id": "t1",
            "event_type": "http_request",
            "message": "GET /api",
        })
        assert result["result"]["trace_id"] == "t1"
        assert result["result"]["event_type"] == "http_request"

    def test_query_events(self, server):
        server.call_tool("log_event", {"trace_id": "t1", "event_type": "step1"})
        server.call_tool("log_event", {"trace_id": "t2", "event_type": "step2"})
        result = server.call_tool("query_events", {"trace_id": "t1"})
        assert result["result"]["count"] == 1


class TestAnalyticsTools:
    def test_get_stats_empty(self, server):
        result = server.call_tool("get_stats", {})
        assert result["result"]["total_calls"] == 0

    def test_get_stats_with_data(self, server):
        session = server.call_tool("start_session", {"agent_id": "a1"})
        sid = session["result"]["id"]
        server.call_tool("record_call", {
            "session_id": sid,
            "tool_name": "tool1",
            "duration_ms": 100,
            "cost_usd": 0.01,
        })
        server.call_tool("record_call", {
            "session_id": sid,
            "tool_name": "tool2",
            "duration_ms": 200,
            "cost_usd": 0.02,
        })
        result = server.call_tool("get_stats", {})
        stats = result["result"]
        assert stats["total_calls"] == 2
        assert stats["total_cost_usd"] == 0.03

    def test_get_agent_report(self, server):
        session = server.call_tool("start_session", {"agent_id": "a1"})
        sid = session["result"]["id"]
        server.call_tool("record_call", {
            "session_id": sid,
            "tool_name": "tool1",
            "agent_id": "a1",
            "duration_ms": 100,
        })
        result = server.call_tool("get_agent_report", {"agent_id": "a1"})
        assert result["result"]["total_calls"] == 1

    def test_get_cost_breakdown(self, server):
        session = server.call_tool("start_session", {})
        sid = session["result"]["id"]
        server.call_tool("record_call", {"session_id": sid, "tool_name": "cheap", "cost_usd": 0.001})
        server.call_tool("record_call", {"session_id": sid, "tool_name": "expensive", "cost_usd": 0.50})
        result = server.call_tool("get_cost_breakdown", {"group_by": "tool"})
        assert result["result"]["total_cost_usd"] == 0.501
        assert result["result"]["breakdown"][0]["name"] == "expensive"


class TestAlertTools:
    def test_create_alert_rule(self, server):
        result = server.call_tool("create_alert_rule", {
            "name": "high_error",
            "metric": "error_rate",
            "operator": ">",
            "threshold": 50.0,
        })
        assert result["result"]["name"] == "high_error"

    def test_list_alert_rules(self, server):
        server.call_tool("create_alert_rule", {
            "name": "r1", "metric": "error_rate", "operator": ">", "threshold": 10.0,
        })
        server.call_tool("create_alert_rule", {
            "name": "r2", "metric": "p95_latency", "operator": ">", "threshold": 1000.0,
        })
        result = server.call_tool("list_alert_rules", {})
        assert result["result"]["count"] == 2

    def test_delete_alert_rule(self, server):
        created = server.call_tool("create_alert_rule", {
            "name": "test", "metric": "error_rate", "operator": ">", "threshold": 10.0,
        })
        rid = created["result"]["id"]
        result = server.call_tool("delete_alert_rule", {"rule_id": rid})
        assert result["result"]["deleted"] is True

    def test_evaluate_alerts_no_breach(self, server):
        server.call_tool("create_alert_rule", {
            "name": "test", "metric": "error_rate", "operator": ">", "threshold": 50.0,
        })
        result = server.call_tool("evaluate_alerts", {})
        assert result["result"]["triggered_count"] == 0

    def test_evaluate_alerts_breach(self, server):
        server.call_tool("create_alert_rule", {
            "name": "test", "metric": "error_rate", "operator": ">", "threshold": 0.0,
        })
        session = server.call_tool("start_session", {})
        sid = session["result"]["id"]
        server.call_tool("record_call", {
            "session_id": sid, "tool_name": "t", "status": "error",
        })
        result = server.call_tool("evaluate_alerts", {})
        assert result["result"]["triggered_count"] == 1


class TestAuditSummary:
    def test_summary_empty(self, server):
        result = server.call_tool("get_audit_summary", {})
        summary = result["result"]
        assert summary["total_calls"] == 0
        assert summary["total_sessions"] == 0
        assert summary["active_sessions"] == 0

    def test_summary_with_data(self, server):
        server.call_tool("start_session", {"agent_id": "a1"})
        result = server.call_tool("get_audit_summary", {})
        summary = result["result"]
        assert summary["total_sessions"] == 1
        assert summary["active_sessions"] == 1
