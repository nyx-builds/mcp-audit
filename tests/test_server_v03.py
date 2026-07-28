"""Tests for new v0.3.0 MCP server tools: get_tool_health, get_recent_calls, export_calls."""
from __future__ import annotations

import csv
import json
import os

import pytest

from mcp_audit import AuditEngine, CallStatus
from mcp_audit.server import MCPServer


@pytest.fixture
def server():
    engine = AuditEngine()
    return MCPServer(engine=engine)


@pytest.fixture
def populated_server():
    engine = AuditEngine()
    server = MCPServer(engine=engine)

    session = engine.start_session(agent_id="mcp-agent")
    for i in range(5):
        server.call_tool("record_call", {
            "session_id": session.id,
            "tool_name": "search",
            "agent_id": "mcp-agent",
            "status": "success" if i < 4 else "error",
            "duration_ms": 100.0 + i * 50,
            "cost_usd": 0.001 * (i + 1),
            "input_tokens": 100,
            "output_tokens": 50,
        })
    for i in range(3):
        server.call_tool("record_call", {
            "session_id": session.id,
            "tool_name": "fetch",
            "agent_id": "mcp-agent",
            "status": "success",
            "duration_ms": 30.0,
            "cost_usd": 0.0001,
        })

    return server


class TestGetToolHealthTool:
    def test_tool_definition_exists(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "get_tool_health" in names

    def test_returns_health(self, populated_server):
        result = populated_server.call_tool("get_tool_health", {})
        data = result["result"]

        assert data["tool_count"] == 2
        assert data["tools"][0]["tool_name"] == "search"
        assert data["tools"][0]["call_count"] == 5
        assert data["tools"][0]["error_rate"] == 20.0
        assert data["tools"][1]["tool_name"] == "fetch"
        assert data["tools"][1]["call_count"] == 3

    def test_empty(self, server):
        result = server.call_tool("get_tool_health", {})
        assert result["result"]["tool_count"] == 0


class TestGetRecentCallsTool:
    def test_tool_definition_exists(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "get_recent_calls" in names

    def test_default_n(self, populated_server):
        result = populated_server.call_tool("get_recent_calls", {})
        data = result["result"]
        assert data["count"] == 8  # 5 + 3 calls

    def test_custom_n(self, populated_server):
        result = populated_server.call_tool("get_recent_calls", {"n": 3})
        assert result["result"]["count"] == 3

    def test_serialized_calls(self, populated_server):
        result = populated_server.call_tool("get_recent_calls", {"n": 1})
        call = result["result"]["calls"][0]
        assert "id" in call
        assert "tool_name" in call
        assert "status" in call
        assert "duration_ms" in call


class TestExportCallsTool:
    def test_tool_definition_exists(self, server):
        tools = server.list_tools()
        names = {t["name"] for t in tools}
        assert "export_calls" in names

    def test_export_jsonl(self, populated_server, tmp_path):
        output_file = str(tmp_path / "export.jsonl")
        result = populated_server.call_tool("export_calls", {
            "format": "jsonl",
            "output_path": output_file,
        })
        data = result["result"]

        assert data["format"] == "jsonl"
        assert data["record_count"] == 8
        assert os.path.exists(output_file)

        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 8
        record = json.loads(lines[0])
        assert "tool_name" in record

    def test_export_csv(self, populated_server, tmp_path):
        output_file = str(tmp_path / "export.csv")
        result = populated_server.call_tool("export_calls", {
            "format": "csv",
            "output_path": output_file,
        })
        data = result["result"]

        assert data["format"] == "csv"
        assert data["record_count"] == 8
        assert os.path.exists(output_file)

        with open(output_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 8

    def test_export_with_session_filter(self, populated_server, tmp_path):
        # Get session id
        summary = populated_server.call_tool("get_audit_summary", {})
        # Just export all and verify count
        output_file = str(tmp_path / "filtered.jsonl")
        result = populated_server.call_tool("export_calls", {
            "format": "jsonl",
            "output_path": output_file,
            "limit": 3,
        })
        assert result["result"]["record_count"] == 3

    def test_invalid_format(self, populated_server, tmp_path):
        result = populated_server.call_tool("export_calls", {
            "format": "xml",
            "output_path": str(tmp_path / "bad.xml"),
        })
        assert "error" in result["result"]


class TestNewToolsInSummary:
    def test_summary_includes_all_data(self, populated_server):
        result = populated_server.call_tool("get_audit_summary", {})
        data = result["result"]

        assert data["total_calls"] == 8
        assert data["total_sessions"] == 1
        assert data["unique_tools"] == 2
