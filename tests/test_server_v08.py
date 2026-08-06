"""Server-level tests for the v0.8 time-series analytics tools."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import CallStatus
from mcp_audit.server import MCPServer, TOOL_DEFINITIONS
from mcp_audit.storage import MemoryStore


@pytest.fixture
def server():
    engine = AuditEngine(store=MemoryStore())
    return MCPServer(engine=engine)


@pytest.fixture
def populated_server():
    engine = AuditEngine(store=MemoryStore())
    server = MCPServer(engine=engine)
    session = engine.start_session(agent_id="agent-1")
    for i in range(30):
        engine.record_call(
            session.id, f"tool_{i % 3}",
            agent_id="agent-1",
            status=CallStatus.ERROR if i % 7 == 0 else CallStatus.SUCCESS,
            duration_ms=50 + i * 10,
            cost_usd=0.01 * (i + 1),
            input_tokens=100,
            output_tokens=50,
        )
    return server


class TestToolDefinitionsV08:
    def test_new_tools_exist(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert "get_timeseries" in names
        assert "detect_anomalies" in names
        assert "analyze_trends" in names
        assert "get_heatmap" in names

    def test_new_tools_have_descriptions(self):
        for name in ["get_timeseries", "detect_anomalies", "analyze_trends", "get_heatmap"]:
            tool = next(t for t in TOOL_DEFINITIONS if t["name"] == name)
            assert len(tool["description"]) > 20

    def test_new_tools_have_input_schema(self):
        for name in ["get_timeseries", "detect_anomalies", "analyze_trends", "get_heatmap"]:
            tool = next(t for t in TOOL_DEFINITIONS if t["name"] == name)
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]


class TestGetTimeseriesTool:
    def test_empty(self, server):
        result = server.call_tool("get_timeseries", {"window": "5m"})
        assert "result" in result
        assert result["result"]["bucket_count"] == 0

    def test_basic(self, populated_server):
        result = populated_server.call_tool("get_timeseries", {"window": "5m"})
        assert "result" in result
        assert result["result"]["bucket_count"] > 0
        assert "buckets" in result["result"]

    def test_with_metric(self, populated_server):
        result = populated_server.call_tool(
            "get_timeseries", {"window": "5m", "metric": "error_rate"}
        )
        assert "result" in result
        for bucket in result["result"]["buckets"]:
            assert "value" in bucket


class TestDetectAnomaliesTool:
    def test_empty(self, server):
        result = server.call_tool("detect_anomalies", {})
        assert "result" in result
        assert result["result"]["total_anomalies"] == 0

    def test_basic(self, populated_server):
        result = populated_server.call_tool("detect_anomalies", {})
        assert "result" in result
        assert result["result"]["status"] == "ok"
        assert "anomalies" in result["result"]

    def test_custom_metrics(self, populated_server):
        result = populated_server.call_tool(
            "detect_anomalies", {"metrics": ["call_count"]}
        )
        assert "result" in result
        assert result["result"]["metrics_analyzed"] == ["call_count"]


class TestAnalyzeTrendsTool:
    def test_empty(self, server):
        result = server.call_tool("analyze_trends", {})
        assert "result" in result
        assert result["result"]["status"] == "ok"

    def test_basic(self, populated_server):
        result = populated_server.call_tool("analyze_trends", {"window": "5m"})
        assert "result" in result
        assert "trends" in result["result"]


class TestGetHeatmapTool:
    def test_empty(self, server):
        result = server.call_tool("get_heatmap", {})
        assert "result" in result
        assert result["result"]["matrix"] == {}

    def test_basic(self, populated_server):
        result = populated_server.call_tool("get_heatmap", {"window": "5m"})
        assert "result" in result
        assert result["result"]["tool_count"] >= 1

    def test_metric_selection(self, populated_server):
        result = populated_server.call_tool(
            "get_heatmap", {"metric": "total_cost_usd"}
        )
        assert "result" in result
        assert result["result"]["metric"] == "total_cost_usd"


class TestUnknownToolError:
    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent", {})
        assert "error" in result
