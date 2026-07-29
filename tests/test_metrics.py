"""Tests for OTLP metrics export (mcp_audit.metrics)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_audit import AuditEngine
from mcp_audit.models import CallStatus
from mcp_audit.metrics import (
    build_metrics,
    build_otlp_metrics_request,
    export_otlp_metrics_http,
    export_otlp_metrics_jsonl,
    export_otlp_metrics_to_string,
    OTLPMetricsExporter,
    _bucketize,
    _LATENCY_BOUNDS,
    _COST_BOUNDS,
    _empty_metrics,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def populated_engine() -> AuditEngine:
    """An engine with varied calls across multiple tools."""
    engine = AuditEngine()
    session = engine.start_session(agent_id="agent-1", name="test-session")

    # Successful web_search call — low cost, fast
    engine.record_call(
        session_id=session.id,
        tool_name="web_search",
        agent_id="agent-1",
        server_name="mcp-web",
        status=CallStatus.SUCCESS,
        duration_ms=50.0,
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.001,
    )
    # Successful web_search call — slightly slower
    engine.record_call(
        session_id=session.id,
        tool_name="web_search",
        agent_id="agent-1",
        server_name="mcp-web",
        status=CallStatus.SUCCESS,
        duration_ms=120.0,
        input_tokens=150,
        output_tokens=300,
        cost_usd=0.002,
    )
    # Failed data_analysis call — error
    engine.record_call(
        session_id=session.id,
        tool_name="data_analysis",
        agent_id="agent-1",
        server_name="mcp-data",
        status=CallStatus.ERROR,
        error="Timeout",
        duration_ms=5000.0,
        input_tokens=500,
        output_tokens=0,
        cost_usd=0.01,
    )
    # Successful expensive code_run call
    engine.record_call(
        session_id=session.id,
        tool_name="code_run",
        agent_id="agent-1",
        server_name="mcp-code",
        status=CallStatus.SUCCESS,
        duration_ms=2000.0,
        input_tokens=2000,
        output_tokens=5000,
        cost_usd=0.15,
    )
    return engine


@pytest.fixture
def empty_engine() -> AuditEngine:
    return AuditEngine()


# ── _bucketize tests ─────────────────────────────────────────────────

class TestBucketize:
    def test_basic_bucketing(self):
        bounds = [10.0, 20.0, 30.0]
        values = [5.0, 15.0, 25.0, 35.0, 10.0]
        counts, explicit_bounds = _bucketize(values, bounds)
        # ≤10: [5, 10] = 2; ≤20: [15] = 1; ≤30: [25] = 1; >30: [35] = 1
        assert counts == [2, 1, 1, 1]
        assert explicit_bounds == bounds

    def test_empty_values(self):
        counts, bounds = _bucketize([], [10.0, 20.0])
        assert counts == [0, 0, 0]

    def test_all_overflow(self):
        counts, _ = _bucketize([100.0, 200.0], [10.0, 20.0])
        assert counts == [0, 0, 2]

    def test_all_underflow(self):
        counts, _ = _bucketize([1.0, 2.0, 3.0], [10.0, 20.0])
        assert counts == [3, 0, 0]

    def test_no_bounds(self):
        counts, explicit_bounds = _bucketize([5.0, 10.0], [])
        assert counts == [2]
        assert explicit_bounds == []

    def test_single_bound(self):
        counts, explicit_bounds = _bucketize([5.0, 15.0], [10.0])
        assert counts == [1, 1]
        assert explicit_bounds == [10.0]

    def test_exact_boundary(self):
        """Values exactly at a boundary go into that bucket."""
        counts, _ = _bucketize([10.0], [10.0, 20.0])
        assert counts == [1, 0, 0]


# ── build_metrics tests ──────────────────────────────────────────────

class TestBuildMetrics:
    def test_returns_list_of_metrics(self, populated_engine):
        metrics = build_metrics(populated_engine)
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_metric_names_present(self, populated_engine):
        metrics = build_metrics(populated_engine)
        names = {m["name"] for m in metrics}
        assert "tool.call.count" in names
        assert "tool.error.count" in names
        assert "tool.duration_ms" in names
        assert "tool.cost.usd" in names
        assert "tool.tokens.input" in names
        assert "tool.tokens.output" in names
        assert "session.count" in names
        assert "error.rate" in names
        assert "total.cost.usd" in names

    def test_call_count_metric_is_sum(self, populated_engine):
        metrics = build_metrics(populated_engine)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        assert "sum" in call_count
        dps = call_count["sum"]["dataPoints"]
        # 3 tools: web_search (2), data_analysis (1), code_run (1)
        assert len(dps) == 3
        counts_by_tool = {}
        for dp in dps:
            for attr in dp["attributes"]:
                if attr["key"] == "tool.name":
                    tool = attr["value"]["stringValue"]
                    counts_by_tool[tool] = int(dp["asInt"])
        assert counts_by_tool["web_search"] == 2
        assert counts_by_tool["data_analysis"] == 1
        assert counts_by_tool["code_run"] == 1

    def test_error_count_metric(self, populated_engine):
        metrics = build_metrics(populated_engine)
        error_metric = next(m for m in metrics if m["name"] == "tool.error.count")
        dps = error_metric["sum"]["dataPoints"]
        assert len(dps) == 1  # only data_analysis had errors
        assert int(dps[0]["asInt"]) == 1

    def test_duration_histogram(self, populated_engine):
        metrics = build_metrics(populated_engine)
        duration_metric = next(m for m in metrics if m["name"] == "tool.duration_ms")
        assert "histogram" in duration_metric
        dps = duration_metric["histogram"]["dataPoints"]
        # One histogram DP per tool
        assert len(dps) == 3
        for dp in dps:
            assert "count" in dp
            assert "sum" in dp
            assert "bucketCounts" in dp
            assert "explicitBounds" in dp
            # bucket counts: len(bounds)+1
            assert len(dp["bucketCounts"]) == len(dp["explicitBounds"]) + 1

    def test_cost_histogram(self, populated_engine):
        metrics = build_metrics(populated_engine)
        cost_metric = next(m for m in metrics if m["name"] == "tool.cost.usd")
        assert "histogram" in cost_metric
        dps = cost_metric["histogram"]["dataPoints"]
        # All tools have cost > 0
        assert len(dps) == 3

    def test_token_counters(self, populated_engine):
        metrics = build_metrics(populated_engine)
        input_metric = next(m for m in metrics if m["name"] == "tool.tokens.input")
        dps = input_metric["sum"]["dataPoints"]
        # web_search: 100+150=250, data_analysis: 500, code_run: 2000
        total = sum(int(dp["asInt"]) for dp in dps)
        assert total == 2750

    def test_session_count_gauge(self, populated_engine):
        metrics = build_metrics(populated_engine)
        session_metric = next(m for m in metrics if m["name"] == "session.count")
        assert "gauge" in session_metric
        dps = session_metric["gauge"]["dataPoints"]
        assert len(dps) >= 2  # all + active

    def test_error_rate_gauge(self, populated_engine):
        metrics = build_metrics(populated_engine)
        error_rate = next(m for m in metrics if m["name"] == "error.rate")
        assert "gauge" in error_rate
        dp = error_rate["gauge"]["dataPoints"][0]
        # 1 error out of 4 calls = 25%
        assert abs(dp["asDouble"] - 25.0) < 0.01

    def test_total_cost_gauge(self, populated_engine):
        metrics = build_metrics(populated_engine)
        cost_metric = next(m for m in metrics if m["name"] == "total.cost.usd")
        assert "gauge" in cost_metric
        dp = cost_metric["gauge"]["dataPoints"][0]
        # 0.001 + 0.002 + 0.01 + 0.15 = 0.163
        assert abs(dp["asDouble"] - 0.163) < 0.001

    def test_empty_engine_returns_metrics(self, empty_engine):
        metrics = build_metrics(empty_engine)
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        names = {m["name"] for m in metrics}
        assert "tool.call.count" in names
        assert "error.rate" in names

    def test_session_id_filter(self, populated_engine):
        other_session = populated_engine.start_session(agent_id="agent-2")
        populated_engine.record_call(
            session_id=other_session.id,
            tool_name="other_tool",
            agent_id="agent-2",
            status=CallStatus.SUCCESS,
            duration_ms=10.0,
            cost_usd=0.0,
        )

        # Filter by original session
        sessions = populated_engine.list_sessions(limit=100)
        original_session = [s for s in sessions if s.agent_id == "agent-1"][0]
        metrics = build_metrics(populated_engine, session_id=original_session.id)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 4  # original calls only

    def test_tool_name_filter(self, populated_engine):
        metrics = build_metrics(populated_engine, tool_name="web_search")
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        dps = call_count["sum"]["dataPoints"]
        total = sum(int(dp["asInt"]) for dp in dps)
        assert total == 2

    def test_sum_is_monotonic(self, populated_engine):
        metrics = build_metrics(populated_engine)
        for m in metrics:
            if "sum" in m:
                assert m["sum"]["isMonotonic"] is True

    def test_temporality_cumulative(self, populated_engine):
        metrics = build_metrics(populated_engine)
        for m in metrics:
            if "sum" in m:
                assert m["sum"]["aggregationTemporality"] == "AGGREGATION_TEMPORALITY_CUMULATIVE"
            if "histogram" in m:
                assert m["histogram"]["aggregationTemporality"] == "AGGREGATION_TEMPORALITY_CUMULATIVE"

    def test_metric_has_description(self, populated_engine):
        metrics = build_metrics(populated_engine)
        for m in metrics:
            assert "description" in m
            assert len(m["description"]) > 0

    def test_metric_has_unit(self, populated_engine):
        metrics = build_metrics(populated_engine)
        for m in metrics:
            assert "unit" in m
            assert len(m["unit"]) > 0

    def test_histogram_has_min_max(self, populated_engine):
        metrics = build_metrics(populated_engine)
        duration_metric = next(m for m in metrics if m["name"] == "tool.duration_ms")
        for dp in duration_metric["histogram"]["dataPoints"]:
            assert "min" in dp
            assert "max" in dp

    def test_data_points_have_timestamp(self, populated_engine):
        metrics = build_metrics(populated_engine)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        for dp in call_count["sum"]["dataPoints"]:
            assert "timeUnixNano" in dp
            assert int(dp["timeUnixNano"]) > 0

    def test_attributes_are_otlp_format(self, populated_engine):
        metrics = build_metrics(populated_engine)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        for dp in call_count["sum"]["dataPoints"]:
            for attr in dp["attributes"]:
                assert "key" in attr
                assert "value" in attr
                # Value must be an OTLP AnyValue
                assert len(attr["value"]) == 1


# ── build_otlp_metrics_request tests ─────────────────────────────────

class TestBuildOtlpMetricsRequest:
    def test_structure(self, populated_engine):
        metrics = build_metrics(populated_engine)
        request = build_otlp_metrics_request(metrics)
        assert "resourceMetrics" in request
        assert len(request["resourceMetrics"]) == 1

        rm = request["resourceMetrics"][0]
        assert "resource" in rm
        assert "scopeMetrics" in rm
        assert len(rm["scopeMetrics"]) == 1

        sm = rm["scopeMetrics"][0]
        assert "scope" in sm
        assert sm["scope"]["name"] == "mcp-audit"
        assert "metrics" in sm
        assert len(sm["metrics"]) == len(metrics)

    def test_resource_attributes(self, populated_engine):
        metrics = build_metrics(populated_engine)
        request = build_otlp_metrics_request(metrics)
        attrs = request["resourceMetrics"][0]["resource"]["attributes"]
        keys = {a["key"] for a in attrs}
        assert "service.name" in keys
        assert "service.version" in keys
        assert "telemetry.sdk.language" in keys

    def test_custom_resource_attrs(self, populated_engine):
        metrics = build_metrics(populated_engine)
        request = build_otlp_metrics_request(
            metrics,
            resource_attrs={"deployment.environment": "production"},
        )
        attrs = request["resourceMetrics"][0]["resource"]["attributes"]
        keys = {a["key"] for a in attrs}
        assert "deployment.environment" in keys

    def test_empty_metrics_list(self):
        request = build_otlp_metrics_request([])
        assert "resourceMetrics" in request
        assert request["resourceMetrics"][0]["scopeMetrics"][0]["metrics"] == []


# ── export_otlp_metrics_to_string tests ──────────────────────────────

class TestExportToString:
    def test_returns_valid_json(self, populated_engine):
        s = export_otlp_metrics_to_string(populated_engine)
        data = json.loads(s)
        assert "resourceMetrics" in data

    def test_contains_all_metrics(self, populated_engine):
        s = export_otlp_metrics_to_string(populated_engine)
        data = json.loads(s)
        metrics = data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
        names = {m["name"] for m in metrics}
        assert "tool.call.count" in names
        assert "tool.duration_ms" in names

    def test_empty_engine_to_string(self, empty_engine):
        s = export_otlp_metrics_to_string(empty_engine)
        data = json.loads(s)
        assert "resourceMetrics" in data

    def test_with_filters(self, populated_engine):
        s = export_otlp_metrics_to_string(populated_engine, tool_name="web_search")
        data = json.loads(s)
        call_count = next(
            m for m in data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            if m["name"] == "tool.call.count"
        )
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 2


# ── export_otlp_metrics_jsonl tests ──────────────────────────────────

class TestExportToJsonl:
    def test_writes_file(self, populated_engine, tmp_path):
        output_path = tmp_path / "metrics.json"
        result = export_otlp_metrics_jsonl(populated_engine, str(output_path))
        assert Path(result["path"]).exists()
        assert result["metric_count"] > 0
        assert result["size_bytes"] > 0
        assert "metric_names" in result

    def test_file_contains_valid_json(self, populated_engine, tmp_path):
        output_path = tmp_path / "metrics.json"
        export_otlp_metrics_jsonl(populated_engine, str(output_path))
        data = json.loads(output_path.read_text())
        assert "resourceMetrics" in data

    def test_empty_engine_jsonl(self, empty_engine, tmp_path):
        output_path = tmp_path / "empty_metrics.json"
        result = export_otlp_metrics_jsonl(empty_engine, str(output_path))
        assert result["metric_count"] > 0  # empty_metrics still returns metrics
        assert Path(result["path"]).exists()

    def test_creates_parent_dirs(self, populated_engine, tmp_path):
        output_path = tmp_path / "subdir" / "nested" / "metrics.json"
        result = export_otlp_metrics_jsonl(populated_engine, str(output_path))
        assert Path(result["path"]).exists()

    def test_with_filters(self, populated_engine, tmp_path):
        output_path = tmp_path / "filtered.json"
        result = export_otlp_metrics_jsonl(
            populated_engine, str(output_path), tool_name="web_search"
        )
        data = json.loads(output_path.read_text())
        call_count = next(
            m for m in data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            if m["name"] == "tool.call.count"
        )
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 2


# ── export_otlp_metrics_http tests ───────────────────────────────────

class TestExportHttp:
    def test_connection_error_returns_dict(self, populated_engine):
        """Export to a non-existent endpoint should return a structured error, not raise."""
        result = export_otlp_metrics_http(
            populated_engine,
            endpoint="http://localhost:59999/v1/metrics",  # nothing listening
            timeout=2,
        )
        assert isinstance(result, dict)
        assert result["status"] in ("connection_error", "error")
        assert result["metric_count"] > 0
        assert result["bytes_sent"] > 0

    def test_empty_engine_http(self, empty_engine):
        result = export_otlp_metrics_http(
            empty_engine,
            endpoint="http://localhost:59999/v1/metrics",
            timeout=2,
        )
        assert isinstance(result, dict)
        assert result["status"] in ("connection_error", "error")
        assert result["metric_count"] > 0

    def test_http_with_headers(self, populated_engine):
        result = export_otlp_metrics_http(
            populated_engine,
            endpoint="http://localhost:59999/v1/metrics",
            headers={"Authorization": "Bearer test-token"},
            timeout=2,
        )
        assert isinstance(result, dict)
        assert result["endpoint"] == "http://localhost:59999/v1/metrics"


# ── OTLPMetricsExporter class tests ──────────────────────────────────

class TestOTLPMetricsExporter:
    def test_default_endpoint(self):
        exporter = OTLPMetricsExporter()
        assert "4318" in exporter.endpoint
        assert "metrics" in exporter.endpoint

    def test_custom_endpoint(self):
        exporter = OTLPMetricsExporter(endpoint="http://custom:9999/metrics")
        assert exporter.endpoint == "http://custom:9999/metrics"

    def test_export_returns_dict(self, populated_engine):
        exporter = OTLPMetricsExporter(
            endpoint="http://localhost:59999/v1/metrics",
            timeout=2,
        )
        result = exporter.export(populated_engine)
        assert isinstance(result, dict)
        assert result["status"] in ("connection_error", "error")

    def test_export_to_string(self, populated_engine):
        exporter = OTLPMetricsExporter()
        s = exporter.export_to_string(populated_engine)
        data = json.loads(s)
        assert "resourceMetrics" in data

    def test_export_to_string_with_filter(self, populated_engine):
        exporter = OTLPMetricsExporter()
        s = exporter.export_to_string(populated_engine, tool_name="web_search")
        data = json.loads(s)
        call_count = next(
            m for m in data["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
            if m["name"] == "tool.call.count"
        )
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 2

    def test_resource_attrs_passthrough(self, populated_engine):
        exporter = OTLPMetricsExporter(
            resource_attrs={"deployment.environment": "staging"},
        )
        s = exporter.export_to_string(populated_engine)
        data = json.loads(s)
        attrs = data["resourceMetrics"][0]["resource"]["attributes"]
        keys = {a["key"] for a in attrs}
        assert "deployment.environment" in keys

    def test_env_var_endpoint(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://from-env:1234/metrics")
        exporter = OTLPMetricsExporter()
        assert exporter.endpoint == "http://from-env:1234/metrics"


# ── _empty_metrics tests ─────────────────────────────────────────────

class TestEmptyMetrics:
    def test_returns_valid_metrics(self):
        metrics = _empty_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_has_call_count(self):
        metrics = _empty_metrics()
        names = {m["name"] for m in metrics}
        assert "tool.call.count" in names

    def test_call_count_is_zero(self):
        metrics = _empty_metrics()
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        dp = call_count["sum"]["dataPoints"][0]
        assert int(dp["asInt"]) == 0

    def test_error_rate_is_zero(self):
        metrics = _empty_metrics()
        error_rate = next(m for m in metrics if m["name"] == "error.rate")
        dp = error_rate["gauge"]["dataPoints"][0]
        assert dp["asDouble"] == 0.0


# ── Integration: via MCP server ──────────────────────────────────────

class TestMCPServerMetricsTool:
    def test_tool_definition_exists(self):
        from mcp_audit.server import TOOL_DEFINITIONS
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert "export_otlp_metrics" in names

    def test_tool_count_increased(self):
        from mcp_audit.server import TOOL_DEFINITIONS
        # v0.5 should have 22 tools (was 21 in v0.4)
        assert len(TOOL_DEFINITIONS) == 22

    def test_tool_handler_jsonl_mode(self, populated_engine, tmp_path):
        from mcp_audit.server import MCPServer
        server = MCPServer(engine=populated_engine)
        output_path = str(tmp_path / "via_server.json")
        result = server.call_tool("export_otlp_metrics", {
            "mode": "jsonl",
            "output_path": output_path,
        })
        assert "result" in result
        assert Path(output_path).exists()

    def test_tool_handler_http_mode_error(self, populated_engine):
        from mcp_audit.server import MCPServer
        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_otlp_metrics", {
            "mode": "http",
            "endpoint": "http://localhost:59999/v1/metrics",
        })
        assert "result" in result
        assert result["result"]["status"] in ("connection_error", "error")

    def test_tool_handler_jsonl_missing_path(self, populated_engine):
        from mcp_audit.server import MCPServer
        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_otlp_metrics", {
            "mode": "jsonl",
        })
        assert "result" in result
        assert "error" in result["result"]

    def test_tool_handler_invalid_mode(self, populated_engine):
        from mcp_audit.server import MCPServer
        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_otlp_metrics", {
            "mode": "invalid",
        })
        assert "result" in result
        assert "error" in result["result"]

    def test_default_mode_is_jsonl(self):
        from mcp_audit.server import TOOL_DEFINITIONS
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "export_otlp_metrics")
        default = tool["inputSchema"]["properties"]["mode"]["default"]
        assert default == "jsonl"


# ── Histogram bucket correctness ─────────────────────────────────────

class TestHistogramBuckets:
    def test_latency_bounds_standard(self):
        """Latency bounds should include common thresholds."""
        assert 0.0 in _LATENCY_BOUNDS
        assert 100.0 in _LATENCY_BOUNDS
        assert 1000.0 in _LATENCY_BOUNDS

    def test_cost_bounds_standard(self):
        """Cost bounds should include common price points."""
        assert 0.0 in _COST_BOUNDS
        assert 0.01 in _COST_BOUNDS
        assert 1.0 in _COST_BOUNDS

    def test_histogram_sum_matches(self, populated_engine):
        metrics = build_metrics(populated_engine)
        duration_metric = next(m for m in metrics if m["name"] == "tool.duration_ms")
        # web_search: 50 + 120 = 170
        web_search_dp = None
        for dp in duration_metric["histogram"]["dataPoints"]:
            for attr in dp["attributes"]:
                if attr["key"] == "tool.name" and attr["value"]["stringValue"] == "web_search":
                    web_search_dp = dp
        assert web_search_dp is not None
        assert int(web_search_dp["count"]) == 2
        assert abs(web_search_dp["sum"] - 170.0) < 0.01

    def test_bucket_counts_sum_to_total(self, populated_engine):
        metrics = build_metrics(populated_engine)
        duration_metric = next(m for m in metrics if m["name"] == "tool.duration_ms")
        for dp in duration_metric["histogram"]["dataPoints"]:
            bucket_total = sum(int(c) for c in dp["bucketCounts"])
            assert bucket_total == int(dp["count"])


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_cost_calls_not_in_cost_histogram(self):
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        engine.record_call(
            session_id=session.id,
            tool_name="free_tool",
            agent_id="a1",
            status=CallStatus.SUCCESS,
            duration_ms=10.0,
            cost_usd=0.0,
        )
        metrics = build_metrics(engine)
        cost_metric = next(m for m in metrics if m["name"] == "tool.cost.usd")
        dps = cost_metric["histogram"]["dataPoints"]
        assert len(dps) == 0  # zero-cost calls excluded

    def test_none_duration_not_in_histogram(self):
        """Calls without explicit duration get 0.0 from the engine,
        so they appear in the histogram with a 0.0 value."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        engine.record_call(
            session_id=session.id,
            tool_name="mystery_tool",
            agent_id="a1",
            status=CallStatus.SUCCESS,
            # no duration_ms — engine sets to 0.0
            cost_usd=0.01,
        )
        metrics = build_metrics(engine)
        duration_metric = next(m for m in metrics if m["name"] == "tool.duration_ms")
        dps = duration_metric["histogram"]["dataPoints"]
        # Engine auto-sets duration to 0.0, so it IS included
        assert len(dps) == 1
        assert int(dps[0]["count"]) == 1

    def test_zero_tokens_not_in_token_counters(self):
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        engine.record_call(
            session_id=session.id,
            tool_name="no_token_tool",
            agent_id="a1",
            status=CallStatus.SUCCESS,
            duration_ms=10.0,
            cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
        )
        metrics = build_metrics(engine)
        input_metric = next(m for m in metrics if m["name"] == "tool.tokens.input")
        output_metric = next(m for m in metrics if m["name"] == "tool.tokens.output")
        assert len(input_metric["sum"]["dataPoints"]) == 0
        assert len(output_metric["sum"]["dataPoints"]) == 0

    def test_single_call(self):
        engine = AuditEngine()
        session = engine.start_session(agent_id="solo")
        engine.record_call(
            session_id=session.id,
            tool_name="only_tool",
            agent_id="solo",
            status=CallStatus.SUCCESS,
            duration_ms=42.0,
            cost_usd=0.005,
            input_tokens=10,
            output_tokens=20,
        )
        metrics = build_metrics(engine)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 1

    def test_many_tools(self):
        """Test with many different tools to verify grouping."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="multi")
        for i in range(20):
            engine.record_call(
                session_id=session.id,
                tool_name=f"tool_{i}",
                agent_id="multi",
                status=CallStatus.SUCCESS,
                duration_ms=float(i * 10),
                cost_usd=0.001 * i,
            )
        metrics = build_metrics(engine)
        call_count = next(m for m in metrics if m["name"] == "tool.call.count")
        assert len(call_count["sum"]["dataPoints"]) == 20
        total = sum(int(dp["asInt"]) for dp in call_count["sum"]["dataPoints"])
        assert total == 20
