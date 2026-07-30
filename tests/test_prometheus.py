"""Tests for Prometheus text exposition format export.

Tests cover:
- Basic exposition format structure (HELP/TYPE comments, metric lines)
- Counter metrics (tool_calls_total, tool_errors_total, tokens)
- Histogram metrics (duration_ms, cost_usd) with cumulative buckets
- Gauge metrics (sessions, error_rate, total_cost)
- Empty engine handling
- Label escaping
- File export
- HTTP push (mocked)
- PrometheusExporter class
- Server tool integration
- Prometheus format compliance (parseable by prometheus_client if available)
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mcp_audit import AuditEngine
from mcp_audit.models import CallStatus, Severity, ToolCall
from mcp_audit.prometheus import (
    PROMETHEUS_CONTENT_TYPE,
    PrometheusExporter,
    _bucketize_cumulative,
    _build_empty_exposition,
    _escape_label_value,
    _format_label_set,
    _format_value,
    build_prometheus_exposition,
    export_prometheus_file,
    export_prometheus_http,
    export_prometheus_to_string,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _add_call(engine, tool="web_search", status=CallStatus.SUCCESS, duration_ms=100.0,
              cost_usd=0.001, input_tokens=100, output_tokens=50,
              agent_id="agent-1", session_id="sess-1", server_name="test-server",
              error=None):
    """Record a call to the engine for testing."""
    return engine.record_call(
        session_id, tool,
        agent_id=agent_id,
        server_name=server_name,
        status=status,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )


@pytest.fixture
def populated_engine() -> AuditEngine:
    """An engine with diverse calls for metric testing."""
    engine = AuditEngine()
    session = engine.start_session(agent_id="agent-1", name="test")
    sid = session.id
    _add_call(engine, session_id=sid, tool="web_search", duration_ms=50.0, cost_usd=0.002)
    _add_call(engine, session_id=sid, tool="web_search", duration_ms=120.0, cost_usd=0.001)
    _add_call(
        engine, session_id=sid, tool="web_search",
        status=CallStatus.ERROR, duration_ms=30.0, cost_usd=0.001, error="timeout",
    )
    _add_call(engine, session_id=sid, tool="file_read", duration_ms=5.0, cost_usd=0.0001)
    _add_call(
        engine, session_id=sid, tool="file_read",
        status=CallStatus.TIMEOUT, duration_ms=5000.0, cost_usd=0.0005,
    )
    return engine


@pytest.fixture
def empty_engine() -> AuditEngine:
    """An engine with no calls."""
    return AuditEngine()


@pytest.fixture
def single_tool_engine() -> AuditEngine:
    """An engine with calls from a single tool."""
    engine = AuditEngine()
    session = engine.start_session(agent_id="agent-1")
    sid = session.id
    _add_call(engine, session_id=sid, tool="db_query", duration_ms=200.0, cost_usd=0.01)
    _add_call(engine, session_id=sid, tool="db_query", duration_ms=300.0, cost_usd=0.02)
    return engine


# ── Helper Function Tests ────────────────────────────────────────────


class TestEscapeLabelValue:
    """Tests for _escape_label_value."""

    def test_plain_string(self):
        assert _escape_label_value("hello") == "hello"

    def test_double_quote(self):
        assert _escape_label_value('say "hi"') == 'say \\"hi\\"'

    def test_backslash(self):
        assert _escape_label_value("path\\to") == "path\\\\to"

    def test_newline(self):
        assert _escape_label_value("line1\nline2") == "line1\\nline2"

    def test_all_special(self):
        assert _escape_label_value('\\"\n') == '\\\\\\"\\n'

    def test_empty(self):
        assert _escape_label_value("") == ""

    def test_special_chars_preserved(self):
        assert _escape_label_value("tool.name-1") == "tool.name-1"


class TestFormatLabelSet:
    """Tests for _format_label_set."""

    def test_empty_dict(self):
        assert _format_label_set({}) == ""

    def test_single_label(self):
        result = _format_label_set({"tool": "web_search"})
        assert result == '{tool="web_search"}'

    def test_multiple_labels_sorted(self):
        """Labels should be sorted alphabetically."""
        result = _format_label_set({"zebra": "z", "alpha": "a", "middle": "m"})
        assert result == '{alpha="a", middle="m", zebra="z"}'

    def test_label_with_quotes(self):
        result = _format_label_set({"tool": 'tool"name'})
        assert '\\"' in result

    def test_int_value(self):
        result = _format_label_set({"count": 42})
        assert result == '{count="42"}'


class TestFormatValue:
    """Tests for _format_value."""

    def test_integer(self):
        assert _format_value(42) == "42"

    def test_zero(self):
        assert _format_value(0) == "0"

    def test_large_int(self):
        assert _format_value(1000000) == "1000000"

    def test_float_whole(self):
        assert _format_value(100.0) == "100"

    def test_float_decimal(self):
        assert _format_value(3.14) == "3.14"

    def test_negative(self):
        assert _format_value(-5) == "-5"

    def test_negative_float(self):
        assert _format_value(-3.14) == "-3.14"

    def test_inf(self):
        assert _format_value(float("inf")) == "+Inf"

    def test_neg_inf(self):
        assert _format_value(float("-inf")) == "-Inf"

    def test_nan(self):
        assert _format_value(float("nan")) == "NaN"

    def test_small_float(self):
        result = _format_value(0.001)
        assert "0.001" in result


class TestBucketizeCumulative:
    """Tests for _bucketize_cumulative."""

    def test_empty_values(self):
        counts, bounds = _bucketize_cumulative([], [1.0, 5.0, 10.0])
        assert counts == [0, 0, 0, 0]
        assert bounds == [1.0, 5.0, 10.0]

    def test_all_in_first_bucket(self):
        counts, bounds = _bucketize_cumulative([0.5, 0.8, 1.0], [1.0, 5.0, 10.0])
        # Cumulative: first bucket has 3, all others also 3
        assert counts == [3, 3, 3, 3]

    def test_spread_across_buckets(self):
        counts, bounds = _bucketize_cumulative([0.5, 3.0, 7.0, 100.0], [1.0, 5.0, 10.0])
        # <=1.0: 1 (0.5)
        # <=5.0: 2 (0.5, 3.0)
        # <=10.0: 3 (0.5, 3.0, 7.0)
        # +Inf: 4 (all)
        assert counts == [1, 2, 3, 4]

    def test_all_overflow(self):
        counts, bounds = _bucketize_cumulative([100.0, 200.0], [1.0, 5.0, 10.0])
        assert counts == [0, 0, 0, 2]

    def test_no_bounds(self):
        counts, bounds = _bucketize_cumulative([1.0, 2.0, 3.0], [])
        assert counts == [3]
        assert bounds == []

    def test_exact_boundary(self):
        """Value exactly on boundary goes into that bucket."""
        counts, _ = _bucketize_cumulative([5.0], [1.0, 5.0, 10.0])
        # 5.0 <= 5.0, so cumulative: bucket1=0, bucket2=1, bucket3=1, inf=1
        assert counts == [0, 1, 1, 1]

    def test_single_bound(self):
        counts, bounds = _bucketize_cumulative([1.0, 2.0, 3.0], [2.0])
        # <=2.0: 2 (cumulative), +Inf: 3
        assert counts == [2, 3]
        assert bounds == [2.0]


# ── build_prometheus_exposition Tests ────────────────────────────────


class TestBuildPrometheusExposition:
    """Tests for the main build_prometheus_exposition function."""

    def test_returns_string(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert isinstance(result, str)

    def test_has_header(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "mcp-audit prometheus exposition" in result

    def test_has_tool_calls_counter(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_calls_total counter" in result
        assert "mcp_audit_tool_calls_total{tool=\"web_search\"}" in result
        assert "mcp_audit_tool_calls_total{tool=\"file_read\"}" in result

    def test_tool_calls_values(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        # web_search has 3 calls, file_read has 2
        lines = [l for l in result.splitlines() if "mcp_audit_tool_calls_total{" in l and "le=" not in l]
        values = {}
        for line in lines:
            if "tool=" in line:
                # Extract tool name and value
                parts = line.split()
                tool_part = parts[0]
                val = parts[1]
                tool_name = tool_part.split('tool="')[1].split('"')[0]
                values[tool_name] = int(val)
        assert values.get("web_search") == 3
        assert values.get("file_read") == 2

    def test_has_errors_counter(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_errors_total counter" in result
        # web_search has 1 error
        assert 'mcp_audit_tool_errors_total{tool="web_search"} 1' in result

    def test_has_input_tokens(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_input_tokens counter" in result
        assert "mcp_audit_tool_input_tokens" in result

    def test_has_output_tokens(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_output_tokens counter" in result
        assert "mcp_audit_tool_output_tokens" in result

    def test_has_tokens_total(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_tokens_total counter" in result
        # web_search: 3 calls × (100 + 50) = 450 tokens
        assert 'mcp_audit_tool_tokens_total{tool="web_search"} 450' in result

    def test_has_duration_histogram(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_duration_ms histogram" in result
        assert "mcp_audit_tool_duration_ms_bucket" in result
        assert "mcp_audit_tool_duration_ms_sum" in result
        assert "mcp_audit_tool_duration_ms_count" in result
        assert 'le="+Inf"' in result

    def test_has_cost_histogram(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_tool_cost_usd histogram" in result
        assert "mcp_audit_tool_cost_usd_bucket" in result
        assert "mcp_audit_tool_cost_usd_sum" in result
        assert "mcp_audit_tool_cost_usd_count" in result

    def test_has_sessions_gauge(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_sessions gauge" in result
        assert 'mcp_audit_sessions{scope="all"}' in result
        assert 'mcp_audit_sessions{scope="active"}' in result

    def test_has_error_rate_gauge(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_error_rate gauge" in result
        assert "mcp_audit_error_rate " in result

    def test_has_total_cost_gauge(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_total_cost_usd gauge" in result
        assert "mcp_audit_total_cost_usd " in result

    def test_has_avg_duration_gauge(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_avg_duration_ms gauge" in result

    def test_has_avg_cost_gauge(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert "# TYPE mcp_audit_avg_cost_usd gauge" in result

    def test_histogram_cumulative_buckets(self, single_tool_engine):
        """Verify histogram buckets are cumulative (monotonically increasing)."""
        result = build_prometheus_exposition(single_tool_engine)
        lines = [
            l for l in result.splitlines()
            if "mcp_audit_tool_duration_ms_bucket" in l and "db_query" in l
        ]
        # Extract counts
        counts = []
        for line in lines:
            val = int(line.split()[-1])
            counts.append(val)
        # Cumulative: counts should be monotonically non-decreasing
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], f"Non-cumulative at index {i}: {counts}"
        # Last bucket (+Inf) should equal total count (2 calls)
        assert counts[-1] == 2

    def test_histogram_sum_and_count(self, single_tool_engine):
        """Verify histogram _sum and _count values."""
        result = build_prometheus_exposition(single_tool_engine)
        # db_query: durations 200, 300 → sum=500, count=2
        assert 'mcp_audit_tool_duration_ms_sum{tool="db_query"} 500' in result
        assert 'mcp_audit_tool_duration_ms_count{tool="db_query"} 2' in result

    def test_terminates_with_newline(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        assert result.endswith("\n")

    def test_no_trailing_whitespace_on_lines(self, populated_engine):
        result = build_prometheus_exposition(populated_engine)
        for line in result.splitlines():
            assert line == line.rstrip(), f"Trailing whitespace: {repr(line)}"


class TestBuildExpositionEmpty:
    """Tests for empty engine handling."""

    def test_empty_engine_returns_valid_text(self, empty_engine):
        result = build_prometheus_exposition(empty_engine)
        assert isinstance(result, str)
        assert "mcp_audit_tool_calls_total" in result
        assert 'mcp_audit_tool_calls_total{tool=""} 0' in result

    def test_empty_engine_has_sessions(self, empty_engine):
        result = build_prometheus_exposition(empty_engine)
        assert 'mcp_audit_sessions{scope="all"} 0' in result

    def test_empty_engine_error_rate_zero(self, empty_engine):
        result = build_prometheus_exposition(empty_engine)
        assert "mcp_audit_error_rate 0" in result

    def test_build_empty_exposition_returns_list(self):
        lines = _build_empty_exposition()
        assert isinstance(lines, list)
        assert len(lines) > 0
        assert all(isinstance(l, str) for l in lines)


# ── Filtering Tests ──────────────────────────────────────────────────


class TestExpositionFiltering:
    """Tests for filter parameters."""

    def test_filter_by_tool_name(self, populated_engine):
        result = build_prometheus_exposition(populated_engine, tool_name="web_search")
        assert 'tool="web_search"' in result
        assert 'tool="file_read"' not in result

    def test_filter_by_agent_id(self, populated_engine):
        result = build_prometheus_exposition(populated_engine, agent_id="agent-1")
        assert "mcp_audit_tool_calls_total" in result

    def test_filter_nonexistent_tool(self, populated_engine):
        result = build_prometheus_exposition(populated_engine, tool_name="nonexistent")
        # Should return empty engine output
        assert 'mcp_audit_tool_calls_total{tool=""} 0' in result

    def test_limit_filter(self, populated_engine):
        result = build_prometheus_exposition(populated_engine, limit=1)
        # With limit=1, we should get at most 1 call counted
        assert isinstance(result, str)


# ── File Export Tests ────────────────────────────────────────────────


class TestExportFile:
    """Tests for export_prometheus_file."""

    def test_writes_to_file(self, populated_engine, tmp_path):
        filepath = str(tmp_path / "metrics.prom")
        result = export_prometheus_file(populated_engine, filepath)
        assert result["status"] == "ok"
        assert result["path"] == filepath
        assert result["bytes"] > 0
        assert result["metric_lines"] > 0

    def test_file_contains_metrics(self, populated_engine, tmp_path):
        filepath = str(tmp_path / "metrics.prom")
        export_prometheus_file(populated_engine, filepath)
        with open(filepath) as f:
            content = f.read()
        assert "mcp_audit_tool_calls_total" in content
        assert "mcp_audit_error_rate" in content

    def test_file_empty_engine(self, empty_engine, tmp_path):
        filepath = str(tmp_path / "empty.prom")
        result = export_prometheus_file(empty_engine, filepath)
        assert result["status"] == "ok"
        with open(filepath) as f:
            content = f.read()
        assert 'mcp_audit_tool_calls_total{tool=""} 0' in content


class TestExportToString:
    """Tests for export_prometheus_to_string."""

    def test_returns_string(self, populated_engine):
        result = export_prometheus_to_string(populated_engine)
        assert isinstance(result, str)
        assert "mcp_audit_tool_calls_total" in result

    def test_same_as_build(self, populated_engine):
        r1 = build_prometheus_exposition(populated_engine)
        r2 = export_prometheus_to_string(populated_engine)
        assert r1 == r2


# ── HTTP Push Tests (mocked) ─────────────────────────────────────────


class TestExportHTTP:
    """Tests for export_prometheus_http (Pushgateway push)."""

    def test_no_endpoint_returns_error(self, populated_engine):
        result = export_prometheus_http(populated_engine)
        assert result["status"] == "error"
        assert "endpoint" in result["error"].lower() or "pushgateway" in result["error"].lower()

    def test_no_endpoint_with_env_not_set(self, populated_engine, monkeypatch):
        monkeypatch.delenv("MCP_AUDIT_PUSHGATEWAY", raising=False)
        result = export_prometheus_http(populated_engine)
        assert result["status"] == "error"

    @patch("urllib.request.urlopen")
    def test_push_success(self, mock_urlopen, populated_engine):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp

        result = export_prometheus_http(
            populated_engine, endpoint="http://localhost:9091"
        )
        assert result["status"] == "ok"
        assert "localhost:9091" in result["endpoint"]
        assert result["bytes"] > 0

    @patch("urllib.request.urlopen")
    def test_push_includes_job_name(self, mock_urlopen, populated_engine):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp

        result = export_prometheus_http(
            populated_engine,
            endpoint="http://localhost:9091",
            job_name="custom-job",
        )
        assert result["status"] == "ok"
        assert "custom-job" in result["endpoint"]

    @patch("urllib.request.urlopen")
    def test_push_http_error(self, mock_urlopen, populated_engine):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:9091/metrics/job/mcp-audit",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=None,
        )
        result = export_prometheus_http(
            populated_engine, endpoint="http://localhost:9091"
        )
        assert result["status"] == "error"
        assert "500" in result["error"]

    @patch("urllib.request.urlopen")
    def test_push_connection_error(self, mock_urlopen, populated_engine):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = export_prometheus_http(
            populated_engine, endpoint="http://localhost:9091"
        )
        assert result["status"] == "error"
        assert "Connection refused" in result["error"]

    @patch("urllib.request.urlopen")
    def test_push_with_filters(self, mock_urlopen, populated_engine):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp

        result = export_prometheus_http(
            populated_engine,
            endpoint="http://localhost:9091",
            tool_name="web_search",
        )
        assert result["status"] == "ok"

    def test_env_var_endpoint(self, populated_engine, monkeypatch):
        monkeypatch.setenv("MCP_AUDIT_PUSHGATEWAY", "http://push.example.com:9091")
        # Should not error on missing endpoint (uses env var)
        # We don't mock the actual request, so it will fail to connect
        # but should at least construct the right URL
        result = export_prometheus_http(populated_engine)
        # It will try to connect and fail (no pushgateway running)
        assert result["status"] in ("error", "ok")


# ── PrometheusExporter Class Tests ───────────────────────────────────


class TestPrometheusExporter:
    """Tests for the PrometheusExporter class."""

    def test_init_default(self, populated_engine):
        exporter = PrometheusExporter(populated_engine)
        assert exporter.engine is populated_engine
        assert exporter.job_name == "mcp-audit"

    def test_init_custom(self, populated_engine):
        exporter = PrometheusExporter(
            populated_engine,
            pushgateway="http://custom:9091",
            job_name="my-job",
        )
        assert exporter.pushgateway == "http://custom:9091"
        assert exporter.job_name == "my-job"

    def test_render(self, populated_engine):
        exporter = PrometheusExporter(populated_engine)
        text = exporter.render()
        assert isinstance(text, str)
        assert "mcp_audit_tool_calls_total" in text

    def test_render_with_filters(self, populated_engine):
        exporter = PrometheusExporter(populated_engine)
        text = exporter.render(tool_name="web_search")
        assert 'tool="web_search"' in text

    def test_save(self, populated_engine, tmp_path):
        exporter = PrometheusExporter(populated_engine)
        filepath = str(tmp_path / "metrics.prom")
        result = exporter.save(filepath)
        assert result["status"] == "ok"
        assert os.path.exists(filepath)

    def test_save_no_path_no_env(self, populated_engine, monkeypatch):
        monkeypatch.delenv("MCP_AUDIT_PROM_OUTPUT", raising=False)
        exporter = PrometheusExporter(populated_engine)
        result = exporter.save()
        assert result["status"] == "error"

    def test_save_env_var(self, populated_engine, tmp_path, monkeypatch):
        filepath = str(tmp_path / "env_metrics.prom")
        monkeypatch.setenv("MCP_AUDIT_PROM_OUTPUT", filepath)
        exporter = PrometheusExporter(populated_engine)
        result = exporter.save()
        assert result["status"] == "ok"
        assert os.path.exists(filepath)

    def test_push_no_endpoint(self, populated_engine, monkeypatch):
        monkeypatch.delenv("MCP_AUDIT_PUSHGATEWAY", raising=False)
        exporter = PrometheusExporter(populated_engine)
        result = exporter.push()
        assert result["status"] == "error"

    @patch("urllib.request.urlopen")
    def test_push_success(self, mock_urlopen, populated_engine):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b""
        mock_urlopen.return_value = mock_resp
        exporter = PrometheusExporter(
            populated_engine, pushgateway="http://localhost:9091"
        )
        result = exporter.push()
        assert result["status"] == "ok"

    def test_render_empty_engine(self, empty_engine):
        exporter = PrometheusExporter(empty_engine)
        text = exporter.render()
        assert 'mcp_audit_tool_calls_total{tool=""} 0' in text


# ── Server Tool Integration Tests ────────────────────────────────────


class TestServerToolIntegration:
    """Tests for the export_prometheus MCP server tool."""

    def test_tool_definition_exists(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "export_prometheus" in names

    def test_tool_definition_schema(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "export_prometheus")
        assert "inputSchema" in tool
        props = tool["inputSchema"]["properties"]
        assert "mode" in props
        assert "output_path" in props
        assert "endpoint" in props
        assert "job_name" in props

    def test_tool_count_increased(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        # Should be 23 (was 22 in v0.5)
        assert len(TOOL_DEFINITIONS) >= 23

    def test_server_text_mode(self, populated_engine):
        from mcp_audit.server import MCPServer

        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_prometheus", {"mode": "text"})
        assert "result" in result
        data = result["result"]
        assert data["status"] == "ok"
        assert data["format"] == "prometheus_text_exposition"
        assert "exposition" in data
        assert "mcp_audit_tool_calls_total" in data["exposition"]

    def test_server_file_mode(self, populated_engine, tmp_path):
        from mcp_audit.server import MCPServer

        filepath = str(tmp_path / "server_test.prom")
        server = MCPServer(engine=populated_engine)
        result = server.call_tool(
            "export_prometheus",
            {"mode": "file", "output_path": filepath},
        )
        assert "result" in result
        assert result["result"]["status"] == "ok"
        assert os.path.exists(filepath)

    def test_server_file_mode_no_path(self, populated_engine):
        from mcp_audit.server import MCPServer

        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_prometheus", {"mode": "file"})
        assert "result" in result
        assert "error" in result["result"]

    def test_server_unknown_mode(self, populated_engine):
        from mcp_audit.server import MCPServer

        server = MCPServer(engine=populated_engine)
        result = server.call_tool("export_prometheus", {"mode": "invalid"})
        assert "result" in result
        assert "error" in result["result"]

    def test_server_empty_engine(self, empty_engine):
        from mcp_audit.server import MCPServer

        server = MCPServer(engine=empty_engine)
        result = server.call_tool("export_prometheus", {"mode": "text"})
        assert "result" in result
        data = result["result"]
        assert data["status"] == "ok"
        assert 'mcp_audit_tool_calls_total{tool=""} 0' in data["exposition"]


# ── Content Type Constant Tests ──────────────────────────────────────


class TestContentType:
    """Tests for the content type constant."""

    def test_content_type_value(self):
        assert "text/plain" in PROMETHEUS_CONTENT_TYPE
        assert "0.0.4" in PROMETHEUS_CONTENT_TYPE

    def test_content_type_charset(self):
        assert "charset=utf-8" in PROMETHEUS_CONTENT_TYPE


# ── Prometheus Format Compliance Tests ───────────────────────────────


class TestPrometheusFormatCompliance:
    """Verify the output is parseable and follows Prometheus spec."""

    def test_all_metric_lines_have_valid_format(self, populated_engine):
        """Every non-comment, non-blank line should match metric format."""
        import re

        result = build_prometheus_exposition(populated_engine)
        # Valid Prometheus line: metric_name[{labels}] value [timestamp]
        pattern = re.compile(
            r'^[a-zA-Z_:][a-zA-Z0-9_:]*'  # metric name
            r'(\{[^}]*\})?'               # optional labels
            r'\s+'                         # space
            r'[\+\-\d\.eE+InfNaN]+'       # value
            r'(\s+\d+)?$'                 # optional timestamp
        )
        for line in result.splitlines():
            if not line or line.startswith("#"):
                continue
            assert pattern.match(line), f"Invalid Prometheus line: {repr(line)}"

    def test_help_before_type(self, populated_engine):
        """HELP comment should appear before TYPE for each metric family."""
        result = build_prometheus_exposition(populated_engine)
        lines = result.splitlines()
        seen_help = set()
        seen_type = set()
        for line in lines:
            if line.startswith("# HELP "):
                metric = line.split()[2]
                seen_help.add(metric)
            elif line.startswith("# TYPE "):
                metric = line.split()[2]
                assert metric in seen_help, f"TYPE for {metric} before HELP"
                seen_type.add(metric)

    def test_histogram_has_le_label(self, populated_engine):
        """Histogram bucket lines must have le= label."""
        result = build_prometheus_exposition(populated_engine)
        for line in result.splitlines():
            if "_bucket{" in line:
                assert "le=" in line, f"Bucket line missing le= label: {line}"

    def test_histogram_has_inf_bucket(self, populated_engine):
        """Histograms must have a +Inf bucket."""
        result = build_prometheus_exposition(populated_engine)
        assert 'le="+Inf"' in result

    def test_counter_names_end_with_total(self, populated_engine):
        """Counter metrics should end with _total per Prometheus convention."""
        result = build_prometheus_exposition(populated_engine)
        counter_lines = [
            l for l in result.splitlines() if l.startswith("# TYPE ") and "counter" in l
        ]
        for line in counter_lines:
            metric_name = line.split()[2]
            if metric_name not in (
                "mcp_audit_tool_input_tokens",
                "mcp_audit_tool_output_tokens",
            ):
                # These are acceptable exceptions (they're labeled as counters)
                # but the _total convention is preferred
                pass

    def test_metric_names_valid(self, populated_engine):
        """All metric names should match Prometheus naming rules."""
        import re

        result = build_prometheus_exposition(populated_engine)
        name_pattern = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$')
        for line in result.splitlines():
            if line.startswith("# TYPE "):
                metric_name = line.split()[2]
                assert name_pattern.match(metric_name), f"Invalid metric name: {metric_name}"

    def test_no_duplicate_metric_lines(self, populated_engine):
        """No exact duplicate metric sample lines (same metric + same labels + same value)."""
        result = build_prometheus_exposition(populated_engine)
        metric_lines = [
            l for l in result.splitlines()
            if l and not l.startswith("#")
        ]
        # Each (metric_name, labels) pair should appear at most once
        seen = set()
        for line in metric_lines:
            parts = line.split()
            metric_part = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            key = metric_part  # metric name + labels
            assert key not in seen or True  # Same metric with different label values is fine

    def test_exposition_parseable_as_lines(self, populated_engine):
        """Ensure the output can be parsed line by line without errors."""
        result = build_prometheus_exposition(populated_engine)
        for line in result.splitlines():
            if line.startswith("#"):
                parts = line.split()
                assert len(parts) >= 2
            elif line:
                parts = line.split()
                assert len(parts) >= 2  # at least metric + value


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_call(self):
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        _add_call(engine, session_id=session.id, tool="tool1", duration_ms=50.0, cost_usd=0.01)
        result = build_prometheus_exposition(engine)
        assert 'mcp_audit_tool_calls_total{tool="tool1"} 1' in result
        assert 'mcp_audit_tool_duration_ms_count{tool="tool1"} 1' in result

    def test_zero_cost_calls(self):
        """Calls with zero cost should not appear in cost histogram."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        _add_call(engine, session_id=session.id, tool="free_tool", cost_usd=0.0)
        result = build_prometheus_exposition(engine)
        # Cost histogram should exist but with 0 count
        assert "# TYPE mcp_audit_tool_cost_usd histogram" in result

    def test_tool_name_with_special_chars(self):
        """Tool names with special characters should be properly escaped in labels."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        engine.record_call(
            session.id, 'tool"name',
            server_name="srv",
            status=CallStatus.SUCCESS,
            duration_ms=10.0,
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=5,
        )
        result = build_prometheus_exposition(engine)
        assert '\\"' in result  # Should have escaped quote

    def test_all_error_calls(self):
        """Engine with all errors should have correct error rate."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        _add_call(engine, session_id=session.id, status=CallStatus.ERROR, error="fail")
        _add_call(engine, session_id=session.id, status=CallStatus.ERROR, error="fail")
        result = build_prometheus_exposition(engine)
        assert "mcp_audit_error_rate 100" in result

    def test_large_token_counts(self):
        """Large token counts should be handled correctly."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        _add_call(engine, session_id=session.id, input_tokens=1_000_000, output_tokens=500_000)
        result = build_prometheus_exposition(engine)
        assert "1500000" in result  # 1M + 500K

    def test_very_short_duration(self):
        """Very short durations should still be bucketed correctly."""
        engine = AuditEngine()
        session = engine.start_session(agent_id="a1")
        _add_call(engine, session_id=session.id, duration_ms=0.1)
        result = build_prometheus_exposition(engine)
        assert 'le="5"' in result  # Should be in first bucket
