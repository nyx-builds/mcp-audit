"""Tests for the OpenTelemetry Protocol (OTLP) export module."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from mcp_audit import AuditEngine, CallStatus, Severity
from mcp_audit.otlp import (
    OTLPExporter,
    _attribute_value,
    _call_to_span,
    _to_nanoseconds,
    _to_otel_span_id,
    _to_otel_trace_id,
    build_otlp_request,
    calls_to_otel_spans,
    export_otlp_http,
    export_otlp_jsonl,
    export_otlp_to_string,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _populate_engine(n: int = 5, *, session_id: str | None = None) -> AuditEngine:
    """Create an engine with n sample calls."""
    engine = AuditEngine()
    if session_id is None:
        session = engine.start_session(agent_id="test-agent")
        session_id = session.id

    for i in range(n):
        status = CallStatus.ERROR if i % 4 == 0 else CallStatus.SUCCESS
        error = f"simulated error {i}" if status == CallStatus.ERROR else None
        engine.record_call(
            session_id=session_id,
            tool_name=f"tool_{i % 3}",
            agent_id="test-agent",
            server_name=f"mcp-server-{i % 2}",
            status=status,
            error=error,
            duration_ms=10.5 + i * 5,
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
            cost_usd=0.001 * (i + 1),
            tags=[f"tag_{i}"] if i % 2 == 0 else [],
        )
    return engine


def _populate_engine_multi_session() -> AuditEngine:
    """Create an engine with calls across multiple sessions."""
    engine = AuditEngine()
    s1 = engine.start_session(agent_id="agent-a")
    s2 = engine.start_session(agent_id="agent-b")

    engine.record_call(session_id=s1.id, tool_name="search", agent_id="agent-a", duration_ms=50, cost_usd=0.01)
    engine.record_call(session_id=s1.id, tool_name="fetch", agent_id="agent-a", duration_ms=100, cost_usd=0.02)
    engine.record_call(session_id=s2.id, tool_name="search", agent_id="agent-b", duration_ms=75, cost_usd=0.015)
    return engine


# ── ID Conversion Tests ──────────────────────────────────────────────

class TestIDConversion:
    def test_trace_id_is_32_hex(self):
        tid = _to_otel_trace_id("session-123")
        assert len(tid) == 32
        int(tid, 16)  # must be valid hex

    def test_trace_id_deterministic(self):
        assert _to_otel_trace_id("abc") == _to_otel_trace_id("abc")

    def test_trace_id_different_inputs_differ(self):
        assert _to_otel_trace_id("a") != _to_otel_trace_id("b")

    def test_span_id_is_16_hex(self):
        sid = _to_otel_span_id("call-456")
        assert len(sid) == 16
        int(sid, 16)

    def test_span_id_deterministic(self):
        assert _to_otel_span_id("x") == _to_otel_span_id("x")

    def test_span_id_different_from_trace_id_format(self):
        tid = _to_otel_trace_id("same-input")
        sid = _to_otel_span_id("same-input")
        assert len(tid) == 32
        assert len(sid) == 16

    def test_to_nanoseconds_utc(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ns = _to_nanoseconds(dt)
        assert ns == 1704067200000000000

    def test_to_nanoseconds_naive_datetime_assumes_utc(self):
        dt = datetime(2024, 1, 1, 0, 0, 0)  # no tzinfo
        ns = _to_nanoseconds(dt)
        assert ns == 1704067200000000000


# ── Attribute Value Tests ────────────────────────────────────────────

class TestAttributeValue:
    def test_bool(self):
        result = _attribute_value(True)
        assert result == {"boolValue": True}

    def test_int(self):
        result = _attribute_value(42)
        assert result == {"intValue": "42"}

    def test_float(self):
        result = _attribute_value(3.14)
        assert result == {"doubleValue": 3.14}

    def test_string(self):
        result = _attribute_value("hello")
        assert result == {"stringValue": "hello"}

    def test_list(self):
        result = _attribute_value([1, 2, 3])
        assert "arrayValue" in result
        assert len(result["arrayValue"]["values"]) == 3

    def test_none_becomes_string(self):
        result = _attribute_value(None)
        assert result == {"stringValue": "None"}


# ── Span Conversion Tests ────────────────────────────────────────────

class TestCallToSpan:
    def test_basic_span_structure(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])

        assert "traceId" in span
        assert "spanId" in span
        assert "name" in span
        assert "kind" in span
        assert "startTimeUnixNano" in span
        assert "endTimeUnixNano" in span
        assert "status" in span
        assert "attributes" in span

    def test_span_name_matches_tool_name(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        assert span["name"] == calls[0].tool_name

    def test_span_kind_is_internal(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        assert span["kind"] == "SPAN_KIND_INTERNAL"

    def test_success_status_ok(self):
        engine = _populate_engine(2)
        # Call index 1 (i=1) is success (only i % 4 == 0 is error)
        calls = engine.query_calls()
        success_calls = [c for c in calls if c.status == CallStatus.SUCCESS]
        assert len(success_calls) > 0
        span = _call_to_span(success_calls[0])
        assert span["status"]["code"] == "STATUS_CODE_OK"

    def test_error_status_error(self):
        engine = _populate_engine(4)  # call index 0 is error
        calls = engine.query_calls()
        # Find the error call
        error_call = [c for c in calls if c.status == CallStatus.ERROR][0]
        span = _call_to_span(error_call)
        assert span["status"]["code"] == "STATUS_CODE_ERROR"
        assert span["status"]["message"] == error_call.error

    def test_attributes_contain_required_keys(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        attr_keys = {a["key"] for a in span["attributes"]}
        assert "tool.name" in attr_keys
        assert "call.id" in attr_keys
        assert "session.id" in attr_keys
        assert "call.status" in attr_keys
        assert "cost.usd" in attr_keys
        assert "tokens.input" in attr_keys
        assert "tokens.output" in attr_keys
        assert "tokens.total" in attr_keys
        assert "call.duration_ms" in attr_keys

    def test_attributes_contain_optional_keys(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        attr_keys = {a["key"] for a in span["attributes"]}
        assert "agent.id" in attr_keys
        assert "mcp.server" in attr_keys

    def test_tags_attribute(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        # First call (i=0) has tags=["tag_0"] since 0 % 2 == 0
        span = _call_to_span(calls[0])
        attr_keys = {a["key"] for a in span["attributes"]}
        assert "tags" in attr_keys

    def test_trace_and_span_id_lengths(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        assert len(span["traceId"]) == 32
        assert len(span["spanId"]) == 16

    def test_start_before_end(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        assert int(span["startTimeUnixNano"]) <= int(span["endTimeUnixNano"])

    def test_no_events_when_none_provided(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        span = _call_to_span(calls[0])
        assert "events" not in span or len(span.get("events", [])) == 0

    def test_with_events(self):
        engine = _populate_engine(1)
        calls = engine.query_calls()
        call = calls[0]
        engine.log_event(
            trace_id=call.session_id,
            call_id=call.id,
            event_type="db_query",
            message="executing SQL",
            severity=Severity.INFO,
        )
        events = engine.query_events(call_id=call.id)
        span = _call_to_span(call, events)
        assert "events" in span
        assert len(span["events"]) == 1
        assert span["events"][0]["name"] == "db_query"

    def test_events_filtered_by_call_id(self):
        engine = _populate_engine(2)
        calls = engine.query_calls()
        call1 = calls[0]
        call2 = calls[1]

        engine.log_event(trace_id=call1.session_id, call_id=call1.id, event_type="ev1", message="m1")
        engine.log_event(trace_id=call2.session_id, call_id=call2.id, event_type="ev2", message="m2")

        all_events = engine.query_events()
        span = _call_to_span(call1, all_events)
        assert len(span["events"]) == 1
        assert span["events"][0]["name"] == "ev1"


# ── calls_to_otel_spans Tests ────────────────────────────────────────

class TestCallsToOtelSpans:
    def test_returns_list_of_spans(self):
        engine = _populate_engine(3)
        spans = calls_to_otel_spans(engine)
        assert isinstance(spans, list)
        assert len(spans) == 3

    def test_empty_engine_returns_empty_list(self):
        engine = AuditEngine()
        spans = calls_to_otel_spans(engine)
        assert spans == []

    def test_filter_by_session(self):
        engine = _populate_engine(3, session_id="test-sess")
        spans = calls_to_otel_spans(engine, session_id="test-sess")
        assert len(spans) == 3

    def test_filter_nonexistent_session(self):
        engine = _populate_engine(3)
        spans = calls_to_otel_spans(engine, session_id="nope")
        assert spans == []

    def test_filter_by_tool(self):
        engine = _populate_engine(3)
        spans = calls_to_otel_spans(engine, tool_name="tool_0")
        # tool_0 appears when i % 3 == 0, so only call 0
        assert all(s["name"] == "tool_0" for s in spans)

    def test_include_events_true(self):
        engine = _populate_engine(2)
        calls = engine.query_calls()
        engine.log_event(trace_id=calls[0].session_id, call_id=calls[0].id, event_type="test", message="msg")
        spans = calls_to_otel_spans(engine, include_events=True)
        assert any("events" in s and len(s["events"]) > 0 for s in spans)

    def test_include_events_false(self):
        engine = _populate_engine(2)
        calls = engine.query_calls()
        engine.log_event(trace_id=calls[0].session_id, call_id=calls[0].id, event_type="test", message="msg")
        spans = calls_to_otel_spans(engine, include_events=False)
        assert all("events" not in s or len(s.get("events", [])) == 0 for s in spans)

    def test_limit(self):
        engine = _populate_engine(10)
        spans = calls_to_otel_spans(engine, limit=3)
        assert len(spans) == 3


# ── build_otlp_request Tests ─────────────────────────────────────────

class TestBuildOTLPRequest:
    def test_basic_structure(self):
        engine = _populate_engine(3)
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans)

        assert "resourceSpans" in request
        assert len(request["resourceSpans"]) == 1
        rs = request["resourceSpans"][0]
        assert "resource" in rs
        assert "scopeSpans" in rs
        assert "attributes" in rs["resource"]

    def test_resource_attributes_default(self):
        engine = _populate_engine(1)
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans)
        attrs = {a["key"]: a["value"]["stringValue"]
                 for a in request["resourceSpans"][0]["resource"]["attributes"]}
        assert attrs["service.name"] == "mcp-audit"
        assert "service.version" in attrs
        assert attrs["telemetry.sdk.language"] == "python"

    def test_custom_resource_attrs_merge(self):
        engine = _populate_engine(1)
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans, resource_attrs={"deployment.environment": "prod"})
        attrs = {a["key"]: a["value"]["stringValue"]
                 for a in request["resourceSpans"][0]["resource"]["attributes"]}
        assert attrs["deployment.environment"] == "prod"
        assert attrs["service.name"] == "mcp-audit"  # default still there

    def test_empty_spans(self):
        request = build_otlp_request([])
        assert "resourceSpans" in request
        assert len(request["resourceSpans"][0]["scopeSpans"]) == 0

    def test_scope_info(self):
        engine = _populate_engine(1)
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans)
        scope = request["resourceSpans"][0]["scopeSpans"][0]["scope"]
        assert scope["name"] == "mcp-audit"
        assert "version" in scope

    def test_multi_trace_grouping(self):
        engine = _populate_engine_multi_session()
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans)
        # Should have multiple scopeSpans entries (one per trace)
        scope_spans = request["resourceSpans"][0]["scopeSpans"]
        assert len(scope_spans) >= 2

    def test_serializable_to_json(self):
        engine = _populate_engine(3)
        spans = calls_to_otel_spans(engine)
        request = build_otlp_request(spans)
        # Should not raise
        json_str = json.dumps(request, default=str)
        assert len(json_str) > 0
        # Should round-trip
        parsed = json.loads(json_str)
        assert "resourceSpans" in parsed


# ── export_otlp_jsonl Tests ──────────────────────────────────────────

class TestExportOTLPJsonl:
    def test_writes_file(self, tmp_path):
        engine = _populate_engine(5)
        out = tmp_path / "traces.otlp.jsonl"
        result = export_otlp_jsonl(engine, str(out))

        assert result["format"] == "otlp_jsonl"
        assert result["span_count"] == 5
        assert result["size_bytes"] > 0
        assert out.exists()

    def test_file_contains_valid_jsonl(self, tmp_path):
        engine = _populate_engine(3)
        out = tmp_path / "traces.jsonl"
        export_otlp_jsonl(engine, str(out))

        lines = out.read_text().strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            assert "resourceSpans" in obj

    def test_one_line_per_trace(self, tmp_path):
        engine = _populate_engine_multi_session()
        out = tmp_path / "traces.jsonl"
        result = export_otlp_jsonl(engine, str(out))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == result["trace_count"]
        assert result["trace_count"] >= 2

    def test_trace_ids_returned(self, tmp_path):
        engine = _populate_engine(2)
        out = tmp_path / "traces.jsonl"
        result = export_otlp_jsonl(engine, str(out))
        assert len(result["trace_ids"]) >= 1

    def test_creates_parent_dirs(self, tmp_path):
        engine = _populate_engine(1)
        out = tmp_path / "subdir" / "deep" / "traces.jsonl"
        result = export_otlp_jsonl(engine, str(out))
        assert out.exists()
        assert result["span_count"] == 1

    def test_empty_engine(self, tmp_path):
        engine = AuditEngine()
        out = tmp_path / "empty.jsonl"
        result = export_otlp_jsonl(engine, str(out))
        assert result["span_count"] == 0
        assert result["trace_count"] == 0
        assert out.exists()

    def test_filtered_export(self, tmp_path):
        engine = _populate_engine(5)
        out = tmp_path / "filtered.jsonl"
        result = export_otlp_jsonl(engine, str(out), tool_name="tool_0")
        assert result["span_count"] <= 5

    def test_custom_resource_attrs(self, tmp_path):
        engine = _populate_engine(1)
        out = tmp_path / "traces.jsonl"
        export_otlp_jsonl(engine, str(out), resource_attrs={"env": "test"})
        lines = out.read_text().strip().split("\n")
        obj = json.loads(lines[0])
        attrs = {a["key"]: a["value"]["stringValue"]
                 for a in obj["resourceSpans"][0]["resource"]["attributes"]}
        assert attrs["env"] == "test"


# ── export_otlp_http Tests ───────────────────────────────────────────

class TestExportOTLPHttp:
    def test_connection_error_handled_gracefully(self):
        """Exporting to a non-existent endpoint should not raise."""
        engine = _populate_engine(3)
        result = export_otlp_http(
            engine,
            endpoint="http://localhost:99999/v1/traces",
            timeout=2,
        )
        assert result["status"] in ("connection_error", "error")
        assert result["span_count"] == 3
        assert result["bytes_sent"] > 0

    def test_empty_engine_no_data(self):
        engine = AuditEngine()
        result = export_otlp_http(engine, endpoint="http://localhost:99999/v1/traces")
        assert result["status"] == "no_data"
        assert result["span_count"] == 0

    def test_endpoint_in_result(self):
        engine = _populate_engine(1)
        result = export_otlp_http(
            engine,
            endpoint="http://my-collector:4318/v1/traces",
            timeout=2,
        )
        assert result["endpoint"] == "http://my-collector:4318/v1/traces"

    def test_custom_headers(self):
        """Test that custom headers are accepted without error."""
        engine = _populate_engine(1)
        result = export_otlp_http(
            engine,
            endpoint="http://localhost:99999/v1/traces",
            headers={"Authorization": "Bearer test-token"},
            timeout=2,
        )
        # Should fail on connection, not on header setup
        assert result["status"] in ("connection_error", "error")

    def test_resource_attrs_in_http(self):
        engine = _populate_engine(1)
        result = export_otlp_http(
            engine,
            endpoint="http://localhost:99999/v1/traces",
            resource_attrs={"deployment.environment": "staging"},
            timeout=2,
        )
        # The request should fail on connection but still process the spans
        assert result["span_count"] == 1


# ── export_otlp_to_string Tests ──────────────────────────────────────

class TestExportOTLPToString:
    def test_returns_valid_json(self):
        engine = _populate_engine(3)
        result = export_otlp_to_string(engine)
        parsed = json.loads(result)
        assert "resourceSpans" in parsed

    def test_empty_engine(self):
        engine = AuditEngine()
        result = export_otlp_to_string(engine)
        parsed = json.loads(result)
        assert "resourceSpans" in parsed

    def test_limit_respected(self):
        engine = _populate_engine(10)
        result_small = export_otlp_to_string(engine, limit=2)
        result_all = export_otlp_to_string(engine, limit=100)

        # Small should be shorter
        assert len(result_small) < len(result_all)

    def test_contains_spans(self):
        engine = _populate_engine(3)
        result = export_otlp_to_string(engine)
        parsed = json.loads(result)
        scope_spans = parsed["resourceSpans"][0]["scopeSpans"]
        total_spans = sum(len(ss["spans"]) for ss in scope_spans)
        assert total_spans == 3


# ── OTLPExporter Tests ───────────────────────────────────────────────

class TestOTLPExporter:
    def test_default_endpoint(self):
        exporter = OTLPExporter()
        assert "4318" in exporter.endpoint

    def test_custom_endpoint(self):
        exporter = OTLPExporter(endpoint="http://custom:1234/traces")
        assert exporter.endpoint == "http://custom:1234/traces"

    def test_env_var_endpoint(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://env:9999/traces")
        exporter = OTLPExporter()
        assert exporter.endpoint == "http://env:9999/traces"

    def test_custom_headers_stored(self):
        exporter = OTLPExporter(headers={"X-Custom": "val"})
        assert exporter.headers == {"X-Custom": "val"}

    def test_export_from_engine(self):
        engine = _populate_engine(3)
        exporter = OTLPExporter(endpoint="http://localhost:99999/v1/traces", timeout=2)
        result = exporter.export(engine)
        assert result["span_count"] == 3
        assert result["status"] in ("connection_error", "error")

    def test_flush_empty_spans(self):
        exporter = OTLPExporter()
        result = exporter.flush([])
        assert result["status"] == "no_data"
        assert result["span_count"] == 0

    def test_flush_with_spans(self):
        engine = _populate_engine(2)
        spans = calls_to_otel_spans(engine)
        exporter = OTLPExporter(endpoint="http://localhost:99999/v1/traces", timeout=2)
        result = exporter.flush(spans)
        assert result["span_count"] == 2
        assert result["status"] in ("connection_error", "error")

    def test_export_with_resource_attrs(self):
        engine = _populate_engine(1)
        exporter = OTLPExporter(
            endpoint="http://localhost:99999/v1/traces",
            timeout=2,
            resource_attrs={"env": "prod"},
        )
        result = exporter.export(engine)
        # Should not raise, spans should be counted
        assert result["span_count"] == 1


# ── MCP Server Tool Tests ────────────────────────────────────────────

class TestMCPServerOTLPTool:
    def test_tool_definition_exists(self):
        from mcp_audit.server import TOOL_DEFINITIONS
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "export_otlp" in names

    def test_tool_count_increased(self):
        from mcp_audit.server import TOOL_DEFINITIONS
        assert len(TOOL_DEFINITIONS) == 28  # v0.8: added 4 time-series analytics tools

    def test_server_jsonl_export(self, tmp_path):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(3)
        server = MCPServer(engine)
        out = tmp_path / "out.jsonl"
        result = server.call_tool("export_otlp", {
            "mode": "jsonl",
            "output_path": str(out),
        })
        assert "result" in result
        assert result["result"]["span_count"] == 3
        assert out.exists()

    def test_server_http_export(self):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(2)
        server = MCPServer(engine)
        result = server.call_tool("export_otlp", {
            "mode": "http",
            "endpoint": "http://localhost:99999/v1/traces",
        })
        assert "result" in result
        assert result["result"]["span_count"] == 2

    def test_server_jsonl_missing_output_path(self):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(1)
        server = MCPServer(engine)
        result = server.call_tool("export_otlp", {
            "mode": "jsonl",
        })
        assert "result" in result
        assert "error" in result["result"]

    def test_server_default_mode_jsonl(self, tmp_path):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(1)
        server = MCPServer(engine)
        out = tmp_path / "default.jsonl"
        result = server.call_tool("export_otlp", {
            "output_path": str(out),
        })
        assert "result" in result
        assert result["result"]["span_count"] == 1

    def test_server_invalid_mode(self):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(1)
        server = MCPServer(engine)
        result = server.call_tool("export_otlp", {
            "mode": "invalid",
        })
        assert "result" in result
        assert "error" in result["result"]

    def test_server_filtered_export(self, tmp_path):
        from mcp_audit.server import MCPServer
        engine = _populate_engine(5)
        server = MCPServer(engine)
        out = tmp_path / "filtered.jsonl"
        result = server.call_tool("export_otlp", {
            "mode": "jsonl",
            "output_path": str(out),
            "tool_name": "tool_0",
        })
        assert "result" in result
        assert result["result"]["span_count"] >= 1


# ── Integration / End-to-End Tests ───────────────────────────────────

class TestIntegration:
    def test_full_pipeline_jsonl(self, tmp_path):
        """Record calls → export to OTLP JSONL → parse back → verify structure."""
        engine = _populate_engine_multi_session()
        out = tmp_path / "integration.jsonl"

        result = export_otlp_jsonl(engine, str(out))
        assert result["status"] if "status" in result else True

        # Parse each line
        lines = out.read_text().strip().split("\n")
        all_spans = []
        for line in lines:
            obj = json.loads(line)
            for rs in obj["resourceSpans"]:
                for ss in rs["scopeSpans"]:
                    all_spans.extend(ss["spans"])

        assert len(all_spans) == result["span_count"]
        # All span IDs should be unique
        span_ids = [s["spanId"] for s in all_spans]
        assert len(set(span_ids)) == len(span_ids)

    def test_otlp_compliance_check(self, tmp_path):
        """Verify the exported OTLP/JSON conforms to expected structure."""
        engine = _populate_engine(2)
        out = tmp_path / "compliance.jsonl"
        export_otlp_jsonl(engine, str(out))

        lines = out.read_text().strip().split("\n")
        for line in lines:
            obj = json.loads(line)
            # Top-level
            assert "resourceSpans" in obj
            for rs in obj["resourceSpans"]:
                # Resource section
                assert "resource" in rs
                assert "attributes" in rs["resource"]
                # ScopeSpans
                assert "scopeSpans" in rs
                for ss in rs["scopeSpans"]:
                    assert "scope" in ss
                    assert "name" in ss["scope"]
                    assert "spans" in ss
                    for span in ss["spans"]:
                        # Span required fields
                        assert len(span["traceId"]) == 32
                        assert len(span["spanId"]) == 16
                        assert "name" in span
                        assert "kind" in span
                        assert "startTimeUnixNano" in span
                        assert "endTimeUnixNano" in span
                        assert "status" in span
                        assert "code" in span["status"]
                        assert span["status"]["code"] in (
                            "STATUS_CODE_OK", "STATUS_CODE_ERROR"
                        )
                        assert "attributes" in span

    def test_roundtrip_with_sqlite_store(self, tmp_path):
        """OTLP export should work with SQLiteStore too."""
        from mcp_audit import SQLiteStore

        db_path = tmp_path / "audit.db"
        store = SQLiteStore(str(db_path))
        engine = AuditEngine(store=store)

        session = engine.start_session(agent_id="sqlite-agent")
        engine.record_call(
            session_id=session.id,
            tool_name="sqlite_tool",
            agent_id="sqlite-agent",
            duration_ms=42,
            cost_usd=0.005,
            input_tokens=100,
            output_tokens=50,
        )

        out = tmp_path / "sqlite_export.jsonl"
        result = export_otlp_jsonl(engine, str(out))
        assert result["span_count"] == 1

        # Verify the span is well-formed
        lines = out.read_text().strip().split("\n")
        obj = json.loads(lines[0])
        spans = obj["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "sqlite_tool"

    def test_decorator_to_otlp(self, tmp_path):
        """Calls recorded via @audit_call should export correctly via OTLP."""
        from mcp_audit import audit_call, bind_session

        engine = AuditEngine()
        session = engine.start_session(agent_id="deco-agent")
        bind_session(engine, session.id)

        @audit_call(tool_name="decorated_fn")
        def my_function(x):
            return x * 2

        result_val = my_function(21)
        assert result_val == 42

        spans = calls_to_otel_spans(engine)
        assert len(spans) == 1
        assert spans[0]["name"] == "decorated_fn"

        # Check attributes
        attr_keys = {a["key"] for a in spans[0]["attributes"]}
        assert "tool.name" in attr_keys
        assert "call.duration_ms" in attr_keys
