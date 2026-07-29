"""OpenTelemetry Protocol (OTLP) export for audit records.

Converts mcp-audit tool calls and trace events into OpenTelemetry-compatible
spans and exports them via OTLP/JSON — the JSON encoding of the OpenTelemetry
Protocol. This lets mcp-audit data flow into **any** OTel-compatible backend:
Jaeger, Tempo, Honeycomb, Grafana, Datadog, SigNoz, or the OTel Collector.

Two export modes:

* **OTLP/HTTP** — POST spans to a collector endpoint (e.g. ``http://localhost:4318/v1/traces``).
* **OTLP/JSON file** — Write spans as OTLP/JSON Lines to disk, for log shippers
  or offline analysis.

Mapping
-------

+-------------------+-----------------------------------+
| mcp-audit         | OpenTelemetry                     |
+===================+===================================+
| ToolCall          | Span                              |
+-------------------+-----------------------------------+
|   .tool_name      | span name                         |
+-------------------+-----------------------------------+
|   .started_at     | span start time                   |
+-------------------+-----------------------------------+
|   .completed_at   | span end time                     |
+-------------------+-----------------------------------+
|   .session_id     | trace ID                          |
+-------------------+-----------------------------------+
|   .agent_id       | resource attribute                |
+-------------------+-----------------------------------+
|   .cost_usd       | span attribute ``cost.usd``       |
+-------------------+-----------------------------------+
|   .status         | span status (OK / ERROR)          |
+-------------------+-----------------------------------+
| TraceEvent        | Span event (annotation)           |
+-------------------+-----------------------------------+

Usage::

    from mcp_audit import AuditEngine
    from mcp_audit.otlp import export_otlp_http, calls_to_otel_spans

    engine = AuditEngine()
    # ... record calls ...

    # Send to a local OTel collector
    result = export_otlp_http(engine, endpoint="http://localhost:4318/v1/traces")

    # Or export to a file for log shipping
    from mcp_audit.otlp import export_otlp_jsonl
    result = export_otlp_jsonl(engine, "traces.otlp.jsonl")
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any

from .engine import AuditEngine
from .models import CallStatus, ToolCall, TraceEvent


# ── Constants ────────────────────────────────────────────────────────

_SERVICE_NAME = "mcp-audit"
_SERVICE_VERSION = "0.5.0"
_DEFAULT_RESOURCE_ATTRS: dict[str, str] = {
    "service.name": _SERVICE_NAME,
    "service.version": _SERVICE_VERSION,
    "telemetry.sdk.language": "python",
    "telemetry.sdk.name": "mcp-audit",
}

# OTLP/HTTP default endpoint (OTel Collector standard)
_DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"
_OTLP_CONTENT_TYPE = "application/json"
_DEFAULT_TIMEOUT = 15  # seconds


# ── ID conversion helpers ────────────────────────────────────────────

def _to_otel_trace_id(session_id: str) -> str:
    """Convert a session ID to a 32-char hex OTel trace ID.

    OTel requires exactly 16 bytes (32 hex chars). We hash or pad the
    session_id to produce a deterministic, valid-length ID.
    """
    import hashlib
    h = hashlib.sha256(session_id.encode()).hexdigest()
    return h[:32]


def _to_otel_span_id(call_id: str) -> str:
    """Convert a call ID to a 16-char hex OTel span ID.

    OTel requires exactly 8 bytes (16 hex chars).
    """
    import hashlib
    h = hashlib.sha256(call_id.encode()).hexdigest()
    return h[:16]


def _to_nanoseconds(dt: datetime) -> int:
    """Convert a datetime to Unix nanoseconds (OTel time format)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


# ── Span conversion ──────────────────────────────────────────────────

def _call_to_span(
    call: ToolCall,
    events: list[TraceEvent] | None = None,
) -> dict[str, Any]:
    """Convert a single ToolCall to an OTLP span dict.

    Parameters
    ----------
    call : ToolCall
        The recorded tool call.
    events : list[TraceEvent], optional
        Trace events to attach as span events (filtered to this call's ID).
    """
    trace_id = _to_otel_trace_id(call.session_id)
    span_id = _to_otel_span_id(call.id)

    start_ns = _to_nanoseconds(call.started_at)
    end_dt = call.completed_at or call.started_at
    end_ns = _to_nanoseconds(end_dt)

    # Span status
    if call.status == CallStatus.SUCCESS:
        status_code = "STATUS_CODE_OK"
    else:
        status_code = "STATUS_CODE_ERROR"

    # Attributes
    attributes: dict[str, Any] = {
        "tool.name": call.tool_name,
        "call.id": call.id,
        "session.id": call.session_id,
        "call.status": call.status.value,
        "call.duration_ms": call.duration_ms or 0.0,
        "cost.usd": call.cost_usd,
        "tokens.input": call.input_tokens,
        "tokens.output": call.output_tokens,
        "tokens.total": call.input_tokens + call.output_tokens,
    }
    if call.agent_id:
        attributes["agent.id"] = call.agent_id
    if call.server_name:
        attributes["mcp.server"] = call.server_name
    if call.error:
        attributes["error.message"] = call.error
    if call.tags:
        attributes["tags"] = call.tags

    # Span events (from TraceEvents)
    span_events: list[dict[str, Any]] = []
    if events:
        for ev in events:
            if ev.call_id and ev.call_id != call.id:
                continue
            span_events.append({
                "name": ev.event_type,
                "timeUnixNano": str(_to_nanoseconds(ev.timestamp)),
                "attributes": [
                    {"key": "message", "value": {"stringValue": ev.message}},
                    {"key": "severity", "value": {"stringValue": ev.severity.value}},
                ] + (
                    [{"key": k, "value": {"stringValue": str(v)}} for k, v in ev.data.items()]
                    if ev.data else []
                ),
            })

    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": call.tool_name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "status": {
            "code": status_code,
        },
        "attributes": [
            {"key": k, "value": _attribute_value(v)}
            for k, v in attributes.items()
        ],
    }

    if call.error:
        span["status"]["message"] = call.error

    if span_events:
        span["events"] = span_events

    return span


def _attribute_value(value: Any) -> dict[str, Any]:
    """Convert a Python value to an OTLP AnyValue."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_attribute_value(v) for v in value]}}
    return {"stringValue": str(value)}


def _group_calls_by_session(calls: list[ToolCall]) -> dict[str, list[ToolCall]]:
    """Group calls by session_id for OTLP trace grouping."""
    groups: dict[str, list[ToolCall]] = {}
    for call in calls:
        groups.setdefault(call.session_id, []).append(call)
    return groups


# ── Public API ───────────────────────────────────────────────────────

def calls_to_otel_spans(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    include_events: bool = True,
) -> list[dict[str, Any]]:
    """Convert engine's tool calls to OpenTelemetry span dicts.

    Returns a flat list of OTLP span dictionaries ready for serialisation.
    Each span maps one ToolCall; trace events are attached as span events.
    """
    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    # Fetch events if requested
    events_by_call: dict[str, list[TraceEvent]] = {}
    if include_events and calls:
        all_events = engine.query_events(limit=limit * 5)
        for ev in all_events:
            if ev.call_id:
                events_by_call.setdefault(ev.call_id, []).append(ev)

    spans = []
    for call in calls:
        call_events = events_by_call.get(call.id)
        spans.append(_call_to_span(call, call_events))

    return spans


def build_otlp_request(
    spans: list[dict[str, Any]],
    *,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a complete OTLP/JSON request body from span dicts.

    The output is a JSON object conforming to the OTLP/JSON schema::

        {
          "resourceSpans": [{
            "resource": {"attributes": [...]},
            "scopeSpans": [{
              "scope": {"name": "mcp-audit"},
              "spans": [...]
            }]
          }]
        }
    """
    attrs = {**_DEFAULT_RESOURCE_ATTRS, **(resource_attrs or {})}
    resource_attributes = [
        {"key": k, "value": {"stringValue": str(v)}}
        for k, v in attrs.items()
    ]

    # Group spans by trace ID for proper trace structure
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = span.get("traceId", "unknown")
        by_trace.setdefault(trace_id, []).append(span)

    scope_spans: list[dict[str, Any]] = []
    for trace_id, trace_spans in by_trace.items():
        scope_spans.append({
            "scope": {
                "name": _SERVICE_NAME,
                "version": _SERVICE_VERSION,
            },
            "spans": trace_spans,
        })

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": resource_attributes,
                },
                "scopeSpans": scope_spans,
            }
        ]
    }


def export_otlp_http(
    engine: AuditEngine,
    *,
    endpoint: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    include_events: bool = True,
    headers: dict[str, str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export tool calls as OTLP/JSON traces to an HTTP endpoint.

    Sends a POST request with the OTLP/JSON payload to the specified
    endpoint (default: local OTel Collector at ``http://localhost:4318/v1/traces``).

    Returns metadata about the export (endpoint, span_count, status_code, bytes_sent).
    """
    spans = calls_to_otel_spans(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
        include_events=include_events,
    )

    if not spans:
        return {
            "endpoint": endpoint or _DEFAULT_OTLP_ENDPOINT,
            "span_count": 0,
            "status": "no_data",
            "bytes_sent": 0,
        }

    payload = build_otlp_request(spans, resource_attrs=resource_attrs)
    body = json.dumps(payload).encode("utf-8")

    url = endpoint or _DEFAULT_OTLP_ENDPOINT
    req_headers = {
        "Content-Type": _OTLP_CONTENT_TYPE,
    }
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(
        url,
        data=body,
        headers=req_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status_code = resp.status
            resp_body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return {
            "endpoint": url,
            "span_count": len(spans),
            "status": "error",
            "status_code": e.code,
            "error": str(e),
            "bytes_sent": len(body),
        }
    except urllib.error.URLError as e:
        return {
            "endpoint": url,
            "span_count": len(spans),
            "status": "connection_error",
            "error": str(e.reason if hasattr(e, "reason") else e),
            "bytes_sent": len(body),
        }
    except Exception as e:
        return {
            "endpoint": url,
            "span_count": len(spans),
            "status": "error",
            "error": str(e),
            "bytes_sent": len(body),
        }

    return {
        "endpoint": url,
        "span_count": len(spans),
        "status": "success",
        "status_code": status_code,
        "bytes_sent": len(body),
    }


def export_otlp_jsonl(
    engine: AuditEngine,
    output_path: str,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    include_events: bool = True,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export tool calls as OTLP/JSON to a file (one OTLP request per line).

    Each line is a complete OTLP/JSON request body — ready for ingestion
    by any OTLP-compatible collector. This is useful for:

    * Shipping to collectors via file-based pipelines (Fluentd, Vector)
    * Offline analysis with ``jq`` or custom tooling
    * Batching exports

    Returns metadata about the export (path, span_count, trace_count, size_bytes).
    """
    from pathlib import Path

    spans = calls_to_otel_spans(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
        include_events=include_events,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group by trace for one OTLP request per trace
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        trace_id = span.get("traceId", "unknown")
        by_trace.setdefault(trace_id, []).append(span)

    with open(path, "w") as f:
        for trace_id, trace_spans in by_trace.items():
            payload = build_otlp_request(trace_spans, resource_attrs=resource_attrs)
            f.write(json.dumps(payload, default=str) + "\n")

    size = path.stat().st_size
    trace_ids = list(by_trace.keys())

    return {
        "format": "otlp_jsonl",
        "path": str(path),
        "span_count": len(spans),
        "trace_count": len(trace_ids),
        "trace_ids": trace_ids[:10],  # first 10 for verification
        "size_bytes": size,
    }


def export_otlp_to_string(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 100,
    include_events: bool = True,
    resource_attrs: dict[str, str] | None = None,
) -> str:
    """Export tool calls as an OTLP/JSON string (for in-memory use, API responses).

    Returns a single OTLP/JSON request body as a string.
    """
    spans = calls_to_otel_spans(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
        include_events=include_events,
    )
    payload = build_otlp_request(spans, resource_attrs=resource_attrs)
    return json.dumps(payload, default=str)


class OTLPExporter:
    """A reusable OTLP exporter that can batch and flush spans.

    Example::

        exporter = OTLPExporter(
            endpoint="http://collector:4318/v1/traces",
            resource_attrs={"deployment.environment": "production"},
        )

        # Export from engine
        result = exporter.export(engine, session_id=sid)

        # Or export pre-built spans
        result = exporter.flush(spans)

    The exporter is stateless between calls — each export is independent.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        resource_attrs: dict[str, str] | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            _DEFAULT_OTLP_ENDPOINT,
        )
        self.headers = headers or {}
        self.timeout = timeout
        self.resource_attrs = resource_attrs or {}

    def export(
        self,
        engine: AuditEngine,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 10_000,
        include_events: bool = True,
    ) -> dict[str, Any]:
        """Export from an AuditEngine via OTLP/HTTP."""
        return export_otlp_http(
            engine,
            endpoint=self.endpoint,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
            include_events=include_events,
            headers=self.headers,
            timeout=self.timeout,
            resource_attrs=self.resource_attrs,
        )

    def flush(self, spans: list[dict[str, Any]]) -> dict[str, Any]:
        """Flush pre-built span dicts via OTLP/HTTP."""
        if not spans:
            return {
                "endpoint": self.endpoint,
                "span_count": 0,
                "status": "no_data",
                "bytes_sent": 0,
            }

        payload = build_otlp_request(spans, resource_attrs=self.resource_attrs)
        body = json.dumps(payload).encode("utf-8")

        req_headers = {"Content-Type": _OTLP_CONTENT_TYPE}
        req_headers.update(self.headers)

        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=req_headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return {
                    "endpoint": self.endpoint,
                    "span_count": len(spans),
                    "status": "success",
                    "status_code": resp.status,
                    "bytes_sent": len(body),
                }
        except Exception as e:
            return {
                "endpoint": self.endpoint,
                "span_count": len(spans),
                "status": "error",
                "error": str(e),
                "bytes_sent": len(body),
            }
