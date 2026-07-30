"""OTLP Metrics export for audit records.

Converts mcp-audit data into OpenTelemetry **metrics** — counters, histograms,
and gauges — and exports them via OTLP/JSON. This complements :mod:`otlp`
(which exports *traces*) to provide full observability coverage.

Metric types generated
----------------------

+-----------------------+-----------+--------------------------------------+
| Metric name           | Type      | Description                          |
+=======================+===========+======================================+
| ``tool.call.count``   | Counter   | Total tool calls, grouped by tool    |
+-----------------------+-----------+--------------------------------------+
| ``tool.error.count``  | Counter   | Error calls, grouped by tool         |
+-----------------------+-----------+--------------------------------------+
| ``tool.duration_ms``  | Histogram | Call latency distribution            |
+-----------------------+-----------+--------------------------------------+
| ``tool.cost.usd``     | Histogram | Cost per call distribution           |
+-----------------------+-----------+--------------------------------------+
| ``tool.tokens.input`` | Counter   | Total input tokens, grouped by tool  |
+-----------------------+-----------+--------------------------------------+
| ``tool.tokens.output``| Counter   | Total output tokens, grouped by tool |
+-----------------------+-----------+--------------------------------------+
| ``session.count``     | Gauge     | Number of sessions                   |
+-----------------------+-----------+--------------------------------------+
| ``error.rate``        | Gauge     | Overall error rate (%)               |
+-----------------------+-----------+--------------------------------------+

Usage::

    from mcp_audit import AuditEngine
    from mcp_audit.metrics import export_otlp_metrics_http, build_metrics

    engine = AuditEngine()
    # ... record calls ...

    # Build metrics dicts (for inspection or custom export)
    metrics = build_metrics(engine)

    # Send to a local OTel collector
    result = export_otlp_metrics_http(
        engine, endpoint="http://localhost:4318/v1/metrics"
    )
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any

from .engine import AuditEngine
from .models import CallStatus, ToolCall


# ── Constants ────────────────────────────────────────────────────────

_SERVICE_NAME = "mcp-audit"
_SERVICE_VERSION = "0.6.0"
_DEFAULT_RESOURCE_ATTRS: dict[str, str] = {
    "service.name": _SERVICE_NAME,
    "service.version": _SERVICE_VERSION,
    "telemetry.sdk.language": "python",
    "telemetry.sdk.name": "mcp-audit",
}

# OTLP/HTTP default endpoint for metrics
_DEFAULT_OTLP_METRICS_ENDPOINT = "http://localhost:4318/v1/metrics"
_OTLP_CONTENT_TYPE = "application/json"
_DEFAULT_TIMEOUT = 15  # seconds

# Standard latency histogram boundaries (ms) — aligned with OTel defaults
_LATENCY_BOUNDS = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
# Standard cost histogram boundaries (USD)
_COST_BOUNDS = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


# ── Helpers ──────────────────────────────────────────────────────────

def _to_nanoseconds(dt: datetime) -> str:
    """Convert a datetime to Unix nanoseconds string (OTel time format)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(int(dt.timestamp() * 1_000_000_000))


def _now_nanoseconds() -> str:
    """Current UTC time in nanoseconds as a string."""
    return _to_nanoseconds(datetime.now(timezone.utc))


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


def _make_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build an OTLP attribute list from a dict."""
    return [{"key": k, "value": _attribute_value(v)} for k, v in attrs.items()]


def _bucketize(values: list[float], bounds: list[float]) -> tuple[list[int], list[float]]:
    """Distribute *values* into histogram buckets defined by *bounds*.

    Returns ``(bucket_counts, explicit_bounds)`` where ``len(bucket_counts) =
    len(bounds) + 1`` (the final bucket catches everything above the last bound).
    """
    if not bounds:
        # Single bucket catches everything
        return [len(values)], []

    counts = [0] * (len(bounds) + 1)
    for v in values:
        placed = False
        for i, b in enumerate(bounds):
            if v <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1  # overflow bucket
    return counts, bounds


# ── Data-point builders ──────────────────────────────────────────────

def _build_sum_data_point(
    value: int,
    attrs: dict[str, Any],
    *,
    time_ns: str | None = None,
) -> dict[str, Any]:
    """Build an OTLP Sum data point."""
    return {
        "attributes": _make_attributes(attrs),
        "timeUnixNano": time_ns or _now_nanoseconds(),
        "asInt": str(value),
    }


def _build_gauge_data_point(
    value: float,
    attrs: dict[str, Any],
    *,
    time_ns: str | None = None,
) -> dict[str, Any]:
    """Build an OTLP Gauge data point (NumberDataPoint with asDouble)."""
    return {
        "attributes": _make_attributes(attrs),
        "timeUnixNano": time_ns or _now_nanoseconds(),
        "asDouble": value,
    }


def _build_histogram_data_point(
    values: list[float],
    bounds: list[float],
    attrs: dict[str, Any],
    *,
    time_ns: str | None = None,
) -> dict[str, Any]:
    """Build an OTLP Histogram data point."""
    bucket_counts, explicit_bounds = _bucketize(values, bounds)
    total = float(sum(values)) if values else 0.0
    count = len(values)

    dp: dict[str, Any] = {
        "attributes": _make_attributes(attrs),
        "timeUnixNano": time_ns or _now_nanoseconds(),
        "count": str(count),
        "sum": total,
        "bucketCounts": [str(c) for c in bucket_counts],
    }
    if explicit_bounds:
        dp["explicitBounds"] = explicit_bounds

    # Min/max if available (OTLP extension)
    if values:
        dp["min"] = min(values)
        dp["max"] = max(values)

    return dp


def _build_sum_metric(
    name: str,
    description: str,
    unit: str,
    data_points: list[dict[str, Any]],
    *,
    is_monotonic: bool = True,
) -> dict[str, Any]:
    """Build a complete OTLP Sum metric."""
    return {
        "name": name,
        "description": description,
        "unit": unit,
        "sum": {
            "dataPoints": data_points,
            "isMonotonic": is_monotonic,
            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
        },
    }


def _build_gauge_metric(
    name: str,
    description: str,
    unit: str,
    data_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a complete OTLP Gauge metric."""
    return {
        "name": name,
        "description": description,
        "unit": unit,
        "gauge": {
            "dataPoints": data_points,
        },
    }


def _build_histogram_metric(
    name: str,
    description: str,
    unit: str,
    data_points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a complete OTLP Histogram metric."""
    return {
        "name": name,
        "description": description,
        "unit": unit,
        "histogram": {
            "dataPoints": data_points,
            "aggregationTemporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
        },
    }


# ── Core metrics builder ─────────────────────────────────────────────

def build_metrics(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> list[dict[str, Any]]:
    """Build a list of OTLP metric dicts from engine data.

    Parameters
    ----------
    engine
        The audit engine to read data from.
    session_id, agent_id, tool_name
        Optional filters (same semantics as ``query_calls``).
    limit
        Maximum number of calls to aggregate.

    Returns
    -------
    list of dict
        A flat list of OTLP metric dictionaries.
    """
    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    if not calls:
        return _empty_metrics()

    # Group calls by tool_name for per-tool metrics
    by_tool: dict[str, list[ToolCall]] = {}
    for call in calls:
        by_tool.setdefault(call.tool_name, []).append(call)

    metrics: list[dict[str, Any]] = []
    now_ns = _now_nanoseconds()

    # ── Counters: tool.call.count ──
    call_count_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        call_count_dps.append(
            _build_sum_data_point(
                len(tool_calls),
                {"tool.name": tool},
                time_ns=now_ns,
            )
        )
    metrics.append(
        _build_sum_metric(
            "tool.call.count",
            "Total number of tool calls by tool name",
            "{calls}",
            call_count_dps,
        )
    )

    # ── Counters: tool.error.count ──
    error_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        error_count = sum(1 for c in tool_calls if c.is_error)
        if error_count > 0:
            error_dps.append(
                _build_sum_data_point(
                    error_count,
                    {"tool.name": tool},
                    time_ns=now_ns,
                )
            )
    metrics.append(
        _build_sum_metric(
            "tool.error.count",
            "Number of failed tool calls by tool name",
            "{errors}",
            error_dps,
        )
    )

    # ── Counters: tool.tokens.input ──
    input_token_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        total_input = sum(c.input_tokens for c in tool_calls)
        if total_input > 0:
            input_token_dps.append(
                _build_sum_data_point(
                    total_input,
                    {"tool.name": tool},
                    time_ns=now_ns,
                )
            )
    metrics.append(
        _build_sum_metric(
            "tool.tokens.input",
            "Total input tokens consumed by tool name",
            "{tokens}",
            input_token_dps,
        )
    )

    # ── Counters: tool.tokens.output ──
    output_token_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        total_output = sum(c.output_tokens for c in tool_calls)
        if total_output > 0:
            output_token_dps.append(
                _build_sum_data_point(
                    total_output,
                    {"tool.name": tool},
                    time_ns=now_ns,
                )
            )
    metrics.append(
        _build_sum_metric(
            "tool.tokens.output",
            "Total output tokens produced by tool name",
            "{tokens}",
            output_token_dps,
        )
    )

    # ── Histograms: tool.duration_ms ──
    duration_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        durations = [c.duration_ms for c in tool_calls if c.duration_ms is not None]
        if durations:
            duration_dps.append(
                _build_histogram_data_point(
                    durations,
                    _LATENCY_BOUNDS,
                    {"tool.name": tool},
                    time_ns=now_ns,
                )
            )
    metrics.append(
        _build_histogram_metric(
            "tool.duration_ms",
            "Distribution of tool call latency in milliseconds",
            "ms",
            duration_dps,
        )
    )

    # ── Histograms: tool.cost.usd ──
    cost_dps = []
    for tool, tool_calls in sorted(by_tool.items()):
        costs = [c.cost_usd for c in tool_calls if c.cost_usd > 0]
        if costs:
            cost_dps.append(
                _build_histogram_data_point(
                    costs,
                    _COST_BOUNDS,
                    {"tool.name": tool},
                    time_ns=now_ns,
                )
            )
    metrics.append(
        _build_histogram_metric(
            "tool.cost.usd",
            "Distribution of per-call cost in USD by tool name",
            "USD",
            cost_dps,
        )
    )

    # ── Gauges ──

    # session.count
    sessions = engine.list_sessions(limit=100_000)
    active_sessions = [s for s in sessions if s.is_active]
    filter_attrs: dict[str, Any] = {}
    if agent_id:
        filter_attrs["agent.id"] = agent_id
    metrics.append(
        _build_gauge_metric(
            "session.count",
            "Total number of audit sessions",
            "{sessions}",
            [
                _build_gauge_data_point(
                    len(sessions), {**filter_attrs, "scope": "all"}, time_ns=now_ns
                ),
                _build_gauge_data_point(
                    len(active_sessions), {**filter_attrs, "scope": "active"}, time_ns=now_ns
                ),
            ],
        )
    )

    # error.rate
    error_calls = sum(1 for c in calls if c.is_error)
    error_rate = (error_calls / len(calls) * 100) if calls else 0.0
    metrics.append(
        _build_gauge_metric(
            "error.rate",
            "Overall error rate across all matching calls (percent)",
            "%",
            [_build_gauge_data_point(error_rate, filter_attrs, time_ns=now_ns)],
        )
    )

    # total.cost.usd gauge
    total_cost = sum(c.cost_usd for c in calls)
    metrics.append(
        _build_gauge_metric(
            "total.cost.usd",
            "Total cost in USD across all matching calls",
            "USD",
            [_build_gauge_data_point(total_cost, filter_attrs, time_ns=now_ns)],
        )
    )

    return metrics


def _empty_metrics() -> list[dict[str, Any]]:
    """Return a minimal set of metrics for the empty-engine case.

    This ensures agents always receive a valid OTLP metrics response even
    when no calls have been recorded yet.
    """
    now_ns = _now_nanoseconds()
    return [
        _build_sum_metric(
            "tool.call.count",
            "Total number of tool calls by tool name",
            "{calls}",
            [_build_sum_data_point(0, {}, time_ns=now_ns)],
        ),
        _build_sum_metric(
            "tool.error.count",
            "Number of failed tool calls by tool name",
            "{errors}",
            [],
        ),
        _build_gauge_metric(
            "session.count",
            "Total number of audit sessions",
            "{sessions}",
            [
                _build_gauge_data_point(0, {"scope": "all"}, time_ns=now_ns),
                _build_gauge_data_point(0, {"scope": "active"}, time_ns=now_ns),
            ],
        ),
        _build_gauge_metric(
            "error.rate",
            "Overall error rate across all matching calls (percent)",
            "%",
            [_build_gauge_data_point(0.0, {}, time_ns=now_ns)],
        ),
        _build_gauge_metric(
            "total.cost.usd",
            "Total cost in USD across all matching calls",
            "USD",
            [_build_gauge_data_point(0.0, {}, time_ns=now_ns)],
        ),
    ]


# ── OTLP request builder ─────────────────────────────────────────────

def build_otlp_metrics_request(
    metrics: list[dict[str, Any]],
    *,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a complete OTLP/JSON metrics request body.

    The output conforms to the OTLP/JSON schema::

        {
          "resourceMetrics": [{
            "resource": {"attributes": [...]},
            "scopeMetrics": [{
              "scope": {"name": "mcp-audit", "version": "..."},
              "metrics": [...]
            }]
          }]
        }
    """
    attrs = {**_DEFAULT_RESOURCE_ATTRS, **(resource_attrs or {})}
    resource_attributes = [
        {"key": k, "value": {"stringValue": str(v)}}
        for k, v in attrs.items()
    ]

    return {
        "resourceMetrics": [
            {
                "resource": {
                    "attributes": resource_attributes,
                },
                "scopeMetrics": [
                    {
                        "scope": {
                            "name": _SERVICE_NAME,
                            "version": _SERVICE_VERSION,
                        },
                        "metrics": metrics,
                    }
                ],
            }
        ]
    }


# ── Export functions ─────────────────────────────────────────────────

def export_otlp_metrics_http(
    engine: AuditEngine,
    *,
    endpoint: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    headers: dict[str, str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export audit metrics as OTLP/JSON to an HTTP endpoint.

    Sends a POST with the OTLP/JSON metrics payload to the specified
    endpoint (default: local OTel Collector at
    ``http://localhost:4318/v1/metrics``).

    Returns metadata about the export (endpoint, metric_count, status_code,
    bytes_sent).
    """
    metrics = build_metrics(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    payload = build_otlp_metrics_request(metrics, resource_attrs=resource_attrs)
    body = json.dumps(payload).encode("utf-8")

    url = endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        _DEFAULT_OTLP_METRICS_ENDPOINT,
    )
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
    except urllib.error.HTTPError as e:
        return {
            "endpoint": url,
            "metric_count": len(metrics),
            "status": "error",
            "status_code": e.code,
            "error": str(e),
            "bytes_sent": len(body),
        }
    except urllib.error.URLError as e:
        return {
            "endpoint": url,
            "metric_count": len(metrics),
            "status": "connection_error",
            "error": str(e.reason if hasattr(e, "reason") else e),
            "bytes_sent": len(body),
        }
    except Exception as e:
        return {
            "endpoint": url,
            "metric_count": len(metrics),
            "status": "error",
            "error": str(e),
            "bytes_sent": len(body),
        }

    return {
        "endpoint": url,
        "metric_count": len(metrics),
        "status": "success",
        "status_code": status_code,
        "bytes_sent": len(body),
    }


def export_otlp_metrics_jsonl(
    engine: AuditEngine,
    output_path: str,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    resource_attrs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Export audit metrics as OTLP/JSON to a file.

    Writes a single OTLP/JSON metrics request body (one JSON object).
    Useful for offline analysis, filebeat/Fluentd shipping, or CI checks.

    Returns metadata about the export (path, metric_count, size_bytes).
    """
    from pathlib import Path

    metrics = build_metrics(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    payload = build_otlp_metrics_request(metrics, resource_attrs=resource_attrs)
    body = json.dumps(payload, default=str)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(body)

    size = path.stat().st_size

    metric_names = [m["name"] for m in metrics]

    return {
        "format": "otlp_jsonl_metrics",
        "path": str(path),
        "metric_count": len(metrics),
        "metric_names": metric_names,
        "size_bytes": size,
    }


def export_otlp_metrics_to_string(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    resource_attrs: dict[str, str] | None = None,
) -> str:
    """Export audit metrics as an OTLP/JSON string.

    Returns a single OTLP/JSON metrics request body as a string —
    useful for API responses or in-memory inspection.
    """
    metrics = build_metrics(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )
    payload = build_otlp_metrics_request(metrics, resource_attrs=resource_attrs)
    return json.dumps(payload, default=str)


class OTLPMetricsExporter:
    """A reusable OTLP metrics exporter.

    Example::

        exporter = OTLPMetricsExporter(
            endpoint="http://collector:4318/v1/metrics",
            resource_attrs={"deployment.environment": "production"},
        )
        result = exporter.export(engine, session_id=sid)

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
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
            _DEFAULT_OTLP_METRICS_ENDPOINT,
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
    ) -> dict[str, Any]:
        """Export metrics from an AuditEngine via OTLP/HTTP."""
        return export_otlp_metrics_http(
            engine,
            endpoint=self.endpoint,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
            headers=self.headers,
            timeout=self.timeout,
            resource_attrs=self.resource_attrs,
        )

    def export_to_string(
        self,
        engine: AuditEngine,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 10_000,
    ) -> str:
        """Export metrics as an OTLP/JSON string."""
        return export_otlp_metrics_to_string(
            engine,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
            resource_attrs=self.resource_attrs,
        )
