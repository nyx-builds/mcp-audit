"""Prometheus text exposition format export for mcp-audit.

Converts audit data into the `Prometheus text exposition format
<https://prometheus.io/docs/instrumenting/exposition_formats/#text-format-details>`_
so that any Prometheus server (or compatible scraper such as Grafana Agent,
VictoriaMetrics, or Datadog Agent) can scrape mcp-audit metrics directly —
**without** an OpenTelemetry Collector in the middle.

Why Prometheus exposition?
--------------------------

OTLP (the OpenTelemetry Line Protocol, provided by :mod:`metrics` and :mod:`otlp`)
is the modern standard, but it requires a running OTel Collector to translate
OTLP into a format Prometheus can scrape. Many teams already have a Prometheus
server and want to add agent observability to their existing dashboards and
alerting rules without standing up new infrastructure. This module gives them
that capability.

Metric families produced
------------------------

All metric names are prefixed with ``mcp_audit_`` and follow Prometheus
naming conventions (snake_case, ``_total`` suffix for counters).

+-----------------------------------+-----------+----------------------------------------+
| Metric                            | Type      | Description                            |
+===================================+===========+========================================+
| ``mcp_audit_tool_calls_total``    | Counter   | Total tool calls by tool               |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_errors_total``   | Counter   | Error calls by tool                    |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_duration_ms_*``  | Histogram | Call latency distribution by tool      |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_cost_usd_*``     | Histogram | Per-call cost distribution by tool     |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_tokens_total``   | Counter   | Total tokens (input + output) by tool  |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_input_tokens``   | Counter   | Input tokens by tool                   |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_tool_output_tokens``  | Counter   | Output tokens by tool                  |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_sessions``            | Gauge     | Session count (scope label: all/active)|
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_error_rate``          | Gauge     | Overall error rate (%)                 |
+-----------------------------------+-----------+----------------------------------------+
| ``mcp_audit_total_cost_usd``      | Gauge     | Total cost in USD                      |
+-----------------------------------+-----------+----------------------------------------+

Usage
-----

Direct call::

    from mcp_audit import AuditEngine
    from mcp_audit.prometheus import build_prometheus_exposition

    engine = AuditEngine()
    # ... record calls ...
    text = build_prometheus_exposition(engine)
    print(text)  # → Prometheus-format text

Export to file::

    from mcp_audit.prometheus import export_prometheus_file

    export_prometheus_file(engine, "/tmp/mcp_audit_metrics.prom")

Serve as a Prometheus endpoint (using any HTTP framework)::

    from mcp_audit.prometheus import build_prometheus_exposition

    # Flask example
    @app.route("/metrics")
    def metrics():
        return build_prometheus_exposition(engine), 200, {
            "Content-Type": "text/plain; version=0.0.4; charset=utf-8"
        }
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable

from .engine import AuditEngine
from .models import CallStatus, ToolCall

# ── Constants ────────────────────────────────────────────────────────

_PROM_VERSION = "0.0.4"

# Histogram bucket boundaries (must be le-buckets in Prometheus)
_LATENCY_BOUNDS = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]
_COST_BOUNDS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

# Content-Type for Prometheus scraping
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


# ── Helpers ──────────────────────────────────────────────────────────


def _escape_label_value(value: str) -> str:
    """Escape a string for use inside a Prometheus label value.

    Per the spec: backslash, double-quote, and newline are escaped.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_label_set(labels: dict[str, str]) -> str:
    """Format a label dict as ``{key="value", ...}`` or empty string."""
    if not labels:
        return ""
    parts = []
    for key in sorted(labels.keys()):
        parts.append(f'{key}="{_escape_label_value(str(labels[key]))}"')
    return "{" + ", ".join(parts) + "}"


def _format_label_set_with_suffix(
    labels: dict[str, str], extra: dict[str, str]
) -> str:
    """Merge *labels* and *extra* dicts, then format as a label set."""
    merged = {**labels, **extra}
    return _format_label_set(merged)


def _format_value(value: float | int) -> str:
    """Format a numeric value for Prometheus.

    Integers print without a decimal point; floats print with maximum
    precision. ``inf``/``nan`` become ``+Inf``/``NaN`` per the spec.
    """
    if isinstance(value, int):
        return str(value)
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    # Avoid scientific notation for small numbers
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def _bucketize_cumulative(
    values: list[float], bounds: list[float]
) -> tuple[list[int], list[float]]:
    """Distribute *values* into cumulative histogram buckets.

    Prometheus histograms use **cumulative** bucket counts (each bucket
    includes all observations ≤ its upper bound), unlike OTLP which uses
    explicit per-bucket counts.

    Returns ``(cumulative_counts, bounds)`` where ``len(cumulative_counts) =
    len(bounds) + 1`` (the ``+Inf`` bucket at the end).
    """
    if not bounds:
        return [len(values)], []

    counts = [0] * (len(bounds) + 1)
    for v in values:
        placed = False
        for i, b in enumerate(bounds):
            if v <= b:
                # Increment all buckets >= i (cumulative)
                for j in range(i, len(bounds) + 1):
                    counts[j] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1  # +Inf bucket
    return counts, bounds


# ── Core builder ─────────────────────────────────────────────────────


def build_prometheus_exposition(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    include_timestamps: bool = False,
) -> str:
    """Build Prometheus text exposition format from engine data.

    Parameters
    ----------
    engine
        The audit engine to read data from.
    session_id, agent_id, tool_name
        Optional filters (same semantics as ``query_calls``).
    limit
        Maximum number of calls to aggregate.
    include_timestamps
        If ``True``, append a millisecond timestamp to each sample
        (Prometheus optional feature — useful for push gateway scenarios).

    Returns
    -------
    str
        A complete Prometheus exposition-format text block, ready to be
        served on a ``/metrics`` endpoint or written to a file.
    """
    lines: list[str] = []

    # ── Header comment ──
    lines.append(f"# mcp-audit prometheus exposition (format version {_PROM_VERSION})")
    lines.append("")

    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    if not calls:
        lines.extend(_build_empty_exposition())
        return "\n".join(lines) + "\n"

    # Group calls by tool_name for per-tool metrics
    by_tool: dict[str, list[ToolCall]] = {}
    for call in calls:
        by_tool.setdefault(call.tool_name, []).append(call)

    # ── Counter: mcp_audit_tool_calls_total ──
    lines.append("# HELP mcp_audit_tool_calls_total Total number of tool calls by tool name")
    lines.append("# TYPE mcp_audit_tool_calls_total counter")
    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        labels = {"tool": tool}
        lines.append(f"mcp_audit_tool_calls_total{_format_label_set(labels)} {len(tool_calls)}")
    lines.append("")

    # ── Counter: mcp_audit_tool_errors_total ──
    lines.append("# HELP mcp_audit_tool_errors_total Number of failed tool calls by tool name")
    lines.append("# TYPE mcp_audit_tool_errors_total counter")
    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        error_count = sum(1 for c in tool_calls if c.is_error)
        if error_count > 0:
            labels = {"tool": tool}
            lines.append(f"mcp_audit_tool_errors_total{_format_label_set(labels)} {error_count}")
    # Also emit a 0-value line if no errors at all (so Prometheus sees the metric)
    total_errors = sum(1 for c in calls if c.is_error)
    if total_errors == 0:
        lines.append(f'mcp_audit_tool_errors_total{{tool=""}} 0')
    lines.append("")

    # ── Counter: mcp_audit_tool_input_tokens ──
    lines.append("# HELP mcp_audit_tool_input_tokens Total input tokens consumed by tool name")
    lines.append("# TYPE mcp_audit_tool_input_tokens counter")
    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        total_input = sum(c.input_tokens for c in tool_calls)
        labels = {"tool": tool}
        lines.append(f"mcp_audit_tool_input_tokens{_format_label_set(labels)} {total_input}")
    lines.append("")

    # ── Counter: mcp_audit_tool_output_tokens ──
    lines.append("# HELP mcp_audit_tool_output_tokens Total output tokens produced by tool name")
    lines.append("# TYPE mcp_audit_tool_output_tokens counter")
    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        total_output = sum(c.output_tokens for c in tool_calls)
        labels = {"tool": tool}
        lines.append(f"mcp_audit_tool_output_tokens{_format_label_set(labels)} {total_output}")
    lines.append("")

    # ── Counter: mcp_audit_tool_tokens_total (combined input + output) ──
    lines.append("# HELP mcp_audit_tool_tokens_total Total tokens (input + output) by tool name")
    lines.append("# TYPE mcp_audit_tool_tokens_total counter")
    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        total_tokens = sum(c.input_tokens + c.output_tokens for c in tool_calls)
        labels = {"tool": tool}
        lines.append(f"mcp_audit_tool_tokens_total{_format_label_set(labels)} {total_tokens}")
    lines.append("")

    # ── Histogram: mcp_audit_tool_duration_ms ──
    lines.extend(
        _build_histogram_lines(
            metric_name="mcp_audit_tool_duration_ms",
            help_text="Distribution of tool call latency in milliseconds",
            by_tool=by_tool,
            bounds=_LATENCY_BOUNDS,
            extractor=lambda c: c.duration_ms,
            filter_fn=lambda v: v is not None,
            cast_fn=float,
        )
    )

    # ── Histogram: mcp_audit_tool_cost_usd ──
    lines.extend(
        _build_histogram_lines(
            metric_name="mcp_audit_tool_cost_usd",
            help_text="Distribution of per-call cost in USD by tool name",
            by_tool=by_tool,
            bounds=_COST_BOUNDS,
            extractor=lambda c: c.cost_usd,
            filter_fn=lambda v: v > 0,
            cast_fn=float,
        )
    )

    # ── Gauge: mcp_audit_sessions ──
    sessions = engine.list_sessions(limit=100_000)
    active_sessions = [s for s in sessions if s.is_active]
    lines.append("# HELP mcp_audit_sessions Number of audit sessions")
    lines.append("# TYPE mcp_audit_sessions gauge")
    lines.append(f'mcp_audit_sessions{{scope="all"}} {len(sessions)}')
    lines.append(f'mcp_audit_sessions{{scope="active"}} {len(active_sessions)}')
    lines.append("")

    # ── Gauge: mcp_audit_error_rate ──
    error_calls = sum(1 for c in calls if c.is_error)
    error_rate = (error_calls / len(calls) * 100) if calls else 0.0
    lines.append("# HELP mcp_audit_error_rate Overall error rate across matching calls (percent)")
    lines.append("# TYPE mcp_audit_error_rate gauge")
    lines.append(f"mcp_audit_error_rate {_format_value(error_rate)}")
    lines.append("")

    # ── Gauge: mcp_audit_total_cost_usd ──
    total_cost = sum(c.cost_usd for c in calls)
    lines.append("# HELP mcp_audit_total_cost_usd Total cost in USD across all matching calls")
    lines.append("# TYPE mcp_audit_total_cost_usd gauge")
    lines.append(f"mcp_audit_total_cost_usd {_format_value(total_cost)}")
    lines.append("")

    # ── Gauge: mcp_audit_avg_duration_ms ──
    durations = [c.duration_ms for c in calls if c.duration_ms is not None]
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    lines.append("# HELP mcp_audit_avg_duration_ms Average call duration in milliseconds")
    lines.append("# TYPE mcp_audit_avg_duration_ms gauge")
    lines.append(f"mcp_audit_avg_duration_ms {_format_value(avg_duration)}")
    lines.append("")

    # ── Gauge: mcp_audit_avg_cost_usd ──
    avg_cost = total_cost / len(calls) if calls else 0.0
    lines.append("# HELP mcp_audit_avg_cost_usd Average cost per call in USD")
    lines.append("# TYPE mcp_audit_avg_cost_usd gauge")
    lines.append(f"mcp_audit_avg_cost_usd {_format_value(avg_cost)}")
    lines.append("")

    return "\n".join(lines) + "\n"


def _build_histogram_lines(
    *,
    metric_name: str,
    help_text: str,
    by_tool: dict[str, list[ToolCall]],
    bounds: list[float],
    extractor: Any,
    filter_fn: Any,
    cast_fn: type,
) -> list[str]:
    """Build Prometheus histogram metric lines for a given value extractor.

    Generates ``_bucket``, ``_sum``, and ``_count`` series per tool.
    """
    lines: list[str] = []
    lines.append(f"# HELP {metric_name} {help_text}")
    lines.append(f"# TYPE {metric_name} histogram")

    for tool in sorted(by_tool.keys()):
        tool_calls = by_tool[tool]
        values: list[float] = []
        for c in tool_calls:
            raw = extractor(c)
            if raw is not None and filter_fn(raw):
                values.append(cast_fn(raw))

        cum_counts, explicit_bounds = _bucketize_cumulative(values, bounds)

        # _bucket lines (including +Inf)
        for i, count in enumerate(cum_counts):
            if i < len(explicit_bounds):
                le_val = _format_value(explicit_bounds[i])
            else:
                le_val = "+Inf"
            labels = {"tool": tool, "le": le_val}
            lines.append(f"{metric_name}_bucket{_format_label_set(labels)} {count}")

        # _sum and _count
        total_sum = sum(values) if values else 0.0
        count = len(values)
        base_labels = {"tool": tool}
        lines.append(f"{metric_name}_sum{_format_label_set(base_labels)} {_format_value(total_sum)}")
        lines.append(f"{metric_name}_count{_format_label_set(base_labels)} {count}")

    lines.append("")
    return lines


def _build_empty_exposition() -> list[str]:
    """Return minimal metric lines for the empty-engine case.

    This ensures Prometheus always sees valid metric definitions even
    when no calls have been recorded yet.
    """
    return [
        "# HELP mcp_audit_tool_calls_total Total number of tool calls by tool name",
        "# TYPE mcp_audit_tool_calls_total counter",
        'mcp_audit_tool_calls_total{tool=""} 0',
        "",
        "# HELP mcp_audit_tool_errors_total Number of failed tool calls by tool name",
        "# TYPE mcp_audit_tool_errors_total counter",
        'mcp_audit_tool_errors_total{tool=""} 0',
        "",
        "# HELP mcp_audit_sessions Number of audit sessions",
        "# TYPE mcp_audit_sessions gauge",
        'mcp_audit_sessions{scope="all"} 0',
        'mcp_audit_sessions{scope="active"} 0',
        "",
        "# HELP mcp_audit_error_rate Overall error rate across matching calls (percent)",
        "# TYPE mcp_audit_error_rate gauge",
        "mcp_audit_error_rate 0",
        "",
        "# HELP mcp_audit_total_cost_usd Total cost in USD across all matching calls",
        "# TYPE mcp_audit_total_cost_usd gauge",
        "mcp_audit_total_cost_usd 0",
        "",
    ]


# ── Export functions ─────────────────────────────────────────────────


def export_prometheus_file(
    engine: AuditEngine,
    output_path: str,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Write Prometheus exposition text to a file.

    Parameters
    ----------
    engine
        The audit engine to read data from.
    output_path
        File path to write the exposition text to.
    session_id, agent_id, tool_name
        Optional filters.
    limit
        Maximum number of calls to aggregate.

    Returns
    -------
    dict
        ``{"status": "ok", "path": ..., "bytes": ..., "metrics_count": ...}``
    """
    text = build_prometheus_exposition(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Count metric families (non-comment, non-blank lines)
    metric_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#")
    ]

    return {
        "status": "ok",
        "path": output_path,
        "bytes": len(text.encode("utf-8")),
        "metric_lines": len(metric_lines),
    }


def export_prometheus_to_string(
    engine: AuditEngine,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> str:
    """Return Prometheus exposition text as a string.

    Convenience wrapper around :func:`build_prometheus_exposition` for
    consistency with the OTLP metrics API.
    """
    return build_prometheus_exposition(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )


def export_prometheus_http(
    engine: AuditEngine,
    endpoint: str | None = None,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
    timeout: int = 15,
    job_name: str = "mcp-audit",
) -> dict[str, Any]:
    """Push metrics to a Prometheus Pushgateway via HTTP POST.

    This uses the Prometheus Pushgateway HTTP API, which accepts the text
    exposition format at ``/metrics/job/<job_name>``.

    Parameters
    ----------
    engine
        The audit engine to read data from.
    endpoint
        Pushgateway URL (e.g. ``http://localhost:9091``). Falls back to
        the ``MCP_AUDIT_PUSHGATEWAY`` environment variable.
    session_id, agent_id, tool_name
        Optional filters.
    limit
        Maximum number of calls to aggregate.
    timeout
        HTTP timeout in seconds.
    job_name
        Prometheus job label for the pushed metrics.

    Returns
    -------
    dict
        ``{"status": "ok"/"error", ...}``
    """
    import urllib.request
    import urllib.error

    if not endpoint:
        endpoint = os.environ.get("MCP_AUDIT_PUSHGATEWAY", "")
    if not endpoint:
        return {
            "status": "error",
            "error": "No Pushgateway endpoint configured. Set endpoint= or MCP_AUDIT_PUSHGATEWAY env var.",
        }

    text = build_prometheus_exposition(
        engine,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    # Normalize endpoint URL
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/metrics/job/{job_name}"

    try:
        req = urllib.request.Request(
            url,
            data=text.encode("utf-8"),
            method="PUT",
            headers={"Content-Type": PROMETHEUS_CONTENT_TYPE},
        )
        urllib.request.urlopen(req, timeout=timeout)
        return {
            "status": "ok",
            "endpoint": url,
            "bytes": len(text.encode("utf-8")),
        }
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "status": "error",
            "error": f"HTTP {e.code}: {e.reason}",
            "endpoint": url,
            "response": body[:500],
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "error": str(e.reason),
            "endpoint": url,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "endpoint": url,
        }


class PrometheusExporter:
    """Reusable Prometheus exporter with env-var configuration.

    Reads configuration from environment variables:

    - ``MCP_AUDIT_PUSHGATEWAY``: Pushgateway URL for :meth:`push`.
    - ``MCP_AUDIT_PROM_OUTPUT``: Default file path for :meth:`save`.

    Example::

        exporter = PrometheusExporter(engine)
        text = exporter.render()        # → exposition string
        exporter.save("/tmp/metrics.prom")
        result = exporter.push()        # → push to Pushgateway
    """

    def __init__(
        self,
        engine: AuditEngine,
        *,
        pushgateway: str | None = None,
        job_name: str = "mcp-audit",
    ) -> None:
        self.engine = engine
        self.pushgateway = pushgateway or os.environ.get("MCP_AUDIT_PUSHGATEWAY", "")
        self.job_name = job_name

    def render(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 10_000,
    ) -> str:
        """Return the current metrics as Prometheus exposition text."""
        return build_prometheus_exposition(
            self.engine,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
        )

    def save(
        self,
        output_path: str | None = None,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 10_000,
    ) -> dict[str, Any]:
        """Write metrics to a file. Uses MCP_AUDIT_PROM_OUTPUT if path is None."""
        path = output_path or os.environ.get("MCP_AUDIT_PROM_OUTPUT", "")
        if not path:
            return {"status": "error", "error": "No output path provided."}
        return export_prometheus_file(
            self.engine,
            path,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
        )

    def push(
        self,
        *,
        endpoint: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        limit: int = 10_000,
        timeout: int = 15,
    ) -> dict[str, Any]:
        """Push metrics to a Prometheus Pushgateway."""
        return export_prometheus_http(
            self.engine,
            endpoint=endpoint or self.pushgateway,
            session_id=session_id,
            agent_id=agent_id,
            tool_name=tool_name,
            limit=limit,
            timeout=timeout,
            job_name=self.job_name,
        )
