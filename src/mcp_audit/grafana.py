"""Grafana dashboard JSON generation for mcp-audit.

Generates ready-to-import Grafana dashboard JSON that visualizes all
``mcp_audit_*`` Prometheus metrics produced by :mod:`prometheus`.

This completes the monitoring pipeline:

    mcp-audit → Prometheus exposition → Prometheus scrape → Grafana dashboard

Why a dashboard generator?
---------------------------

Prometheus gives you raw metrics; Grafana gives you **visualizations**. But
building a Grafana dashboard from scratch is tedious — you need to know panel
types, PromQL queries, thresholds, grid layouts, and templating. This module
auto-generates a professional dashboard JSON tuned specifically for
``mcp_audit_*`` metrics, so an operator can:

1. Start the mcp-audit Prometheus exporter
2. Configure Prometheus to scrape it
3. Import the generated dashboard JSON into Grafana
4. See agent observability dashboards immediately

Dashboard structure
-------------------

The generated dashboard contains these rows/sections:

+---------------------------------------------+-------------------+
| Section                                     | Panels            |
+=============================================+===================+
| Overview                                    | Call rate,        |
|                                             | Error rate,       |
|                                             | Total cost,       |
|                                             | Session count     |
+---------------------------------------------+-------------------+
| Latency                                     | Duration histogram |
|                                             | (heatmap), p50/   |
|                                             | p95/p99 trends    |
+---------------------------------------------+-------------------+
| Cost                                        | Cost histogram,   |
|                                             | avg cost trend,   |
|                                             | cost by tool      |
+---------------------------------------------+-------------------+
| Token Usage                                 | Input tokens,     |
|                                             | Output tokens,    |
|                                             | total tokens by   |
|                                             | tool              |
+---------------------------------------------+-------------------+
| Per-Tool Breakdown                          | Table of all      |
|                                             | metrics grouped   |
|                                             | by tool           |
+---------------------------------------------+-------------------+

Usage
-----

.. code-block:: python

    from mcp_audit.grafana import generate_dashboard_json, save_dashboard

    # Generate dashboard JSON
    json_str = generate_dashboard_json()

    # Save to file for manual import
    save_dashboard("/tmp/mcp_audit_dashboard.json")

    # Or with custom datasource
    json_str = generate_dashboard_json(
        datasource="Prometheus",
        refresh_interval="10s",
        time_range="now-6h",
    )
"""
from __future__ import annotations

import json
import os
from typing import Any

# ── Constants ────────────────────────────────────────────────────────

_GRAFANA_VERSION = "1.0"

# Panel grid positions (Grafana uses 24-column grid)
_GRID_TOTAL_W = 24


def _uid() -> str:
    """Generate a stable dashboard UID."""
    return "mcp-audit-overview"


def _make_target(
    expr: str,
    legend_format: str = "",
    ref_id: str = "A",
) -> dict[str, Any]:
    """Create a Prometheus query target for a Grafana panel."""
    target: dict[str, Any] = {
        "expr": expr,
        "intervalFactor": 1,
        "refId": ref_id,
    }
    if legend_format:
        target["legendFormat"] = legend_format
    return target


def _make_panel(
    *,
    title: str,
    panel_type: str,
    targets: list[dict[str, Any]],
    grid_pos: dict[str, int],
    datasource: str,
    description: str = "",
    unit: str = "",
    thresholds: list[dict[str, Any]] | None = None,
    field_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a Grafana panel definition.

    Parameters
    ----------
    title
        Panel title shown in the dashboard.
    panel_type
        Grafana panel type: ``stat``, ``timeseries``, ``heatmap``, ``table``,
        ``bargauge``, ``gauge``.
    targets
        List of query target dicts (from :func:`_make_target`).
    grid_pos
        Grid position ``{"x": n, "y": n, "w": n, "h": n}``.
    datasource
        Grafana datasource name (e.g. ``"Prometheus"``).
    description
        Optional panel description shown on hover.
    unit
        Grafana unit shorthand (e.g. ``"ms"``, ``"short"``, ``"percent"``,
        ``"currencyusd"``).
    thresholds
        Optional list of threshold dicts for stat/gauge panels.
    field_config
        Optional field config overrides.
    """
    panel: dict[str, Any] = {
        "title": title,
        "type": panel_type,
        "datasource": {"type": "prometheus", "uid": datasource} if datasource != "default"
        else None,
        "targets": targets,
        "gridPos": grid_pos,
        "options": {},
    }

    if description:
        panel["description"] = description

    # Build field config
    fc: dict[str, Any] = {
        "defaults": {},
        "overrides": [],
    }
    if unit:
        fc["defaults"]["unit"] = unit
    if thresholds:
        fc["defaults"]["thresholds"] = {
            "mode": "absolute",
            "steps": [
                {"color": "green", "value": None},
            ]
            + thresholds,
        }
    if field_config:
        fc["defaults"].update(field_config)

    panel["fieldConfig"] = fc

    # Panel-specific options
    if panel_type == "stat":
        panel["options"] = {
            "reduceOptions": {
                "values": False,
                "calcs": ["lastNotNull"],
                "fields": "",
            },
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
        }
    elif panel_type == "timeseries":
        panel["options"] = {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
                "showLegend": True,
            },
            "tooltip": {
                "mode": "single",
                "sort": "none",
            },
        }
    elif panel_type == "heatmap":
        panel["options"] = {
            "calculate": False,
            "cellGap": 1,
            "color": {
                "scheme": "Spectral",
                "mode": "scheme",
                "fill": "dark-orange",
                "reverse": True,
            },
            "yAxis": {
                "axisPlacement": "left",
                "unit": "ms",
            },
        }
    elif panel_type == "table":
        panel["options"] = {
            "showHeader": True,
            "cellHeight": "sm",
            "footer": {"show": False},
        }
    elif panel_type == "bargauge":
        panel["options"] = {
            "reduceOptions": {
                "values": False,
                "calcs": ["lastNotNull"],
                "fields": "",
            },
            "orientation": "horizontal",
            "displayMode": "gradient",
            "showUnfilled": True,
        }
    elif panel_type == "gauge":
        panel["options"] = {
            "reduceOptions": {
                "values": False,
                "calcs": ["lastNotNull"],
                "fields": "",
            },
            "orientation": "auto",
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
        }

    return panel


# ── Panel builders by section ────────────────────────────────────────


def _build_overview_panels(datasource: str) -> list[dict[str, Any]]:
    """Build overview stat panels: call rate, error rate, cost, sessions."""
    panels: list[dict[str, Any]] = []

    # Row 1: 4 stat panels side by side (each 6 wide)
    panels.append(
        _make_panel(
            title="Total Tool Calls",
            panel_type="stat",
            datasource=datasource,
            description="Sum of all tool calls across all tools",
            unit="short",
            targets=[
                _make_target(
                    expr="sum(mcp_audit_tool_calls_total)",
                    legend_format="Total Calls",
                )
            ],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "yellow", "value": 1000},
                {"color": "red", "value": 10000},
            ],
        )
    )

    panels.append(
        _make_panel(
            title="Error Rate",
            panel_type="stat",
            datasource=datasource,
            description="Overall error rate as a percentage",
            unit="percent",
            targets=[
                _make_target(
                    expr="mcp_audit_error_rate",
                    legend_format="Error Rate",
                )
            ],
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "yellow", "value": 5},
                {"color": "red", "value": 10},
            ],
        )
    )

    panels.append(
        _make_panel(
            title="Total Cost (USD)",
            panel_type="stat",
            datasource=datasource,
            description="Total cost across all tool calls",
            unit="currencyUSD",
            targets=[
                _make_target(
                    expr="mcp_audit_total_cost_usd",
                    legend_format="Total Cost",
                )
            ],
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
            thresholds=[
                {"color": "yellow", "value": 10},
                {"color": "red", "value": 100},
            ],
        )
    )

    panels.append(
        _make_panel(
            title="Active Sessions",
            panel_type="stat",
            datasource=datasource,
            description="Number of currently active audit sessions",
            unit="short",
            targets=[
                _make_target(
                    expr='mcp_audit_sessions{scope="active"}',
                    legend_format="Active",
                )
            ],
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
        )
    )

    # Row 2: Calls over time + Errors over time
    panels.append(
        _make_panel(
            title="Tool Calls Over Time",
            panel_type="timeseries",
            datasource=datasource,
            description="Rate of tool calls per tool over time",
            unit="short",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_calls_total[5m])) by (tool)",
                    legend_format="{{tool}}",
                )
            ],
            grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
        )
    )

    panels.append(
        _make_panel(
            title="Errors Over Time",
            panel_type="timeseries",
            datasource=datasource,
            description="Rate of errors per tool over time",
            unit="short",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_errors_total[5m])) by (tool)",
                    legend_format="{{tool}}",
                )
            ],
            grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
        )
    )

    return panels


def _build_latency_panels(datasource: str, y_offset: int) -> list[dict[str, Any]]:
    """Build latency analysis panels."""
    panels: list[dict[str, Any]] = []

    # Duration heatmap
    panels.append(
        _make_panel(
            title="Call Latency Distribution",
            panel_type="heatmap",
            datasource=datasource,
            description="Distribution of tool call latency (ms) over time",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_duration_ms_bucket[5m])) by (le)",
                    legend_format="{{le}}",
                )
            ],
            grid_pos={"x": 0, "y": y_offset, "w": 12, "h": 8},
        )
    )

    # p50, p95, p99 trend
    panels.append(
        _make_panel(
            title="Latency Percentiles (p50, p95, p99)",
            panel_type="timeseries",
            datasource=datasource,
            description="Key latency percentiles across all tools",
            unit="ms",
            targets=[
                _make_target(
                    expr=(
                        "histogram_quantile(0.50, "
                        "sum(rate(mcp_audit_tool_duration_ms_bucket[5m])) by (le))"
                    ),
                    legend_format="p50",
                    ref_id="A",
                ),
                _make_target(
                    expr=(
                        "histogram_quantile(0.95, "
                        "sum(rate(mcp_audit_tool_duration_ms_bucket[5m])) by (le))"
                    ),
                    legend_format="p95",
                    ref_id="B",
                ),
                _make_target(
                    expr=(
                        "histogram_quantile(0.99, "
                        "sum(rate(mcp_audit_tool_duration_ms_bucket[5m])) by (le))"
                    ),
                    legend_format="p99",
                    ref_id="C",
                ),
            ],
            grid_pos={"x": 12, "y": y_offset, "w": 12, "h": 8},
        )
    )

    return panels


def _build_cost_panels(datasource: str, y_offset: int) -> list[dict[str, Any]]:
    """Build cost analysis panels."""
    panels: list[dict[str, Any]] = []

    # Cost by tool (bar gauge)
    panels.append(
        _make_panel(
            title="Cost by Tool",
            panel_type="bargauge",
            datasource=datasource,
            description="Total cost (USD) broken down by tool",
            unit="currencyUSD",
            targets=[
                _make_target(
                    expr="sum(mcp_audit_total_cost_usd) by (tool)",
                    legend_format="{{tool}}",
                )
            ],
            grid_pos={"x": 0, "y": y_offset, "w": 12, "h": 8},
        )
    )

    # Cost histogram
    panels.append(
        _make_panel(
            title="Cost Distribution",
            panel_type="heatmap",
            datasource=datasource,
            description="Distribution of per-call cost (USD) over time",
            unit="currencyUSD",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_cost_usd_bucket[5m])) by (le)",
                    legend_format="{{le}}",
                )
            ],
            grid_pos={"x": 12, "y": y_offset, "w": 12, "h": 8},
        )
    )

    return panels


def _build_token_panels(datasource: str, y_offset: int) -> list[dict[str, Any]]:
    """Build token usage panels."""
    panels: list[dict[str, Any]] = []

    # Input tokens over time
    panels.append(
        _make_panel(
            title="Input Tokens by Tool",
            panel_type="timeseries",
            datasource=datasource,
            description="Input token consumption rate by tool",
            unit="short",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_input_tokens[5m])) by (tool)",
                    legend_format="{{tool}}",
                )
            ],
            grid_pos={"x": 0, "y": y_offset, "w": 12, "h": 8},
        )
    )

    # Output tokens over time
    panels.append(
        _make_panel(
            title="Output Tokens by Tool",
            panel_type="timeseries",
            datasource=datasource,
            description="Output token production rate by tool",
            unit="short",
            targets=[
                _make_target(
                    expr="sum(rate(mcp_audit_tool_output_tokens[5m])) by (tool)",
                    legend_format="{{tool}}",
                )
            ],
            grid_pos={"x": 12, "y": y_offset, "w": 12, "h": 8},
        )
    )

    return panels


def _build_breakdown_panels(datasource: str, y_offset: int) -> list[dict[str, Any]]:
    """Build per-tool breakdown table panel."""
    panels: list[dict[str, Any]] = []

    panels.append(
        _make_panel(
            title="Per-Tool Breakdown",
            panel_type="table",
            datasource=datasource,
            description="All metrics grouped by tool name",
            targets=[
                _make_target(
                    expr=(
                        'label_replace(mcp_audit_tool_calls_total, "metric", "calls", "__name__", ".*")'
                        ' or label_replace(mcp_audit_tool_errors_total, "metric", "errors", "__name__", ".*")'
                    ),
                    legend_format="{{tool}} - {{metric}}",
                ),
                _make_target(
                    expr="mcp_audit_tool_calls_total",
                    legend_format="{{tool}} calls",
                    ref_id="A",
                ),
                _make_target(
                    expr="mcp_audit_tool_errors_total",
                    legend_format="{{tool}} errors",
                    ref_id="B",
                ),
                _make_target(
                    expr="mcp_audit_tool_tokens_total",
                    legend_format="{{tool}} tokens",
                    ref_id="C",
                ),
            ],
            grid_pos={"x": 0, "y": y_offset, "w": _GRID_TOTAL_W, "h": 6},
        )
    )

    return panels


def _build_templating() -> dict[str, Any]:
    """Build templating variables for the dashboard."""
    return {
        "list": [
            {
                "name": "tool",
                "type": "query",
                "label": "Tool",
                "datasource": {"type": "prometheus"},
                "query": "label_values(mcp_audit_tool_calls_total, tool)",
                "refresh": 2,  # On dashboard load
                "sort": 1,  # Alphabetical
                "multi": True,
                "includeAll": True,
                "current": {
                    "selected": True,
                    "text": "All",
                    "value": "$__all",
                },
            }
        ]
    }


def _build_annotations() -> dict[str, Any]:
    """Build annotations configuration."""
    return {
        "list": [
            {
                "builtIn": 1,
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "hide": True,
                "iconColor": "rgba(0, 211, 255, 1)",
                "name": "Annotations & Alerts",
                "type": "dashboard",
            }
        ]
    }


# ── Main dashboard builder ───────────────────────────────────────────


def generate_dashboard_json(
    *,
    title: str = "MCP Audit — Agent Observability",
    datasource: str = "default",
    refresh_interval: str = "10s",
    time_range: str = "now-6h",
    timezone: str = "browser",
    tags: list[str] | None = None,
) -> str:
    """Generate a complete Grafana dashboard JSON string.

    Parameters
    ----------
    title
        Dashboard title shown in Grafana.
    datasource
        Grafana datasource UID or name. Use ``"default"`` for the default
        Prometheus datasource, or pass a specific datasource name/UID.
    refresh_interval
        Auto-refresh interval (e.g. ``"10s"``, ``"1m"``, ``"5m"``).
    time_range
        Default time range (e.g. ``"now-6h"``, ``"now-1d"``).
    timezone
        Dashboard timezone (``"browser"``, ``"utc"``).
    tags
        List of tags for the dashboard.

    Returns
    -------
    str
        A JSON string containing the complete Grafana dashboard definition,
        ready to be imported via Grafana's **Import Dashboard** feature.
    """
    if tags is None:
        tags = ["mcp", "mcp-audit", "agent-observability", "ai-agents"]

    # Build all panels, tracking Y position
    panels: list[dict[str, Any]] = []

    # Overview section
    panels.extend(_build_overview_panels(datasource))
    current_y = 12  # After 4 stat panels (h=4) + 2 timeseries (h=8)

    # Latency section
    panels.extend(_build_latency_panels(datasource, y_offset=current_y))
    current_y += 8

    # Cost section
    panels.extend(_build_cost_panels(datasource, y_offset=current_y))
    current_y += 8

    # Token section
    panels.extend(_build_token_panels(datasource, y_offset=current_y))
    current_y += 8

    # Breakdown table
    panels.extend(_build_breakdown_panels(datasource, y_offset=current_y))

    dashboard: dict[str, Any] = {
        "title": title,
        "uid": _uid(),
        "schemaVersion": 39,
        "version": 1,
        "timezone": timezone,
        "refresh": refresh_interval,
        "time": {
            "from": time_range,
            "to": "now",
        },
        "tags": tags,
        "templating": _build_templating(),
        "annotations": _build_annotations(),
        "panels": panels,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,  # Shared crosshair
        "links": [],
        "liveNow": False,
        "weekStart": "",
    }

    return json.dumps(dashboard, indent=2, ensure_ascii=False)


def generate_dashboard_dict(
    *,
    title: str = "MCP Audit — Agent Observability",
    datasource: str = "default",
    refresh_interval: str = "10s",
    time_range: str = "now-6h",
    timezone: str = "browser",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Generate the dashboard as a Python dict instead of a JSON string.

    Useful for programmatic manipulation or API-based import.
    """
    return json.loads(
        generate_dashboard_json(
            title=title,
            datasource=datasource,
            refresh_interval=refresh_interval,
            time_range=time_range,
            timezone=timezone,
            tags=tags,
        )
    )


def save_dashboard(
    output_path: str | None = None,
    *,
    title: str = "MCP Audit — Agent Observability",
    datasource: str = "default",
    refresh_interval: str = "10s",
    time_range: str = "now-6h",
    timezone: str = "browser",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Write the Grafana dashboard JSON to a file.

    Parameters
    ----------
    output_path
        File path. Falls back to ``MCP_AUDIT_GRAFANA_OUTPUT`` env var,
        then ``./mcp_audit_dashboard.json``.
    title, datasource, refresh_interval, time_range, timezone, tags
        Same as :func:`generate_dashboard_json`.

    Returns
    -------
    dict
        ``{"status": "ok", "path": ..., "panels": ..., "bytes": ...}``
    """
    path = output_path or os.environ.get("MCP_AUDIT_GRAFANA_OUTPUT", "mcp_audit_dashboard.json")

    dashboard_json = generate_dashboard_json(
        title=title,
        datasource=datasource,
        refresh_interval=refresh_interval,
        time_range=time_range,
        timezone=timezone,
        tags=tags,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(dashboard_json)

    dashboard = json.loads(dashboard_json)
    panel_count = len(dashboard.get("panels", []))

    return {
        "status": "ok",
        "path": path,
        "panels": panel_count,
        "bytes": len(dashboard_json.encode("utf-8")),
    }


def export_dashboard_http(
    grafana_url: str | None = None,
    api_key: str | None = None,
    *,
    title: str = "MCP Audit — Agent Observability",
    datasource: str = "default",
    overwrite: bool = True,
) -> dict[str, Any]:
    """Import the dashboard directly into a Grafana instance via HTTP API.

    Uses the Grafana HTTP API endpoint ``POST /api/dashboards/db``.

    Parameters
    ----------
    grafana_url
        Base URL of the Grafana instance (e.g. ``http://localhost:3000``).
        Falls back to ``MCP_AUDIT_GRAFANA_URL`` env var.
    api_key
        Grafana API key or service account token. Falls back to
        ``MCP_AUDIT_GRAFANA_KEY`` env var.
    title
        Dashboard title.
    datasource
        Datasource to use.
    overwrite
        If True, overwrite existing dashboard with same UID.

    Returns
    -------
    dict
        ``{"status": "ok"/"error", ...}``
    """
    import urllib.request
    import urllib.error

    if not grafana_url:
        grafana_url = os.environ.get("MCP_AUDIT_GRAFANA_URL", "")
    if not api_key:
        api_key = os.environ.get("MCP_AUDIT_GRAFANA_KEY", "")

    if not grafana_url:
        return {
            "status": "error",
            "error": "No Grafana URL configured. Set grafana_url= or MCP_AUDIT_GRAFANA_URL env var.",
        }
    if not api_key:
        return {
            "status": "error",
            "error": "No Grafana API key configured. Set api_key= or MCP_AUDIT_GRAFANA_KEY env var.",
        }

    dashboard_dict = generate_dashboard_dict(
        title=title,
        datasource=datasource,
    )

    # Grafana API expects {"dashboard": {...}, "overwrite": true}
    payload = json.dumps(
        {
            "dashboard": dashboard_dict,
            "folderId": 0,
            "overwrite": overwrite,
        }
    ).encode("utf-8")

    url = grafana_url.rstrip("/") + "/api/dashboards/db"

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        resp = urllib.request.urlopen(req, timeout=15)
        body = json.loads(resp.read().decode("utf-8"))
        return {
            "status": "ok",
            "url": body.get("url", ""),
            "slug": body.get("slug", ""),
            "version": body.get("version", ""),
            "id": body.get("id", ""),
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
            "response": body[:500],
        }
    except urllib.error.URLError as e:
        return {
            "status": "error",
            "error": str(e.reason),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


class GrafanaDashboardExporter:
    """Reusable Grafana dashboard exporter with env-var configuration.

    Reads configuration from environment variables:

    - ``MCP_AUDIT_GRAFANA_URL``: Grafana instance URL for :meth:`import_to_grafana`.
    - ``MCP_AUDIT_GRAFANA_KEY``: Grafana API key/token.
    - ``MCP_AUDIT_GRAFANA_OUTPUT``: Default file path for :meth:`save`.

    Example::

        exporter = GrafanaDashboardExporter()
        json_str = exporter.render()           # → dashboard JSON
        exporter.save("/tmp/dashboard.json")   # → write to file
        result = exporter.import_to_grafana()  # → POST to Grafana API
    """

    def __init__(
        self,
        *,
        grafana_url: str | None = None,
        api_key: str | None = None,
        datasource: str = "default",
    ) -> None:
        self.grafana_url = grafana_url or os.environ.get("MCP_AUDIT_GRAFANA_URL", "")
        self.api_key = api_key or os.environ.get("MCP_AUDIT_GRAFANA_KEY", "")
        self.datasource = datasource

    def render(
        self,
        *,
        title: str = "MCP Audit — Agent Observability",
        refresh_interval: str = "10s",
        time_range: str = "now-6h",
    ) -> str:
        """Return the dashboard as a JSON string."""
        return generate_dashboard_json(
            title=title,
            datasource=self.datasource,
            refresh_interval=refresh_interval,
            time_range=time_range,
        )

    def save(
        self,
        output_path: str | None = None,
        *,
        title: str = "MCP Audit — Agent Observability",
    ) -> dict[str, Any]:
        """Write dashboard to a file."""
        path = output_path or os.environ.get("MCP_AUDIT_GRAFANA_OUTPUT", "mcp_audit_dashboard.json")
        return save_dashboard(
            path,
            title=title,
            datasource=self.datasource,
        )

    def import_to_grafana(
        self,
        *,
        title: str = "MCP Audit — Agent Observability",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Import dashboard directly to a Grafana instance via HTTP API."""
        return export_dashboard_http(
            grafana_url=self.grafana_url,
            api_key=self.api_key,
            title=title,
            datasource=self.datasource,
            overwrite=overwrite,
        )
