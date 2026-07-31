"""Tests for the Grafana dashboard JSON generation module."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from mcp_audit.grafana import (
    GrafanaDashboardExporter,
    export_dashboard_http,
    generate_dashboard_dict,
    generate_dashboard_json,
    save_dashboard,
)
from mcp_audit.grafana import (
    _build_annotations,
    _build_breakdown_panels,
    _build_cost_panels,
    _build_latency_panels,
    _build_overview_panels,
    _build_token_panels,
    _build_templating,
    _make_panel,
    _make_target,
    _uid,
)


# ── Helper tests ─────────────────────────────────────────────────────


class TestMakeTarget:
    def test_basic_target(self):
        t = _make_target(expr="rate(mcp_audit_tool_calls_total[5m])")
        assert t["expr"] == "rate(mcp_audit_tool_calls_total[5m])"
        assert t["refId"] == "A"
        assert "legendFormat" not in t

    def test_with_legend(self):
        t = _make_target(
            expr="sum(rate(mcp_audit_tool_calls_total[5m])) by (tool)",
            legend_format="{{tool}}",
        )
        assert t["legendFormat"] == "{{tool}}"

    def test_custom_ref_id(self):
        t = _make_target(expr="up", ref_id="B")
        assert t["refId"] == "B"

    def test_interval_factor(self):
        t = _make_target(expr="up")
        assert t["intervalFactor"] == 1


class TestMakePanel:
    def test_basic_stat_panel(self):
        p = _make_panel(
            title="Test",
            panel_type="stat",
            targets=[_make_target(expr="up")],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
            datasource="default",
        )
        assert p["title"] == "Test"
        assert p["type"] == "stat"
        assert p["gridPos"]["w"] == 6

    def test_datasource_assignment(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="my-prom",
        )
        assert p["datasource"]["uid"] == "my-prom"
        assert p["datasource"]["type"] == "prometheus"

    def test_default_datasource_is_none(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert p["datasource"] is None

    def test_unit_assignment(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
            unit="ms",
        )
        assert p["fieldConfig"]["defaults"]["unit"] == "ms"

    def test_description(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
            description="My description",
        )
        assert p["description"] == "My description"

    def test_thresholds(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
            thresholds=[{"color": "yellow", "value": 5}],
        )
        steps = p["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert steps[0]["color"] == "green"
        assert steps[1]["color"] == "yellow"
        assert steps[1]["value"] == 5

    def test_field_config_defaults_empty(self):
        p = _make_panel(
            title="T",
            panel_type="stat",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert p["fieldConfig"]["defaults"] == {}
        assert p["fieldConfig"]["overrides"] == []

    def test_timeseries_options(self):
        p = _make_panel(
            title="T",
            panel_type="timeseries",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert "legend" in p["options"]
        assert "tooltip" in p["options"]

    def test_heatmap_options(self):
        p = _make_panel(
            title="T",
            panel_type="heatmap",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert "color" in p["options"]

    def test_table_options(self):
        p = _make_panel(
            title="T",
            panel_type="table",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert p["options"]["showHeader"] is True

    def test_bargauge_options(self):
        p = _make_panel(
            title="T",
            panel_type="bargauge",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert p["options"]["displayMode"] == "gradient"

    def test_gauge_options(self):
        p = _make_panel(
            title="T",
            panel_type="gauge",
            targets=[],
            grid_pos={"x": 0, "y": 0, "w": 1, "h": 1},
            datasource="default",
        )
        assert p["options"]["showThresholdMarkers"] is True


class TestUid:
    def test_uid_stable(self):
        assert _uid() == "mcp-audit-overview"

    def test_uid_returns_string(self):
        assert isinstance(_uid(), str)


# ── Section builders ─────────────────────────────────────────────────


class TestOverviewPanels:
    def test_returns_6_panels(self):
        panels = _build_overview_panels("default")
        assert len(panels) == 6

    def test_has_stat_panels(self):
        panels = _build_overview_panels("default")
        stat_types = [p for p in panels if p["type"] == "stat"]
        assert len(stat_types) == 4  # calls, errors, cost, sessions

    def test_has_timeseries_panels(self):
        panels = _build_overview_panels("default")
        ts_types = [p for p in panels if p["type"] == "timeseries"]
        assert len(ts_types) == 2  # calls over time, errors over time

    def test_total_calls_panel(self):
        panels = _build_overview_panels("default")
        assert any(p["title"] == "Total Tool Calls" for p in panels)

    def test_error_rate_panel(self):
        panels = _build_overview_panels("default")
        assert any(p["title"] == "Error Rate" for p in panels)

    def test_cost_panel(self):
        panels = _build_overview_panels("default")
        assert any(p["title"] == "Total Cost (USD)" for p in panels)

    def test_sessions_panel(self):
        panels = _build_overview_panels("default")
        assert any(p["title"] == "Active Sessions" for p in panels)

    def test_queries_use_mcp_audit_prefix(self):
        panels = _build_overview_panels("default")
        for p in panels:
            for t in p["targets"]:
                assert "mcp_audit_" in t["expr"]


class TestLatencyPanels:
    def test_returns_2_panels(self):
        panels = _build_latency_panels("default", y_offset=12)
        assert len(panels) == 2

    def test_has_heatmap(self):
        panels = _build_latency_panels("default", y_offset=0)
        assert any(p["type"] == "heatmap" for p in panels)

    def test_has_percentile_timeseries(self):
        panels = _build_latency_panels("default", y_offset=0)
        pct_panels = [p for p in panels if "Percentile" in p["title"]]
        assert len(pct_panels) == 1

    def test_percentile_queries(self):
        panels = _build_latency_panels("default", y_offset=0)
        pct_panel = next(p for p in panels if "Percentile" in p["title"])
        # Should have 3 targets: p50, p95, p99
        assert len(pct_panel["targets"]) == 3
        ref_ids = {t["refId"] for t in pct_panel["targets"]}
        assert ref_ids == {"A", "B", "C"}

    def test_histogram_quantile_used(self):
        panels = _build_latency_panels("default", y_offset=0)
        pct_panel = next(p for p in panels if "Percentile" in p["title"])
        for t in pct_panel["targets"]:
            assert "histogram_quantile" in t["expr"]


class TestCostPanels:
    def test_returns_2_panels(self):
        panels = _build_cost_panels("default", y_offset=20)
        assert len(panels) == 2

    def test_has_bargauge(self):
        panels = _build_cost_panels("default", y_offset=0)
        assert any(p["type"] == "bargauge" for p in panels)

    def test_has_heatmap(self):
        panels = _build_cost_panels("default", y_offset=0)
        assert any(p["type"] == "heatmap" for p in panels)

    def test_currency_unit(self):
        panels = _build_cost_panels("default", y_offset=0)
        bargauge = next(p for p in panels if p["type"] == "bargauge")
        assert bargauge["fieldConfig"]["defaults"]["unit"] == "currencyUSD"


class TestTokenPanels:
    def test_returns_2_panels(self):
        panels = _build_token_panels("default", y_offset=28)
        assert len(panels) == 2

    def test_both_timeseries(self):
        panels = _build_token_panels("default", y_offset=0)
        assert all(p["type"] == "timeseries" for p in panels)

    def test_input_and_output(self):
        panels = _build_token_panels("default", y_offset=0)
        titles = {p["title"] for p in panels}
        assert "Input Tokens by Tool" in titles
        assert "Output Tokens by Tool" in titles


class TestBreakdownPanels:
    def test_returns_1_panel(self):
        panels = _build_breakdown_panels("default", y_offset=36)
        assert len(panels) == 1

    def test_is_table(self):
        panels = _build_breakdown_panels("default", y_offset=0)
        assert panels[0]["type"] == "table"

    def test_full_width(self):
        panels = _build_breakdown_panels("default", y_offset=0)
        assert panels[0]["gridPos"]["w"] == 24


class TestTemplating:
    def test_has_tool_variable(self):
        t = _build_templating()
        assert len(t["list"]) >= 1
        assert t["list"][0]["name"] == "tool"

    def test_tool_query_uses_label_values(self):
        t = _build_templating()
        query = t["list"][0]["query"]
        assert "label_values" in query
        assert "mcp_audit_tool_calls_total" in query

    def test_tool_multi_select(self):
        t = _build_templating()
        assert t["list"][0]["multi"] is True
        assert t["list"][0]["includeAll"] is True


class TestAnnotations:
    def test_has_builtin(self):
        a = _build_annotations()
        assert len(a["list"]) >= 1


# ── Full dashboard generation ────────────────────────────────────────


class TestGenerateDashboardJSON:
    def test_returns_valid_json(self):
        result = generate_dashboard_json()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_has_title(self):
        result = generate_dashboard_json(title="My Dashboard")
        data = json.loads(result)
        assert data["title"] == "My Dashboard"

    def test_default_title(self):
        result = generate_dashboard_json()
        data = json.loads(result)
        assert "MCP Audit" in data["title"]

    def test_has_uid(self):
        data = json.loads(generate_dashboard_json())
        assert data["uid"] == "mcp-audit-overview"

    def test_has_schema_version(self):
        data = json.loads(generate_dashboard_json())
        assert "schemaVersion" in data
        assert data["schemaVersion"] >= 30

    def test_has_refresh(self):
        result = generate_dashboard_json(refresh_interval="5s")
        data = json.loads(result)
        assert data["refresh"] == "5s"

    def test_default_refresh(self):
        data = json.loads(generate_dashboard_json())
        assert data["refresh"] == "10s"

    def test_time_range(self):
        result = generate_dashboard_json(time_range="now-1d")
        data = json.loads(result)
        assert data["time"]["from"] == "now-1d"
        assert data["time"]["to"] == "now"

    def test_has_tags(self):
        data = json.loads(generate_dashboard_json())
        assert "mcp" in data["tags"]
        assert "mcp-audit" in data["tags"]

    def test_custom_tags(self):
        data = json.loads(generate_dashboard_json(tags=["custom", "test"]))
        assert "custom" in data["tags"]

    def test_default_tags(self):
        data = json.loads(generate_dashboard_json())
        assert len(data["tags"]) >= 3

    def test_timezone(self):
        data = json.loads(generate_dashboard_json(timezone="utc"))
        assert data["timezone"] == "utc"

    def test_has_panels(self):
        data = json.loads(generate_dashboard_json())
        assert len(data["panels"]) >= 10

    def test_panel_count(self):
        data = json.loads(generate_dashboard_json())
        # 6 overview + 2 latency + 2 cost + 2 tokens + 1 breakdown = 13
        assert len(data["panels"]) == 13

    def test_panels_have_valid_grid_positions(self):
        data = json.loads(generate_dashboard_json())
        for p in data["panels"]:
            assert "gridPos" in p
            assert p["gridPos"]["w"] > 0
            assert p["gridPos"]["h"] > 0

    def test_all_panel_titles_unique(self):
        data = json.loads(generate_dashboard_json())
        titles = [p["title"] for p in data["panels"]]
        assert len(titles) == len(set(titles)), f"Duplicate titles: {titles}"

    def test_has_templating(self):
        data = json.loads(generate_dashboard_json())
        assert "templating" in data
        assert len(data["templating"]["list"]) >= 1

    def test_has_annotations(self):
        data = json.loads(generate_dashboard_json())
        assert "annotations" in data

    def test_graph_tooltip(self):
        data = json.loads(generate_dashboard_json())
        assert data["graphTooltip"] == 1  # shared crosshair

    def test_all_queries_reference_mcp_audit(self):
        data = json.loads(generate_dashboard_json())
        for p in data["panels"]:
            for t in p.get("targets", []):
                assert "mcp_audit_" in t["expr"], (
                    f"Query in panel '{p['title']}' does not reference mcp_audit_"
                )

    def test_json_string_is_indented(self):
        result = generate_dashboard_json()
        assert "\n" in result  # Multi-line (pretty-printed)


class TestGenerateDashboardDict:
    def test_returns_dict(self):
        d = generate_dashboard_dict()
        assert isinstance(d, dict)

    def test_matches_json_version(self):
        d1 = generate_dashboard_dict(title="Test")
        d2 = json.loads(generate_dashboard_json(title="Test"))
        assert d1 == d2

    def test_panels_accessible(self):
        d = generate_dashboard_dict()
        assert isinstance(d["panels"], list)


# ── save_dashboard ──────────────────────────────────────────────────


class TestSaveDashboard:
    def test_writes_file(self, tmp_path):
        path = str(tmp_path / "dashboard.json")
        result = save_dashboard(path)
        assert result["status"] == "ok"
        assert os.path.exists(path)

    def test_file_contains_valid_json(self, tmp_path):
        path = str(tmp_path / "dashboard.json")
        save_dashboard(path)
        with open(path) as f:
            data = json.load(f)
        assert "title" in data
        assert "panels" in data

    def test_returns_panel_count(self, tmp_path):
        path = str(tmp_path / "dashboard.json")
        result = save_dashboard(path)
        assert result["panels"] == 13

    def test_returns_bytes(self, tmp_path):
        path = str(tmp_path / "dashboard.json")
        result = save_dashboard(path)
        assert result["bytes"] > 0

    def test_custom_title(self, tmp_path):
        path = str(tmp_path / "dash.json")
        save_dashboard(path, title="Custom Title")
        with open(path) as f:
            data = json.load(f)
        assert data["title"] == "Custom Title"

    def test_env_var_fallback(self, tmp_path, monkeypatch):
        path = str(tmp_path / "env.json")
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_OUTPUT", path)
        result = save_dashboard()
        assert result["status"] == "ok"
        assert os.path.exists(path)


# ── export_dashboard_http ────────────────────────────────────────────


class TestExportDashboardHTTP:
    def test_no_url_returns_error(self):
        result = export_dashboard_http(grafana_url=None, api_key="key")
        assert result["status"] == "error"
        assert "Grafana URL" in result["error"]

    def test_no_api_key_returns_error(self):
        result = export_dashboard_http(grafana_url="http://localhost:3000", api_key=None)
        assert result["status"] == "error"
        assert "API key" in result["error"]

    def test_env_var_url(self, monkeypatch):
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_URL", "")
        result = export_dashboard_http()
        assert result["status"] == "error"
        assert "Grafana URL" in result["error"]

    def test_connection_error_handled(self, monkeypatch):
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_URL", "")
        result = export_dashboard_http(
            grafana_url="http://localhost:59999",  # Nothing listening
            api_key="fake-key",
        )
        assert result["status"] == "error"


# ── GrafanaDashboardExporter ─────────────────────────────────────────


class TestGrafanaDashboardExporter:
    def test_init_default(self):
        exporter = GrafanaDashboardExporter()
        assert exporter.datasource == "default"

    def test_init_custom(self):
        exporter = GrafanaDashboardExporter(
            grafana_url="http://grafana:3000",
            api_key="tok123",
            datasource="prometheus",
        )
        assert exporter.grafana_url == "http://grafana:3000"
        assert exporter.api_key == "tok123"
        assert exporter.datasource == "prometheus"

    def test_render(self):
        exporter = GrafanaDashboardExporter()
        text = exporter.render()
        data = json.loads(text)
        assert "panels" in data

    def test_render_custom_title(self):
        exporter = GrafanaDashboardExporter()
        text = exporter.render(title="My Dashboard")
        assert "My Dashboard" in text

    def test_save(self, tmp_path):
        exporter = GrafanaDashboardExporter()
        path = str(tmp_path / "out.json")
        result = exporter.save(path)
        assert result["status"] == "ok"
        assert os.path.exists(path)

    def test_save_env_var(self, tmp_path, monkeypatch):
        path = str(tmp_path / "env_out.json")
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_OUTPUT", path)
        exporter = GrafanaDashboardExporter()
        result = exporter.save()
        assert result["status"] == "ok"

    def test_import_no_url(self):
        exporter = GrafanaDashboardExporter()
        result = exporter.import_to_grafana()
        assert result["status"] == "error"

    def test_import_no_key(self):
        exporter = GrafanaDashboardExporter(grafana_url="http://localhost:3000")
        result = exporter.import_to_grafana()
        assert result["status"] == "error"

    def test_env_var_configuration(self, monkeypatch):
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_URL", "http://my-grafana:3000")
        monkeypatch.setenv("MCP_AUDIT_GRAFANA_KEY", "my-token")
        exporter = GrafanaDashboardExporter()
        assert exporter.grafana_url == "http://my-grafana:3000"
        assert exporter.api_key == "my-token"


# ── Integration: server tool ─────────────────────────────────────────


class TestServerToolIntegration:
    def test_tool_definition_exists(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        tool_names = [t["name"] for t in TOOL_DEFINITIONS]
        assert "export_grafana_dashboard" in tool_names

    def test_tool_definition_schema(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        tool = next(
            t for t in TOOL_DEFINITIONS if t["name"] == "export_grafana_dashboard"
        )
        props = tool["inputSchema"]["properties"]
        assert "mode" in props
        assert "output_path" in props
        assert "datasource" in props
        assert "title" in props

    def test_tool_count_increased(self):
        from mcp_audit.server import TOOL_DEFINITIONS

        # v0.6 had 23, v0.7 adds 1 = 24
        assert len(TOOL_DEFINITIONS) >= 24

    def test_server_text_mode(self):
        from mcp_audit.server import MCPServer
        from mcp_audit.engine import AuditEngine

        engine = AuditEngine()
        server = MCPServer(engine)
        wrapper = server.call_tool("export_grafana_dashboard", {"mode": "text"})
        result = wrapper["result"]

        assert result["status"] == "ok"
        assert "dashboard" in result
        assert isinstance(result["dashboard"], str)

    def test_server_file_mode(self, tmp_path):
        from mcp_audit.server import MCPServer
        from mcp_audit.engine import AuditEngine

        engine = AuditEngine()
        server = MCPServer(engine)
        path = str(tmp_path / "dash.json")
        wrapper = server.call_tool(
            "export_grafana_dashboard",
            {"mode": "file", "output_path": path},
        )
        result = wrapper["result"]
        assert result["status"] == "ok"
        assert os.path.exists(path)

    def test_server_import_mode_no_config(self):
        from mcp_audit.server import MCPServer
        from mcp_audit.engine import AuditEngine

        engine = AuditEngine()
        server = MCPServer(engine)
        wrapper = server.call_tool("export_grafana_dashboard", {"mode": "import"})
        result = wrapper["result"]
        assert result["status"] == "error"

    def test_server_unknown_mode(self):
        from mcp_audit.server import MCPServer
        from mcp_audit.engine import AuditEngine

        engine = AuditEngine()
        server = MCPServer(engine)
        wrapper = server.call_tool(
            "export_grafana_dashboard", {"mode": "invalid_mode"}
        )
        result = wrapper["result"]
        assert "error" in result


# ── Dashboard structure validation ───────────────────────────────────


class TestDashboardStructure:
    def test_all_panel_types_valid(self):
        data = generate_dashboard_dict()
        valid_types = {
            "stat", "timeseries", "heatmap", "table",
            "bargauge", "gauge",
        }
        for p in data["panels"]:
            assert p["type"] in valid_types, f"Invalid panel type: {p['type']}"

    def test_grid_width_within_bounds(self):
        data = generate_dashboard_dict()
        for p in data["panels"]:
            w = p["gridPos"]["w"]
            assert 0 < w <= 24, f"Panel width {w} out of bounds"

    def test_grid_positions_no_horizontal_overlap(self):
        """Verify panels on the same Y row don't overlap."""
        data = generate_dashboard_dict()
        # Group by Y start position
        by_y: dict[int, list[dict]] = {}
        for p in data["panels"]:
            gp = p["gridPos"]
            y = gp["y"]
            if y not in by_y:
                by_y[y] = []
            by_y[y].append(gp)

        for y, panels in by_y.items():
            # Sort by x
            panels.sort(key=lambda g: g["x"])
            for i in range(len(panels) - 1):
                right_edge = panels[i]["x"] + panels[i]["w"]
                left_next = panels[i + 1]["x"]
                assert right_edge <= left_next, (
                    f"Panels overlap at y={y}: {right_edge} > {left_next}"
                )

    def test_version_field(self):
        data = generate_dashboard_dict()
        assert data["version"] == 1

    def test_links_field(self):
        data = generate_dashboard_dict()
        assert isinstance(data["links"], list)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_tags_list(self):
        data = generate_dashboard_dict(tags=[])
        assert data["tags"] == []

    def test_special_chars_in_title(self):
        data = generate_dashboard_dict(title="Dashboard: It's \"Great\" & <cool>")
        assert data["title"] == "Dashboard: It's \"Great\" & <cool>"

    def test_very_long_title(self):
        long_title = "A" * 500
        data = generate_dashboard_dict(title=long_title)
        assert data["title"] == long_title

    def test_datasource_persists_to_panels(self):
        data = generate_dashboard_dict(datasource="my-prometheus")
        for p in data["panels"]:
            if p["datasource"] is not None:
                assert p["datasource"]["uid"] == "my-prometheus"

    def test_dashboard_is_serializable(self):
        """Ensure the full dashboard can be round-tripped through JSON."""
        original = generate_dashboard_dict()
        serialized = json.dumps(original)
        deserialized = json.loads(serialized)
        assert deserialized == original
