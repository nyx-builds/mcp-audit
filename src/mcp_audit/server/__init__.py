"""MCP server — exposes audit / observability tools via Model Context Protocol.

Agents connect to this server to log their own tool calls, query execution
history, compute cost analytics, and receive alert notifications.
"""
from __future__ import annotations

from typing import Any

from ..engine import AuditEngine
from ..models import CallStatus, Severity


# ── MCP tool definitions ────────────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ── Session management ──
    {
        "name": "start_session",
        "description": "Start a new audit session for an agent. Returns a session_id to use in subsequent record_call calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Identifier for the agent"},
                "name": {"type": "string", "description": "Human-readable session name"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Session tags"},
                "metadata": {"type": "object", "description": "Arbitrary key-value metadata"},
            },
        },
    },
    {
        "name": "end_session",
        "description": "Mark a session as ended and compute final aggregates (total calls, errors, cost, tokens).",
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "get_session",
        "description": "Get details for a specific session including aggregate metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "list_sessions",
        "description": "List sessions, optionally filtered by agent or active status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "active_only": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "default": 50},
            },
        },
    },
    # ── Tool call recording ──
    {
        "name": "record_call",
        "description": "Record a completed tool call with timing, cost, and token data. This is the primary audit ingestion method — call this after every tool invocation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID from start_session"},
                "tool_name": {"type": "string", "description": "Name of the tool that was called"},
                "server_name": {"type": "string", "description": "MCP server hosting the tool"},
                "agent_id": {"type": "string"},
                "status": {"type": "string", "enum": [s.value for s in CallStatus], "default": "success"},
                "error": {"type": "string", "description": "Error message if status is error/timeout"},
                "duration_ms": {"type": "number", "description": "Wall-clock duration in milliseconds"},
                "input_tokens": {"type": "integer", "default": 0},
                "output_tokens": {"type": "integer", "default": 0},
                "cost_usd": {"type": "number", "default": 0.0, "description": "Monetary cost in USD"},
                "arguments": {"type": "object", "description": "Input arguments to the tool"},
                "result": {"description": "Return value (will be JSON-serialised)"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["session_id", "tool_name"],
        },
    },
    {
        "name": "get_call",
        "description": "Look up a specific tool call by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"call_id": {"type": "string"}},
            "required": ["call_id"],
        },
    },
    {
        "name": "query_calls",
        "description": "Search tool calls with filters. Returns calls newest-first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "server_name": {"type": "string"},
                "status": {"type": "string"},
                "min_cost": {"type": "number"},
                "min_duration": {"type": "number"},
                "tag": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    # ── Trace events ──
    {
        "name": "log_event",
        "description": "Log a structured trace event (sub-step within a call or session). Useful for fine-grained observability.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string", "description": "Trace/session ID this event belongs to"},
                "call_id": {"type": "string", "description": "Optional parent tool call ID"},
                "event_type": {"type": "string", "description": "E.g. 'http_request', 'db_query', 'llm_call'"},
                "message": {"type": "string"},
                "severity": {"type": "string", "enum": [s.value for s in Severity], "default": "info"},
                "data": {"type": "object"},
                "duration_ms": {"type": "number"},
            },
            "required": ["trace_id", "event_type"],
        },
    },
    {
        "name": "query_events",
        "description": "Query trace events with filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
                "call_id": {"type": "string"},
                "severity": {"type": "string"},
                "limit": {"type": "integer", "default": 100},
            },
        },
    },
    # ── Analytics ──
    {
        "name": "get_stats",
        "description": "Compute aggregate statistics (error rate, p95/p99 latency, total cost, top tools) over matching calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "tool_name": {"type": "string"},
            },
        },
    },
    {
        "name": "get_agent_report",
        "description": "Generate a comprehensive performance report for an agent across all sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "session_id": {"type": "string", "description": "Optional: limit to one session"},
            },
            "required": ["agent_id"],
        },
    },
    {
        "name": "get_cost_breakdown",
        "description": "Break down costs by tool, server, or session. Useful for identifying expensive tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "group_by": {"type": "string", "enum": ["tool", "server", "session"], "default": "tool"},
            },
        },
    },
    # ── Alert rules ──
    {
        "name": "create_alert_rule",
        "description": "Create an alert rule that triggers when a metric breaches a threshold (e.g. error_rate > 50%).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Human-readable rule name"},
                "metric": {"type": "string", "enum": ["error_rate", "p95_latency", "cost_per_call", "total_cost", "call_volume"]},
                "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "=="]},
                "threshold": {"type": "number"},
                "window": {"type": "integer", "default": 100, "description": "Number of recent calls to evaluate"},
            },
            "required": ["name", "metric", "operator", "threshold"],
        },
    },
    {
        "name": "list_alert_rules",
        "description": "List all configured alert rules.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "delete_alert_rule",
        "description": "Delete an alert rule by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"rule_id": {"type": "string"}},
            "required": ["rule_id"],
        },
    },
    {
        "name": "evaluate_alerts",
        "description": "Evaluate all enabled alert rules against recent calls. Returns any rules that are currently breached.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
        },
    },
    # ── Tool Health ──
    {
        "name": "get_tool_health",
        "description": "Get per-tool health metrics: call count, error rate, p95 latency, cost for each tool. Sorted by usage (most-called first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
        },
    },
    {
        "name": "get_recent_calls",
        "description": "Get the N most recent tool calls (newest first). Quick way to see what the agent just did.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 10, "description": "Number of recent calls to return"},
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
        },
    },
    # ── Export ──
    {
        "name": "export_calls",
        "description": "Export tool calls to JSONL or CSV format. Writes to a file path and returns metadata (path, count, size).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["jsonl", "csv"], "default": "jsonl"},
                "output_path": {"type": "string", "description": "File path to write to"},
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "limit": {"type": "integer", "default": 10000},
            },
            "required": ["format", "output_path"],
        },
    },
    {
        "name": "export_otlp",
        "description": "Export tool calls as OpenTelemetry traces. Supports HTTP (send to OTel Collector/Jaeger/Tempo/Honeycomb) or JSONL file output. Converts each tool call to an OTel span with cost, latency, token, and status attributes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["http", "jsonl"], "default": "jsonl",
                         "description": "http = POST to OTel collector, jsonl = write OTLP/JSON lines to file"},
                "endpoint": {"type": "string", "description": "OTLP/HTTP endpoint (default: http://localhost:4318/v1/traces). Only for mode=http."},
                "output_path": {"type": "string", "description": "File path for JSONL output. Only for mode=jsonl."},
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "limit": {"type": "integer", "default": 10000},
            },
        },
    },
    # ── Metrics Export ──
    {
        "name": "export_otlp_metrics",
        "description": "Export audit metrics as OpenTelemetry metrics (counters, histograms, gauges). Supports HTTP (send to OTel Collector/Prometheus/Grafana/Datadog) or JSONL file output. Generates tool.call.count, tool.error.count, tool.duration_ms histogram, tool.cost.usd histogram, tool.tokens.input/output counters, session.count gauge, error.rate gauge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["http", "jsonl"], "default": "jsonl",
                         "description": "http = POST metrics to OTel collector, jsonl = write OTLP/JSON metrics to file"},
                "endpoint": {"type": "string", "description": "OTLP/HTTP metrics endpoint (default: http://localhost:4318/v1/metrics). Only for mode=http."},
                "output_path": {"type": "string", "description": "File path for JSONL output. Only for mode=jsonl."},
                "session_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "limit": {"type": "integer", "default": 10000},
            },
        },
    },
    # ── Utility ──
    {
        "name": "get_audit_summary",
        "description": "Get a high-level summary of all audit data: total calls, sessions, cost, active alerts.",
        "inputSchema": { "type": "object", "properties": {} },
    },
]


class MCPServer:
    """MCP server that exposes audit / observability tools."""

    def __init__(self, engine: AuditEngine | None = None) -> None:
        self.engine = engine or AuditEngine()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all available tools (MCP tools/list)."""
        return TOOL_DEFINITIONS

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle a tool call (MCP tools/call)."""
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return {"error": f"Unknown tool: {name}"}
            result = handler(arguments)
            return {"result": result}
        except Exception as exc:
            return {"error": str(exc)}

    @property
    def tool_count(self) -> int:
        return len(TOOL_DEFINITIONS)

    # ── Tool Handlers ──────────────────────────────────────────────

    def _tool_start_session(self, args: dict) -> dict:
        session = self.engine.start_session(
            agent_id=args.get("agent_id"),
            name=args.get("name"),
            tags=args.get("tags"),
            metadata=args.get("metadata"),
        )
        return self._serialize_session(session)

    def _tool_end_session(self, args: dict) -> dict:
        session = self.engine.end_session(args["session_id"])
        if session is None:
            return {"error": "Session not found"}
        return self._serialize_session(session)

    def _tool_get_session(self, args: dict) -> dict:
        session = self.engine.get_session(args["session_id"])
        if session is None:
            return {"error": "Session not found"}
        return self._serialize_session(session)

    def _tool_list_sessions(self, args: dict) -> dict:
        sessions = self.engine.list_sessions(
            agent_id=args.get("agent_id"),
            active_only=args.get("active_only", False),
            limit=args.get("limit", 50),
        )
        return {
            "count": len(sessions),
            "sessions": [self._serialize_session(s) for s in sessions],
        }

    def _tool_record_call(self, args: dict) -> dict:
        status = CallStatus(args.get("status", "success"))
        call = self.engine.record_call(
            session_id=args["session_id"],
            tool_name=args["tool_name"],
            agent_id=args.get("agent_id"),
            server_name=args.get("server_name"),
            arguments=args.get("arguments", {}),
            result=args.get("result"),
            status=status,
            error=args.get("error"),
            duration_ms=args.get("duration_ms"),
            input_tokens=args.get("input_tokens", 0),
            output_tokens=args.get("output_tokens", 0),
            cost_usd=args.get("cost_usd", 0.0),
            tags=args.get("tags", []),
            metadata=args.get("metadata", {}),
        )
        return self._serialize_call(call)

    def _tool_get_call(self, args: dict) -> dict:
        call = self.engine.get_call(args["call_id"])
        if call is None:
            return {"error": "Call not found"}
        return self._serialize_call(call)

    def _tool_query_calls(self, args: dict) -> dict:
        calls = self.engine.query_calls(**_filter_args(args))
        return {
            "count": len(calls),
            "calls": [self._serialize_call(c) for c in calls],
        }

    def _tool_log_event(self, args: dict) -> dict:
        severity = Severity(args.get("severity", "info"))
        event = self.engine.log_event(
            trace_id=args["trace_id"],
            event_type=args["event_type"],
            message=args.get("message", ""),
            call_id=args.get("call_id"),
            severity=severity,
            data=args.get("data", {}),
            duration_ms=args.get("duration_ms"),
        )
        return self._serialize_event(event)

    def _tool_query_events(self, args: dict) -> dict:
        events = self.engine.query_events(**_filter_args(args))
        return {
            "count": len(events),
            "events": [self._serialize_event(e) for e in events],
        }

    def _tool_get_stats(self, args: dict) -> dict:
        return self.engine.get_stats(**_filter_args(args))

    def _tool_get_agent_report(self, args: dict) -> dict:
        report = self.engine.get_agent_report(
            args["agent_id"],
            session_id=args.get("session_id"),
        )
        return report.model_dump(mode="json")

    def _tool_get_cost_breakdown(self, args: dict) -> dict:
        return self.engine.get_cost_breakdown(**_filter_args(args))

    def _tool_create_alert_rule(self, args: dict) -> dict:
        rule = self.engine.create_rule(
            name=args["name"],
            metric=args["metric"],
            operator=args["operator"],
            threshold=args["threshold"],
            window=args.get("window", 100),
        )
        return self._serialize_rule(rule)

    def _tool_list_alert_rules(self, args: dict) -> dict:
        rules = self.engine.list_rules()
        return {
            "count": len(rules),
            "rules": [self._serialize_rule(r) for r in rules],
        }

    def _tool_delete_alert_rule(self, args: dict) -> dict:
        deleted = self.engine.delete_rule(args["rule_id"])
        return {"deleted": deleted}

    def _tool_evaluate_alerts(self, args: dict) -> dict:
        triggered = self.engine.evaluate_rules(**_filter_args(args))
        return {
            "evaluated": True,
            "triggered_count": len(triggered),
            "alerts": triggered,
        }

    def _tool_get_audit_summary(self, args: dict) -> dict:
        stats = self.engine.get_stats()
        sessions = self.engine.list_sessions(limit=1)
        rules = self.engine.list_rules()
        active_rules = [r for r in rules if r.enabled]
        return {
            "total_calls": stats["total_calls"],
            "total_sessions": len(self.engine.list_sessions(limit=100_000)),
            "active_sessions": len(self.engine.list_sessions(active_only=True, limit=100_000)),
            "total_cost_usd": stats["total_cost_usd"],
            "error_rate": stats["error_rate"],
            "p95_latency_ms": stats["p95_latency_ms"],
            "total_rules": len(rules),
            "active_rules": len(active_rules),
            "unique_tools": stats["unique_tools"],
        }

    def _tool_get_tool_health(self, args: dict) -> dict:
        health = self.engine.get_tool_health(**_filter_args(args))
        return {
            "tool_count": len(health),
            "tools": health,
        }

    def _tool_get_recent_calls(self, args: dict) -> dict:
        calls = self.engine.get_recent_calls(
            n=args.get("n", 10),
            session_id=args.get("session_id"),
            agent_id=args.get("agent_id"),
        )
        return {
            "count": len(calls),
            "calls": [self._serialize_call(c) for c in calls],
        }

    def _tool_export_calls(self, args: dict) -> dict:
        from ..export import export_calls_jsonl, export_calls_csv

        fmt = args.get("format", "jsonl")
        output_path = args["output_path"]
        kwargs = {
            "session_id": args.get("session_id"),
            "agent_id": args.get("agent_id"),
            "tool_name": args.get("tool_name"),
            "limit": args.get("limit", 10_000),
        }
        if fmt == "jsonl":
            return export_calls_jsonl(self.engine, output_path, **kwargs)
        elif fmt == "csv":
            return export_calls_csv(self.engine, output_path, **kwargs)
        else:
            return {"error": f"Unknown format: {fmt}. Use 'jsonl' or 'csv'."}

    def _tool_export_otlp(self, args: dict) -> dict:
        from ..otlp import export_otlp_http, export_otlp_jsonl

        mode = args.get("mode", "jsonl")
        kwargs = {
            "session_id": args.get("session_id"),
            "agent_id": args.get("agent_id"),
            "tool_name": args.get("tool_name"),
            "limit": args.get("limit", 10_000),
        }
        if mode == "http":
            return export_otlp_http(
                self.engine,
                endpoint=args.get("endpoint"),
                **kwargs,
            )
        elif mode == "jsonl":
            output_path = args.get("output_path")
            if not output_path:
                return {"error": "output_path is required for jsonl mode"}
            return export_otlp_jsonl(self.engine, output_path, **kwargs)
        else:
            return {"error": f"Unknown mode: {mode}. Use 'http' or 'jsonl'."}

    def _tool_export_otlp_metrics(self, args: dict) -> dict:
        from ..metrics import export_otlp_metrics_http, export_otlp_metrics_jsonl

        mode = args.get("mode", "jsonl")
        kwargs = {
            "session_id": args.get("session_id"),
            "agent_id": args.get("agent_id"),
            "tool_name": args.get("tool_name"),
            "limit": args.get("limit", 10_000),
        }
        if mode == "http":
            return export_otlp_metrics_http(
                self.engine,
                endpoint=args.get("endpoint"),
                **kwargs,
            )
        elif mode == "jsonl":
            output_path = args.get("output_path")
            if not output_path:
                return {"error": "output_path is required for jsonl mode"}
            return export_otlp_metrics_jsonl(self.engine, output_path, **kwargs)
        else:
            return {"error": f"Unknown mode: {mode}. Use 'http' or 'jsonl'."}

    # ── Serializers ────────────────────────────────────────────────

    @staticmethod
    def _serialize_call(call) -> dict:
        return {
            "id": call.id,
            "session_id": call.session_id,
            "agent_id": call.agent_id,
            "tool_name": call.tool_name,
            "server_name": call.server_name,
            "status": call.status.value,
            "error": call.error,
            "duration_ms": call.duration_ms,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": call.cost_usd,
            "tags": call.tags,
            "started_at": call.started_at.isoformat(),
            "completed_at": call.completed_at.isoformat() if call.completed_at else None,
        }

    @staticmethod
    def _serialize_session(session) -> dict:
        return {
            "id": session.id,
            "agent_id": session.agent_id,
            "name": session.name,
            "is_active": session.is_active,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "total_calls": session.total_calls,
            "error_count": session.error_count,
            "total_cost_usd": session.total_cost_usd,
            "total_tokens": session.total_tokens,
            "tags": session.tags,
        }

    @staticmethod
    def _serialize_event(event) -> dict:
        return {
            "id": event.id,
            "trace_id": event.trace_id,
            "call_id": event.call_id,
            "event_type": event.event_type,
            "message": event.message,
            "severity": event.severity.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "duration_ms": event.duration_ms,
        }

    @staticmethod
    def _serialize_rule(rule) -> dict:
        return {
            "id": rule.id,
            "name": rule.name,
            "metric": rule.metric,
            "operator": rule.operator,
            "threshold": rule.threshold,
            "window": rule.window,
            "enabled": rule.enabled,
            "trigger_count": rule.trigger_count,
            "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
            "created_at": rule.created_at.isoformat(),
        }


def _filter_args(args: dict) -> dict:
    """Remove None values and keys that engine methods don't accept."""
    return {k: v for k, v in args.items() if v is not None}
