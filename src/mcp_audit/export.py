"""Data export utilities for audit records.

Export tool calls, sessions, and events to JSONL (for log shippers like
Datadog, Splunk, ELK) or CSV (for spreadsheet analysis, BI tools).

Usage::

    from mcp_audit import AuditEngine
    from mcp_audit.export import export_calls_jsonl, export_calls_csv

    engine = AuditEngine()

    # Export all calls as JSONL
    path = export_calls_jsonl(engine, "audit_export.jsonl")

    # Export session calls as CSV
    path = export_calls_csv(engine, "session_calls.csv", session_id=sid)
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .engine import AuditEngine


def _call_to_flat_dict(call: Any) -> dict[str, Any]:
    """Flatten a ToolCall into a dict suitable for serialization."""
    return {
        "id": call.id,
        "session_id": call.session_id,
        "agent_id": call.agent_id,
        "tool_name": call.tool_name,
        "server_name": call.server_name,
        "status": call.status.value,
        "error": call.error,
        "started_at": call.started_at.isoformat() if call.started_at else None,
        "completed_at": call.completed_at.isoformat() if call.completed_at else None,
        "duration_ms": call.duration_ms,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "total_tokens": call.input_tokens + call.output_tokens,
        "cost_usd": call.cost_usd,
        "tags": call.tags,
        "arguments": call.arguments,
        "result": call.result,
        "metadata": call.metadata,
    }


def export_calls_jsonl(
    engine: AuditEngine,
    output_path: str | Path,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Export tool calls to a JSONL file (one JSON object per line).

    Returns metadata about the export (path, count, size_bytes).
    """
    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for call in calls:
            f.write(json.dumps(_call_to_flat_dict(call), default=str) + "\n")

    size = path.stat().st_size
    return {
        "format": "jsonl",
        "path": str(path),
        "record_count": len(calls),
        "size_bytes": size,
    }


def export_calls_csv(
    engine: AuditEngine,
    output_path: str | Path,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Export tool calls to a CSV file.

    Tags and arguments are serialized as JSON strings within their cells.
    """
    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id", "session_id", "agent_id", "tool_name", "server_name",
        "status", "error",
        "started_at", "completed_at", "duration_ms",
        "input_tokens", "output_tokens", "total_tokens", "cost_usd",
        "tags", "arguments",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for call in calls:
            row = _call_to_flat_dict(call)
            row["tags"] = json.dumps(row["tags"])
            row["arguments"] = json.dumps(row["arguments"], default=str)
            writer.writerow(row)

    size = path.stat().st_size
    return {
        "format": "csv",
        "path": str(path),
        "record_count": len(calls),
        "size_bytes": size,
    }


def export_events_jsonl(
    engine: AuditEngine,
    output_path: str | Path,
    *,
    trace_id: str | None = None,
    call_id: str | None = None,
    severity: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Export trace events to a JSONL file."""
    events = engine.query_events(
        trace_id=trace_id,
        call_id=call_id,
        severity=severity,
        limit=limit,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for event in events:
            obj = {
                "id": event.id,
                "trace_id": event.trace_id,
                "call_id": event.call_id,
                "event_type": event.event_type,
                "message": event.message,
                "severity": event.severity.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "duration_ms": event.duration_ms,
            }
            f.write(json.dumps(obj, default=str) + "\n")

    size = path.stat().st_size
    return {
        "format": "jsonl",
        "path": str(path),
        "record_count": len(events),
        "size_bytes": size,
    }


def export_to_string(
    engine: AuditEngine,
    fmt: str = "jsonl",
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 100,
) -> str:
    """Export calls as a string (for in-memory use, API responses, etc.).

    Parameters
    ----------
    fmt: "jsonl" or "csv"
    """
    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    if fmt == "jsonl":
        lines = [json.dumps(_call_to_flat_dict(c), default=str) for c in calls]
        return "\n".join(lines)
    elif fmt == "csv":
        fields = [
            "id", "session_id", "agent_id", "tool_name", "server_name",
            "status", "error", "duration_ms",
            "input_tokens", "output_tokens", "total_tokens", "cost_usd",
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for call in calls:
            writer.writerow(_call_to_flat_dict(call))
        return output.getvalue()
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'jsonl' or 'csv'.")
