"""mcp-audit: Audit, trace, and observe MCP tool calls."""
from __future__ import annotations

from .engine import AuditEngine
from .models import (
    AgentReport,
    AlertRule,
    CallStatus,
    Session,
    Severity,
    ToolCall,
    TraceEvent,
)
from .tracer import traced_call
from .transport import create_fastmcp_server, run_stdio

__version__ = "0.5.0"

__all__ = [
    "AuditEngine",
    "traced_call",
    "CallStatus",
    "Severity",
    "ToolCall",
    "TraceEvent",
    "Session",
    "AlertRule",
    "AgentReport",
    "create_fastmcp_server",
    "run_stdio",
    "__version__",
]


def __getattr__(name: str):
    """Lazy imports for optional modules."""
    if name == "SQLiteStore":
        from .sqlite_store import SQLiteStore
        return SQLiteStore
    if name == "audit_call":
        from .decorator import audit_call
        return audit_call
    if name == "bind_session":
        from .decorator import bind_session
        return bind_session
    if name == "OTLPExporter":
        from .otlp import OTLPExporter
        return OTLPExporter
    if name == "export_otlp_http":
        from .otlp import export_otlp_http
        return export_otlp_http
    if name == "export_otlp_jsonl":
        from .otlp import export_otlp_jsonl
        return export_otlp_jsonl
    if name == "OTLPMetricsExporter":
        from .metrics import OTLPMetricsExporter
        return OTLPMetricsExporter
    if name == "export_otlp_metrics_http":
        from .metrics import export_otlp_metrics_http
        return export_otlp_metrics_http
    if name == "export_otlp_metrics_jsonl":
        from .metrics import export_otlp_metrics_jsonl
        return export_otlp_metrics_jsonl
    if name == "build_metrics":
        from .metrics import build_metrics
        return build_metrics
    raise AttributeError(f"module 'mcp_audit' has no attribute '{name}'")
