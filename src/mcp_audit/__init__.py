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

__version__ = "0.2.0"

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
