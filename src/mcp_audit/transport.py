"""MCP stdio transport — bridges the audit tool handlers to the real MCP protocol.

This lets agents connect to mcp-audit via stdio transport (the standard MCP
transport used by Claude Desktop, Cursor, etc.).  We dynamically register all
tool definitions from the ``MCPServer`` class onto an MCP SDK server instance.

Usage::

    # In an MCP client config (e.g. Claude Desktop ``claude_desktop_config.json``):
    {
      "mcpServers": {
        "mcp-audit": {
          "command": "mcp-audit",
          "args": ["stdio"]
        }
      }
    }

    # Or programmatically:
    from mcp_audit.transport import run_stdio
    run_stdio()
"""
from __future__ import annotations

import inspect
import json
from typing import Any

# The MCP SDK renamed FastMCP → MCPServer in v2.0.0
# Support both for compatibility
try:
    from mcp.server.mcpserver import MCPServer as _McpSdkServer
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _McpSdkServer  # type: ignore[assignment]
    except ImportError:
        _McpSdkServer = None  # type: ignore[assignment]

from .engine import AuditEngine
from .server import MCPServer as _AuditServer
from .server import TOOL_DEFINITIONS

# JSON Schema property → Python type mapping
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_annotation(prop_schema: dict[str, Any]) -> Any:
    """Convert a JSON-schema property to a Python type annotation."""
    json_type = prop_schema.get("type", "string")
    return _JSON_TYPE_MAP.get(json_type, str)


def _build_signature(input_schema: dict[str, Any]) -> inspect.Signature:
    """Build a Python function signature from an MCP input-schema dict."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    params: list[inspect.Parameter] = []

    for name, prop in properties.items():
        annotation = _schema_to_annotation(prop)
        if name in required:
            default = inspect.Parameter.empty
        else:
            # Use the schema default if present, otherwise None
            default = prop.get("default", None)
        params.append(
            inspect.Parameter(
                name=name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )

    return inspect.Signature(parameters=params)


def _make_tool_function(
    tool_name: str,
    description: str,
    input_schema: dict[str, Any],
    inner_server: _AuditServer,
) -> Any:
    """Create a callable with a proper signature that delegates to inner server."""
    sig = _build_signature(input_schema)

    def _handler(*args: Any, **kwargs: Any) -> str:
        result = inner_server.call_tool(tool_name, kwargs)
        return json.dumps(result, default=str)

    _handler.__name__ = tool_name
    _handler.__qualname__ = tool_name
    _handler.__doc__ = description
    _handler.__signature__ = sig  # type: ignore[attr-defined]

    return _handler


def create_fastmcp_server(
    engine: AuditEngine | None = None,
) -> Any:
    """Build an MCP SDK server with all audit tools registered.

    Parameters
    ----------
    engine
        Optional pre-configured :class:`AuditEngine` (e.g. with persistent
        storage).  A fresh in-memory engine is created when *None*.

    Returns
    -------
    MCPServer (SDK v2) or FastMCP (SDK v1)
        A server ready to ``.run(transport="stdio")``.
    """
    if _McpSdkServer is None:
        raise ImportError(
            "MCP SDK is not installed. Install with: pip install mcp"
        )

    inner = _AuditServer(engine=engine)
    server = _McpSdkServer(
        name="mcp-audit",
        instructions=(
            "mcp-audit: observability for AI agent tool calls. "
            "Use start_session to begin, record_call after every tool "
            "invocation, and get_stats / get_agent_report for analytics. "
            "Create alert rules to detect runaway costs or error spikes. "
            "Export to OpenTelemetry with export_otlp."
        ),
    )

    for tool_def in TOOL_DEFINITIONS:
        tool_name: str = tool_def["name"]
        description: str = tool_def["description"]
        input_schema: dict[str, Any] = tool_def.get("inputSchema", {})

        handler_fn = _make_tool_function(
            tool_name, description, input_schema, inner
        )

        server.add_tool(
            fn=handler_fn,
            name=tool_name,
            description=description,
        )

    return server


def run_stdio() -> None:
    """Run the MCP server on stdio transport (blocking)."""
    server = create_fastmcp_server()
    server.run(transport="stdio")
