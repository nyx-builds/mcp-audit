"""CLI for mcp-audit."""
from __future__ import annotations

import json
import sys

import click

from .engine import AuditEngine
from .server import MCPServer


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """mcp-audit — observability for AI agent tool calls."""


@cli.command()
@click.option("--agent-id", default=None, help="Agent identifier")
@click.option("--name", default=None, help="Session name")
def start_session(agent_id: str | None, name: str | None) -> None:
    """Start a new audit session."""
    engine = AuditEngine()
    session = engine.start_session(agent_id=agent_id, name=name)
    click.echo(json.dumps({"session_id": session.id}, indent=2))


@cli.command()
@click.option("--limit", default=20, help="Number of calls to show")
def recent(limit: int) -> None:
    """Show recent tool calls."""
    engine = AuditEngine()
    calls = engine.query_calls(limit=limit)
    if not calls:
        click.echo("No calls recorded.")
        return
    for c in calls:
        status_icon = "✓" if not c.is_error else "✗"
        click.echo(
            f"  {status_icon} {c.tool_name:<30} {c.status.value:<8} "
            f"{c.duration_ms or 0:>8.1f}ms  ${c.cost_usd:.6f}  "
            f"{c.started_at.isoformat()}"
        )


@cli.command()
@click.option("--agent-id", default=None)
@click.option("--session-id", default=None)
@click.option("--tool-name", default=None)
def stats(agent_id: str | None, session_id: str | None, tool_name: str | None) -> None:
    """Show aggregate statistics."""
    engine = AuditEngine()
    result = engine.get_stats(
        agent_id=agent_id, session_id=session_id, tool_name=tool_name
    )
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command()
def tools() -> None:
    """List all MCP tools exposed by the server."""
    server = MCPServer()
    for tool in server.list_tools():
        click.echo(f"  {tool['name']:<30} {tool.get('description', '')[:60]}")


@cli.command()
def serve() -> None:
    """Print MCP server configuration for stdio transport."""
    server = MCPServer()
    config = {
        "mcpServers": {
            "mcp-audit": {
                "command": "mcp-audit",
                "args": ["stdio"],
            }
        },
        "tools": [t["name"] for t in server.list_tools()],
        "tool_count": server.tool_count,
    }
    click.echo(json.dumps(config, indent=2))


if __name__ == "__main__":
    cli()
