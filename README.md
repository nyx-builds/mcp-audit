# mcp-audit

<p align="center">
  <strong>Audit, trace, and observe MCP tool calls.</strong><br>
  The observability MCP server for AI agents.
</p>

<p align="center">
  <a href="https://github.com/nyx-builds/mcp-audit/actions"><img src="https://github.com/nyx-builds/mcp-audit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/MCP-tools-20-green.svg" alt="MCP Tools">
  <img src="https://img.shields.io/badge/tests-263-passing-brightgreen.svg" alt="Tests">
</p>

---

**mcp-audit** is a [Model Context Protocol](https://modelcontextprotocol.io) server that gives AI agents observability over their own tool calls. Every time your agent invokes a tool — whether it's an MCP tool, a custom function, or an external API — `mcp-audit` records who called what, when, how long it took, what it cost, and whether it succeeded.

## Why?

AI agents are increasingly autonomous, calling dozens of tools per session. But agents are **blind to their own behavior**:

- ❌ No way to see *which* tools an agent called or *how often*
- ❌ No cost tracking — agents can silently rack up API bills
- ❌ No latency visibility — one slow tool tanks the whole session
- ❌ No error correlation — which tool is failing 30% of the time?
- ❌ No alerting — you find out about runaway costs after the fact

`mcp-audit` fixes this. It's an MCP server that agents use to **audit themselves**.

## Features

- 📊 **Call Recording** — log every tool invocation with timing, cost, tokens, and result
- 🔍 **Flexible Querying** — filter by session, agent, tool, server, status, cost, tags
- 📈 **Aggregate Analytics** — error rate, p50/p95/p99 latency, total cost, top tools
- 💰 **Cost Breakdown** — see exactly which tools are consuming budget, grouped by tool/server/session
- 🚨 **Alert Rules** — set thresholds (e.g. "alert if error_rate > 50%" or "alert if total_cost > $10")
- 📝 **Trace Events** — fine-grained structured logging within calls (sub-steps, HTTP requests, DB queries)
- 📊 **Tool Health Dashboard** — per-tool metrics at a glance (error rate, p95 latency, cost)
- 💾 **SQLite Persistence** — durable storage that survives restarts (drop-in replacement for memory store)
- 🎯 **Auto-Instrumentation** — `@audit_call` decorator for zero-code tracing of any Python function
- 📤 **Data Export** — JSONL & CSV export for feeding data to Grafana, Datadog, Splunk, ELK
- 🏷️ **Agent Reports** — comprehensive per-agent performance summaries
- 🪝 **Context Manager** — Python `with` block for automatic call tracing

## Quick Start

### Install

```bash
pip install mcp-audit
```

### Use as a Python library

```python
from mcp_audit import AuditEngine, traced_call

engine = AuditEngine()
session = engine.start_session(agent_id="my-agent")

# Wrap any function call with automatic tracing
with traced_call(engine, session_id=session.id, tool_name="web_search") as tc:
    tc.set_cost(0.003)
    tc.set_tokens(input_tokens=500, output_tokens=200)
    result = search("best MCP servers")
    tc.set_result(result)

# Query analytics
stats = engine.get_stats(session_id=session.id)
print(f"Total cost: ${stats['total_cost_usd']}")
print(f"P95 latency: {stats['p95_latency_ms']}ms")
print(f"Error rate: {stats['error_rate']}%")
```

### Use as an MCP server

Add to your MCP client config (Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "mcp-audit": {
      "command": "mcp-audit",
      "args": ["stdio"]
    }
  }
}
```

This runs mcp-audit as a real MCP server over stdio transport. Your agent can now call audit tools like `record_call`, `get_stats`, `create_alert_rule`, and `evaluate_alerts` directly through the MCP protocol.

> **Note:** Use `mcp-audit stdio` (not `mcp-audit serve`). The `serve` command prints configuration JSON for reference; `stdio` runs the actual MCP stdio transport.

### Use as a Python library with FastMCP

For programmatic integration:

```python
from mcp_audit import create_fastmcp_server

# Get a FastMCP instance with all 17 tools registered
server = create_fastmcp_server()
server.run(transport="stdio")
```

## MCP Tools (20)

| Tool | Description |
|------|-------------|
| `start_session` | Start a new audit session for an agent |
| `end_session` | End a session and compute final aggregates |
| `get_session` | Get session details with aggregate metrics |
| `list_sessions` | List sessions with optional agent/active filters |
| `record_call` | Record a completed tool call (primary ingestion) |
| `get_call` | Look up a specific tool call by ID |
| `query_calls` | Search calls with flexible filters |
| `log_event` | Log a structured trace event (sub-step) |
| `query_events` | Query trace events with filters |
| `get_stats` | Aggregate statistics (error rate, percentiles, cost) |
| `get_agent_report` | Comprehensive per-agent performance report |
| `get_cost_breakdown` | Cost analysis grouped by tool/server/session |
| `create_alert_rule` | Set threshold-based alert rules |
| `list_alert_rules` | List configured alert rules |
| `delete_alert_rule` | Remove an alert rule |
| `evaluate_alerts` | Check which alert rules are currently breached |
| `get_tool_health` | Per-tool health metrics (error rate, p95, cost) |
| `get_recent_calls` | Get the N most recent tool calls |
| `export_calls` | Export calls to JSONL or CSV file |
| `get_audit_summary` | High-level dashboard summary |

## SQLite Persistence

For production use, persist audit data across restarts with the SQLite backend:

```python
from mcp_audit import AuditEngine
from mcp_audit.sqlite_store import SQLiteStore

store = SQLiteStore("audit.db")  # persists to disk
engine = AuditEngine(store=store)

# All calls, sessions, events, and rules now survive process restarts
session = engine.start_session(agent_id="prod-agent")
```

SQLiteStore is a drop-in replacement for the default MemoryStore — same interface, durable storage. Uses indexed columns for efficient querying on session_id, agent_id, tool_name, status, and timestamps.

## Auto-Instrumentation with @audit_call

Skip manual tracing — decorate any function and it's automatically audited:

```python
from mcp_audit import AuditEngine
from mcp_audit.decorator import audit_call, bind_session

engine = AuditEngine()
session = engine.start_session(agent_id="my-agent")

# Option 1: explicit engine + session
@audit_call(engine, session_id=session.id, cost_fn=lambda *_: 0.001)
def search(query: str) -> list:
    return [{"title": "result"}]

# Option 2: bind once, decorate everywhere
ctx = bind_session(engine, session.id)

@audit_call()  # uses bound engine + session
def fetch(url: str) -> dict:
    return requests.get(url).json()

@audit_call(tool_name="llm_complete", cost_fn=compute_cost)
def complete(prompt: str) -> str:
    return llm.generate(prompt)

ctx.reset()  # unbind when done
```

Every call is automatically recorded with timing, status, and errors. Exceptions are recorded as errors and re-raised.

## Data Export

Export audit data for external tools:

```python
from mcp_audit.export import export_calls_jsonl, export_calls_csv

# JSONL for log shippers (Datadog, Splunk, ELK)
export_calls_jsonl(engine, "audit.jsonl", session_id=sid)

# CSV for spreadsheet analysis
export_calls_csv(engine, "costs.csv", agent_id="prod-agent")

# Or export to a string
from mcp_audit.export import export_to_string
text = export_to_string(engine, fmt="jsonl", limit=100)
```

Or call the `export_calls` MCP tool directly from your agent.

## Alert Rules

Set up automatic monitoring:

```python
engine.create_rule(
    name="high_error_rate",
    metric="error_rate",       # error_rate | p95_latency | cost_per_call | total_cost | call_volume
    operator=">",              # > | >= | < | <= | ==
    threshold=50.0,            # threshold value
    window=100,                # evaluate last N calls
)

engine.create_rule(
    name="budget_exceeded",
    metric="total_cost",
    operator=">=",
    threshold=10.0,
)

# Check if any rules are breached
alerts = engine.evaluate_rules()
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AI Agent (Claude, etc.)              │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ MCP Tool │  │ MCP Tool │  │   mcp-audit       │  │
│  │ Server A │  │ Server B │  │   (this server)   │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │                  │             │
│       └──────┬───────┘                  │             │
│              │  Agent calls record_call │             │
│              └──────────────────────────┘             │
└─────────────────────────────────────────────────────┘
```

The agent calls tools on other MCP servers, then calls `record_call` on mcp-audit to log what happened. Alternatively, wrap tool calls programmatically using the `traced_call` context manager.

## Development

```bash
git clone https://github.com/nyx-builds/mcp-audit.git
cd mcp-audit
uv venv .venv
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

## License

MIT
