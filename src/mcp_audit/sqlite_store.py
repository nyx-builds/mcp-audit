"""SQLite persistent storage backend for audit records.

Provides durability across restarts and efficient querying with indexed
lookups.  Use this in production instead of ``MemoryStore`` when you
need data to survive process restarts.

Usage::

    from mcp_audit import AuditEngine
    from mcp_audit.sqlite_store import SQLiteStore

    store = SQLiteStore("audit.db")
    engine = AuditEngine(store=store)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .models import AlertRule, CallStatus, Session, Severity, ToolCall, TraceEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    agent_id      TEXT,
    tool_name     TEXT NOT NULL,
    server_name   TEXT,
    arguments     TEXT,   -- JSON
    result        TEXT,   -- JSON or null
    status        TEXT NOT NULL DEFAULT 'success',
    error         TEXT,
    started_at    TEXT NOT NULL,
    completed_at  TEXT,
    duration_ms   REAL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      REAL DEFAULT 0.0,
    tags          TEXT,   -- JSON array
    metadata      TEXT    -- JSON object
);

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    agent_id       TEXT,
    name           TEXT,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    tags           TEXT,
    metadata       TEXT,
    total_calls    INTEGER DEFAULT 0,
    error_count    INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    total_tokens   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trace_events (
    id          TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    call_id     TEXT,
    event_type  TEXT NOT NULL,
    message     TEXT DEFAULT '',
    severity    TEXT DEFAULT 'info',
    data        TEXT,
    timestamp   TEXT NOT NULL,
    duration_ms REAL
);

CREATE TABLE IF NOT EXISTS alert_rules (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    metric         TEXT NOT NULL,
    operator       TEXT NOT NULL,
    threshold      REAL NOT NULL,
    window         INTEGER DEFAULT 100,
    enabled        INTEGER DEFAULT 1,
    created_at     TEXT NOT NULL,
    last_triggered TEXT,
    trigger_count  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_agent ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_calls_tool ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_calls_status ON tool_calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_started ON tool_calls(started_at);

CREATE INDEX IF NOT EXISTS idx_events_trace ON trace_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_events_call ON trace_events(call_id);
CREATE INDEX IF NOT EXISTS idx_events_severity ON trace_events(severity);

CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id);
"""


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SQLiteStore:
    """Persistent SQLite storage backend.

    Thread-safe via a per-instance lock.  Connections are opened per-call
    to avoid cross-thread issues with SQLite's default threading model.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._local = threading.local()

        # Initialize schema
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ── writes ──────────────────────────────────────────────────────

    def save_call(self, call: ToolCall) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO tool_calls
                   (id, session_id, agent_id, tool_name, server_name,
                    arguments, result, status, error,
                    started_at, completed_at, duration_ms,
                    input_tokens, output_tokens, cost_usd,
                    tags, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    call.id,
                    call.session_id,
                    call.agent_id,
                    call.tool_name,
                    call.server_name,
                    json.dumps(call.arguments, default=str),
                    json.dumps(call.result, default=str) if call.result is not None else None,
                    call.status.value,
                    call.error,
                    _dt_to_str(call.started_at),
                    _dt_to_str(call.completed_at),
                    call.duration_ms,
                    call.input_tokens,
                    call.output_tokens,
                    call.cost_usd,
                    json.dumps(call.tags),
                    json.dumps(call.metadata, default=str),
                ),
            )
            conn.commit()

    def save_session(self, session: Session) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, agent_id, name, started_at, ended_at,
                    tags, metadata,
                    total_calls, error_count, total_cost_usd, total_tokens)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session.id,
                    session.agent_id,
                    session.name,
                    _dt_to_str(session.started_at),
                    _dt_to_str(session.ended_at),
                    json.dumps(session.tags),
                    json.dumps(session.metadata, default=str),
                    session.total_calls,
                    session.error_count,
                    session.total_cost_usd,
                    session.total_tokens,
                ),
            )
            conn.commit()

    def save_event(self, event: TraceEvent) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO trace_events
                   (id, trace_id, call_id, event_type, message,
                    severity, data, timestamp, duration_ms)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    event.id,
                    event.trace_id,
                    event.call_id,
                    event.event_type,
                    event.message,
                    event.severity.value,
                    json.dumps(event.data, default=str),
                    _dt_to_str(event.timestamp),
                    event.duration_ms,
                ),
            )
            conn.commit()

    def save_rule(self, rule: AlertRule) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO alert_rules
                   (id, name, metric, operator, threshold,
                    window, enabled, created_at, last_triggered, trigger_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    rule.id,
                    rule.name,
                    rule.metric,
                    rule.operator,
                    rule.threshold,
                    rule.window,
                    1 if rule.enabled else 0,
                    _dt_to_str(rule.created_at),
                    _dt_to_str(rule.last_triggered),
                    rule.trigger_count,
                ),
            )
            conn.commit()

    def delete_rule(self, rule_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
            conn.commit()
            return cur.rowcount > 0

    # ── single reads ────────────────────────────────────────────────

    def get_call(self, call_id: str) -> ToolCall | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tool_calls WHERE id = ?", (call_id,)).fetchone()
        return self._row_to_call(row) if row else None

    def get_session(self, session_id: str) -> Session | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def get_rule(self, rule_id: str) -> AlertRule | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,)).fetchone()
        return self._row_to_rule(row) if row else None

    # ── queries ─────────────────────────────────────────────────────

    def query_calls(
        self,
        *,
        session_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        server_name: str | None = None,
        status: str | None = None,
        min_cost: float | None = None,
        min_duration: float | None = None,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolCall]:
        clauses: list[str] = []
        params: list[Any] = []

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if tool_name:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if server_name:
            clauses.append("server_name = ?")
            params.append(server_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if min_cost is not None:
            clauses.append("cost_usd >= ?")
            params.append(min_cost)
        if min_duration is not None:
            clauses.append("duration_ms >= ?")
            params.append(min_duration)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM tool_calls{where} ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_call(r) for r in rows]

    def query_sessions(
        self,
        *,
        agent_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[Session]:
        clauses: list[str] = []
        params: list[Any] = []

        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if active_only:
            clauses.append("ended_at IS NULL")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM sessions{where} ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_session(r) for r in rows]

    def query_events(
        self,
        *,
        trace_id: str | None = None,
        call_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[TraceEvent]:
        clauses: list[str] = []
        params: list[Any] = []

        if trace_id:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if call_id:
            clauses.append("call_id = ?")
            params.append(call_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM trace_events{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def all_rules(self) -> list[AlertRule]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM alert_rules").fetchall()
        return [self._row_to_rule(r) for r in rows]

    def count_calls(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM tool_calls")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM trace_events")
            conn.execute("DELETE FROM alert_rules")
            conn.commit()

    # ── row mappers ─────────────────────────────────────────────────

    @staticmethod
    def _row_to_call(row: sqlite3.Row) -> ToolCall:
        return ToolCall(
            id=row["id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            tool_name=row["tool_name"],
            server_name=row["server_name"],
            arguments=json.loads(row["arguments"] or "{}"),
            result=json.loads(row["result"]) if row["result"] else None,
            status=CallStatus(row["status"]),
            error=row["error"],
            started_at=_str_to_dt(row["started_at"]) or datetime.now(timezone.utc),
            completed_at=_str_to_dt(row["completed_at"]),
            duration_ms=row["duration_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost_usd=row["cost_usd"],
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            agent_id=row["agent_id"],
            name=row["name"],
            started_at=_str_to_dt(row["started_at"]) or datetime.now(timezone.utc),
            ended_at=_str_to_dt(row["ended_at"]),
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            total_calls=row["total_calls"],
            error_count=row["error_count"],
            total_cost_usd=row["total_cost_usd"],
            total_tokens=row["total_tokens"],
        )

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            id=row["id"],
            trace_id=row["trace_id"],
            call_id=row["call_id"],
            event_type=row["event_type"],
            message=row["message"],
            severity=Severity(row["severity"]),
            data=json.loads(row["data"] or "{}"),
            timestamp=_str_to_dt(row["timestamp"]) or datetime.now(timezone.utc),
            duration_ms=row["duration_ms"],
        )

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> AlertRule:
        return AlertRule(
            id=row["id"],
            name=row["name"],
            metric=row["metric"],
            operator=row["operator"],
            threshold=row["threshold"],
            window=row["window"],
            enabled=bool(row["enabled"]),
            created_at=_str_to_dt(row["created_at"]) or datetime.now(timezone.utc),
            last_triggered=_str_to_dt(row["last_triggered"]),
            trigger_count=row["trigger_count"],
        )
