"""Tests for the data export module (JSONL, CSV, string)."""
from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from mcp_audit import AuditEngine, CallStatus
from mcp_audit.export import (
    export_calls_csv,
    export_calls_jsonl,
    export_events_jsonl,
    export_to_string,
)


@pytest.fixture
def engine():
    return AuditEngine()


@pytest.fixture
def populated_engine(engine):
    """Engine with some calls for export testing."""
    session = engine.start_session(agent_id="export-agent", name="export test")
    for i in range(5):
        engine.record_call(
            session_id=session.id,
            tool_name="search" if i < 3 else "fetch",
            agent_id="export-agent",
            status=CallStatus.SUCCESS if i < 4 else CallStatus.ERROR,
            duration_ms=float(i * 100 + 50),
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
            cost_usd=0.001 * (i + 1),
            tags=["test", f"batch-{i}"],
            result={"index": i},
        )
    return engine


class TestExportJSONL:
    def test_export_calls_jsonl(self, populated_engine, tmp_path):
        output = tmp_path / "calls.jsonl"
        result = export_calls_jsonl(populated_engine, str(output))

        assert result["format"] == "jsonl"
        assert result["record_count"] == 5
        assert result["size_bytes"] > 0
        assert os.path.exists(str(output))

        # Read and validate
        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 5

        first = json.loads(lines[0])
        assert "id" in first
        assert "tool_name" in first
        assert "status" in first
        assert "cost_usd" in first
        assert "started_at" in first

    def test_export_calls_jsonl_with_filter(self, populated_engine, tmp_path):
        output = tmp_path / "filtered.jsonl"

        # Get session id from the engine
        sessions = populated_engine.list_sessions()
        session_id = sessions[0].id

        result = export_calls_jsonl(
            populated_engine, str(output), session_id=session_id
        )
        assert result["record_count"] == 5

    def test_export_empty_engine(self, engine, tmp_path):
        output = tmp_path / "empty.jsonl"
        result = export_calls_jsonl(engine, str(output))
        assert result["record_count"] == 0

        with open(output) as f:
            content = f.read()
        assert content == ""


class TestExportCSV:
    def test_export_calls_csv(self, populated_engine, tmp_path):
        output = tmp_path / "calls.csv"
        result = export_calls_csv(populated_engine, str(output))

        assert result["format"] == "csv"
        assert result["record_count"] == 5

        with open(output, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert "id" in rows[0]
        assert "tool_name" in rows[0]
        assert "cost_usd" in rows[0]
        assert "total_tokens" in rows[0]

    def test_csv_has_header(self, populated_engine, tmp_path):
        output = tmp_path / "header.csv"
        export_calls_csv(populated_engine, str(output))

        with open(output) as f:
            header_line = f.readline().strip()

        fields = header_line.split(",")
        assert "tool_name" in fields
        assert "cost_usd" in fields

    def test_csv_tags_are_json(self, populated_engine, tmp_path):
        output = tmp_path / "tags.csv"
        export_calls_csv(populated_engine, str(output))

        with open(output, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        tags = json.loads(rows[0]["tags"])
        assert isinstance(tags, list)
        assert "test" in tags


class TestExportEvents:
    def test_export_events_jsonl(self, engine, tmp_path):
        session = engine.start_session(agent_id="a1")
        for i in range(3):
            engine.log_event(
                trace_id=session.id,
                event_type="http_request",
                message=f"request {i}",
                data={"status": 200},
            )

        output = tmp_path / "events.jsonl"
        result = export_events_jsonl(engine, str(output), trace_id=session.id)

        assert result["record_count"] == 3

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert first["event_type"] == "http_request"
        assert first["data"]["status"] == 200


class TestExportToString:
    def test_export_jsonl_string(self, populated_engine):
        text = export_to_string(populated_engine, fmt="jsonl", limit=3)

        lines = text.strip().split("\n")
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert "tool_name" in first

    def test_export_csv_string(self, populated_engine):
        text = export_to_string(populated_engine, fmt="csv", limit=3)

        lines = text.strip().split("\n")
        assert len(lines) == 4  # header + 3 rows

    def test_invalid_format(self, engine):
        with pytest.raises(ValueError, match="Unknown format"):
            export_to_string(engine, fmt="xml")

    def test_export_string_empty(self, engine):
        text = export_to_string(engine, fmt="jsonl")
        assert text == ""


class TestExportWithToolFilter:
    def test_export_by_tool_name(self, populated_engine, tmp_path):
        output = tmp_path / "search_only.jsonl"
        result = export_calls_jsonl(
            populated_engine, str(output), tool_name="search"
        )
        assert result["record_count"] == 3

        with open(output) as f:
            for line in f:
                record = json.loads(line)
                assert record["tool_name"] == "search"


class TestExportCreatesParentDirs:
    def test_creates_parent_directory(self, populated_engine, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "calls.jsonl"
        result = export_calls_jsonl(populated_engine, str(nested))
        assert result["record_count"] == 5
        assert nested.exists()
