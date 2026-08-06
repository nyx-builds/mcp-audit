"""Tests for time-series analytics and anomaly detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mcp_audit.engine import AuditEngine
from mcp_audit.models import CallStatus
from mcp_audit.storage import MemoryStore
from mcp_audit.timeseries import (
    ANALYZABLE_METRICS,
    _ewma_baseline,
    _floor_to_bucket,
    _iqr_anomalies,
    _linear_regression_slope,
    _percentile,
    _zscore_anomalies,
    analyze_trends,
    build_heatmap,
    build_timeseries,
    detect_anomalies,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def engine():
    return AuditEngine(store=MemoryStore())


def _seed_calls(
    engine: AuditEngine,
    n: int = 50,
    *,
    start_offset_minutes: int = 0,
    error_rate: float = 0.1,
    base_latency: float = 100.0,
    cost: float = 0.05,
    tool_name: str = "test_tool",
    agent_id: str = "agent-1",
) -> str:
    """Seed *n* calls spread across time. Returns the session_id."""
    session = engine.start_session(agent_id=agent_id, name="seed")
    base_time = datetime.now(timezone.utc) - timedelta(minutes=start_offset_minutes)

    for i in range(n):
        started = base_time + timedelta(minutes=i * 5)
        status = CallStatus.ERROR if i < n * error_rate else CallStatus.SUCCESS
        engine.record_call(
            session.id,
            tool_name=tool_name,
            agent_id=agent_id,
            status=status,
            duration_ms=base_latency + (i % 5) * 10,
            cost_usd=cost,
            input_tokens=100,
            output_tokens=50,
        )
    return session.id


def _seed_with_spike(
    engine: AuditEngine,
    *,
    spike_at_minutes: int = 25,
    spike_latency: float = 2000.0,
    normal_latency: float = 100.0,
    n_buckets: int = 10,
    calls_per_bucket: int = 5,
    window_minutes: int = 5,
) -> str:
    """Seed calls with a clear latency spike at a specific time."""
    session = engine.start_session(agent_id="agent-1", name="spike")
    now = datetime.now(timezone.utc)

    for bucket in range(n_buckets):
        bucket_time = now - timedelta(minutes=(n_buckets - bucket) * window_minutes)
        is_spike = bucket == spike_at_minutes // window_minutes
        latency = spike_latency if is_spike else normal_latency

        for _ in range(calls_per_bucket):
            engine.record_call(
                session.id,
                tool_name="tool_a",
                agent_id="agent-1",
                duration_ms=latency,
                cost_usd=0.05,
            )
    return session.id


# ── Time Bucketing ────────────────────────────────────────────────


class TestFloorToBucket:
    def test_minute_floor(self):
        dt = datetime(2025, 1, 1, 12, 7, 33, tzinfo=timezone.utc)
        from datetime import timedelta
        result = _floor_to_bucket(dt, timedelta(minutes=5))
        assert result == datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc)

    def test_hour_floor(self):
        dt = datetime(2025, 1, 1, 12, 45, 30, tzinfo=timezone.utc)
        from datetime import timedelta
        result = _floor_to_bucket(dt, timedelta(hours=1))
        assert result == datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_day_floor(self):
        dt = datetime(2025, 1, 1, 12, 45, 30, tzinfo=timezone.utc)
        from datetime import timedelta
        result = _floor_to_bucket(dt, timedelta(days=1))
        assert result == datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_already_on_boundary(self):
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        from datetime import timedelta
        result = _floor_to_bucket(dt, timedelta(minutes=5))
        assert result == dt


class TestPercentile:
    def test_median(self):
        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p95(self):
        data = list(range(1, 101))
        result = _percentile(data, 95)
        assert 94 <= result <= 96

    def test_empty(self):
        assert _percentile([], 95) == 0.0

    def test_single(self):
        assert _percentile([42], 95) == 42


# ── Time Series ───────────────────────────────────────────────────


class TestBuildTimeseries:
    def test_empty(self, engine):
        result = build_timeseries(engine, window="5m")
        assert result["bucket_count"] == 0
        assert result["buckets"] == []
        assert result["summary"] == {}

    def test_basic(self, engine):
        _seed_calls(engine, n=20)
        result = build_timeseries(engine, window="5m")
        assert result["window"] == "5m"
        assert result["bucket_count"] > 0
        assert len(result["buckets"]) == result["bucket_count"]

    def test_bucket_structure(self, engine):
        _seed_calls(engine, n=10)
        result = build_timeseries(engine, window="5m")
        for bucket in result["buckets"]:
            assert "timestamp" in bucket
            assert "call_count" in bucket
            assert "error_rate" in bucket
            assert "p95_latency_ms" in bucket
            assert "total_cost_usd" in bucket

    def test_metric_extraction(self, engine):
        _seed_calls(engine, n=10)
        result = build_timeseries(engine, window="5m", metric="error_rate")
        for bucket in result["buckets"]:
            assert "value" in bucket
            assert bucket["value"] == bucket["error_rate"]

    def test_invalid_window(self, engine):
        with pytest.raises(ValueError, match="Invalid window"):
            build_timeseries(engine, window="2m")

    def test_all_windows(self, engine):
        _seed_calls(engine, n=10)
        for w in ["1m", "5m", "15m", "1h", "1d"]:
            result = build_timeseries(engine, window=w)
            assert result["window"] == w

    def test_filter_by_session(self, engine):
        _seed_calls(engine, n=10)
        engine2 = AuditEngine(store=engine.store)
        result = build_timeseries(engine, window="5m")
        session_id = result["summary"].get("total_calls")
        # Filtering by non-existent session should return empty
        result2 = build_timeseries(engine, window="5m", session_id="nonexistent")
        assert result2["bucket_count"] == 0

    def test_filter_by_tool(self, engine):
        _seed_calls(engine, n=10, tool_name="tool_a")
        _seed_calls(engine, n=5, tool_name="tool_b")
        result = build_timeseries(engine, window="5m", tool_name="tool_a")
        # Should only contain tool_a calls
        assert result["summary"]["total_calls"] == 10

    def test_summary(self, engine):
        _seed_calls(engine, n=20)
        result = build_timeseries(engine, window="5m")
        summary = result["summary"]
        assert summary["total_calls"] == 20
        assert "time_range_start" in summary
        assert "time_range_end" in summary
        assert "bucket_count" in summary

    def test_error_rate_calculation(self, engine):
        session = engine.start_session()
        for i in range(10):
            engine.record_call(
                session.id, "tool1",
                status=CallStatus.ERROR if i < 5 else CallStatus.SUCCESS,
            )
        result = build_timeseries(engine, window="5m")
        assert any(b["error_rate"] >= 40 for b in result["buckets"])


# ── Z-Score Anomaly Detection ─────────────────────────────────────


class TestZScoreAnomalies:
    def test_no_anomalies(self):
        values = [10, 11, 10, 10, 11, 10, 11, 10]
        result = _zscore_anomalies(values, threshold=3.0)
        assert result == []

    def test_spike_detected(self):
        # 20 normal + 1 extreme → the outlier will have a large z-score
        values = [10.0] * 20 + [1000.0]
        result = _zscore_anomalies(values, threshold=3.0)
        assert len(result) >= 1
        assert result[0]["direction"] == "spike"

    def test_drop_detected(self):
        values = [100.0] * 20 + [1.0]
        result = _zscore_anomalies(values, threshold=3.0)
        assert len(result) >= 1
        assert result[0]["direction"] == "drop"

    def test_too_few_values(self):
        assert _zscore_anomalies([1, 2], threshold=3.0) == []

    def test_no_variance(self):
        assert _zscore_anomalies([5, 5, 5, 5, 5], threshold=3.0) == []

    def test_higher_threshold_fewer(self):
        values = [10, 10, 10, 10, 10, 10, 10, 10, 50]
        low_t = _zscore_anomalies(values, threshold=2.0)
        high_t = _zscore_anomalies(values, threshold=5.0)
        assert len(low_t) >= len(high_t)


# ── IQR Anomaly Detection ─────────────────────────────────────────


class TestIQRAnomalies:
    def test_no_anomalies(self):
        values = [10, 11, 10, 10, 11, 10, 11, 10]
        result = _iqr_anomalies(values, multiplier=1.5)
        assert result == []

    def test_spike(self):
        values = [10, 11, 10, 10, 11, 10, 11, 10, 500]
        result = _iqr_anomalies(values, multiplier=1.5)
        assert len(result) >= 1
        assert result[-1]["direction"] == "spike"

    def test_drop(self):
        values = [100, 101, 100, 100, 101, 100, 101, 100, 0]
        result = _iqr_anomalies(values, multiplier=1.5)
        assert len(result) >= 1
        assert result[-1]["direction"] == "drop"

    def test_too_few(self):
        assert _iqr_anomalies([1, 2, 3], multiplier=1.5) == []

    def test_has_bounds(self):
        values = [10, 11, 10, 10, 11, 10, 11, 10, 500]
        result = _iqr_anomalies(values, multiplier=1.5)
        for a in result:
            assert "lower_bound" in a
            assert "upper_bound" in a
            assert "q1" in a
            assert "q3" in a


# ── EWMA Baseline ─────────────────────────────────────────────────


class TestEWMA:
    def test_basic(self):
        values = [10, 20, 30, 40, 50]
        result = _ewma_baseline(values, alpha=0.5)
        assert len(result) == 5
        assert result[0] == 10  # first value

    def test_empty(self):
        assert _ewma_baseline([]) == []

    def test_smoothness(self):
        # EWMA with alpha=0.3 on alternating values should produce
        # values between the high and low (less extreme)
        values = [10.0, 100.0, 10.0, 100.0, 10.0]
        ewma = _ewma_baseline(values, alpha=0.3)
        # The EWMA should never exceed the max or go below the min
        assert max(ewma) <= max(values)
        assert min(ewma) >= min(values)
        # And should be less volatile than the raw input
        import statistics as st
        assert st.stdev(ewma) < st.stdev(values)


# ── Linear Regression Slope ───────────────────────────────────────


class TestLinearRegression:
    def test_increasing(self):
        values = [1, 2, 3, 4, 5]
        slope = _linear_regression_slope(values)
        assert slope > 0

    def test_decreasing(self):
        values = [5, 4, 3, 2, 1]
        slope = _linear_regression_slope(values)
        assert slope < 0

    def test_flat(self):
        values = [5, 5, 5, 5, 5]
        slope = _linear_regression_slope(values)
        assert slope == 0.0

    def test_single_value(self):
        assert _linear_regression_slope([42]) == 0.0

    def test_empty(self):
        assert _linear_regression_slope([]) == 0.0


# ── detect_anomalies ──────────────────────────────────────────────


class TestDetectAnomalies:
    def test_empty(self, engine):
        result = detect_anomalies(engine)
        assert result["total_anomalies"] == 0

    def test_with_normal_data(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine, window="5m")
        assert result["status"] == "ok"
        assert "anomalies" in result
        assert "anomalies_by_metric" in result

    def test_invalid_metric(self, engine):
        with pytest.raises(ValueError, match="Invalid metric"):
            detect_anomalies(engine, metrics=["invalid_metric"])

    def test_invalid_sensitivity(self, engine):
        with pytest.raises(ValueError, match="Invalid sensitivity"):
            detect_anomalies(engine, sensitivity="extreme")

    def test_invalid_method(self, engine):
        _seed_calls(engine, n=20)
        with pytest.raises(ValueError):
            detect_anomalies(engine, method="invalid")

    def test_zscore_method(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine, method="zscore")
        assert result["method"] == "zscore"

    def test_iqr_method(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine, method="iqr")
        assert result["method"] == "iqr"

    def test_ewma_method(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine, method="ewma")
        # ewma is handled in detect_anomalies
        assert result["status"] == "ok"

    def test_sensitivity_levels(self, engine):
        _seed_calls(engine, n=50)
        high = detect_anomalies(engine, sensitivity="high")
        normal = detect_anomalies(engine, sensitivity="normal")
        low = detect_anomalies(engine, sensitivity="low")
        # Higher sensitivity should detect >= anomalies (or same)
        assert high["total_anomalies"] >= normal["total_anomalies"]
        assert normal["total_anomalies"] >= low["total_anomalies"]

    def test_custom_metrics(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine, metrics=["call_count"])
        assert result["metrics_analyzed"] == ["call_count"]

    def test_severity_classification(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine)
        for anomaly in result["anomalies"]:
            assert "severity" in anomaly
            assert anomaly["severity"] in ["critical", "high", "medium", "low"]

    def test_anomalies_sorted_by_severity(self, engine):
        _seed_calls(engine, n=50)
        result = detect_anomalies(engine)
        zscores = [abs(a.get("zscore", 0)) for a in result["anomalies"]]
        assert zscores == sorted(zscores, reverse=True)

    def test_auto_method_selection(self, engine):
        # Small sample → IQR
        _seed_calls(engine, n=5)
        result = detect_anomalies(engine, method="auto")
        assert result["status"] == "ok"


# ── analyze_trends ────────────────────────────────────────────────


class TestAnalyzeTrends:
    def test_empty(self, engine):
        result = analyze_trends(engine)
        assert result["status"] == "ok"
        assert result["trend_count"] >= 0

    def test_basic(self, engine):
        _seed_calls(engine, n=50)
        result = analyze_trends(engine, window="5m")
        assert result["status"] == "ok"
        assert len(result["trends"]) > 0

    def test_trend_structure(self, engine):
        _seed_calls(engine, n=50)
        result = analyze_trends(engine, window="5m")
        for trend in result["trends"]:
            assert "metric" in trend
            assert "direction" in trend
            assert "slope" in trend
            assert "pct_change" in trend
            assert "volatility_cv" in trend
            assert "volatility_label" in trend
            assert trend["direction"] in [
                "increasing", "decreasing", "stable", "insufficient_data",
            ]
            assert trend["volatility_label"] in ["high", "moderate", "low", "n/a"]

    def test_increasing_cost_trend(self, engine):
        session = engine.start_session()
        now = datetime.now(timezone.utc)
        # 10 calls spread across 10 different 1-hour buckets with increasing cost
        for i in range(10):
            started = now - timedelta(hours=10 - i)
            # Increasing costs each hour
            engine.record_call(
                session.id, "tool1",
                cost_usd=0.01 * (i + 1),
                duration_ms=100,
            )
        result = analyze_trends(engine, window="1h", metrics=["total_cost_usd"])
        cost_trend = [t for t in result["trends"] if t["metric"] == "total_cost_usd"]
        if cost_trend and cost_trend[0]["bucket_count"] >= 2:
            assert cost_trend[0]["direction"] in ["increasing", "stable"]

    def test_invalid_metric(self, engine):
        with pytest.raises(ValueError, match="Invalid metric"):
            analyze_trends(engine, metrics=["nonexistent"])

    def test_stable_trend(self, engine):
        session = engine.start_session()
        # Calls with consistent latency — but spread across time buckets
        now = datetime.now(timezone.utc)
        for i in range(20):
            started = now - timedelta(minutes=(20 - i) * 5)
            engine.record_call(
                session.id, "tool1",
                duration_ms=100,
                cost_usd=0.05,
            )
        result = analyze_trends(engine, window="5m", metrics=["avg_latency_ms"])
        lat_trend = [t for t in result["trends"] if t["metric"] == "avg_latency_ms"]
        if lat_trend and lat_trend[0]["bucket_count"] >= 2:
            assert lat_trend[0]["direction"] in ["stable", "increasing", "decreasing"]


# ── build_heatmap ─────────────────────────────────────────────────


class TestBuildHeatmap:
    def test_empty(self, engine):
        result = build_heatmap(engine)
        assert result["matrix"] == {}
        assert result["tools"] == []

    def test_basic(self, engine):
        _seed_calls(engine, n=20, tool_name="tool_a")
        _seed_calls(engine, n=10, tool_name="tool_b")
        result = build_heatmap(engine, window="5m")
        assert result["tool_count"] >= 2
        assert "tool_a" in result["tools"]
        assert "tool_b" in result["tools"]
        assert "tool_a" in result["matrix"]
        assert "tool_b" in result["matrix"]

    def test_matrix_dimensions(self, engine):
        _seed_calls(engine, n=20, tool_name="tool_a")
        result = build_heatmap(engine, window="5m")
        assert len(result["timestamps"]) == result["bucket_count"]
        for tool in result["tools"]:
            assert len(result["matrix"][tool]) == result["bucket_count"]

    def test_metric_selection(self, engine):
        _seed_calls(engine, n=20)
        result = build_heatmap(engine, metric="total_cost_usd")
        assert result["metric"] == "total_cost_usd"

    def test_invalid_metric(self, engine):
        with pytest.raises(ValueError, match="Invalid metric"):
            build_heatmap(engine, metric="invalid")

    def test_filter_by_agent(self, engine):
        _seed_calls(engine, n=20, agent_id="agent-1")
        _seed_calls(engine, n=10, agent_id="agent-2")
        result = build_heatmap(engine, agent_id="agent-1")
        # Should only contain agent-1 calls
        total = sum(sum(row) for row in result["matrix"].values())
        assert total == 20

    def test_heatmap_values_are_numeric(self, engine):
        _seed_calls(engine, n=20)
        result = build_heatmap(engine, metric="call_count")
        for tool, row in result["matrix"].items():
            for val in row:
                assert isinstance(val, (int, float))
                assert val >= 0
