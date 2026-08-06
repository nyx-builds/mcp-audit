"""Time-series analytics and anomaly detection.

This module provides time-bucketed aggregation of tool calls, trend analysis,
and statistical anomaly detection. It lets operators answer questions like:

* "How did error rate change over the last hour, bucketed by 5-minute windows?"
* "Is the current p95 latency anomalous compared to the recent baseline?"
* "Are costs trending up or down?"

The anomaly detector uses a robust statistical approach: it builds a baseline
from the rolling mean and standard deviation of recent observations, then flags
data points whose z-score exceeds a configurable threshold.  For small sample
sizes it falls back to an IQR (interquartile range) method that is resilient
to outliers.

No external dependencies — pure Python statistics.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .engine import AuditEngine
from .models import CallStatus, ToolCall


# ── Time bucketing ────────────────────────────────────────────────


class TimeWindow(str, Enum):
    """Supported bucket sizes for time-series aggregation."""

    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    HOUR = "1h"
    DAY = "1d"

    @property
    def timedelta(self) -> timedelta:
        return {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }[self.value]


def _window_from_string(s: str) -> TimeWindow:
    """Parse a window string into a TimeWindow enum."""
    try:
        return TimeWindow(s)
    except ValueError:
        valid = [w.value for w in TimeWindow]
        raise ValueError(f"Invalid window '{s}'. Must be one of {valid}") from None


def _floor_to_bucket(dt: datetime, bucket_td: timedelta) -> datetime:
    """Floor a datetime to the nearest bucket boundary."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = (dt - epoch).total_seconds()
    bucket_seconds = bucket_td.total_seconds()
    floored = math.floor(seconds / bucket_seconds) * bucket_seconds
    return epoch + timedelta(seconds=floored)


def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def _bucket_metrics(calls: list[ToolCall]) -> dict[str, Any]:
    """Compute metrics for a set of calls within one time bucket."""
    if not calls:
        return {
            "call_count": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "avg_cost_usd": 0.0,
            "total_tokens": 0,
            "unique_tools": 0,
            "unique_sessions": 0,
        }

    total = len(calls)
    errors = sum(1 for c in calls if c.status == CallStatus.ERROR)
    timeouts = sum(1 for c in calls if c.status == CallStatus.TIMEOUT)
    blocked = sum(1 for c in calls if c.status == CallStatus.BLOCKED)
    durations = [c.duration_ms or 0.0 for c in calls]
    costs = [c.cost_usd for c in calls]
    tokens = [c.input_tokens + c.output_tokens for c in calls]

    return {
        "call_count": total,
        "error_count": errors,
        "timeout_count": timeouts,
        "blocked_count": blocked,
        "error_rate": round(errors / total * 100, 2) if total else 0.0,
        "timeout_rate": round(timeouts / total * 100, 2) if total else 0.0,
        "avg_latency_ms": round(statistics.mean(durations), 2) if durations else 0.0,
        "p50_latency_ms": round(_percentile(durations, 50), 2),
        "p95_latency_ms": round(_percentile(durations, 95), 2),
        "p99_latency_ms": round(_percentile(durations, 99), 2),
        "total_cost_usd": round(sum(costs), 6),
        "avg_cost_usd": round(sum(costs) / total, 6) if total else 0.0,
        "total_tokens": sum(tokens),
        "avg_tokens_per_call": round(sum(tokens) / total, 1) if total else 0.0,
        "unique_tools": len({c.tool_name for c in calls}),
        "unique_sessions": len({c.session_id for c in calls}),
    }


def build_timeseries(
    engine: AuditEngine,
    *,
    window: str = "5m",
    metric: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Build a time-series of call metrics bucketed by *window*.

    Returns a dict with:
    - ``window``: the bucket size
    - ``metric``: which metric was extracted (if specified)
    - ``buckets``: list of ``{timestamp, ...metrics}`` dicts, oldest first
    - ``summary``: aggregate stats across all buckets
    """
    tw = _window_from_string(window)
    bucket_td = tw.timedelta

    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    if not calls:
        return {
            "window": window,
            "metric": metric,
            "bucket_count": 0,
            "buckets": [],
            "summary": {},
        }

    # Group calls into time buckets
    grouped: dict[datetime, list[ToolCall]] = defaultdict(list)
    for call in calls:
        bucket_start = _floor_to_bucket(call.started_at, bucket_td)
        grouped[bucket_start].append(call)

    # Build sorted bucket list (oldest first)
    sorted_starts = sorted(grouped.keys())
    buckets: list[dict[str, Any]] = []
    for start in sorted_starts:
        bucket_calls = grouped[start]
        metrics = _bucket_metrics(bucket_calls)
        entry: dict[str, Any] = {
            "timestamp": start.isoformat(),
        }
        # If a specific metric is requested, add a top-level 'value' key
        if metric and metric in metrics:
            entry["value"] = metrics[metric]
        entry.update(metrics)
        buckets.append(entry)

    # Compute summary across all buckets
    all_durations = [c.duration_ms or 0.0 for c in calls]
    all_costs = [c.cost_usd for c in calls]
    total_calls = len(calls)
    total_errors = sum(1 for c in calls if c.status == CallStatus.ERROR)
    summary = {
        "total_calls": total_calls,
        "total_errors": total_errors,
        "overall_error_rate": round(total_errors / total_calls * 100, 2) if total_calls else 0.0,
        "overall_p95_latency_ms": round(_percentile(all_durations, 95), 2),
        "total_cost_usd": round(sum(all_costs), 6),
        "time_range_start": sorted_starts[0].isoformat(),
        "time_range_end": (sorted_starts[-1] + bucket_td).isoformat(),
        "bucket_count": len(buckets),
    }

    return {
        "window": window,
        "metric": metric,
        "bucket_count": len(buckets),
        "buckets": buckets,
        "summary": summary,
    }


# ── Anomaly Detection ─────────────────────────────────────────────


def _zscore_anomalies(
    values: list[float],
    threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Detect anomalies using z-score (mean ± threshold × std).

    Returns a list of ``{index, value, zscore, direction}`` dicts.
    Uses the full series to compute baseline statistics.
    """
    if len(values) < 3:
        return []

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return []  # no variance, no anomalies

    anomalies: list[dict[str, Any]] = []
    for i, val in enumerate(values):
        z = (val - mean) / stdev
        if abs(z) >= threshold:
            anomalies.append({
                "bucket_index": i,
                "value": round(val, 6),
                "zscore": round(z, 2),
                "direction": "spike" if z > 0 else "drop",
                "baseline_mean": round(mean, 6),
                "baseline_stddev": round(stdev, 6),
                "method": "zscore",
            })
    return anomalies


def _iqr_anomalies(
    values: list[float],
    multiplier: float = 1.5,
) -> list[dict[str, Any]]:
    """Detect anomalies using the IQR (interquartile range) method.

    Points below Q1 − multiplier×IQR or above Q3 + multiplier×IQR are flagged.
    More robust to outliers than z-score for small datasets.
    """
    if len(values) < 4:
        return []

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = _percentile(sorted_vals, 25)
    q3 = _percentile(sorted_vals, 75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    anomalies: list[dict[str, Any]] = []
    for i, val in enumerate(values):
        if val < lower:
            anomalies.append({
                "bucket_index": i,
                "value": round(val, 6),
                "direction": "drop",
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "method": "iqr",
            })
        elif val > upper:
            anomalies.append({
                "bucket_index": i,
                "value": round(val, 6),
                "direction": "spike",
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
                "q1": round(q1, 6),
                "q3": round(q3, 6),
                "method": "iqr",
            })
    return anomalies


def _ewma_baseline(values: list[float], alpha: float = 0.3) -> list[float]:
    """Compute an EWMA (exponentially weighted moving average) baseline."""
    if not values:
        return []
    baseline = [values[0]]
    for i in range(1, len(values)):
        ewma = alpha * values[i] + (1 - alpha) * baseline[-1]
        baseline.append(ewma)
    return baseline


# Metrics that can be analyzed for anomalies
ANALYZABLE_METRICS = [
    "call_count",
    "error_rate",
    "timeout_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "total_cost_usd",
    "avg_cost_usd",
    "total_tokens",
]


def detect_anomalies(
    engine: AuditEngine,
    *,
    window: str = "5m",
    metrics: list[str] | None = None,
    method: str = "auto",
    sensitivity: str = "normal",
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Detect anomalous buckets across one or more metrics.

    Parameters
    ----------
    window : str
        Time bucket size (1m, 5m, 15m, 1h, 1d).
    metrics : list[str] | None
        Which metrics to analyze.  Default: error_rate, p95_latency_ms,
        total_cost_usd.
    method : str
        Detection method: ``auto`` (pick based on sample size), ``zscore``,
        ``iqr``, or ``ewma``.
    sensitivity : str
        ``high`` (more sensitive, lower threshold), ``normal``, or ``low``
        (fewer false positives, higher threshold).
    """
    if metrics is None:
        metrics = ["error_rate", "p95_latency_ms", "total_cost_usd"]

    # Validate metrics
    invalid = [m for m in metrics if m not in ANALYZABLE_METRICS]
    if invalid:
        raise ValueError(
            f"Invalid metric(s): {invalid}. Must be one of {ANALYZABLE_METRICS}"
        )

    # Sensitivity to threshold mapping
    sensitivity_thresholds = {
        "high": {"zscore": 2.0, "iqr": 1.0, "ewma": 2.0},
        "normal": {"zscore": 3.0, "iqr": 1.5, "ewma": 3.0},
        "low": {"zscore": 4.0, "iqr": 2.0, "ewma": 4.0},
    }
    if sensitivity not in sensitivity_thresholds:
        raise ValueError(
            f"Invalid sensitivity '{sensitivity}'. Must be 'high', 'normal', or 'low'."
        )
    thresholds = sensitivity_thresholds[sensitivity]

    # Validate method
    if method not in ("auto", "zscore", "iqr", "ewma"):
        raise ValueError(
            f"Invalid method '{method}'. Must be 'auto', 'zscore', 'iqr', or 'ewma'."
        )

    # Build timeseries once
    ts = build_timeseries(
        engine,
        window=window,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    buckets = ts.get("buckets", [])
    bucket_count = len(buckets)
    anomalies_by_metric: dict[str, list[dict[str, Any]]] = {}
    all_anomalies: list[dict[str, Any]] = []

    for m in metrics:
        values = [b.get(m, 0) for b in buckets]

        # Determine method
        if method == "auto":
            if len(values) >= 8:
                use_method = "zscore"
            elif len(values) >= 4:
                use_method = "iqr"
            else:
                # Not enough data — skip
                anomalies_by_metric[m] = []
                continue
        else:
            use_method = method

        # Detect
        if use_method == "zscore":
            detected = _zscore_anomalies(values, threshold=thresholds["zscore"])
        elif use_method == "iqr":
            detected = _iqr_anomalies(values, multiplier=thresholds["iqr"])
        elif use_method == "ewma":
            detected = _ewma_anomaly_check(values, threshold=thresholds["ewma"])
        else:
            detected = []

        # Enrich with timestamp info
        for a in detected:
            idx = a["bucket_index"]
            if idx < bucket_count:
                a["timestamp"] = buckets[idx]["timestamp"]
                a["metric"] = m
            all_anomalies.append(a)

        anomalies_by_metric[m] = detected

    # Severity classification
    for a in all_anomalies:
        z = a.get("zscore", 0)
        if abs(z) >= 4:
            a["severity"] = "critical"
        elif abs(z) >= 3:
            a["severity"] = "high"
        elif abs(z) >= 2:
            a["severity"] = "medium"
        else:
            a["severity"] = "low"

    # Sort by absolute zscore descending (most severe first)
    all_anomalies.sort(key=lambda x: abs(x.get("zscore", 0)), reverse=True)

    return {
        "status": "ok",
        "window": window,
        "method": method,
        "sensitivity": sensitivity,
        "bucket_count": bucket_count,
        "metrics_analyzed": metrics,
        "total_anomalies": len(all_anomalies),
        "anomalies_by_metric": anomalies_by_metric,
        "anomalies": all_anomalies,
        "summary": ts.get("summary", {}),
    }


def _ewma_anomaly_check(
    values: list[float],
    threshold: float = 3.0,
    alpha: float = 0.3,
) -> list[dict[str, Any]]:
    """Detect anomalies using EWMA baseline deviation.

    Computes a moving baseline, then flags points where the deviation
    exceeds ``threshold × rolling_std``.
    """
    if len(values) < 3:
        return []

    baseline = _ewma_baseline(values, alpha=alpha)
    deviations = [values[i] - baseline[i] for i in range(len(values))]

    # Use rolling std of deviations (minimum window of 3)
    anomalies: list[dict[str, Any]] = []
    for i in range(len(values)):
        window_start = max(0, i - 20)  # look-back window
        window_devs = deviations[window_start : i + 1]
        if len(window_devs) < 2:
            continue
        std = statistics.stdev(window_devs) if len(window_devs) >= 2 else 0
        if std == 0:
            continue
        z = deviations[i] / std
        if abs(z) >= threshold:
            anomalies.append({
                "bucket_index": i,
                "value": round(values[i], 6),
                "zscore": round(z, 2),
                "direction": "spike" if z > 0 else "drop",
                "baseline_mean": round(baseline[i], 6),
                "method": "ewma",
            })
    return anomalies


# ── Trend Analysis ────────────────────────────────────────────────


def _linear_regression_slope(values: list[float]) -> float:
    """Compute the slope of a simple linear regression over the values.

    Positive slope = increasing trend, negative = decreasing.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def analyze_trends(
    engine: AuditEngine,
    *,
    window: str = "1h",
    metrics: list[str] | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    tool_name: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Analyze trends across time-bucketed metrics.

    For each metric, computes:
    - Linear regression slope (rate of change per bucket)
    - Percentage change (first vs last bucket)
    - Trend direction: ``increasing``, ``decreasing``, or ``stable``
    - Coefficient of variation (volatility)
    """
    if metrics is None:
        metrics = [
            "error_rate",
            "p95_latency_ms",
            "total_cost_usd",
            "call_count",
        ]

    invalid = [m for m in metrics if m not in ANALYZABLE_METRICS + ["call_count"]]
    if invalid:
        raise ValueError(
            f"Invalid metric(s): {invalid}. Must be one of {ANALYZABLE_METRICS}"
        )

    ts = build_timeseries(
        engine,
        window=window,
        session_id=session_id,
        agent_id=agent_id,
        tool_name=tool_name,
        limit=limit,
    )

    buckets = ts.get("buckets", [])
    trends: list[dict[str, Any]] = []

    for m in metrics:
        values = [b.get(m, 0) for b in buckets]

        if len(values) < 2:
            trends.append({
                "metric": m,
                "direction": "insufficient_data",
                "slope": 0.0,
                "pct_change": 0.0,
                "first_value": values[0] if values else 0,
                "last_value": values[-1] if values else 0,
                "volatility_cv": 0.0,
                "volatility_label": "n/a",
                "mean": round(statistics.mean(values), 6) if values else 0.0,
                "stdev": 0.0,
                "bucket_count": len(values),
            })
            continue

        slope = _linear_regression_slope(values)

        # Percentage change from first to last
        first = values[0]
        last = values[-1]
        if first != 0:
            pct_change = round(((last - first) / abs(first)) * 100, 2)
        else:
            pct_change = 0.0 if last == 0 else 100.0

        # Volatility: coefficient of variation
        mean_val = statistics.mean(values)
        stdev_val = statistics.stdev(values) if len(values) >= 2 else 0.0
        cv = round(stdev_val / mean_val * 100, 2) if mean_val != 0 else 0.0

        # Determine direction — use slope relative to mean for significance
        if mean_val != 0:
            relative_slope = abs(slope) / abs(mean_val)
        else:
            relative_slope = 0.0

        # 2% change per bucket is the threshold for significance
        if relative_slope > 0.02:
            direction = "increasing" if slope > 0 else "decreasing"
        else:
            direction = "stable"

        # Volatility classification
        if cv > 50:
            volatility_label = "high"
        elif cv > 20:
            volatility_label = "moderate"
        else:
            volatility_label = "low"

        trends.append({
            "metric": m,
            "direction": direction,
            "slope": round(slope, 6),
            "pct_change": pct_change,
            "first_value": round(first, 6),
            "last_value": round(last, 6),
            "volatility_cv": cv,
            "volatility_label": volatility_label,
            "mean": round(mean_val, 6),
            "stdev": round(stdev_val, 6),
            "bucket_count": len(values),
        })

    return {
        "status": "ok",
        "window": window,
        "trend_count": len(trends),
        "trends": trends,
        "summary": ts.get("summary", {}),
    }


# ── Heatmap (tool × time bucket) ──────────────────────────────────


def build_heatmap(
    engine: AuditEngine,
    *,
    window: str = "1h",
    metric: str = "call_count",
    session_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Build a tool × time-bucket heatmap.

    Useful for visualizing which tools are hot at which times.  Returns a
    matrix of values: rows are tools, columns are time buckets.
    """
    if metric not in ANALYZABLE_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of {ANALYZABLE_METRICS}"
        )

    tw = _window_from_string(window)
    bucket_td = tw.timedelta

    calls = engine.query_calls(
        session_id=session_id,
        agent_id=agent_id,
        limit=limit,
    )

    if not calls:
        return {
            "window": window,
            "metric": metric,
            "tools": [],
            "timestamps": [],
            "matrix": {},
        }

    # Group by (tool, bucket)
    grouped: dict[tuple[str, datetime], list[ToolCall]] = defaultdict(list)
    all_tools: set[str] = set()
    all_buckets: set[datetime] = set()

    for call in calls:
        bucket_start = _floor_to_bucket(call.started_at, bucket_td)
        grouped[(call.tool_name, bucket_start)].append(call)
        all_tools.add(call.tool_name)
        all_buckets.add(bucket_start)

    sorted_tools = sorted(all_tools)
    sorted_buckets = sorted(all_buckets)
    timestamps = [b.isoformat() for b in sorted_buckets]

    matrix: dict[str, list[Any]] = {}
    for tool in sorted_tools:
        row: list[Any] = []
        for bucket_start in sorted_buckets:
            bucket_calls = grouped.get((tool, bucket_start), [])
            if bucket_calls:
                m = _bucket_metrics(bucket_calls)
                row.append(m.get(metric, 0))
            else:
                row.append(0)
        matrix[tool] = row

    return {
        "window": window,
        "metric": metric,
        "tools": sorted_tools,
        "timestamps": timestamps,
        "matrix": matrix,
        "bucket_count": len(sorted_buckets),
        "tool_count": len(sorted_tools),
    }
