"""Detect statistically meaningful provider-level failure-rate spikes."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Sequence

from src.domain import Incident, PaymentStatus, TransactionEvent


def _bucket_key(event: TransactionEvent) -> tuple[str, str]:
    return event.provider, event.response_code or "NO_RESPONSE_CODE"


def detect_failure_spikes(
    events: Sequence[TransactionEvent],
    *,
    now: datetime | None = None,
    recent_window: timedelta = timedelta(minutes=15),
    baseline_window: timedelta = timedelta(hours=24),
    minimum_recent_failures: int = 5,
    minimum_rate_increase: float = 0.20,
    minimum_rate_ratio: float = 3.0,
) -> list[Incident]:
    """Compare recent provider/code rates with a preceding baseline window.

    A spike is emitted only when both the absolute-rate increase and relative
    increase pass their configured thresholds, reducing noise from tiny counts.
    """
    supplied_now = now or datetime.now(timezone.utc)
    if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
        raise ValueError("now must include a timezone")
    current_time = datetime.fromtimestamp(
        int(supplied_now.timestamp()) // 60 * 60, tz=timezone.utc
    )
    if recent_window <= timedelta(0) or baseline_window <= recent_window:
        raise ValueError("baseline_window must be longer than a positive recent_window")
    if minimum_recent_failures < 1:
        raise ValueError("minimum_recent_failures must be at least 1")
    if minimum_rate_increase <= 0 or minimum_rate_ratio <= 1:
        raise ValueError("Spike thresholds must be positive and rate ratio must exceed 1")

    recent_start = current_time - recent_window
    baseline_start = current_time - baseline_window
    current_by_transaction: dict[str, TransactionEvent] = {}
    baseline_by_transaction: dict[str, TransactionEvent] = {}
    for event in sorted(events, key=lambda item: item.occurred_at):
        timestamp = event.occurred_at
        if timestamp >= recent_start and timestamp <= current_time:
            current_by_transaction[event.transaction_id] = event
        elif baseline_start <= timestamp < recent_start:
            baseline_by_transaction[event.transaction_id] = event
    for transaction_id in current_by_transaction:
        baseline_by_transaction.pop(transaction_id, None)

    current_total: dict[str, int] = defaultdict(int)
    baseline_total: dict[str, int] = defaultdict(int)
    current_failures: dict[tuple[str, str], int] = defaultdict(int)
    baseline_failures: dict[tuple[str, str], int] = defaultdict(int)
    for event in current_by_transaction.values():
        current_total[event.provider] += 1
        if event.status == PaymentStatus.FAILED:
            current_failures[_bucket_key(event)] += 1
    for event in baseline_by_transaction.values():
        baseline_total[event.provider] += 1
        if event.status == PaymentStatus.FAILED:
            baseline_failures[_bucket_key(event)] += 1

    incidents: list[Incident] = []
    for key, recent_failures in current_failures.items():
        provider, response_code = key
        recent_count = current_total[provider]
        baseline_failure_count = baseline_failures.get(key, 0)
        baseline_count = baseline_total[provider]
        if recent_failures < minimum_recent_failures or recent_count == 0:
            continue
        recent_rate = recent_failures / recent_count
        baseline_rate = (
            baseline_failure_count / baseline_count if baseline_count else 0.0
        )
        absolute_increase = recent_rate - baseline_rate
        ratio = recent_rate / max(baseline_rate, 1 / max(baseline_count + 1, 1))
        if (
            absolute_increase < minimum_rate_increase
            or ratio < minimum_rate_ratio
        ):
            continue

        identity = f"{provider}|{response_code}|{recent_start.isoformat()}"
        incident_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        severity = "critical" if recent_rate >= 0.80 and recent_failures >= 10 else "high"
        incidents.append(
            Incident(
                incident_id=incident_id,
                provider=provider,
                response_code=response_code,
                window_start=recent_start,
                window_end=current_time,
                recent_failures=recent_failures,
                recent_total=recent_count,
                recent_failure_rate=round(recent_rate, 4),
                baseline_failures=baseline_failure_count,
                baseline_total=baseline_count,
                baseline_failure_rate=round(baseline_rate, 4),
                severity=severity,
                evidence=[
                    f"Recent failure rate {recent_rate:.1%} exceeded baseline {baseline_rate:.1%}.",
                    f"Rate increase: {absolute_increase:.1%}; relative factor: {ratio:.2f}x.",
                ],
            )
        )

    return sorted(
        incidents,
        key=lambda incident: (
            incident.severity != "critical",
            -incident.recent_failure_rate,
            incident.provider,
        ),
    )
