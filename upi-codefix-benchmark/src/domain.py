"""Typed domain objects and transaction-history projections."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaymentStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class PaymentType(str, Enum):
    TRANSFER = "transfer"
    MERCHANT = "merchant"


class ReversalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class FailureCategory(str, Enum):
    NETWORK_TIMEOUT = "network_timeout"
    PAYER_BANK_DECLINE = "payer_bank_decline"
    BENEFICIARY_BANK_UNAVAILABLE = "beneficiary_bank_unavailable"
    MERCHANT_CONFIRMATION_TIMEOUT = "merchant_confirmation_timeout"
    SETTLEMENT_MISMATCH = "settlement_mismatch"
    UNKNOWN = "unknown"


class ReviewStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class TransactionEvent(BaseModel):
    """Sanitized event emitted by a payment-system simulator or test fixture."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=80)
    payment_type: PaymentType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: PaymentStatus
    response_code: str | None = Field(default=None, max_length=64)
    message: str | None = Field(default=None, max_length=1000)
    component: str | None = Field(default=None, max_length=80)
    payer_debited: bool | None = None
    beneficiary_credited: bool | None = None
    merchant_confirmed: bool | None = None
    reversal_status: ReversalStatus = ReversalStatus.UNKNOWN

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("provider", "component", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("response_code", mode="before")
    @classmethod
    def normalize_response_code(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper() or None
        return value

    @field_validator("message")
    @classmethod
    def redact_obvious_secrets(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = re.sub(
            r"\b(?:otp|pin|cvv|password)\s*[:=]\s*\S+",
            "[REDACTED_SECRET]",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "[REDACTED_ADDRESS]",
            value,
        )
        value = re.sub(
            r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b",
            "[REDACTED_ADDRESS]",
            value,
        )
        value = re.sub(r"\b\d{10,19}\b", "[REDACTED_NUMBER]", value)
        return value


class TransactionProjection(BaseModel):
    transaction_id: str
    provider: str
    payment_type: PaymentType
    status: PaymentStatus
    first_seen: datetime
    last_seen: datetime
    payer_debited: bool | None
    beneficiary_credited: bool | None
    merchant_confirmed: bool | None
    reversal_status: ReversalStatus
    response_codes: list[str]
    event_count: int
    state_conflicts: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    transaction_id: str
    category: FailureCategory
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    recommended_action: str
    requires_human_review: bool
    classifier: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class Incident(BaseModel):
    incident_id: str
    provider: str
    response_code: str
    window_start: datetime
    window_end: datetime
    recent_failures: int
    recent_total: int
    recent_failure_rate: float
    baseline_failures: int
    baseline_total: int
    baseline_failure_rate: float
    severity: str
    evidence: list[str]


class ReviewDecision(BaseModel):
    actor: str = Field(min_length=1, max_length=128)
    status: ReviewStatus
    confirmed_category: FailureCategory | None = None
    note: str = Field(min_length=1, max_length=2000)


def _latest_non_null(events: list[TransactionEvent], field_name: str):
    for event in reversed(events):
        value = getattr(event, field_name)
        if value is not None:
            return value
    return None


def project_transaction(
    events: Iterable[TransactionEvent],
) -> TransactionProjection:
    ordered = sorted(events, key=lambda event: (event.occurred_at, event.event_id))
    if not ordered:
        raise ValueError("Cannot project a transaction without events")

    transaction_ids = {event.transaction_id for event in ordered}
    if len(transaction_ids) != 1:
        raise ValueError("All events in a projection must share one transaction_id")

    latest = ordered[-1]
    codes = list(
        dict.fromkeys(
            event.response_code for event in ordered if event.response_code is not None
        )
    )
    terminal_statuses = {
        event.status
        for event in ordered
        if event.status in (PaymentStatus.SUCCESS, PaymentStatus.FAILED)
    }
    state_conflicts: list[str] = []
    if len(terminal_statuses) > 1:
        state_conflicts.append(
            "The event history contains both SUCCESS and FAILED terminal statuses."
        )
    return TransactionProjection(
        transaction_id=latest.transaction_id,
        provider=latest.provider,
        payment_type=latest.payment_type,
        status=latest.status,
        first_seen=ordered[0].occurred_at,
        last_seen=latest.occurred_at,
        payer_debited=_latest_non_null(ordered, "payer_debited"),
        beneficiary_credited=_latest_non_null(ordered, "beneficiary_credited"),
        merchant_confirmed=_latest_non_null(ordered, "merchant_confirmed"),
        reversal_status=_latest_non_null(ordered, "reversal_status"),
        response_codes=codes,
        event_count=len(ordered),
        state_conflicts=state_conflicts,
    )
