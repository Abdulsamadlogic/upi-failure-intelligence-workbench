
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

from src.domain import (
    FailureCategory,
    PaymentStatus,
    PaymentType,
    ReversalStatus,
    TransactionEvent,
    TransactionProjection,
    TriageResult,
    project_transaction,
)
from src.model import FailureClassifier


@dataclass(frozen=True)
class RulePrediction:
    category: FailureCategory
    confidence: float
    evidence: tuple[str, ...]


_RULES: tuple[tuple[FailureCategory, tuple[str, ...], float], ...] = (
    (
        FailureCategory.NETWORK_TIMEOUT,
        ("network timeout", "gateway timeout", "connection timed out", "read timeout"),
        0.97,
    ),
    (
        FailureCategory.PAYER_BANK_DECLINE,
        ("insufficient funds", "payer bank declined", "debit not permitted", "limit exceeded"),
        0.98,
    ),
    (
        FailureCategory.BENEFICIARY_BANK_UNAVAILABLE,
        ("beneficiary bank unavailable", "payee bank unavailable", "beneficiary host down"),
        0.97,
    ),
    (
        FailureCategory.MERCHANT_CONFIRMATION_TIMEOUT,
        ("merchant confirmation timeout", "merchant acknowledgement missing", "pos ack timeout"),
        0.97,
    ),
    (
        FailureCategory.SETTLEMENT_MISMATCH,
        ("debit confirmed but beneficiary not credited", "settlement mismatch", "credit missing after debit"),
        0.96,
    ),
)


def classify_with_rules(text: str) -> RulePrediction:
    if not text.strip():
        raise ValueError("Triage text must not be empty")
    normalized = re.sub(r"\s+", " ", text.lower())
    for category, phrases, confidence in _RULES:
        matches = [phrase for phrase in phrases if phrase in normalized]
        if matches:
            return RulePrediction(
                category=category,
                confidence=confidence,
                evidence=tuple(f"Matched diagnostic phrase: {phrase}" for phrase in matches),
            )
    return RulePrediction(
        category=FailureCategory.UNKNOWN,
        confidence=0.0,
        evidence=("No deterministic diagnostic rule matched the event history.",),
    )


def build_model_text(
    projection: TransactionProjection, events: Sequence[TransactionEvent]
) -> str:
    recent = sorted(events, key=lambda event: (event.occurred_at, event.event_id))[-8:][::-1]
    lines = [
        f"provider={projection.provider}",
        f"payment_type={projection.payment_type.value}",
        f"status={projection.status.value}",
        f"payer_debited={projection.payer_debited}",
        f"beneficiary_credited={projection.beneficiary_credited}",
        f"merchant_confirmed={projection.merchant_confirmed}",
        f"reversal_status={projection.reversal_status.value}",
        f"state_conflicts={projection.state_conflicts}",
        "recent_events:",
    ]
    for event in recent:
        lines.append(
            " | ".join(
                (
                    event.occurred_at.isoformat(),
                    f"component={event.component or 'unspecified'}",
                    f"status={event.status.value}",
                    f"code={event.response_code or 'none'}",
                    f"message={event.message or 'none'}",
                )
            )
        )
    return "\n".join(lines)


def evidence_from_projection(
    projection: TransactionProjection, events: Sequence[TransactionEvent]
) -> list[str]:
    evidence: list[str] = []
    if projection.payer_debited is True:
        evidence.append("The payer-side debit is recorded as confirmed.")
    if projection.beneficiary_credited is False:
        evidence.append("The beneficiary-side credit is recorded as not completed.")
    if projection.merchant_confirmed is False:
        evidence.append("Merchant confirmation is recorded as missing.")
    if projection.reversal_status == ReversalStatus.PENDING:
        evidence.append("The latest reversal state is pending.")
    if projection.reversal_status == ReversalStatus.COMPLETED:
        evidence.append("A reversal is recorded as completed.")
    if projection.response_codes:
        evidence.append(f"Observed response codes: {', '.join(projection.response_codes)}.")
    evidence.extend(projection.state_conflicts)
    if projection.event_count > 1:
        evidence.append(f"The transaction has {projection.event_count} distinct events.")
    if not evidence:
        evidence.append("The available event history does not establish a specific cause.")
    return evidence


def recommended_action(category: FailureCategory) -> str:
    actions = {
        FailureCategory.NETWORK_TIMEOUT: (
            "Check the relevant network/provider health and reconcile the final transaction "
            "state before considering any retry."
        ),
        FailureCategory.PAYER_BANK_DECLINE: (
            "Review the payer-side response and confirm the customer-facing decline reason."
        ),
        FailureCategory.BENEFICIARY_BANK_UNAVAILABLE: (
            "Check beneficiary-bank availability and verify settlement status before closure."
        ),
        FailureCategory.MERCHANT_CONFIRMATION_TIMEOUT: (
            "Reconcile the merchant confirmation with the transaction ledger; do not assume "
            "the payment failed solely from a missing acknowledgement."
        ),
        FailureCategory.SETTLEMENT_MISMATCH: (
            "Reconcile debit, beneficiary credit, and reversal records; route any unresolved "
            "difference to an authorized operator."
        ),
        FailureCategory.UNKNOWN: (
            "Human review required: compare the full event trace with authoritative bank and "
            "payment-network records."
        ),
    }
    return actions[category]


class FailureTriageService:
    def __init__(
        self,
        model: FailureClassifier | None = None,
        confidence_threshold: float = 0.70,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model = model
        self.confidence_threshold = confidence_threshold

    def analyze(self, events: Sequence[TransactionEvent]) -> TriageResult:
        projection = project_transaction(events)
        if (
            projection.status not in (PaymentStatus.FAILED, PaymentStatus.PENDING)
            and not projection.state_conflicts
        ):
            raise ValueError("Only failed or pending transactions can be triaged")

        model_text = build_model_text(projection, events)
        rule_result = classify_with_rules(model_text)
        category = rule_result.category
        confidence = rule_result.confidence
        probabilities: dict[str, float] = {}
        classifier = "rules"
        evidence = list(rule_result.evidence)

        if category == FailureCategory.UNKNOWN and self.model is not None:
            prediction = self.model.predict(model_text)
            confidence = prediction.confidence
            probabilities = prediction.probabilities
            classifier = "qwen"
            if confidence >= self.confidence_threshold:
                category = prediction.category
                evidence = [
                    "Category suggested by the fine-tuned classifier; verify against source events."
                ]
            else:
                evidence = [
                    "Classifier confidence is below the review threshold.",
                    *evidence_from_projection(projection, events),
                ]

        evidence.extend(
            item
            for item in evidence_from_projection(projection, events)
            if item not in evidence
        )
        requires_review = (
            category == FailureCategory.UNKNOWN
            or classifier == "qwen"
            or confidence < self.confidence_threshold
            or bool(projection.state_conflicts)
            or (
                projection.payer_debited is True
                and projection.beneficiary_credited is not True
                and projection.reversal_status != ReversalStatus.COMPLETED
            )
            or (
                projection.payment_type == PaymentType.MERCHANT
                and projection.payer_debited is True
                and projection.merchant_confirmed is not True
            )
        )
        return TriageResult(
            transaction_id=projection.transaction_id,
            category=category,
            confidence=confidence,
            evidence=evidence,
            recommended_action=recommended_action(category),
            requires_human_review=requires_review,
            classifier=classifier,
            probabilities=probabilities,
        )


@dataclass(frozen=True)
class ReversalDeadline:
    deadline: datetime
    overdue: bool
    hours_remaining: float


class ReversalSlaPolicy:
    """Configurable demonstration policy; values are not legal advice."""

    def __init__(self, transfer_hours: int, merchant_hours: int) -> None:
        if transfer_hours <= 0 or merchant_hours <= 0:
            raise ValueError("SLA durations must be positive")
        self.transfer_hours = transfer_hours
        self.merchant_hours = merchant_hours

    def assess(
        self,
        projection: TransactionProjection,
        now: datetime | None = None,
    ) -> ReversalDeadline | None:
        if (
            projection.payer_debited is not True
            or projection.reversal_status == ReversalStatus.COMPLETED
            or projection.status == PaymentStatus.SUCCESS
        ):
            return None
        hours = (
            self.transfer_hours
            if projection.payment_type == PaymentType.TRANSFER
            else self.merchant_hours
        )
        deadline = projection.first_seen + timedelta(hours=hours)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must include a timezone")
        remaining = (deadline - current).total_seconds() / 3600
        return ReversalDeadline(
            deadline=deadline,
            overdue=remaining < 0,
            hours_remaining=round(remaining, 2),
        )
