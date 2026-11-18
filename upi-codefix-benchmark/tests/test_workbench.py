import csv
import os
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import psycopg
from psycopg import sql
import pytest

from src.api import create_app
from src.domain import (
    FailureCategory,
    PaymentStatus,
    PaymentType,
    ReversalStatus,
    ReviewDecision,
    ReviewStatus,
    TransactionEvent,
    project_transaction,
)
from src.evaluate import load_test_rows
from src.incidents import detect_failure_spikes
from src.model import ModelPrediction
from src.storage import (
    EventConflictError,
    InvalidCaseTransitionError,
    WorkbenchStore,
)
from src.triage import FailureTriageService, ReversalSlaPolicy, classify_with_rules
from src.train import generate_synthetic_dataset, load_split


NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)


def event(
    event_id: str,
    transaction_id: str,
    *,
    minutes_ago: int = 0,
    status: PaymentStatus = PaymentStatus.FAILED,
    provider: str = "bank-a",
    code: str | None = "GW_TIMEOUT",
    message: str | None = "network timeout",
    payment_type: PaymentType = PaymentType.TRANSFER,
    payer_debited: bool | None = False,
    beneficiary_credited: bool | None = False,
    merchant_confirmed: bool | None = None,
    reversal_status: ReversalStatus = ReversalStatus.UNKNOWN,
) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        transaction_id=transaction_id,
        provider=provider,
        payment_type=payment_type,
        occurred_at=NOW - timedelta(minutes=minutes_ago),
        status=status,
        response_code=code,
        message=message,
        payer_debited=payer_debited,
        beneficiary_credited=beneficiary_credited,
        merchant_confirmed=merchant_confirmed,
        reversal_status=reversal_status,
    )


@pytest.fixture
def store():
    database_url = os.getenv("UPI_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set UPI_TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema_name = f"workbench_test_{uuid.uuid4().hex}"
    with psycopg.connect(database_url) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
        )
    test_store = WorkbenchStore(database_url, schema_name=schema_name)
    try:
        yield test_store
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )


def test_event_redacts_common_secrets_and_normalizes_fields() -> None:
    item = event(
        "evt-1",
        "txn-1",
        provider=" BANK-A ",
        code=" gw_timeout ",
        message="OTP: 887766; account 1234567890123456",
    )

    assert item.provider == "bank-a"
    assert item.response_code == "GW_TIMEOUT"
    assert "887766" not in item.message
    assert "1234567890123456" not in item.message


def test_event_history_projection_uses_latest_observations() -> None:
    first = event(
        "evt-1",
        "txn-1",
        minutes_ago=3,
        payer_debited=True,
        beneficiary_credited=False,
    )
    second = event(
        "evt-2",
        "txn-1",
        minutes_ago=1,
        status=PaymentStatus.SUCCESS,
        code=None,
        message=None,
        payer_debited=True,
        beneficiary_credited=True,
        reversal_status=ReversalStatus.COMPLETED,
    )

    result = project_transaction([second, first])

    assert result.status == PaymentStatus.SUCCESS
    assert result.event_count == 2
    assert result.beneficiary_credited is True
    assert result.reversal_status == ReversalStatus.COMPLETED
    assert result.state_conflicts == [
        "The event history contains both SUCCESS and FAILED terminal statuses."
    ]


def test_rule_classifier_handles_known_categories_and_abstains() -> None:
    known = classify_with_rules("payer bank reported insufficient funds")
    unknown = classify_with_rules("upstream returned an unmapped diagnostic")

    assert known.category == FailureCategory.PAYER_BANK_DECLINE
    assert known.confidence >= 0.9
    assert unknown.category == FailureCategory.UNKNOWN
    assert unknown.confidence == 0


def test_triage_routes_unresolved_debit_to_human_review() -> None:
    item = event(
        "evt-1",
        "txn-1",
        message="beneficiary bank unavailable",
        code="HOST_DOWN",
        payer_debited=True,
        beneficiary_credited=False,
        reversal_status=ReversalStatus.PENDING,
    )

    result = FailureTriageService().analyze([item])

    assert result.category == FailureCategory.BENEFICIARY_BANK_UNAVAILABLE
    assert result.requires_human_review is True
    assert result.recommended_action
    assert any("debit" in evidence.lower() for evidence in result.evidence)


def test_qwen_suggestions_always_require_human_review() -> None:
    class ConfidentClassifier:
        def predict(self, text: str) -> ModelPrediction:
            return ModelPrediction(
                category=FailureCategory.BENEFICIARY_BANK_UNAVAILABLE,
                confidence=0.99,
                probabilities={
                    FailureCategory.BENEFICIARY_BANK_UNAVAILABLE.value: 0.99,
                    FailureCategory.UNKNOWN.value: 0.01,
                },
            )

    item = event(
        "evt-unknown",
        "txn-unknown",
        code="UNMAPPED",
        message="no diagnostic details are available",
        payer_debited=False,
        beneficiary_credited=False,
    )
    result = FailureTriageService(model=ConfidentClassifier()).analyze([item])

    assert result.classifier == "qwen"
    assert result.confidence == 0.99
    assert result.requires_human_review is True


def test_store_ingestion_is_idempotent_and_rejects_conflicting_event_ids(
    store: WorkbenchStore,
) -> None:
    original = event("evt-1", "txn-1")

    assert store.append_event(original) is True
    assert store.append_event(original) is False

    conflicting = event("evt-1", "txn-other")
    try:
        store.append_event(conflicting)
    except EventConflictError:
        pass
    else:
        raise AssertionError("Reused event IDs with different payloads must be rejected")
    assert len(store.events_for_transaction("txn-1")) == 1


def test_batch_event_conflict_rolls_back_the_entire_batch(
    store: WorkbenchStore,
) -> None:
    store.append_event(event("existing", "txn-existing"))

    try:
        store.append_events(
            [
                event("new-event", "txn-new"),
                event("existing", "txn-different"),
            ]
        )
    except EventConflictError:
        pass
    else:
        raise AssertionError("Conflicting event batch should be rejected")

    assert store.events_for_transaction("txn-new") == []


def test_case_review_is_audited_and_terminal_cases_cannot_be_reopened(
    store: WorkbenchStore,
) -> None:
    store.save_projection(project_transaction([event("evt-1", "txn-1")]))
    case = store.upsert_review_case(
        transaction_id="txn-1",
        category=FailureCategory.NETWORK_TIMEOUT.value,
        confidence=0.97,
        evidence=["gateway timed out"],
        recommended_action="reconcile final state",
    )
    reviewed = store.record_review(
        case["case_id"],
        ReviewDecision(
            actor="reviewer-1",
            status=ReviewStatus.RESOLVED,
            confirmed_category=FailureCategory.NETWORK_TIMEOUT,
            note="Confirmed from simulator trace",
        ),
    )

    assert reviewed["status"] == "resolved"
    assert store.audit_history("case", str(case["case_id"]))[-1]["actor"] == "reviewer-1"
    try:
        store.record_review(
            case["case_id"],
            ReviewDecision(
                actor="reviewer-2",
                status=ReviewStatus.IN_PROGRESS,
                note="Reopen",
            ),
        )
    except InvalidCaseTransitionError:
        pass
    else:
        raise AssertionError("A terminal review case must not be changed")


def test_reversal_sla_uses_configured_durations_and_skips_completed_reversal() -> None:
    policy = ReversalSlaPolicy(transfer_hours=24, merchant_hours=120)
    projection = project_transaction(
        [
            event(
                "evt-1",
                "txn-1",
                minutes_ago=1500,
                payer_debited=True,
                reversal_status=ReversalStatus.PENDING,
            )
        ]
    )

    result = policy.assess(projection, NOW)

    assert result is not None
    assert result.overdue is True
    assert result.hours_remaining < 0


def test_incident_detector_compares_recent_failures_with_provider_baseline() -> None:
    history = [
        event(
            f"base-{index}",
            f"base-txn-{index}",
            minutes_ago=60 + index,
            status=PaymentStatus.SUCCESS if index < 19 else PaymentStatus.FAILED,
            code="HOST_DOWN",
            message="routine response",
        )
        for index in range(20)
    ]
    history.extend(
        event(
            f"recent-{index}",
            f"recent-txn-{index}",
            minutes_ago=1,
            status=PaymentStatus.FAILED,
            code="HOST_DOWN",
            message="service unavailable",
        )
        for index in range(8)
    )

    incidents = detect_failure_spikes(
        history,
        now=NOW,
        minimum_recent_failures=5,
        minimum_rate_increase=0.2,
        minimum_rate_ratio=3,
    )

    assert len(incidents) == 1
    assert incidents[0].provider == "bank-a"
    assert incidents[0].recent_failures == 8
    assert incidents[0].recent_failure_rate > incidents[0].baseline_failure_rate


def test_api_ingests_deduplicates_triages_and_records_review(
    store: WorkbenchStore,
) -> None:
    app = create_app(store=store, triage_mode="rules")
    client = TestClient(app)
    payload = {
        "events": [
            {
                "event_id": "event-api-1",
                "transaction_id": "txn-api-1",
                "provider": "bank-a",
                "payment_type": "transfer",
                "occurred_at": NOW.isoformat(),
                "status": "FAILED",
                "response_code": "HOST_DOWN",
                "message": "beneficiary bank unavailable",
                "payer_debited": True,
                "beneficiary_credited": False,
                "reversal_status": "pending",
            }
        ]
    }

    first = client.post("/api/v1/events", json=payload)
    duplicate = client.post("/api/v1/events", json=payload)
    cases = client.get("/api/v1/cases")

    assert first.status_code == 200
    assert first.json()["accepted_count"] == 1
    assert first.json()["results"][0]["requires_human_review"] is True
    assert duplicate.json()["duplicate_count"] == 1
    assert len(cases.json()) == 1

    case_id = cases.json()[0]["case_id"]
    reviewed = client.post(
        f"/api/v1/cases/{case_id}/review",
        json={
            "actor": "reviewer-1",
            "status": "resolved",
            "confirmed_category": "beneficiary_bank_unavailable",
            "note": "Validated in simulated event history",
        },
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["status"] == "resolved"


def test_synthetic_dataset_generator_writes_2000_marked_rows(tmp_path) -> None:
    dataset_path = tmp_path / "synthetic.csv"

    count = generate_synthetic_dataset(dataset_path, seed=7)

    assert count == 2000
    with dataset_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2000
    assert {row["synthetic"] for row in rows} == {"true"}
    assert {row["split"] for row in rows} == {"train", "validation", "test"}


def test_dataset_readers_accept_utf8_bom_headers(tmp_path) -> None:
    dataset_path = tmp_path / "bom_dataset.csv"
    with dataset_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("text", "label", "split", "scenario_id"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "text": "No response from gateway",
                "label": "network_timeout",
                "split": "test",
                "scenario_id": "test-001",
            }
        )

    assert load_test_rows(dataset_path)[0]["scenario_id"] == "test-001"
    assert load_split(dataset_path, "test")[0]["label"] == "network_timeout"
