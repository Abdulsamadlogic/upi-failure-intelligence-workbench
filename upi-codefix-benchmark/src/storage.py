"""PostgreSQL persistence for transaction events and human-review history."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from src.domain import (
    Incident,
    ReviewDecision,
    TransactionEvent,
    TransactionProjection,
)


class EventConflictError(ValueError):
    """Raised when an existing event ID is reused with different content."""


class CaseNotFoundError(LookupError):
    pass


class InvalidCaseTransitionError(ValueError):
    pass


class WorkbenchStore:
    def __init__(
        self,
        database_url: str,
        *,
        schema_name: str | None = None,
        initialize: bool = True,
    ) -> None:
        if not database_url or not database_url.startswith(
            ("postgresql://", "postgres://")
        ):
            raise ValueError("A PostgreSQL URL is required")
        if schema_name is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", schema_name
        ):
            raise ValueError("schema_name must be a simple PostgreSQL identifier")
        self.database_url = database_url
        self.schema_name = schema_name
        if initialize:
            self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            if self.schema_name is not None:
                connection.execute(
                    sql.SQL("SET search_path TO {}").format(
                        sql.Identifier(self.schema_name)
                    )
                )
            yield connection

    def initialize(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                occurred_at TIMESTAMPTZ NOT NULL,
                payment_type TEXT NOT NULL,
                status TEXT NOT NULL,
                response_code TEXT,
                raw_json JSONB NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_events_transaction_time
                ON events(transaction_id, occurred_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_events_provider_time
                ON events(provider, occurred_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                payment_type TEXT NOT NULL,
                status TEXT NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL,
                last_seen TIMESTAMPTZ NOT NULL,
                payer_debited BOOLEAN,
                beneficiary_credited BOOLEAN,
                merchant_confirmed BOOLEAN,
                reversal_status TEXT NOT NULL,
                response_codes_json JSONB NOT NULL,
                event_count INTEGER NOT NULL,
                state_conflicts_json JSONB NOT NULL DEFAULT '[]'::jsonb
            )
            """,
            """
            ALTER TABLE transactions ADD COLUMN IF NOT EXISTS
                state_conflicts_json JSONB NOT NULL DEFAULT '[]'::jsonb
            """,
            """
            CREATE TABLE IF NOT EXISTS review_cases (
                case_id BIGSERIAL PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                evidence_json JSONB NOT NULL,
                recommended_action TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_to TEXT,
                confirmed_category TEXT,
                reviewer_note TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_cases_status_created
                ON review_cases(status, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                response_code TEXT NOT NULL,
                window_start TIMESTAMPTZ NOT NULL,
                window_end TIMESTAMPTZ NOT NULL,
                severity TEXT NOT NULL,
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                details_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_audit_entity
                ON audit_log(entity_type, entity_id, audit_id)
            """,
        )
        with self._connect() as db:
            for statement in statements:
                db.execute(statement)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _event_payload(event: TransactionEvent) -> str:
        return WorkbenchStore._json(event.model_dump(mode="json"))

    def append_event(self, event: TransactionEvent) -> bool:
        return self.append_events([event])[0]

    def append_events(self, events: list[TransactionEvent]) -> list[bool]:
        """Insert a batch atomically; return whether each event was newly stored."""
        if not events:
            raise ValueError("At least one event is required")
        results: list[bool] = []
        seen_in_batch: dict[str, str] = {}
        with self._connect() as db:
            for event in events:
                raw_json = self._event_payload(event)
                if event.event_id in seen_in_batch:
                    if seen_in_batch[event.event_id] != raw_json:
                        raise EventConflictError(
                            f"event_id {event.event_id!r} has conflicting "
                            "payloads in the same batch"
                        )
                    results.append(False)
                    continue

                inserted = db.execute(
                    """
                    INSERT INTO events (
                        event_id, transaction_id, provider, occurred_at,
                        payment_type, status, response_code, raw_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT(event_id) DO NOTHING
                    RETURNING event_id
                    """,
                    (
                        event.event_id,
                        event.transaction_id,
                        event.provider,
                        event.occurred_at,
                        event.payment_type.value,
                        event.status.value,
                        event.response_code,
                        raw_json,
                    ),
                ).fetchone()
                if inserted is None:
                    existing = db.execute(
                        "SELECT raw_json FROM events WHERE event_id = %s",
                        (event.event_id,),
                    ).fetchone()
                    if existing is None or self._json(existing["raw_json"]) != raw_json:
                        raise EventConflictError(
                            f"event_id {event.event_id!r} was reused with different content"
                        )
                    seen_in_batch[event.event_id] = raw_json
                    results.append(False)
                    continue

                seen_in_batch[event.event_id] = raw_json
                results.append(True)
        return results

    def events_for_transaction(self, transaction_id: str) -> list[TransactionEvent]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT raw_json FROM events
                WHERE transaction_id = %s
                ORDER BY occurred_at, event_id
                """,
                (transaction_id,),
            ).fetchall()
        return [
            TransactionEvent.model_validate(row["raw_json"])
            for row in rows
        ]

    def list_events(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 10000,
    ) -> list[TransactionEvent]:
        if not 1 <= limit <= 100000:
            raise ValueError("limit must be between 1 and 100000")
        clauses: list[str] = []
        values: list[Any] = []
        if since is not None:
            clauses.append("occurred_at >= %s")
            values.append(since)
        if until is not None:
            clauses.append("occurred_at < %s")
            values.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT raw_json FROM events {where}
                ORDER BY occurred_at LIMIT %s
                """,
                (*values, limit),
            ).fetchall()
        return [
            TransactionEvent.model_validate(row["raw_json"])
            for row in rows
        ]

    def save_projection(self, projection: TransactionProjection) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO transactions (
                    transaction_id, provider, payment_type, status, first_seen,
                    last_seen, payer_debited, beneficiary_credited,
                    merchant_confirmed, reversal_status, response_codes_json,
                    event_count, state_conflicts_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s::jsonb
                )
                ON CONFLICT(transaction_id) DO UPDATE SET
                    provider=EXCLUDED.provider,
                    payment_type=EXCLUDED.payment_type,
                    status=EXCLUDED.status,
                    first_seen=EXCLUDED.first_seen,
                    last_seen=EXCLUDED.last_seen,
                    payer_debited=EXCLUDED.payer_debited,
                    beneficiary_credited=EXCLUDED.beneficiary_credited,
                    merchant_confirmed=EXCLUDED.merchant_confirmed,
                    reversal_status=EXCLUDED.reversal_status,
                    response_codes_json=EXCLUDED.response_codes_json,
                    event_count=EXCLUDED.event_count,
                    state_conflicts_json=EXCLUDED.state_conflicts_json
                """,
                (
                    projection.transaction_id,
                    projection.provider,
                    projection.payment_type.value,
                    projection.status.value,
                    projection.first_seen,
                    projection.last_seen,
                    projection.payer_debited,
                    projection.beneficiary_credited,
                    projection.merchant_confirmed,
                    projection.reversal_status.value,
                    self._json(projection.response_codes),
                    projection.event_count,
                    self._json(projection.state_conflicts),
                ),
            )

    def get_projection(self, transaction_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM transactions WHERE transaction_id = %s",
                (transaction_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["response_codes"] = result.pop("response_codes_json")
        result["state_conflicts"] = result.pop("state_conflicts_json")
        return result

    def upsert_review_case(
        self,
        *,
        transaction_id: str,
        category: str,
        confidence: float,
        evidence: list[str],
        recommended_action: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO review_cases (
                    transaction_id, category, confidence, evidence_json,
                    recommended_action, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, 'open', %s, %s)
                ON CONFLICT(transaction_id) DO UPDATE SET
                    category=EXCLUDED.category,
                    confidence=EXCLUDED.confidence,
                    evidence_json=EXCLUDED.evidence_json,
                    recommended_action=EXCLUDED.recommended_action,
                    updated_at=EXCLUDED.updated_at
                WHERE review_cases.status IN ('open', 'in_progress')
                """,
                (
                    transaction_id,
                    category,
                    confidence,
                    self._json(evidence),
                    recommended_action,
                    now,
                    now,
                ),
            )
            db.execute(
                """
                INSERT INTO audit_log (
                    entity_type, entity_id, action, actor, details_json, created_at
                ) VALUES ('case', %s, 'triage_updated', 'system', %s::jsonb, %s)
                """,
                (
                    transaction_id,
                    self._json({"category": category, "confidence": confidence}),
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM review_cases WHERE transaction_id = %s",
                (transaction_id,),
            ).fetchone()
        return self._case_row(row)

    @staticmethod
    def _case_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["evidence"] = result.pop("evidence_json")
        return result

    def list_cases(
        self, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            if status is None:
                rows = db.execute(
                    "SELECT * FROM review_cases ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT * FROM review_cases
                    WHERE status = %s ORDER BY created_at DESC LIMIT %s
                    """,
                    (status, limit),
                ).fetchall()
        return [
            self._case_row(row)
            for row in rows
            if row is not None
        ]

    def get_case(self, case_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM review_cases WHERE case_id = %s",
                (case_id,),
            ).fetchone()
        return self._case_row(row)

    def record_review(
        self, case_id: int, decision: ReviewDecision
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self._connect() as db:
            row = db.execute(
                "SELECT status FROM review_cases WHERE case_id = %s FOR UPDATE",
                (case_id,),
            ).fetchone()
            if row is None:
                raise CaseNotFoundError(f"Review case {case_id} does not exist")
            if row["status"] in ("resolved", "dismissed"):
                raise InvalidCaseTransitionError(
                    f"Cannot review a case already marked {row['status']}"
                )

            db.execute(
                """
                UPDATE review_cases SET status=%s, assigned_to=%s,
                    confirmed_category=%s, reviewer_note=%s, updated_at=%s
                WHERE case_id=%s
                """,
                (
                    decision.status.value,
                    decision.actor,
                    decision.confirmed_category.value
                    if decision.confirmed_category
                    else None,
                    decision.note,
                    now,
                    case_id,
                ),
            )
            db.execute(
                """
                INSERT INTO audit_log (
                    entity_type, entity_id, action, actor, details_json, created_at
                ) VALUES ('case', %s, 'review_recorded', %s, %s::jsonb, %s)
                """,
                (
                    str(case_id),
                    decision.actor,
                    self._json(decision.model_dump(mode="json")),
                    now,
                ),
            )
            result = db.execute(
                "SELECT * FROM review_cases WHERE case_id = %s",
                (case_id,),
            ).fetchone()
        return self._case_row(result)

    def save_incidents(self, incidents: list[Incident]) -> int:
        now = datetime.now(timezone.utc)
        inserted = 0
        with self._connect() as db:
            for incident in incidents:
                cursor = db.execute(
                    """
                    INSERT INTO incidents (
                        incident_id, provider, response_code, window_start,
                        window_end, severity, payload_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT(incident_id) DO NOTHING
                    """,
                    (
                        incident.incident_id,
                        incident.provider,
                        incident.response_code,
                        incident.window_start,
                        incident.window_end,
                        incident.severity,
                        incident.model_dump_json(),
                        now,
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def list_incidents(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT payload_json FROM incidents
                ORDER BY created_at DESC LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [
            row["payload_json"]
            if isinstance(row["payload_json"], dict)
            else json.loads(row["payload_json"])
            for row in rows
        ]

    def audit_history(
        self, entity_type: str, entity_id: str
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT action, actor, details_json, created_at FROM audit_log
                WHERE entity_type=%s AND entity_id=%s ORDER BY audit_id
                """,
                (entity_type, entity_id),
            ).fetchall()
        return [
            {
                "action": row["action"],
                "actor": row["actor"],
                "details": row["details_json"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def summary(self) -> dict[str, int]:
        with self._connect() as db:
            transactions = db.execute(
                "SELECT COUNT(*) AS n FROM transactions"
            ).fetchone()["n"]
            events = db.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            open_cases = db.execute(
                """
                SELECT COUNT(*) AS n FROM review_cases
                WHERE status IN ('open', 'in_progress')
                """
            ).fetchone()["n"]
            incidents = db.execute(
                "SELECT COUNT(*) AS n FROM incidents"
            ).fetchone()["n"]
        return {
            "transactions": transactions,
            "events": events,
            "open_cases": open_cases,
            "incidents": incidents,
        }
