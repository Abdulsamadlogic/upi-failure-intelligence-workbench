"""HTTP API for simulated UPI event ingestion, triage, incidents, and review."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain import (
    FailureCategory,
    Incident,
    ReviewDecision,
    ReviewStatus,
    TransactionEvent,
    project_transaction,
)
from src.incidents import detect_failure_spikes
from src.model import QwenFailureClassifier
from src.storage import (
    CaseNotFoundError,
    EventConflictError,
    InvalidCaseTransitionError,
    WorkbenchStore,
)
from src.triage import FailureTriageService, ReversalSlaPolicy


class EventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[TransactionEvent] = Field(min_length=1, max_length=500)


class IngestedEventResult(BaseModel):
    event_id: str
    transaction_id: str
    accepted: bool
    duplicate: bool
    case_id: int | None = None
    category: FailureCategory | None = None
    confidence: float | None = None
    requires_human_review: bool | None = None


class IngestResponse(BaseModel):
    accepted_count: int
    duplicate_count: int
    results: list[IngestedEventResult]


class ReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: str = Field(min_length=1, max_length=128)
    status: ReviewStatus
    confirmed_category: FailureCategory | None = None
    note: str = Field(min_length=1, max_length=2000)

    @field_validator("status")
    @classmethod
    def require_terminal_or_active_review(cls, value: ReviewStatus) -> ReviewStatus:
        if value == ReviewStatus.OPEN:
            raise ValueError("A reviewer cannot change a case back to open")
        return value


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UPI Failure Workbench</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f3f6fb; }
    body { margin: 0 auto; max-width: 1240px; padding: 28px; }
    header, section { background: white; border: 1px solid #dfe5ef; border-radius: 12px; padding: 20px; margin-bottom: 18px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 20px; }
    h1, h2 { margin: 0 0 12px; } h1 { font-size: 24px; } h2 { font-size: 18px; }
    p { color: #526078; } button { cursor: pointer; border: 0; border-radius: 7px; padding: 9px 13px; background: #2456a6; color: white; }
    button.secondary { background: #e8eef8; color: #18345e; } button:disabled { opacity: .55; cursor: wait; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    .metric { background: #f5f8fc; border-radius: 8px; padding: 14px; } .metric strong { display: block; font-size: 24px; }
    .table-wrap { overflow-x: auto; } table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 10px; text-align: left; vertical-align: top; border-bottom: 1px solid #e8edf4; }
    th { color: #526078; font-weight: 600; } textarea { box-sizing: border-box; width: 100%; min-height: 145px; margin: 8px 0; padding: 10px; font: 13px ui-monospace, monospace; }
    #message { min-height: 20px; color: #18345e; } .muted { color: #68758a; }
    @media (max-width: 720px) { body { padding: 12px; } header { align-items: flex-start; flex-direction: column; } .metrics { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header>
    <div><h1>UPI Failure Intelligence</h1><p>Simulated events · human-reviewed triage · no payment actions</p></div>
    <button id="refresh" class="secondary">Refresh</button>
  </header>
  <section><h2>Operations summary</h2><div id="metrics" class="metrics"></div></section>
  <section>
    <h2>Review queue</h2><p class="muted">Predictions are suggestions. Verify evidence before resolving a case.</p>
    <div class="table-wrap"><table><thead><tr><th>Transaction</th><th>Category</th><th>Confidence</th><th>Evidence</th><th>Action</th></tr></thead><tbody id="cases"></tbody></table></div>
  </section>
  <section>
    <h2>Provider incidents</h2><button id="detect" class="secondary">Analyze recent failure spikes</button>
    <div class="table-wrap"><table><thead><tr><th>Provider</th><th>Response code</th><th>Severity</th><th>Recent rate</th><th>Baseline rate</th></tr></thead><tbody id="incidents"></tbody></table></div>
  </section>
  <section>
    <h2>Replay sanitized events</h2>
    <p class="muted">Use synthetic examples only. Do not paste account numbers, UPI PINs, OTPs, or customer data.</p>
    <textarea id="event-json" spellcheck="false">[
  {
    "event_id": "demo-event-001",
    "transaction_id": "demo-transaction-001",
    "provider": "demo-bank",
    "payment_type": "transfer",
    "status": "FAILED",
    "response_code": "HOST_DOWN",
    "message": "beneficiary bank unavailable",
    "payer_debited": true,
    "beneficiary_credited": false,
    "reversal_status": "pending"
  }
]</textarea>
    <button id="ingest">Ingest events</button><div id="message" role="status"></div>
  </section>
  <script>
    const api = "/api/v1";
    const message = document.getElementById("message");
    function cell(row, value) { const td = document.createElement("td"); td.textContent = value ?? ""; row.appendChild(td); return td; }
    function rowFor(body, values) { const tr = document.createElement("tr"); values.forEach(value => cell(tr, value)); body.appendChild(tr); return tr; }
    async function request(path, options) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error((await response.text()) || response.statusText);
      return response.json();
    }
    async function refresh() {
      const [summary, cases, incidents] = await Promise.all([
        request(api + "/summary"), request(api + "/cases?limit=100"), request(api + "/incidents?limit=100")
      ]);
      const metrics = document.getElementById("metrics");
      metrics.replaceChildren();
      for (const [label, value] of Object.entries(summary)) {
        const card = document.createElement("div"); card.className = "metric";
        const title = document.createElement("span"); title.textContent = label.replaceAll("_", " ");
        const count = document.createElement("strong"); count.textContent = value;
        card.append(title, count); metrics.appendChild(card);
      }
      const caseBody = document.getElementById("cases"); caseBody.replaceChildren();
      for (const item of cases) {
        const tr = rowFor(caseBody, [item.transaction_id, item.category, (item.confidence * 100).toFixed(1) + "%", (item.evidence || []).join(" ")]);
        const actionCell = document.createElement("td");
        if (item.status === "open" || item.status === "in_progress") {
          const button = document.createElement("button"); button.textContent = "Resolve";
          button.addEventListener("click", async () => {
            const note = window.prompt("Add a short review note:");
            if (!note) return;
            try {
              await request(api + "/cases/" + item.case_id + "/review", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify({actor: "dashboard-reviewer", status: "resolved", confirmed_category: item.category, note})
              });
              await refresh();
            } catch (error) { message.textContent = error.message; }
          });
          actionCell.appendChild(button);
        } else { actionCell.textContent = item.status; }
        tr.appendChild(actionCell);
      }
      const incidentBody = document.getElementById("incidents"); incidentBody.replaceChildren();
      for (const item of incidents) {
        rowFor(incidentBody, [item.provider, item.response_code, item.severity, (item.recent_failure_rate * 100).toFixed(1) + "%", (item.baseline_failure_rate * 100).toFixed(1) + "%"]);
      }
    }
    document.getElementById("refresh").addEventListener("click", () => refresh().catch(error => message.textContent = error.message));
    document.getElementById("detect").addEventListener("click", async () => {
      try { await request(api + "/incidents/detect", {method: "POST"}); await refresh(); message.textContent = "Incident analysis completed."; }
      catch (error) { message.textContent = error.message; }
    });
    document.getElementById("ingest").addEventListener("click", async () => {
      try {
        const events = JSON.parse(document.getElementById("event-json").value);
        const result = await request(api + "/events", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({events})});
        message.textContent = result.accepted_count + " accepted; " + result.duplicate_count + " duplicate.";
        await refresh();
      } catch (error) { message.textContent = error.message; }
    });
    refresh().catch(error => message.textContent = error.message);
  </script>
</body>
</html>"""


def _read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def create_app(
    *,
    database_url: str | None = None,
    database_schema: str | None = None,
    store: WorkbenchStore | None = None,
    triage_mode: str | None = None,
    model_path: str | Path | None = None,
) -> FastAPI:
    mode = (triage_mode or os.getenv("TRIAGE_MODE", "rules")).strip().lower()
    if mode not in {"rules", "hybrid"}:
        raise ValueError("TRIAGE_MODE must be 'rules' or 'hybrid'")

    resolved_model_path = model_path or os.getenv("MODEL_PATH")
    classifier = None
    if mode == "hybrid":
        if not resolved_model_path:
            raise ValueError("Hybrid triage requires MODEL_PATH for a fine-tuned model")
        classifier = QwenFailureClassifier(resolved_model_path)

    resolved_database_url = database_url or os.getenv("UPI_DATABASE_URL")
    store_instance = store
    if store_instance is None and resolved_database_url:
        store_instance = WorkbenchStore(
            resolved_database_url,
            schema_name=database_schema,
            initialize=False,
        )
    triage = FailureTriageService(
        model=classifier,
        confidence_threshold=float(os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", "0.70")),
    )
    sla_policy = ReversalSlaPolicy(
        transfer_hours=_read_positive_int("UPI_TRANSFER_SLA_HOURS", 24),
        merchant_hours=_read_positive_int("UPI_MERCHANT_SLA_HOURS", 120),
    )

    app = FastAPI(
        title="UPI Failure Intelligence Workbench",
        version="0.1.0",
        description=(
            "A local, simulated payment-operations workbench. It never connects "
            "to a bank or initiates a refund."
        ),
    )
    app.state.store = store_instance
    app.state.triage = triage
    app.state.sla_policy = sla_policy
    app.state.triage_mode = mode

    @app.on_event("startup")
    def initialize_database() -> None:
        if store_instance is None:
            raise RuntimeError(
                "Set UPI_DATABASE_URL to a PostgreSQL connection URL before starting the API"
            )
        store_instance.initialize()

    def get_store() -> WorkbenchStore:
        if store_instance is None:
            raise RuntimeError("PostgreSQL storage is not configured")
        return store_instance

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "triage_mode": app.state.triage_mode}

    @app.post("/api/v1/events", response_model=IngestResponse)
    def ingest_events(batch: EventBatch) -> IngestResponse:
        store = get_store()
        results: list[IngestedEventResult] = []
        accepted_count = 0
        duplicate_count = 0
        try:
            insertion_results = store.append_events(batch.events)
        except EventConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        for event, inserted in zip(batch.events, insertion_results):
            if inserted:
                accepted_count += 1
            else:
                duplicate_count += 1

            history = store.events_for_transaction(event.transaction_id)
            projection = project_transaction(history)
            store.save_projection(projection)
            if (
                projection.status.value not in {"FAILED", "PENDING"}
                and not projection.state_conflicts
            ):
                results.append(
                    IngestedEventResult(
                        event_id=event.event_id,
                        transaction_id=event.transaction_id,
                        accepted=inserted,
                        duplicate=not inserted,
                    )
                )
                continue

            result = triage.analyze(history)
            case = None
            if result.requires_human_review:
                case = store.upsert_review_case(
                    transaction_id=result.transaction_id,
                    category=result.category.value,
                    confidence=result.confidence,
                    evidence=result.evidence,
                    recommended_action=result.recommended_action,
                )
            results.append(
                IngestedEventResult(
                    event_id=event.event_id,
                    transaction_id=event.transaction_id,
                    accepted=inserted,
                    duplicate=not inserted,
                    case_id=case["case_id"] if case else None,
                    category=result.category,
                    confidence=result.confidence,
                    requires_human_review=result.requires_human_review,
                )
            )
        return IngestResponse(
            accepted_count=accepted_count,
            duplicate_count=duplicate_count,
            results=results,
        )

    @app.get("/api/v1/transactions/{transaction_id}")
    def get_transaction(transaction_id: str) -> dict:
        store = get_store()
        projection = store.get_projection(transaction_id)
        if projection is None:
            raise HTTPException(status_code=404, detail="Transaction not found")
        history = store.events_for_transaction(transaction_id)
        return {
            "projection": projection,
            "events": [event.model_dump(mode="json") for event in history],
            "audit": store.audit_history("case", transaction_id),
        }

    @app.get("/api/v1/transactions/{transaction_id}/sla")
    def get_transaction_sla(transaction_id: str) -> dict:
        store = get_store()
        events = store.events_for_transaction(transaction_id)
        if not events:
            raise HTTPException(status_code=404, detail="Transaction not found")
        projection = project_transaction(events)
        assessment = sla_policy.assess(projection)
        if assessment is None:
            return {
                "transaction_id": transaction_id,
                "tracked": False,
                "reason": "No unresolved debit requiring reversal tracking.",
            }
        return {
            "transaction_id": transaction_id,
            "tracked": True,
            "deadline": assessment.deadline.isoformat(),
            "overdue": assessment.overdue,
            "hours_remaining": assessment.hours_remaining,
            "policy": "demo configuration; verify applicable current requirements",
        }

    @app.get("/api/v1/cases")
    def list_cases(
        status: ReviewStatus | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        store = get_store()
        return store.list_cases(status.value if status else None, limit)

    @app.get("/api/v1/cases/{case_id}")
    def get_case(case_id: int) -> dict:
        store = get_store()
        case = store.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Review case not found")
        return case

    @app.post("/api/v1/cases/{case_id}/review")
    def review_case(case_id: int, payload: ReviewPayload) -> dict:
        store = get_store()
        try:
            decision = ReviewDecision.model_validate(payload.model_dump())
            return store.record_review(case_id, decision)
        except CaseNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except InvalidCaseTransitionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/incidents/detect")
    def analyze_incidents(
        recent_minutes: int = Query(default=15, ge=1, le=1440),
        baseline_hours: int = Query(default=24, ge=1, le=720),
        minimum_failures: int = Query(default=5, ge=1, le=10000),
    ) -> dict:
        store = get_store()
        now = datetime.now(timezone.utc)
        events = store.list_events(
            since=now - timedelta(hours=baseline_hours),
            until=now,
            limit=100000,
        )
        incidents: list[Incident] = detect_failure_spikes(
            events,
            now=now,
            recent_window=timedelta(minutes=recent_minutes),
            baseline_window=timedelta(hours=baseline_hours),
            minimum_recent_failures=minimum_failures,
        )
        stored_count = store.save_incidents(incidents)
        return {
            "detected_count": len(incidents),
            "newly_stored_count": stored_count,
            "incidents": [incident.model_dump(mode="json") for incident in incidents],
        }

    @app.get("/api/v1/incidents")
    def list_incidents(limit: int = Query(default=100, ge=1, le=1000)) -> list[dict]:
        store = get_store()
        return store.list_incidents(limit)

    @app.get("/api/v1/summary")
    def get_summary() -> dict[str, int]:
        store = get_store()
        return store.summary()

    return app


app = create_app()
