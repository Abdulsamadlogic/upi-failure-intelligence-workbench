"""Generate demonstration data and fine-tune Qwen for failure classification."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYNTHETIC_ROW_COUNT = 2000
CSV_COLUMNS = ("text", "label", "split", "scenario_id", "synthetic")
FAILURE_LABELS = (
    "network_timeout",
    "payer_bank_decline",
    "beneficiary_bank_unavailable",
    "merchant_confirmation_timeout",
    "settlement_mismatch",
    "unknown",
)


def _failure_trace(
    label: str,
    rng: random.Random,
    sequence: int,
) -> str:
    providers = [f"sim_bank_{index:02d}" for index in range(1, 13)]
    components = {
        "network_timeout": [
            "upi_switch",
            "gateway_adapter",
            "bank_connector",
            "status_poller",
            "shared_adapter",
            "reconciliation_worker",
        ],
        "payer_bank_decline": [
            "issuer_adapter",
            "payer_bank_host",
            "risk_gateway",
            "shared_adapter",
            "upi_switch",
        ],
        "beneficiary_bank_unavailable": [
            "beneficiary_route",
            "payee_bank_host",
            "routing_adapter",
            "shared_adapter",
            "bank_connector",
        ],
        "merchant_confirmation_timeout": [
            "merchant_callback",
            "pos_gateway",
            "confirmation_listener",
            "shared_adapter",
            "upi_switch",
        ],
        "settlement_mismatch": [
            "reconciliation_worker",
            "ledger_comparator",
            "settlement_adapter",
            "shared_adapter",
            "bank_connector",
        ],
        "unknown": [
            "event_consumer",
            "status_adapter",
            "ledger_listener",
            "shared_adapter",
            "upi_switch",
        ],
    }
    ambiguous_codes = [
        "UPSTREAM_ERROR",
        "STATUS_UNAVAILABLE",
        "PROCESSING_ERROR",
    ]
    codes = {
        "network_timeout": [
            "GW_TIMEOUT", "READ_TIMEOUT", "HTTP_504", "NO_RESPONSE",
        ],
        "payer_bank_decline": [
            "ISSUER_DECLINED", "LIMIT_REACHED", "ACCOUNT_RESTRICTED",
        ],
        "beneficiary_bank_unavailable": [
            "HOST_UNAVAILABLE", "ROUTE_OFFLINE", "CREDIT_HOST_BUSY",
        ],
        "merchant_confirmation_timeout": [
            "ACK_TIMEOUT", "CALLBACK_MISSING", "POS_NO_ACK",
        ],
        "settlement_mismatch": [
            "LEDGER_MISMATCH", "CREDIT_NOT_FOUND", "RECONCILE_REQUIRED",
        ],
        "unknown": [
            "UNMAPPED_17", "UPSTREAM_UNKNOWN", "INCOMPLETE_TRACE",
        ],
    }
    messages = {
        "network_timeout": [
            "No response arrived from {component} within {latency} ms during {phase}.",
            "The connector closed the request after {latency} ms without a final status.",
            "Status polling ended without an upstream reply; elapsed time was {latency} ms.",
            "A gateway read deadline expired before the remote bank returned a result.",
            "The route remained pending after a socket timeout during {phase}.",
        ],
        "payer_bank_decline": [
            "Issuer rejected the debit request with reason {reason}.",
            "Payer-side authorization was refused; the account limit response was returned.",
            "The payer bank declined the request before a debit confirmation was recorded.",
            "Issuer response indicates that this debit is not permitted for the account.",
            "The debit authorization was rejected and no payer debit was observed.",
        ],
        "beneficiary_bank_unavailable": [
            "The beneficiary route returned {reason}; credit confirmation is absent.",
            "Payee-bank host did not accept the credit request during {phase}.",
            "The destination bank route was unavailable after payer debit confirmation.",
            "Credit status remains pending because the beneficiary host is not responding.",
            "The receiving-bank connector returned a temporary unavailable response.",
        ],
        "merchant_confirmation_timeout": [
            "No merchant acknowledgement was received within {latency} ms.",
            "The POS callback is missing although the payment request was accepted.",
            "Merchant confirmation polling expired before a final acknowledgement arrived.",
            "The merchant adapter has no final response for this transaction.",
            "Payment status is pending while the merchant endpoint acknowledgement is absent.",
        ],
        "settlement_mismatch": [
            "Ledger shows a payer debit but no matching beneficiary credit entry.",
            "Reconciliation found different payer-side and receiving-side transaction states.",
            "Debit confirmation exists; the expected credit record is missing from the ledger.",
            "Settlement comparison could not match the credit to the confirmed debit.",
            "The transaction remains unresolved because debit and credit records disagree.",
        ],
        "unknown": [
            "Upstream response was not mapped and the trace lacks a definitive final state.",
            "The adapter returned an unclassified result; source evidence is incomplete.",
            "Available events do not establish whether the payment completed or failed.",
            "The latest status conflicts with an earlier observation; manual reconciliation needed.",
            "No diagnostic cause was supplied by the upstream component.",
        ],
    }
    payment_type = (
        "merchant"
        if label == "merchant_confirmation_timeout"
        or (
            label == "unknown"
            and rng.random() < 0.25
        )
        else "transfer"
    )
    status = "PENDING" if rng.random() < 0.3 else "FAILED"
    component = rng.choice(components[label])
    response_code = (
        rng.choice(ambiguous_codes)
        if rng.random() < 0.2
        else rng.choice(codes[label])
    )
    phase = rng.choice(
        ["authorization", "debit confirmation", "credit routing", "status reconciliation"]
    )
    latency = rng.choice([800, 1200, 2500, 5000, 10000, 15000, 30000])
    reason_options = {
        "network_timeout": ["upstream response deadline", "socket read deadline"],
        "payer_bank_decline": [
            "issuer policy",
            "account limit",
            "debit not permitted",
            "account restriction",
        ],
        "beneficiary_bank_unavailable": [
            "temporary host error",
            "route unavailable",
            "receiving bank busy",
        ],
        "merchant_confirmation_timeout": [
            "merchant endpoint did not acknowledge",
            "callback deadline expired",
        ],
        "settlement_mismatch": [
            "credit record missing",
            "ledger comparison failed",
        ],
        "unknown": [
            "unmapped response",
            "incomplete event history",
            "source status unavailable",
        ],
    }
    reason = rng.choice(reason_options[label])
    message = rng.choice(messages[label]).format(
        component=component,
        latency=latency,
        phase=phase,
        reason=reason,
    )
    payer_debited = label in (
        "beneficiary_bank_unavailable",
        "merchant_confirmation_timeout",
        "settlement_mismatch",
    )
    if label == "network_timeout":
        payer_debited = rng.choice([True, False, None])
    elif label == "unknown":
        payer_debited = rng.choice([True, False, None])
    beneficiary_credited = (
        False
        if payer_debited is True
        else rng.choice([False, None])
    )
    if label == "unknown":
        beneficiary_credited = rng.choice([True, False, None])
    merchant_confirmed = (
        False
        if payment_type == "merchant"
        and label == "merchant_confirmation_timeout"
        else None
    )
    reversal = (
        rng.choice(["pending", "unknown"])
        if payer_debited is True
        else "not_required"
    )
    payer_value = "unknown" if payer_debited is None else str(payer_debited).lower()
    beneficiary_value = (
        "unknown" if beneficiary_credited is None else str(beneficiary_credited).lower()
    )
    merchant_value = (
        "unknown" if merchant_confirmed is None else str(merchant_confirmed).lower()
    )
    trace = f"trace-{sequence:06d}-{rng.getrandbits(24):06x}"
    amount = rng.choice([75, 120, 249, 499, 850, 1250, 2400, 5100])
    provider = rng.choice(providers)
    if sequence % 4 == 0:
        return (
            f"[{trace}] {provider}/{component} returned {response_code} during {phase}. "
            f"status={status}; payment_type={payment_type}; amount_inr={amount}. "
            f"{message} ledger(payer_debited={payer_value}, "
            f"beneficiary_credited={beneficiary_value}, "
            f"merchant_confirmed={merchant_value}, reversal={reversal})"
        )
    if sequence % 4 == 1:
        return (
            f"trace_id={trace} provider={provider} component={component} "
            f"payment_type={payment_type} amount_inr={amount} | "
            f"event_1 status={status} response_code={response_code} "
            f"latency_ms={latency} phase={phase} | "
            f"event_2 message=\"{message}\" payer_debited={payer_value} "
            f"beneficiary_credited={beneficiary_value} "
            f"merchant_confirmed={merchant_value} reversal_status={reversal}"
        )
    if sequence % 4 == 2:
        return (
            f"{provider} | {component} | {response_code} | {status} | {phase} | "
            f"{latency}ms | {message} | debit={payer_value} | "
            f"credit={beneficiary_value} | merchant_ack={merchant_value} | "
            f"reversal={reversal} | amount={amount} INR | ref={trace}"
        )
    return (
        f"provider={provider} payment_type={payment_type} status={status} "
        f"amount_inr={amount} component={component} response_code={response_code} "
        f"trace_id={trace} phase={phase} latency_ms={latency} "
        f"message={message} payer_debited={payer_value} "
        f"beneficiary_credited={beneficiary_value} "
        f"merchant_confirmed={merchant_value} reversal_status={reversal}"
    )


def generate_synthetic_dataset(path: Path, seed: int = 42) -> int:
    """Write 2,000 varied, explicitly synthetic training examples."""
    rng = random.Random(seed)
    labels = list(FAILURE_LABELS)
    rows_by_label: dict[str, list[dict[str, str]]] = {
        label: [] for label in labels
    }
    label_counts = {
        label: SYNTHETIC_ROW_COUNT // len(labels)
        for label in labels
    }
    for label in labels[: SYNTHETIC_ROW_COUNT % len(labels)]:
        label_counts[label] += 1

    sequence = 0
    for label in labels:
        for _ in range(label_counts[label]):
            sequence += 1
            rows_by_label[label].append(
                {
                    "text": "",
                    "label": label,
                    "scenario_id": f"synthetic-{label}-{sequence:06d}",
                    "synthetic": "true",
                    "_sequence": str(sequence),
                }
            )

    for records in rows_by_label.values():
        rng.shuffle(records)
        total = len(records)
        train_count = round(total * 0.70)
        validation_count = round(total * 0.15)
        for index, row in enumerate(records):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "test"
            row["text"] = _failure_trace(
                row["label"],
                rng,
                int(row["_sequence"]),
            )
            del row["_sequence"]

    rows = [row for label in labels for row in rows_by_label[label]]
    rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


@dataclass
class EncodedRecord:
    input_ids: list[int]
    attention_mask: list[int]
    labels: int


class IncidentDataset:
    def __init__(
        self,
        records: list[dict[str, str]],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_input_tokens: int = 384,
    ) -> None:
        self.rows: list[EncodedRecord] = []
        self.max_input_tokens = max_input_tokens
        for index, record in enumerate(records, start=1):
            text = record.get("text", "").strip()
            label = record.get("label", "").strip().lower()
            if not text:
                raise ValueError(f"Record {index} has empty text")
            if label not in label_to_id:
                raise ValueError(f"Record {index} has unsupported label {label!r}")

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            self.rows.append(
                EncodedRecord(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=label_to_id[label],
                )
            )

        if not self.rows:
            raise ValueError("Dataset split contains no records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
            "labels": row.labels,
        }


def load_split(csv_path: Path, expected_split: str) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "label", "split", "scenario_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required_columns)}"
            )
        rows = [
            row
            for row in reader
            if row["split"].strip().lower() == expected_split
        ]

    if not rows:
        raise ValueError(f"No {expected_split!r} rows found in {csv_path}")
    return rows


def validate_split_isolation(splits: dict[str, list[dict[str, str]]]) -> None:
    scenario_ids: dict[str, set[str]] = {
        name: {row["scenario_id"] for row in rows}
        for name, rows in splits.items()
    }
    names = list(scenario_ids)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = scenario_ids[left_name] & scenario_ids[right_name]
            if overlap:
                raise ValueError(
                    f"Scenario leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )


def compute_metrics(prediction: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/labeled_incidents.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen-upi-triage"))
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Write 2,000 varied but synthetic rows to --dataset and exit",
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.generate_synthetic:
        count = generate_synthetic_dataset(args.dataset, args.seed)
        print(
            f"Wrote {count} synthetic examples to {args.dataset}. "
            "They are for pipeline development, not real-world model claims."
        )
        return

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("--epochs and --learning-rate must be positive")

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from src.model import MODEL_ID, MAX_INPUT_TOKENS

    set_seed(args.seed)

    splits = {
        name: load_split(args.dataset, name)
        for name in ("train", "validation")
    }
    validate_split_isolation(splits)

    labels = list(FAILURE_LABELS)
    label_to_id = {label: index for index, label in enumerate(labels)}
    for split_name, rows in splits.items():
        observed = {row["label"].strip().lower() for row in rows}
        missing = set(labels) - observed
        if missing:
            raise ValueError(
                f"{split_name} split is missing classes: {sorted(missing)}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = IncidentDataset(
        splits["train"], tokenizer, label_to_id, MAX_INPUT_TOKENS
    )
    validation_dataset = IncidentDataset(
        splits["validation"], tokenizer, label_to_id, MAX_INPUT_TOKENS
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
            fp16=torch.cuda.is_available(),
            logging_strategy="steps",
            logging_steps=10,
        ),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()