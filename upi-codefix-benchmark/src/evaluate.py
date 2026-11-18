"""Evaluate rules and a fine-tuned classifier on an untouched CSV test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from src.domain import FailureCategory
from src.model import QwenFailureClassifier
from src.triage import classify_with_rules


def load_test_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"text", "label", "split", "scenario_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset must contain columns {sorted(required)}")
        rows = [row for row in reader if row["split"].strip().lower() == "test"]
    if not rows:
        raise ValueError(f"No test rows found in {path}")
    if len({row["scenario_id"] for row in rows}) != len(rows):
        raise ValueError("Test split contains repeated scenario IDs")
    return rows


def validate_split_isolation(path: Path) -> None:
    split_ids: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = row["split"].strip().lower()
            if split not in split_ids:
                raise ValueError(f"Unsupported dataset split: {split!r}")
            scenario_id = row["scenario_id"].strip()
            if not scenario_id:
                raise ValueError("Every dataset row requires a scenario_id")
            split_ids[split].add(scenario_id)

    if any(not scenario_ids for scenario_ids in split_ids.values()):
        raise ValueError("Dataset must contain train, validation, and test splits")
    names = list(split_ids)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = split_ids[left_name] & split_ids[right_name]
            if overlap:
                raise ValueError(
                    f"Scenario leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )


def expected_calibration_error(
    truth: list[str],
    predicted: list[str],
    confidence: list[float],
    bins: int = 10,
) -> float:
    if not (len(truth) == len(predicted) == len(confidence)) or not truth:
        raise ValueError("Calibration inputs must be non-empty and have equal lengths")
    total = len(truth)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, score in enumerate(confidence)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1)
        ]
        if members:
            accuracy = sum(truth[index] == predicted[index] for index in members) / len(members)
            mean_confidence = sum(confidence[index] for index in members) / len(members)
            error += len(members) / total * abs(accuracy - mean_confidence)
    return float(error)


def evaluate_predictions(
    truth: list[str],
    predicted: list[str],
    confidence: list[float],
    labels: list[str],
    abstain_threshold: float,
) -> dict[str, Any]:
    if not (len(truth) == len(predicted) == len(confidence)):
        raise ValueError("Truth, predictions, and confidence lengths must match")

    routed = [
        prediction if score >= abstain_threshold else FailureCategory.UNKNOWN.value
        for prediction, score in zip(predicted, confidence)
    ]
    report = classification_report(
        truth, routed, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "sample_count": len(truth),
        "accuracy": float(accuracy_score(truth, routed)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, routed)),
        "macro_f1": float(f1_score(truth, routed, labels=labels, average="macro", zero_division=0)),
        "expected_calibration_error": expected_calibration_error(
            truth, predicted, confidence
        ),
        "coverage": float(sum(score >= abstain_threshold for score in confidence) / len(confidence)),
        "abstain_threshold": abstain_threshold,
        "confusion_matrix": confusion_matrix(truth, routed, labels=labels).tolist(),
        "per_class": report,
    }


def evaluate_rules(rows: list[dict[str, str]]) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    confidences: list[float] = []
    for row in rows:
        result = classify_with_rules(row["text"])
        predictions.append(result.category.value)
        confidences.append(result.confidence)
    return predictions, confidences


def evaluate_model(
    rows: list[dict[str, str]], model_path: Path
) -> tuple[list[str], list[float]]:
    classifier = QwenFailureClassifier(model_path)
    predictions: list[str] = []
    confidences: list[float] = []
    for row in rows:
        result = classifier.predict(row["text"])
        predictions.append(result.category.value)
        confidences.append(result.confidence)
    return predictions, confidences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/labeled_incidents.csv"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--abstain-threshold", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.abstain_threshold <= 1:
        raise ValueError("--abstain-threshold must be between 0 and 1")

    validate_split_isolation(args.dataset)
    rows = load_test_rows(args.dataset)
    labels = [category.value for category in FailureCategory]
    truth = [row["label"].strip().lower() for row in rows]
    unsupported = set(truth) - set(labels)
    if unsupported:
        raise ValueError(f"Test split has unsupported labels: {sorted(unsupported)}")
    missing_test_labels = set(labels) - set(truth)
    if missing_test_labels:
        raise ValueError(
            f"Test split is missing classes: {sorted(missing_test_labels)}"
        )

    results: dict[str, Any] = {}
    rules_predictions, rules_confidence = evaluate_rules(rows)
    results["rules"] = evaluate_predictions(
        truth, rules_predictions, rules_confidence, labels, args.abstain_threshold
    )
    if args.model is not None:
        model_predictions, model_confidence = evaluate_model(rows, args.model)
        results["fine_tuned_model"] = evaluate_predictions(
            truth, model_predictions, model_confidence, labels, args.abstain_threshold
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
