"""Inference for a fine-tuned Qwen transaction-failure classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.domain import FailureCategory


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_INPUT_TOKENS = 384


@dataclass(frozen=True)
class ModelPrediction:
    category: FailureCategory
    confidence: float
    probabilities: dict[str, float]


class FailureClassifier(Protocol):
    def predict(self, text: str) -> ModelPrediction: ...


class QwenFailureClassifier:
    """Loads a fine-tuned sequence classifier from a local model directory."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Fine-tuned model directory not found: {model_path}")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()

        configured_labels = self.model.config.id2label
        self.id_to_label = {
            int(index): str(label).lower()
            for index, label in configured_labels.items()
        }
        expected = {category.value for category in FailureCategory}
        actual = set(self.id_to_label.values())
        if actual != expected:
            raise ValueError(
                f"Model labels {sorted(actual)} do not match required labels "
                f"{sorted(expected)}"
            )

    @classmethod
    def from_base_model(
        cls,
        *,
        device: str | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> "QwenFailureClassifier":
        """Create an un-fine-tuned classifier for a baseline experiment."""
        instance = cls.__new__(cls)
        instance.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        instance.max_input_tokens = max_input_tokens
        labels = [category.value for category in FailureCategory]
        label_to_id = {label: index for index, label in enumerate(labels)}
        instance.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=len(labels),
            id2label={index: label for label, index in label_to_id.items()},
            label2id=label_to_id,
        )
        instance.model.config.pad_token_id = instance.tokenizer.pad_token_id
        instance.model.to(instance.device)
        instance.model.eval()
        instance.id_to_label = {index: label for label, index in enumerate(labels)}
        return instance

    def predict(self, text: str) -> ModelPrediction:
        if not text.strip():
            raise ValueError("Classifier input must not be empty")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        probability_by_label = {
            self.id_to_label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
        winning_label = max(probability_by_label, key=probability_by_label.get)
        return ModelPrediction(
            category=FailureCategory(winning_label),
            confidence=probability_by_label[winning_label],
            probabilities=probability_by_label,
        )
