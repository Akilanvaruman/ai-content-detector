"""Desklib AI text detector — load + inference.

The desklib/ai-text-detector-v1.01 model is published as a DeBERTa-v3-large
backbone with a custom single-logit classification head trained with
BCEWithLogitsLoss (P(AI) = sigmoid(logit)).

We support both interfaces transparently so this file works whether the
checkpoint exposes:
  * shape (B, 1)  — single logit, BCE-trained → sigmoid
  * shape (B, 2)  — two-class softmax        → softmax(...)[:, 1]

Loading proceeds in two attempts:
  1. The published custom architecture (`DesklibAIDetectionModel`).
  2. Standard `AutoModelForSequenceClassification` (e.g. for fine-tuned forks).

Whichever loads first is used; the inference path then dispatches on the
output shape, so probabilities are always P(AI) regardless of the head.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from huggingface_hub import HfApi
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
)

MODEL_NAME = "desklib/ai-text-detector-v1.01"
MAX_LEN = 512


class DesklibAIDetectionModel(PreTrainedModel):
    """Custom architecture published by desklib. DeBERTa backbone + linear head."""

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Tokenizers may emit `token_type_ids` etc.; the DeBERTa backbone
        # ignores them, but Python rejects unknown kwargs without **kwargs.
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
        return {"logits": logits, "loss": loss}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_commit_hash(model_name: str) -> str:
    """Best-effort lookup of the current model revision for citation purposes."""
    try:
        info = HfApi().model_info(model_name)
        return info.sha or "unknown"
    except Exception:
        return "unknown"


def _extract_logits(output) -> torch.Tensor:
    """Pull the logits tensor out of either a HF ModelOutput or a plain dict."""
    if isinstance(output, dict):
        return output["logits"]
    return output.logits


def _logits_to_ai_prob(logits: torch.Tensor) -> torch.Tensor:
    """Convert a logits tensor to P(AI) regardless of head shape.

    Accepts:
      (B,)        single-logit binary head        → sigmoid
      (B, 1)      single-logit binary head        → sigmoid
      (B, 2)      two-class softmax head          → softmax(..., -1)[:, 1]
    Raises ValueError on anything else (we don't want to silently mislabel).
    """
    if logits.dim() == 1:
        return torch.sigmoid(logits)

    if logits.dim() == 2:
        n_classes = logits.size(-1)
        if n_classes == 1:
            return torch.sigmoid(logits.view(-1))
        if n_classes == 2:
            return torch.softmax(logits, dim=-1)[:, 1]

    raise ValueError(
        f"Unexpected logits shape {tuple(logits.shape)}; "
        "expected (B,), (B,1), or (B,2)."
    )


@dataclass
class DetectorInfo:
    model_name: str
    commit_hash: str
    device: str
    head: str  # "single-sigmoid" or "two-class-softmax"


class Detector:
    """Wraps the AI detector. Construct once, reuse across many predictions."""

    def __init__(self, model_name: str = MODEL_NAME, device: torch.device | None = None):
        self.model_name = model_name
        self.device = device or _pick_device()
        self.commit_hash = _resolve_commit_hash(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model, self.head = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_model(model_name: str) -> tuple[nn.Module, str]:
        """Try the custom Desklib head first, then fall back to a standard
        AutoModelForSequenceClassification (covers fine-tuned forks that may
        use a 2-class head).
        """
        config = AutoConfig.from_pretrained(model_name)

        # Attempt 1: custom Desklib architecture (single sigmoid).
        try:
            model = DesklibAIDetectionModel.from_pretrained(model_name, config=config)
            return model, "single-sigmoid"
        except Exception:
            pass

        # Attempt 2: standard sequence-classification head (1- or 2-class).
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        n_labels = getattr(model.config, "num_labels", 2) or 2
        head = "single-sigmoid" if n_labels == 1 else "two-class-softmax"
        return model, head

    def info(self) -> DetectorInfo:
        return DetectorInfo(
            model_name=self.model_name,
            commit_hash=self.commit_hash,
            device=str(self.device),
            head=self.head,
        )

    @torch.no_grad()
    def predict(self, text: str) -> float:
        """Run a single chunk (<= 512 tokens) through the model. Returns P(AI)."""
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        ).to(self.device)
        logits = _extract_logits(self.model(**encoded))
        prob = _logits_to_ai_prob(logits)
        return prob.view(-1)[0].item()

    @torch.no_grad()
    def predict_batch(self, chunks: Iterable[str], batch_size: int = 2) -> list[float]:
        """Run several chunks in mini-batches. Returns one probability per chunk.

        DeBERTa-v3-large activations at seq=512 are large; on MPS we keep
        batch_size small and explicitly drop the per-batch cache to avoid the
        gradual memory creep that MPS exhibits during long runs.
        """
        chunks = list(chunks)
        if not chunks:
            return []

        is_mps = self.device.type == "mps"
        is_cuda = self.device.type == "cuda"

        out: list[float] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
            ).to(self.device)
            logits = _extract_logits(self.model(**encoded))
            probs = _logits_to_ai_prob(logits).detach().cpu().tolist()
            out.extend(probs)

            # Free the per-batch activations before the next iteration.
            del encoded, logits
            if is_mps:
                torch.mps.empty_cache()
            elif is_cuda:
                torch.cuda.empty_cache()
        return out


def to_label(prob: float) -> str:
    """Bucket the average AI probability into the four research labels."""
    if prob < 0.25:
        return "Low"
    if prob < 0.50:
        return "Moderate"
    if prob < 0.75:
        return "High"
    return "Very High"


_DETECTOR: Detector | None = None


def get_detector() -> Detector:
    """Process-wide singleton so we only pay the load cost once."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return _DETECTOR


if __name__ == "__main__":
    # Smoke test: load the model and run two short samples.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    det = get_detector()
    info = det.info()
    print(f"Loaded {info.model_name}")
    print(f"  commit: {info.commit_hash}")
    print(f"  device: {info.device}")
    print(f"  head:   {info.head}")

    samples = [
        "The quick brown fox jumps over the lazy dog. It was a clear afternoon "
        "and the small village square smelled of bread and pine.",
        "In conclusion, the importance of leveraging cutting-edge synergies "
        "cannot be overstated. By holistically integrating multifaceted "
        "paradigms, organizations can unlock unprecedented value.",
    ]
    for s in samples:
        p = det.predict(s)
        print(f"  P(AI)={p:.4f}  label={to_label(p)}  text={s[:60]!r}…")
