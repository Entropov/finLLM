from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "LLaMA-Factory" / "src"))

from llamafactory.hparams.finetuning_args import FinetuningArguments  # noqa: E402
from llamafactory.train.trainer_utils import (  # noqa: E402
    _mark_citation_spans,
    audit_weighted_loss_func,
    make_audit_weighted_loss,
)


def _outputs(probability_by_position: list[float], vocab_size: int = 2) -> dict[str, torch.Tensor]:
    logits = torch.zeros((1, len(probability_by_position), vocab_size), dtype=torch.float32)
    for index, probability in enumerate(probability_by_position):
        logit = torch.logit(torch.tensor(probability).clamp(1e-5, 1 - 1e-5))
        logits[0, index, 1] = logit
    return {"logits": logits}


def _loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, **overrides: object) -> torch.Tensor:
    options = {
        "numeric_token_ids": set(),
        "citation_open_patterns": [],
        "citation_close_ids": set(),
        "keyphrase_patterns": [],
        "eos_token_id": None,
        "numeric_weight": 3.0,
        "citation_weight": 3.0,
        "keyphrase_weight": 2.0,
        "eos_weight": 2.0,
    }
    options.update(overrides)
    return audit_weighted_loss_func(outputs, labels, **options)


def test_audit_weighted_loss_is_sample_balanced() -> None:
    logits = torch.zeros((2, 4, 2), dtype=torch.float32)
    logits[0, :, 1] = torch.logit(torch.tensor(0.9))
    logits[1, :, 1] = torch.logit(torch.tensor(0.1))
    labels = torch.tensor([[-100, 1, -100, -100], [-100, 1, 1, 1]])

    loss = _loss({"logits": logits}, labels)
    expected = (-torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(0.1))) / 2
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_numeric_error_receives_more_weight() -> None:
    labels = torch.tensor([[-100, 1, 0]])
    outputs = _outputs([0.1, 0.1, 0.5])
    plain = _loss(outputs, labels)
    weighted = _loss(outputs, labels, numeric_token_ids={1}, numeric_weight=4.0)

    assert weighted.item() != pytest.approx(plain.item())


def test_citation_and_keyphrase_spans_are_supported() -> None:
    labels = torch.tensor([[-100, 3, 4, 5, 6, 7]])
    outputs = _outputs([0.5] * labels.size(-1), vocab_size=8)
    loss = _loss(
        outputs,
        labels,
        citation_open_patterns=[[3, 4]],
        citation_close_ids={5},
        keyphrase_patterns=[[6, 7]],
    )
    assert torch.isfinite(loss)


def test_citation_mask_requires_evidence_prefix_and_handles_adjacent_ids() -> None:
    labels = torch.tensor([[3, 4, 1, 5, 3, 4, 2, 5, 8, 1, 5]])
    mask = _mark_citation_spans(labels, open_patterns=[[3, 4]], close_ids={5})
    assert mask.tolist() == [[True, True, True, True, True, True, True, True, False, False, False]]


def test_audit_loss_factory_builds_evidence_specific_open_patterns() -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def get_vocab(self) -> dict[str, int]:
            return {"0": 0, "x": 1, "%": 2}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(character) for character in text]

    loss = make_audit_weighted_loss(
        FakeTokenizer(),
        numeric_weight=3.0,
        citation_weight=3.0,
        keyphrase_weight=2.0,
        eos_weight=2.0,
    )
    assert loss.keywords["citation_open_patterns"] == [[ord("["), ord("E")], [ord(" "), ord("["), ord("E")]]
    assert loss.keywords["citation_close_ids"] == {ord("]")}
    assert [ord(character) for character in "render_quant_artifact"] in loss.keywords["keyphrase_patterns"]


def test_audit_loss_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        FinetuningArguments(use_dft_loss=True, use_audit_weighted_loss=True)


def test_audit_loss_rejects_non_positive_weights() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        FinetuningArguments(use_audit_weighted_loss=True, audit_numeric_weight=0.0)
