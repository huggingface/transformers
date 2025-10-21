import torch

from transformers.moderation import DummySafetyChecker
from transformers.generation.moderation import ModerationLogitsProcessor, ModerationStoppingCriteria
from transformers.moderation.base import SafetyConfig


class FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def batch_decode(self, input_ids, skip_special_tokens=True):
        texts = []
        for ids in input_ids:
            texts.append(" ".join(self.vocab[i.item()] for i in ids))
        return texts


def test_logits_processor_masks_forbidden_token():
    vocab = {0: "[PAD]", 1: "hello", 2: "badword", 3: "world"}
    tokenizer = FakeTokenizer(vocab)
    checker = DummySafetyChecker(forbidden_token_id=2)

    processor = ModerationLogitsProcessor(checker, tokenizer)

    # batch of 2 sequences: one safe, one unsafe
    input_ids = torch.tensor([[1, 3], [1, 2]])
    scores = torch.zeros((2, 4))
    out = processor(input_ids, scores.clone())

    # For the unsafe second example, the forbidden token id 2 should be suppressed
    assert out[1, 2] == -float("inf")
    # For the safe example, token 2 should remain untouched
    assert out[0, 2] == 0.0


def test_stopping_criteria_stops_on_unsafe():
    vocab = {0: "[PAD]", 1: "hello", 2: "badword", 3: "world"}
    tokenizer = FakeTokenizer(vocab)
    checker = DummySafetyChecker(forbidden_token_id=2)
    cfg = SafetyConfig(threshold=0.5, stop_on_unsafe=True)

    criteria = ModerationStoppingCriteria(checker, tokenizer, cfg)

    input_ids_safe = torch.tensor([[1, 3]])
    input_ids_unsafe = torch.tensor([[1, 2]])
    dummy_scores = torch.zeros((1, 4))

    assert criteria(input_ids_safe, dummy_scores) is False
    assert criteria(input_ids_unsafe, dummy_scores) is True
