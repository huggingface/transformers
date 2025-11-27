import pytest

from transformers import Evo2Tokenizer


@pytest.fixture
def tokenizer():
    return Evo2Tokenizer()


def test_round_trip_ascii(tokenizer):
    text = "Hello, Evo2!"
    encoded = tokenizer(text)["input_ids"]
    expected = list(text.encode("utf-8"))
    assert encoded == expected
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_clamp_behavior(tokenizer):
    tokens = [0, 1, 255, 300, -5]
    decoded = tokenizer.decode(tokens)
    expected = "".join(chr(max(0, min(255, token))) for token in tokens)
    assert decoded == expected


def test_tokenize_returns_bytes(tokenizer):
    text = "ABcd"
    tokens = tokenizer.tokenize(text)
    assert tokens == list(text.encode("utf-8"))
