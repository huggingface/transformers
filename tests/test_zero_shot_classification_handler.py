import pytest

from transformers.pipelines import ZeroShotClassificationArgumentHandler


def test_valid_hypothesis_template():
    handler = ZeroShotClassificationArgumentHandler()
    sequences = ["I love this movie"]
    labels = ["positive", "negative"]
    template = "This movie is {}."

    # Should work fine with simple template
    sequence_pairs, _ = handler(sequences, labels, template)
    assert len(sequence_pairs) == 2
    assert sequence_pairs[0][1] == "This movie is positive."


def test_invalid_hypothesis_templates():
    handler = ZeroShotClassificationArgumentHandler()
    sequences = ["I love this movie"]
    labels = ["positive"]

    invalid_templates = [
        "{:>10}",  # Format specifier
        "{!r}",  # Conversion flag
        "{}{}",  # Multiple placeholders
        "No placeholder",  # Missing placeholder
        "{0}",  # Indexed placeholder
    ]

    for template in invalid_templates:
        with pytest.raises(ValueError, match=".*must contain exactly one simple {} placeholder.*"):
            handler(sequences, labels, template)


def test_template_without_placeholder_usage():
    handler = ZeroShotClassificationArgumentHandler()
    sequences = ["I love this movie"]
    labels = ["positive"]
    template = "This template doesn't use the placeholder"

    with pytest.raises(ValueError, match=".*must contain exactly one simple {} placeholder.*"):
        handler(sequences, labels, template)
