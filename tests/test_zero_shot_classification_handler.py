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
