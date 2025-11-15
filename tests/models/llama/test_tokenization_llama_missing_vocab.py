import pytest

from transformers.models.llama.tokenization_llama import LlamaTokenizer


def test_llama_tokenizer_missing_vocab_shows_mistral_hint():
    # Construct tokenizer with missing vocab_file and from_slow=True to exercise the legacy Load path
    with pytest.raises(Exception) as excinfo:
        # from_slow is passed via kwargs in __init__ through get_spm_processor
        LlamaTokenizer(vocab_file=None, from_slow=True)

    msg = str(excinfo.value)
    assert "mistral-common" in msg or "mistral" in msg.lower()
