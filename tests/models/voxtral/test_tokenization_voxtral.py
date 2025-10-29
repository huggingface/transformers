import pytest

from transformers import AutoTokenizer
from transformers.models.auto import tokenization_auto
from transformers.models.voxtral import VoxtralConfig

def test_voxtral_tokenizer_requires_mistral_common(monkeypatch):
    monkeypatch.setattr(tokenization_auto, "is_mistral_common_available", lambda: False)
    monkeypatch.setattr(tokenization_auto, "get_tokenizer_config", lambda *args, **kwargs: {})
    with pytest.raises(ImportError, match="mistral-common"):
        AutoTokenizer.from_pretrained("dummy", config=VoxtralConfig())
