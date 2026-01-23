import pytest

from transformers import AutoTokenizer
from transformers.models.voxtral import VoxtralConfig
import transformers.models.auto.tokenization_auto as ta


def test_voxtral_tokenizer_requires_mistral_common(monkeypatch):
    # Simulate that mistral_common is not available for the auto-tokenizer logic
    monkeypatch.setattr(ta, "is_mistral_common_available", lambda: False)
    # Avoid network access by short-circuiting tokenizer_config retrieval
    monkeypatch.setattr(ta, "get_tokenizer_config", lambda *args, **kwargs: {})
    with pytest.raises(ImportError, match="mistral-common"):
        # Using a dummy path since the guard should raise before any file access
        AutoTokenizer.from_pretrained("dummy", config=VoxtralConfig())


