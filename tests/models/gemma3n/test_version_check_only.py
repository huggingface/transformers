import pytest
import timm
from transformers import Gemma3nConfig, Gemma3nForConditionalGeneration

def test_gemma3n_model_raises_for_old_timm(monkeypatch):
    # Force timm to look "too old"
    monkeypatch.setattr(timm, "__version__", "0.9.10")

    # Instantiating the model should now raise our ImportError guard
    with pytest.raises(ImportError, match="Requires timm >= 0.9.16"):
        config = Gemma3nConfig()
        Gemma3nForConditionalGeneration(config)
