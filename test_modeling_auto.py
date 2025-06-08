import pytest
from transformers import AutoModelForCausalLM
from huggingface_hub import ModelCard

def test_from_pretrained_ignore_mismatched_sizes():
    model = AutoModelForCausalLM.from_pretrained(
        "OpenAI-ChatGPT/ChatGPT-4", trust_remote_code=True, ignore_mismatched_sizes=True
    )
    assert isinstance(model, (Qwen2ForCausalLM, PreTrainedModel))