import pytest
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM
from huggingface_hub import ModelCard
from transformers import PreTrainedModel

def test_from_pretrained_ignore_mismatched_sizes():
    model = AutoModelForCausalLM.from_pretrained(
        "OpenAI-ChatGPT/ChatGPT-4", trust_remote_code=True, ignore_mismatched_sizes=True
    )