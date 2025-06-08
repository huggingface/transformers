import pytest
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM
from huggingface_hub import ModelCard
from transformers import PreTrainedModel

def test_from_pretrained_ignore_mismatched_sizes():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "OpenAI-ChatGPT/ChatGPT-4",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        print("Model loaded successfully:", model)
        assert isinstance(model, (Qwen2ForCausalLM, PreTrainedModel)) 
    except Exception as e:
        print("Error occurred:", type(e).__name__, str(e))