import pytest
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM
from huggingface_hub import ModelCard
from transformers import PreTrainedModel

def test_from_pretrained_ignore_mismatched_sizes():
    # Loading the model from the specified checkpoint, allowing mismatched sizes and trusting remote code
    model = AutoModelForCausalLM.from_pretrained(
        "OpenAI-ChatGPT/ChatGPT-4", trust_remote_code=True, ignore_mismatched_sizes=True
    )
    # Assert that the model is successfully loaded, raising an error if it fails
    assert model is not None, "Model failed to load with ignore_mismatched_sizes=True"