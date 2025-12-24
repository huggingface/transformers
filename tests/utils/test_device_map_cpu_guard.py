import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.integrations.accelerate import _get_device_map


def test_device_map_auto_no_cpu_does_not_crash():
    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        vocab_size=100,
    )
    model = GPT2LMHeadModel(config)

    # Should not crash even if inferred max_memory has no "cpu"
    device_map = _get_device_map(
        model=model,
        device_map="auto",
        max_memory=None,
        hf_quantizer=None,
    )

    assert device_map is not None
