import torch

from transformers.models.blueberry import BlueberryConfig
from transformers.models.blueberry import modeling_blueberry as modeling


def test_blueberry_model_forward_shapes():
    config = BlueberryConfig(
        vocab_size=101,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    )
    model = modeling.BlueberryModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    outputs = model(input_ids=input_ids)
    hidden = outputs.last_hidden_state
    assert hidden.shape == (2, 12, config.hidden_size)


def test_blueberry_causallm_head():
    config = BlueberryConfig(
        vocab_size=77,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        layer_types=["full_attention", "sliding_attention"],
    )
    model = modeling.BlueberryForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    outputs = model(input_ids=input_ids)
    assert outputs.logits.shape == (1, 8, config.vocab_size)

