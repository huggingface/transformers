import torch

from transformers import T5GemmaConfig, T5GemmaForConditionalGeneration, T5GemmaModuleConfig


def _tiny():
    return T5GemmaModuleConfig(
        vocab_size=33,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        layer_types=["full_attention"] * 2,
        rope_theta=10000,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )


def test_generate_use_cache_works_for_t5gemma():
    cfg = T5GemmaConfig(encoder=_tiny(), decoder=_tiny(), vocab_size=33, attn_implementation="eager")
    model = T5GemmaForConditionalGeneration(cfg)

    output = model.generate(torch.randint(0, 33, (1, 10)), use_cache=True, max_new_tokens=2)

    assert output.shape[0] == 1
    assert output.shape[1] > 0
