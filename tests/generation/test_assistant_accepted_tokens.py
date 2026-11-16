import copy

import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM


@pytest.mark.parametrize("do_sample", [False, True])
def test_accepted_tokens_per_round(do_sample):
    torch.manual_seed(1337)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    target = LlamaForCausalLM(config).eval()
    assistant = copy.deepcopy(target).eval()
    assistant.generation_config.num_assistant_tokens = 4
    assistant.generation_config.num_assistant_tokens_schedule = "constant"
    assistant.generation_config.assistant_confidence_threshold = 0.0

    prompt = torch.tensor([[1, 11, 12, 13, 14, 15, 16, 17]])
    kwargs = {
        "assistant_model": assistant,
        "max_new_tokens": 8,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
    }

    with torch.inference_mode():
        torch.manual_seed(101)
        baseline = target.generate(prompt, **kwargs)
        torch.manual_seed(101)
        result = target.generate(prompt, output_accepted_tokens_per_round=True, **kwargs)

    assert torch.equal(baseline.sequences, result.sequences)
    assert "accepted_tokens_per_round" not in baseline
    counts = result.accepted_tokens_per_round
    assert counts
    assert all(type(n) is int and 0 <= n <= 4 for n in counts)
    assert sum(counts) <= 8

    if not do_sample:
        eos = int(result.sequences[0, prompt.shape[1]])
        with torch.inference_mode():
            stopped = target.generate(
                prompt,
                assistant_model=assistant,
                eos_token_id=eos,
                max_new_tokens=8,
                return_dict_in_generate=True,
                output_accepted_tokens_per_round=True,
            )
        assert stopped.sequences.shape[1] == prompt.shape[1] + 1
        assert stopped.accepted_tokens_per_round == [1]
