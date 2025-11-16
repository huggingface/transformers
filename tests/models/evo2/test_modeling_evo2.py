import unittest

import pytest

pytest.importorskip("parameterized")

from transformers import is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester

if is_torch_available():
    from transformers import Evo2ForCausalLM, Evo2Model


class Evo2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Evo2Model

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            pad_token_id=1,
            bos_token_id=None,
            eos_token_id=0,
            vocab_size=256,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            use_input_mask=True,
            use_token_type_ids=False,
            use_labels=True,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.layer_types = ["attention"] * config.num_hidden_layers
        config.hyena_filters = 8
        config.hyena_kernel_size = 3
        config.hyena_order = 2
        config.tie_word_embeddings = True
        return config


@require_torch
class Evo2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Evo2ModelTester


if __name__ == "__main__":
    unittest.main()
