import unittest

from transformers import HumanVConfig, HumanVForCausalLM
from transformers.testing_utils import require_torch
from transformers.tests.test_modeling_common import ModelTesterMixin, ids_tensor


all_model_classes = (HumanVForCausalLM,)


class HumanVModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.seq_length = 11
        self.vocab_size = 97
        self.hidden_size = 32
        self.intermediate_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.head_dim = 8
        self.max_position_embeddings = 64

    def get_config(self):
        return HumanVConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            layer_types=["full_attention"] * self.num_hidden_layers,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        return config, input_ids

    def create_and_check_model(self, config, input_ids):
        model = HumanVForCausalLM(config).eval()
        outputs = model(input_ids=input_ids)
        self.parent.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))


@require_torch
class HumanVModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = all_model_classes

    def setUp(self):
        self.model_tester = HumanVModelTester(self)

    def test_model(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids)

    def test_generate(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForCausalLM(config).eval()
        out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False)
        self.assertEqual(out.shape, (input_ids.shape[0], input_ids.shape[1] + 3))
