import unittest

from transformers import (
    HumanVConfig,
    HumanVForCausalLM,
    HumanVForQuestionAnswering,
    HumanVForSequenceClassification,
    HumanVForTokenClassification,
    HumanVModel,
)
from transformers.testing_utils import require_torch
from transformers.tests.test_modeling_common import ModelTesterMixin, ids_tensor


all_model_classes = (
    HumanVModel,
    HumanVForCausalLM,
    HumanVForSequenceClassification,
    HumanVForTokenClassification,
    HumanVForQuestionAnswering,
)

all_generative_model_classes = (HumanVForCausalLM,)


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
        self.num_labels = 3

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
            num_labels=self.num_labels,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        return config, input_ids


@require_torch
class HumanVModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = all_model_classes
    all_generative_model_classes = all_generative_model_classes

    def setUp(self):
        self.model_tester = HumanVModelTester(self)

    def test_base_model_output_shape(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVModel(config).eval()
        outputs = model(input_ids=input_ids)
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    def test_causal_lm_output_shape(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForCausalLM(config).eval()
        outputs = model(input_ids=input_ids)
        self.assertEqual(
            outputs.logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length, config.vocab_size)
        )

    def test_sequence_classification_output_shape(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForSequenceClassification(config).eval()
        outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.logits.shape, (self.model_tester.batch_size, config.num_labels))

    def test_token_classification_output_shape(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForTokenClassification(config).eval()
        outputs = model(input_ids=input_ids)
        self.assertEqual(
            outputs.logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length, config.num_labels)
        )

    def test_question_answering_output_shape(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForQuestionAnswering(config).eval()
        outputs = model(input_ids=input_ids)
        self.assertEqual(outputs.start_logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length))
        self.assertEqual(outputs.end_logits.shape, (self.model_tester.batch_size, self.model_tester.seq_length))

    def test_generate(self):
        config, input_ids = self.model_tester.prepare_config_and_inputs()
        model = HumanVForCausalLM(config).eval()
        out = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False)
        self.assertEqual(out.shape, (input_ids.shape[0], input_ids.shape[1] + 3))
