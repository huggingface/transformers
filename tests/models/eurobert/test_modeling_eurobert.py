# Copyright 2025 Nicolas Boizard, Duarte M. Alves, Hippolyte Gisserot-Boukhlef and the EuroBert team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing suite for the PyTorch EuroBERT model."""

import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, EuroBertConfig, is_torch_available, set_seed
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import _config_supports_rope_scaling, _set_config_rope_params
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        EuroBertForMaskedLM,
        EuroBertForSequenceClassification,
        EuroBertForTokenClassification,
        EuroBertModel,
    )


class EuroBertModelTester:
    if is_torch_available():
        base_model_class = EuroBertModel

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length]).to(torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return EuroBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = EuroBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = EuroBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = EuroBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = EuroBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class EuroBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            EuroBertModel,
            EuroBertForMaskedLM,
            EuroBertForSequenceClassification,
            EuroBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": EuroBertModel,
            "fill-mask": EuroBertForMaskedLM,
            "text-classification": EuroBertForSequenceClassification,
            "token-classification": EuroBertForTokenClassification,
            "zero-shot": EuroBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = EuroBertModelTester
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = EuroBertForMaskedLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = EuroBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EuroBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_eurobert_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = EuroBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_eurobert_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = EuroBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_eurobert_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = EuroBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @unittest.skip(reason="EuroBert buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """
        Tests that we can initialize a model with RoPE scaling in the config, that it can run a forward pass, and
        that a few basic model output properties are honored.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        if not _config_supports_rope_scaling(config):
            self.skipTest("This model does not support RoPE scaling")

        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
                "original_max_position_embeddings": 16384,
            },
        )
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": scaling_type,
                "factor": 10.0,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        scaled_model = self.model_tester_class.base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        if not _config_supports_rope_scaling(config):
            self.skipTest("This model does not support RoPE scaling")

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        _set_config_rope_params(
            config, {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor}
        )
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        _set_config_rope_params(
            config,
            {
                "rope_type": "linear",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        _set_config_rope_params(
            config,
            {
                "rope_type": "dynamic",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        _set_config_rope_params(
            config,
            {
                "rope_type": "yarn",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_model_loading_old_rope_configs(self):
        def _reinitialize_config(base_config, new_kwargs):
            # Reinitialize the config with the new kwargs, forcing the config to go through its __init__ validation
            # steps.
            base_config_dict = base_config.to_dict()
            new_config = EuroBertConfig.from_dict(config_dict={**base_config_dict, **new_kwargs})
            return new_config

        # from untouched config -> ✅
        base_config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        original_model = EuroBertForMaskedLM(base_config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the expected rope configuration -> ✅
        config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0}})
        original_model = EuroBertForMaskedLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the old rope configuration ('type' instead of 'rope_type')  -> ✅ we gracefully handle BC
        config = _reinitialize_config(base_config, {"rope_scaling": {"type": "linear", "factor": 10.0}})
        original_model = EuroBertForMaskedLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with both 'type' and 'rope_type'  -> ✅ they can coexist (and both are present in the config)
        config = _reinitialize_config(
            base_config, {"rope_scaling": {"type": "linear", "rope_type": "linear", "factor": 10.0}}
        )
        self.assertTrue(config.rope_scaling["type"] == "linear")
        self.assertTrue(config.rope_scaling["rope_type"] == "linear")
        original_model = EuroBertForMaskedLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with parameters in a bad range ('factor' should be >= 1.0) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": -999.0}})
            original_model = EuroBertForMaskedLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("factor field", logs.output[0])

        # from a config with unknown parameters ('foo' isn't a rope option) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(
                base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0, "foo": "bar"}}
            )
            original_model = EuroBertForMaskedLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("Unrecognized keys", logs.output[0])

        # from a config with specific rope type but missing one of its mandatory parameters -> ❌ throws exception
        with self.assertRaises(KeyError):
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear"}})  # missing "factor"


@require_torch
class EuroBertIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = EuroBertForMaskedLM.from_pretrained("EuroBERT/EuroBERT-210m", attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-210m")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 4, 128256))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor([[[2.2926, 2.4539, 1.8910], [5.9669, 3.8567, 0.0723], [2.4965, 2.7193, 1.9904]]])
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_no_head(self):
        model = EuroBertModel.from_pretrained("EuroBERT/EuroBERT-210m", attn_implementation="sdpa")
        tokenizer = AutoTokenizer.from_pretrained("EuroBERT/EuroBERT-210m")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 4, 768))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[1.2437, 1.8956, 50.9435], [-4.5560, -0.1686, -1.2776], [1.6557, 1.9383, 50.1393]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_token_classification(self):
        model = EuroBertForTokenClassification.from_pretrained(
            "hf-internal-testing/tiny-random-EuroBertForTokenClassification",
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-EuroBertForTokenClassification")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 4, 2))
        self.assertEqual(output.shape, expected_shape)

        expected = torch.tensor([[[-1.0817, -5.3000], [5.6100, -5.2878], [3.4393, -8.8765], [-0.0329, -3.8588]]])
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_sequence_classification(self):
        model = EuroBertForSequenceClassification.from_pretrained(
            "hf-internal-testing/tiny-random-EuroBertForSequenceClassification",
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-EuroBertForSequenceClassification")

        inputs = tokenizer("Hello World!", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected = torch.tensor([[-1.8948, 6.2092]])
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
