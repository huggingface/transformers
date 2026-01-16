# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Jamba model."""

import math
import tempfile
import unittest

import pytest

from transformers import AutoTokenizer, BitsAndBytesConfig, JambaConfig, is_torch_available
from transformers.testing_utils import (
    DeviceProperties,
    Expectations,
    get_device_properties,
    is_flaky,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        JambaForCausalLM,
        JambaForSequenceClassification,
        JambaModel,
    )
    from transformers.models.jamba.modeling_jamba import (
        HybridMambaAttentionDynamicCache,
    )


class JambaConfigTester(ConfigTester):
    def _create_attn_config(self, attn_layer_offset: int, attn_layer_period: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["attn_layer_offset"] = attn_layer_offset
        _input_dict["attn_layer_period"] = attn_layer_period
        return self.config_class(**_input_dict)

    def _create_expert_config(self, expert_layer_offset: int, expert_layer_period: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["expert_layer_offset"] = expert_layer_offset
        _input_dict["expert_layer_period"] = expert_layer_period
        return self.config_class(**_input_dict)

    def test_attn_offsets(self):
        self._create_attn_config(attn_layer_offset=0, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=1, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=2, attn_layer_period=4)
        self._create_attn_config(attn_layer_offset=3, attn_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_attn_config(attn_layer_offset=4, attn_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_attn_config(attn_layer_offset=5, attn_layer_period=4)

    def test_expert_offsets(self):
        self._create_expert_config(expert_layer_offset=0, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=1, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=2, expert_layer_period=4)
        self._create_expert_config(expert_layer_offset=3, expert_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_expert_config(expert_layer_offset=4, expert_layer_period=4)
        with self.parent.assertRaises(ValueError):
            self._create_expert_config(expert_layer_offset=5, expert_layer_period=4)

    def test_jamba_offset_properties(self):
        self.test_attn_offsets()
        self.test_expert_offsets()

    def run_common_tests(self):
        self.test_jamba_offset_properties()
        return super().run_common_tests()


class JambaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        attn_layer_offset=1,
        attn_layer_period=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=40,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_layer_offset = attn_layer_offset
        self.attn_layer_period = attn_layer_period
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return JambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            attn_layer_offset=self.attn_layer_offset,
            attn_layer_period=self.attn_layer_period,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            use_mamba_kernels=False,
            num_experts=2,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = JambaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = JambaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = JambaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        # Attention: Jamba needs the cache to be initialized to return a cache!
        past_key_values = HybridMambaAttentionDynamicCache(
            config, input_ids.shape[0], model.dtype, device=model.device
        )
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            cache_position=torch.arange(
                input_ids.shape[1], input_ids.shape[1] + next_tokens.shape[1], device=model.device
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = JambaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class JambaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            JambaModel,
            JambaForCausalLM,
            JambaForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "text-embedding": JambaModel,
            "text-classification": JambaForSequenceClassification,
            "text-generation": JambaForCausalLM,
            "zero-shot": JambaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, HybridMambaAttentionDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)
        conv_shape = (batch_size, config.mamba_expand * config.hidden_size, config.mamba_d_conv)
        ssm_shape = (batch_size, config.mamba_expand * config.hidden_size, config.mamba_d_state)

        self.assertTrue(config.num_hidden_layers, len(past_key_values))

        for idx in range(len(past_key_values)):
            if config.layers_block_type[idx] == "mamba":
                self.assertEqual(past_key_values.conv_states[idx].shape, conv_shape)
                self.assertEqual(past_key_values.ssm_states[idx].shape, ssm_shape)
            else:
                self.assertEqual(past_key_values.key_cache[idx].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[idx].shape, attention_shape)

    def _check_caches_are_equal(
        self, cache1: HybridMambaAttentionDynamicCache, cache2: HybridMambaAttentionDynamicCache
    ):
        if not isinstance(cache1, HybridMambaAttentionDynamicCache) or not isinstance(
            cache2, HybridMambaAttentionDynamicCache
        ):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_states[idx], cache2.conv_states[idx])
            torch.testing.assert_close(cache1.ssm_states[idx], cache2.ssm_states[idx])

    def setUp(self):
        self.model_tester = JambaModelTester(self)
        self.config_tester = JambaConfigTester(self, config_class=JambaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    # After #40617, we still have 0.01 % of failure rate here.
    @is_flaky(max_attempts=2)
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 3
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(config.pad_token_id).to(torch_device)
        model = JambaForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        bs, seqlen = input_ids.shape
        self.assertEqual(result.router_logits[0].shape, (bs * seqlen, config.num_experts))
        # After #40617, we still have 0.01 % of failure rate here.
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        # (This length is selected from experiments)
        pad_length = input_ids.shape[1] * 4
        # Add padding tokens to input_ids
        padding_block = config.pad_token_id * torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(
            torch_device
        )
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(config.pad_token_id).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        # After #40617, we still have 0.003 % of failure rate here.
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Jamba model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = math.ceil(
            (self.model_tester.num_hidden_layers - self.model_tester.attn_layer_offset)
            / self.model_tester.attn_layer_period
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    @require_flash_attn
    @require_torch_accelerator
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        r"""
        Overriding the test_flash_attn_2_fp32_ln test as the Jamba model, like Mixtral, doesn't support
        right padding + use cache with FA2
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: Jamba does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                _ = model(dummy_input)
                # with attention mask
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @unittest.skip("Jamba has a non standard cache which is not compatible with dp/ddp")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
@slow
class JambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None
    # This variable is used to determine which acclerator are we using for our runners (e.g. A10 or T4)
    # Depending on the hardware we get different logits / generations
    device_properties: DeviceProperties = (None, None, None)

    @classmethod
    def setUpClass(cls):
        model_id = "ai21labs/Jamba-tiny-dev"
        cls.model = JambaForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            use_mamba_kernels=False,
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.device_properties = get_device_properties()

    def test_simple_generate(self):
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4.
        #
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh<|reserved_797|>cw algunas",
                ("cuda", 8): "<|startoftext|>Hey how are you doing on this lovely evening? I'm so glad you're here.",
                ("rocm", 9): "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew llam bb",
            }
        )
        # fmt: on
        expected_sentence = EXPECTED_TEXTS.get_expectation()

        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(output_sentence, expected_sentence)

    def test_simple_batched_generate_with_padding(self):
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4.
        #
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): ["<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a storyptus Nets Madison El chamadamodern updximVaparsed",],
                ("cuda", 8): ["<|startoftext|>Hey how are you doing on this lovely evening? I'm so glad you're here.", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a story about a woman who was born in the United States",],
                ("rocm", 9): ["<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh<|reserved_797|>cw algunas", "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a storyptus Nets Madison El chamadamodern updximVaparsed",],
            }
        )
        # fmt: on
        expected_sentences = EXPECTED_TEXTS.get_expectation()

        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "Tell me a story"], padding=True, return_tensors="pt"
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(output_sentences[0], expected_sentences[0])
        self.assertEqual(output_sentences[1], expected_sentences[1])
