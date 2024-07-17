# coding=utf-8
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
"""Testing suite for the PyTorch Mamba2 model."""

import math
import unittest
from typing import Dict, List, Tuple

from parameterized import parameterized

from transformers import Mamba2Config, is_torch_available, set_seed
from transformers.testing_utils import (
    require_einops,
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Mamba2ForCausalLM,
        Mamba2ForSequenceClassification,
        Mamba2Model,
    )
    from transformers.models.mamba2.modeling_mamba2 import (
        HybridMamba2AttentionDynamicCache,
        Mamba2DynamicNTKScalingRotaryEmbedding,
        Mamba2LinearScalingRotaryEmbedding,
        Mamba2RotaryEmbedding,
    )


class Mamba2ModelTester:
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
        A_initializer_range=None,
        mlp_intermediate_size=64,
        num_hidden_layers=5,
        attention_layers_idx=None,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=True,
        classifier_dropout=0.1,
    ):
        if attention_layers_idx is None:
            self.attention_layers_idx = [1]
        if A_initializer_range is None:
            self.A_initializer_range = [2, 2]

        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_intermediate_size = mlp_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.tie_word_embeddings = tie_word_embeddings
        self.classifier_dropout = classifier_dropout

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
        return Mamba2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            A_initializer_range=self.A_initializer_range,
            num_hidden_layers=self.num_hidden_layers,
            attention_layers_idx=self.attention_layers_idx,
            mlp_intermediate_size=self.mlp_intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            tie_word_embeddings=self.tie_word_embeddings,
            gradient_checkpointing=False,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_mamba2_model(self, config, input_ids, input_mask, *args):
        model = Mamba2Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)
        result = model(input_ids, attention_mask=input_mask, output_hidden_states=True)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def create_and_check_mamba2_causal_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Mamba2ForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_mamba2_lm_head_forward_and_backwards(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_label, gradient_checkpointing=False
    ):
        model = Mamba2ForCausalLM(config)
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def create_and_check_mamba2_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Mamba2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, labels=sequence_labels)
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_state_equivalency(self, config, input_ids, input_mask, *args):
        model = Mamba2Model(config=config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids, attention_mask=input_mask)
        output_whole = outputs.last_hidden_state

        outputs = model(input_ids[:, :-1], attention_mask=input_mask[:, :-1], use_cache=True)
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(input_ids[:, -1:], attention_mask=input_mask, past_key_values=outputs.past_key_values)
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5))

    def create_and_check_einops_torch_equivalence(self):
        from einops import rearrange, repeat

        # cover all operations in the original mamba2 repo that used einops operations as we use pure torch only
        d = torch.randn(size=(4,))
        v = torch.randn(size=(2, 4))
        w = torch.randn(size=(2, 3, 4, 5, 6))
        x = torch.randn(size=(2, 6, 4))
        y = torch.randn(size=(2, 6, 4, 5))
        z = torch.randn(size=(2, 1, 4))

        self.parent.assertTrue(d.unsqueeze(-1).expand(d.shape[0], 3).equal(repeat(d, "h -> h p", p=3)))
        self.parent.assertTrue(
            d.unsqueeze(-1).unsqueeze(-1).expand(d.shape[0], 2, 3).equal(repeat(d, "h -> h p n", p=2, n=3))
        )

        self.parent.assertTrue(v.reshape(v.shape[0], -1, 2).equal(rearrange(v, "b (h p) -> b h p", p=2)))
        self.parent.assertTrue(v.unsqueeze(-1).unsqueeze(-1).equal(rearrange(v, "b h -> b h 1 1")))

        self.parent.assertTrue(
            w.reshape(w.shape[0], -1, w.shape[-2], w.shape[-1]).equal(rearrange(w, "b c l h p -> b (c l) h p"))
        )

        self.parent.assertTrue(x.transpose(1, 2).equal(rearrange(x, "b l d -> b d l")))
        self.parent.assertTrue(x.unsqueeze(-1).expand(*x.size(), 5).equal(repeat(x, "... d -> ... d e", e=5)))
        self.parent.assertTrue(
            x.reshape(x.shape[0], -1, 3, x.shape[2]).equal(rearrange(x, "b (c l) ... -> b c l ...", l=3))
        )
        self.parent.assertTrue(x.unsqueeze(-2).equal(rearrange(x, pattern="b l n -> b l 1 n")))
        self.parent.assertTrue(
            x.reshape(x.shape[0], x.shape[1], -1, 2).equal(rearrange(x, pattern="b l (h p) -> b l h p", p=2))
        )
        self.parent.assertTrue(x.reshape(x.shape[0], -1).unsqueeze(1).equal((rearrange(x, "b h p -> b 1 (h p)"))))

        self.parent.assertTrue(y.permute(0, 3, 1, 2).equal(rearrange(y, "b c l h -> b h c l")))
        self.parent.assertTrue(y.unsqueeze(-1).expand(*y.size(), 5).equal(repeat(y, "... d -> ... d e", e=5)))
        self.parent.assertTrue(
            y.reshape(y.shape[0], -1, 3, y.shape[2], y.shape[3]).equal(rearrange(y, "b (c l) ... -> b c l ...", l=3))
        )
        self.parent.assertTrue(y.reshape(y.shape[0], y.shape[1], -1).equal(rearrange(y, "b l h p -> b l (h p)")))

        self.parent.assertTrue(z.squeeze(1).equal(rearrange(z, "d 1 w -> d w")))
        self.parent.assertTrue(
            z.transpose(1, 2).expand(z.shape[0], z.shape[-1], 2).equal(repeat(z, "b 1 h -> b h p", p=2))
        )

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
class Mamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (Mamba2Model, Mamba2ForCausalLM, Mamba2ForSequenceClassification) if is_torch_available() else ()
    )
    all_generative_model_classes = (Mamba2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Mamba2Model,
            "text-generation": Mamba2ForCausalLM,
            "text-classification": Mamba2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    # TODO: check if True is ok
    test_torchscript = False
    # TODO: check if True is ok
    fx_compatible = False

    def setUp(self):
        self.model_tester = Mamba2ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Mamba2Config,
            hidden_size=37,
            common_properties=["hidden_size", "mamba2_head_dim", "attention_head_dim", "num_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    # TODO: add test_model_from_pretrained test
    # TODO: add test_multi_gpu_parallel_forward test

    def test_mamba2_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_model(*config_and_inputs)

    def test_mamba2_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_causal_lm(*config_and_inputs)

    def test_mamba2_lm_head_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_lm_head_forward_and_backwards(*config_and_inputs)

    def test_mamba2_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_sequence_classification(*config_and_inputs)

    def test_state_equivalency(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_state_equivalency(*config_and_inputs)

    @require_einops
    # TODO: is this necessary, was initially used to test it internally
    def test_einops_torch_equivalence(self):
        self.model_tester.create_and_check_einops_torch_equivalence()

    def test_initialization(self):
        r"""
        Overriding the test_initialization test as the dt_bias and A_log params of the Mamba2 block are done differently
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        # We force them to be initialized on the same range, would return a float either way which is problematic
        configs_no_init.A_initializer_range = config.A_initializer_range

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "dt_bias" in name:
                    dt = torch.exp(
                        torch.tensor([0, 1]) * (math.log(config.time_step_max) - math.log(config.time_step_min))
                        + math.log(config.time_step_min)
                    ).clamp(min=config.time_step_floor)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))

                    if param.requires_grad:
                        self.assertTrue(param.data.max().item() <= inv_dt[1])
                        self.assertTrue(param.data.min().item() >= inv_dt[0])
                elif "A_log" in name:
                    A = torch.empty(config.mamba2_num_heads, dtype=torch.float32).uniform_(*config.A_initializer_range)
                    self.assertTrue(param.data.equal(torch.log(A)))
                elif "D" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log, D, and dt_bias params of the
        Mamba2 block are initialized differently and we tested that in test_initialization
        """
        self.skipTest("Cumbersome and redundant for Mamba2")

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Mamba2 model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = len(config.attention_layers_idx)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
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

    def test_left_padding_compatibility(self):
        r"""
        Overriding the test_left_padding_compatibility test as the mamba2 layers accentuate the numerical differences
        effect of the left padding discussed in the issue in the note. Using a more permissive tolerance value.
        """
        import inspect
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding - generative and decoder-only.
        # Mamba2 is a decoder-only architecture
        decoder_only_classes = self.all_generative_model_classes

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, input_ids, attention_mask = self._get_input_ids_and_config()
            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * config.pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            # TODO: this is quite large, what is causing this? My hw?
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=3e-1))

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, HybridMamba2AttentionDynamicCache):  # MODIFIED PART START
                        recursive_check(tuple_object.conv_states, dict_object.conv_states)
                        recursive_check(tuple_object.ssm_states, dict_object.ssm_states)
                        recursive_check(tuple_object.key_cache, dict_object.key_cache)
                        recursive_check(tuple_object.value_cache, dict_object.value_cache)
                    elif isinstance(tuple_object, (List, Tuple)):  # MODIFIED PART END
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(tuple_object, dict_object, atol=1e-5),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, *_ = self.model_tester.prepare_config_and_inputs()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = Mamba2Model(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = Mamba2Model(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = Mamba2RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        linear_scaling_rope = Mamba2LinearScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
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
        ntk_scaling_rope = Mamba2DynamicNTKScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

    @unittest.skip(reason="Mamba2 has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @unittest.skip(
        reason="Mamba2 does not follow the standard format for head_dim and emb_dim. "
        "Additionally, outputting attentions is different as we only handle the specific layers doing so."
    )
    def test_past_key_values_format(self):
        pass

    # TODO: check test_flash_attn_2_fp32_ln and test_flash_attn_2_generate_padding_right
    # TODO: check test_flash_attn_2_generate_padding_right
    # TODO: check test_flash_attn_2_generate_use_cache
    # TODO: check test_flash_attn_2_inference_equivalence_right_padding


# TODO: in total, add integration tests for Mamba2
"""@require_torch
class JambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    def setUpClass(cls):
        model_id = "ai21labs/Jamba-tiny-random"
        cls.model = JambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @slow
    def test_simple_generate(self):
        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        self.assertEqual(
            output_sentence,
            "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats",
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = torch.tensor(
            [
                0.0140, -0.2246,  0.0408, -0.1016,  0.0471,  0.2715, -0.1465,  0.1631,
                -0.2949, -0.0297,  0.0250, -0.5586, -0.2139, -0.1426, -0.1602,  0.1309,
                0.0703,  0.2236,  0.1729, -0.2285, -0.1152, -0.1177, -0.1367,  0.0289,
                0.1245,  0.2363,  0.0442,  0.1094, -0.1348, -0.2295,  0.1494, -0.3945,
                0.1777, -0.4570, -0.0408,  0.2412,  0.1562, -0.1943,  0.2373, -0.0593
            ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1e-3)

    @slow
    def test_simple_batched_generate_with_padding(self):
        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "Tell me a story"], padding=True, return_tensors="pt"
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(
            output_sentences[0],
            "<|startoftext|>Hey how are you doing on this lovely evening? Canyon rins hugaughter glamour Rutgers Singh Hebrew cases Cats",
        )
        self.assertEqual(
            output_sentences[1],
            "<|pad|><|pad|><|pad|><|pad|><|pad|><|pad|><|startoftext|>Tell me a storyptus Nets Madison El chamadamodern updximVaparsed",
        )

        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"]).logits

        EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
            [
                0.0140, -0.2246,  0.0408, -0.1016,  0.0471,  0.2715, -0.1465,  0.1631,
                -0.2949, -0.0297,  0.0250, -0.5586, -0.2139, -0.1426, -0.1602,  0.1309,
                0.0703,  0.2236,  0.1729, -0.2285, -0.1152, -0.1177, -0.1367,  0.0289,
                0.1245,  0.2363,  0.0442,  0.1094, -0.1348, -0.2295,  0.1494, -0.3945,
                0.1777, -0.4570, -0.0408,  0.2412,  0.1562, -0.1943,  0.2373, -0.0593
            ]
            , dtype=torch.float32)  # fmt: skip

        EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
            [
                -0.1289,  0.2363, -0.4180, -0.0302, -0.0476,  0.0327,  0.2578,  0.0874,
                0.1484,  0.2305, -0.1152, -0.1396, -0.1494, -0.1113, -0.0021, -0.2832,
                0.2002, -0.2676,  0.0598, -0.1982, -0.2539, -0.1133, -0.1973,  0.2148,
                0.0559,  0.1670,  0.1846,  0.1270,  0.1680, -0.1250, -0.2656, -0.2871,
                0.2344,  0.2637,  0.0510, -0.1855,  0.2158, -0.1289,  0.1758,  0.0074
            ]
            , dtype=torch.float32)  # fmt: skip

        torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[1, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1e-3)
"""
