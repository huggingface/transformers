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

from parameterized import parameterized

from transformers import Mamba2Config, is_torch_available
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
        Mamba2Model,
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
        A_initializer_range=(2, 2),
        mlp_intermediate_size=64,
        num_hidden_layers=5,
        attention_layers_idx=None,
        attention_num_heads=4,
        attention_num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=True,
    ):
        if attention_layers_idx is None:
            self.attention_layers_idx = [1]

        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.A_initializer_range = A_initializer_range
        self.mlp_intermediate_size = mlp_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_num_heads = attention_num_heads
        self.attention_num_key_value_heads = attention_num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.tie_word_embeddings = tie_word_embeddings

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
            attention_num_heads=self.attention_num_heads,
            attention_num_key_value_heads=self.attention_num_key_value_heads,
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

    def create_and_check_mamba2_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Mamba2Model(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)
        result = model(input_ids, attention_mask=input_mask)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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

    def create_and_check_state_equivalency(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
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

        self.parent.assertTrue(v.view(v.shape[0], -1, 2).equal(rearrange(v, "b (h p) -> b h p", p=2)))
        self.parent.assertTrue(v.unsqueeze(-1).unsqueeze(-1).equal(rearrange(v, "b h -> b h 1 1")))

        self.parent.assertTrue(
            w.view(w.shape[0], -1, w.shape[-2], w.shape[-1]).equal(rearrange(w, "b c l h p -> b (c l) h p"))
        )

        self.parent.assertTrue(x.transpose(1, 2).equal(rearrange(x, "b l d -> b d l")))
        self.parent.assertTrue(x.unsqueeze(-1).expand(*x.size(), 5).equal(repeat(x, "... d -> ... d e", e=5)))
        self.parent.assertTrue(
            x.view(x.shape[0], -1, 3, x.shape[2]).equal(rearrange(x, "b (c l) ... -> b c l ...", l=3))
        )
        self.parent.assertTrue(x.unsqueeze(-2).equal(rearrange(x, pattern="b l n -> b l 1 n")))
        self.parent.assertTrue(
            x.view(x.shape[0], x.shape[1], -1, 2).equal(rearrange(x, pattern="b l (h p) -> b l h p", p=2))
        )
        self.parent.assertTrue(x.view(x.shape[0], -1).unsqueeze(1).equal((rearrange(x, "b h p -> b 1 (h p)"))))

        self.parent.assertTrue(y.permute(0, 3, 1, 2).equal(rearrange(y, "b c l h -> b h c l")))
        self.parent.assertTrue(y.unsqueeze(-1).expand(*y.size(), 5).equal(repeat(y, "... d -> ... d e", e=5)))
        self.parent.assertTrue(
            y.view(y.shape[0], -1, 3, y.shape[2], y.shape[3]).equal(rearrange(y, "b (c l) ... -> b c l ...", l=3))
        )
        self.parent.assertTrue(y.view(y.shape[0], y.shape[1], -1).equal(rearrange(y, "b l h p -> b l (h p)")))

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
    all_model_classes = (Mamba2Model, Mamba2ForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Mamba2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Mamba2Model, "text-generation": Mamba2ForCausalLM} if is_torch_available() else {}
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
            common_properties=["hidden_size", "mamba2_head_dim", "attention_head_dim", "attention_num_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mamba2_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_model(*config_and_inputs)

    def test_mamba2_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_causal_lm(*config_and_inputs)

    def test_mamba2_lm_head_forward_and_backwards(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_lm_head_forward_and_backwards(*config_and_inputs)

    def test_state_equivalency(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_state_equivalency(*config_and_inputs)

    @require_einops
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
                [self.model_tester.attention_num_heads, encoder_seq_length, encoder_key_length],
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
                [self.model_tester.attention_num_heads, encoder_seq_length, encoder_key_length],
            )

    def test_left_padding_compatibility(self):
        r"""
        TODO: is this also the case over here?
        Overriding the test_left_padding_compatibility test as the mamba layers accentuate the numerical differences
        effect of the left padding discussed in the issue in the note. Using a more permissive tolerance value.
        """
        import inspect
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding - generative and decoder-only.
        # Jamba is a decoder-only architecture
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
            self.assertTrue(torch.allclose(next_logits_wo_padding, next_logits_with_padding, atol=3e-3))

    @unittest.skip(reason="Mamba2 has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass


# TODO: in total
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
