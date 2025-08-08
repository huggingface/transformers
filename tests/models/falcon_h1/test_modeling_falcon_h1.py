# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch FalconH1 model."""

import inspect
import unittest

import pytest

from transformers import FalconH1Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    get_device_properties,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, FalconH1ForCausalLM, FalconH1Model
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconHybridMambaAttentionDynamicCache,
    )


class FalconH1ModelTester:
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
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        hidden_act="silu",
        attention_dropout=0.0,
        attn_layer_indices=None,
        attn_rotary_emb=8,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        num_labels=3,
        pad_token_id=0,
        mamba_n_groups=1,
        mamba_n_heads=16,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
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
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.attn_layer_indices = attn_layer_indices
        self.attn_rotary_emb = attn_rotary_emb
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.mamba_n_groups = mamba_n_groups
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def get_config(self):
        # Fix for SDPA tests, force at least 4 layers
        if self.num_hidden_layers < 4:
            self.num_hidden_layers = 4
        if self.attn_layer_indices is None:
            d = [x for x in range(2, self.num_hidden_layers) if self.num_hidden_layers % x == 0]
            if len(d) == 0:
                raise ValueError("num_hidden_layers is prime, cannot automatically set attn_layer_indices.")
            d = d[-1]  # get the largest divisor
            self.attn_layer_indices = [x + 1 for x in range(0, self.num_hidden_layers, d)]

        return FalconH1Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_dropout,
            attn_layer_indices=self.attn_layer_indices,
            attn_rotary_emb=self.attn_rotary_emb,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            mamba_n_groups=self.mamba_n_groups,
            mamba_n_heads=self.mamba_n_heads,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_chunk_size=self.mamba_chunk_size,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        model = FalconH1Model(config=config)
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
        token_labels,
    ):
        model = FalconH1ForCausalLM(config=config)
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
        token_labels,
    ):
        # config.is_decoder = True
        # config.add_cross_attention = True
        model = FalconH1ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        # Attention: Jamba needs the cache to be initialized to return a cache!
        past_key_values = FalconHybridMambaAttentionDynamicCache(
            config,
            input_ids.shape[0],
            model.dtype,
            devices=[model.device for _ in range(model.config.num_hidden_layers)],
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


@require_torch
class FalconH1ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (FalconH1Model, FalconH1ForCausalLM) if is_torch_available() else ()
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    pipeline_model_mapping = (
        {"feature-extraction": FalconH1Model, "text-generation": FalconH1ForCausalLM} if is_torch_available() else {}
    )

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, FalconHybridMambaAttentionDynamicCache)

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            cache_length,
            config.hidden_size // config.num_attention_heads,
        )

        self.assertListEqual(
            [key_tensor.shape for key_tensor in decoder_past_key_values.key_cache],
            [expected_shape] * len(decoder_past_key_values.key_cache),
        )
        self.assertListEqual(
            [value_cache.shape for value_cache in decoder_past_key_values.value_cache],
            [expected_shape] * len(decoder_past_key_values.value_cache),
        )

    def setUp(self):
        self.model_tester = FalconH1ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FalconH1Config, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_casual_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    # def test_initialization(self):
    #     r"""
    #     Overriding the test_initialization test as the A_log and D params of the FalconH1 mixer are initialized differently
    #     """
    #     config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

    #     configs_no_init = _config_zero_init(config)
    #     for model_class in self.all_model_classes:
    #         model = model_class(config=configs_no_init)
    #         for name, param in model.named_parameters():
    #             if param.requires_grad:
    #                 if "A_log" in name:
    #                     A = torch.arange(1, config.mamba_n_heads + 1, dtype=torch.float32)
    #                     torch.testing.assert_close(param.data, torch.log(A), rtol=1e-5, atol=1e-5)
    #                 elif "D" in name:
    #                     D = torch.ones(config.mamba_n_heads, dtype=torch.float32)
    #                     torch.testing.assert_close(param.data, D, rtol=1e-5, atol=1e-5)
    #                 else:
    #                     self.assertIn(
    #                         ((param.data.mean() * 1e9).round() / 1e9).item(),
    #                         [0.0, 1.0],
    #                         msg=f"Parameter {name} of model {model_class} seems not properly initialized",
    #                     )

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        FalconH1 mixer are initialized differently and we tested that in test_initialization
        """
        self.skipTest(reason="Cumbersome and redundant for FalconH1")

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the FalconH1 model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = self.model_tester.num_hidden_layers

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

    def test_batching_equivalence(self):
        # need to disable the tril input mask
        orig = self.model_tester.use_input_mask
        self.model_tester.use_input_mask = False
        super().test_batching_equivalence()
        self.model_tester.use_input_mask = orig

    # essentially the same test in test_utils, just adjustment for rtol for this model
    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

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
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]

            # - for left padding we absolutely need to use an all ones
            #   attention mask, so we do not use the one in inputs_dict
            attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)


@slow
@require_torch
@require_torch_gpu
class FalconH1ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_falcon_h1_hard(self):
        """
        An integration test for Falcon-H1.
        """
        EXPECTED_TEXT_DEFAULT = """
            user
            Tell me about the french revolution.
            assistant
            The French Revolution (1789–1799) was a period of radical social and political upheaval in France that fundamentally transformed the nation and had profound effects on the rest of Europe and the world. Here are the key aspects of the revolution:

            ### **Causes**
            1. **Economic Crisis**: France was in severe financial trouble due to costly wars (particularly the American Revolution), extravagant spending by the monarchy, and inefficient taxation.
            2. **Social Inequality**: The rigid class system (the Ancien Régime) divided society into the privileged nobility and clergy (First Estate) and the commoners (Third Estate), who bore the brunt of taxation and had few rights.
            3. **Enlightenment Ideas**: Philosophers like Voltaire, Rousseau, and Montesquieu inspired ideas of liberty, equality, and popular sovereignty.
            4. **Settlement of 1789**: The Estates-General convened to address the financial crisis, leading to the Third Estate's assertion of its rights and the eventual abolition of the feudal system.

            ### **Key Events**
            1. **Storming of the Bastille (July 14, 1789)**: A symbol of royal tyranny, the Bastille fortress was stormed by revolutionaries, sparking widespread rebellion.
            2. **Declaration of the Rights of Man and of the Citizen (August 1789)**: A foundational document proclaiming liberty, equality, and fraternity.
            3. **National Assembly and King’s Trial (1791–1792)**: King Louis XVI and his ministers were tried and executed (King Louis was guillotined, Marie Antoinette was banished), marking the end of the monarchy.
            4. **Rise of the Jacobins and Reign of Terror (1793–1794)**: Radical leaders like Maximilien Robespierre sought to purge France of counter-revolutionaries, leading to mass executions and widespread fear.
            5. **Thermidorian Reaction
        """

        EXPECTED_TEXT_A10 = """
            user
            Tell me about the french revolution.
            assistant
            The French Revolution (1789–1799) was a period of profound social upheaval and radical political change in France that fundamentally transformed the nation and had far-reaching effects on the rest of Europe and the world. Here are the key aspects of the revolution:

            ### **Causes**
            1. **Economic Crisis**: France was in severe financial trouble due to costly wars (particularly the American Revolution), extravagant spending by the monarchy, and an inefficient tax system.
            2. **Social Inequality**: The privileged classes (the nobility and clergy) enjoyed immense wealth and power, while the majority of the population (the Third Estate, comprising commoners) faced poverty and lack of representation.
            3. **Enlightenment Ideas**: Philosophers like Voltaire, Rousseau, and Montesquieu inspired ideas of liberty, equality, and popular sovereignty, which fueled revolutionary fervor.
            4. **Political Instability**: The absolute monarchy under King Louis XVI proved unable to address the nation's problems, leading to growing discontent.

            ### **Key Events**
            1. **Estates-General (1789)**: The Third Estate broke away and formed the National Assembly, forcing King Louis XVI to convene the Estates-General, an old legislative body, to address the financial crisis.
            2. **Storming of the Bastille (July 14, 1789)**: A symbol of royal tyranny, the Bastille fortress was stormed by revolutionaries, sparking widespread rebellion.
            3. **Declaration of the Rights of Man and of the Citizen (August 1789)**: This foundational document proclaimed liberty, equality, and fraternity as fundamental rights.
            4. **Abolition of Feudalism (November 1789)**: The National Assembly abolished feudal privileges, redistributing church lands to the people.
            5. **Tennis Court Oath (May 5, 1789)**: The National Assembly members, meeting on a tennis court, pledged to continue their work until a new constitution was established.
            6.
        """

        expected_texts = Expectations(
            {
                (None, None): EXPECTED_TEXT_DEFAULT,
                ("cuda", 8): EXPECTED_TEXT_A10,
            }
        )
        EXPECTED_TEXT = expected_texts.get_expectation()
        # Remove the first char (`\n`) and the consecutive whitespaces caused by the formatting.
        EXPECTED_TEXT = EXPECTED_TEXT.strip().replace(" " * 12, "")

        device_properties = get_device_properties()
        # For A10, there is an ending " "
        if device_properties[0] == "cuda" and device_properties[1] == 8:
            EXPECTED_TEXT = EXPECTED_TEXT + " "

        model_id = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FalconH1ForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        device = "cuda"
        messages = [{"role": "user", "content": "Tell me about the french revolution."}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.assertEqual(generated_text, EXPECTED_TEXT)
