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
"""Testing suite for the PyTorch Bamba model."""

import inspect
import tempfile
import unittest

import pytest
from pytest import mark

from transformers import (
    AutoTokenizer,
    BambaConfig,
    DataCollatorWithFlattening,
    is_torch_available,
)
from transformers.testing_utils import (
    DeviceProperties,
    Expectations,
    get_device_properties,
    require_deterministic_for_xpu,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        BambaForCausalLM,
        BambaModel,
    )
    from transformers.models.bamba.modeling_bamba import (
        HybridMambaAttentionDynamicCache,
    )


class BambaModelTester:
    config_class = BambaConfig
    if is_torch_available():
        model_class = BambaModel
        for_causal_lm_class = BambaForCausalLM

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

        self._update_layer_configs()
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

    def _update_layer_configs(self):
        """Configures hidden layers and attn layer indices if they are not set."""
        # Fix for SDPA tests, force at least 4 layers
        if self.num_hidden_layers < 4:
            self.num_hidden_layers = 4

        if self.attn_layer_indices is None:
            d = [x for x in range(2, self.num_hidden_layers) if self.num_hidden_layers % x == 0]
            if len(d) == 0:
                raise ValueError("num_hidden_layers is prime, cannot automatically set attn_layer_indices.")
            d = d[-1]  # get the largest divisor
            self.attn_layer_indices = [x + 1 for x in range(0, self.num_hidden_layers, d)]

    def get_config(self, **kwargs):
        return self.config_class(
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
            **kwargs,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        model = self.model_class(config=config)
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
        model = self.for_causal_lm_class(config=config)
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
        model = self.for_causal_lm_class(config=config)
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


@require_torch
class BambaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    model_tester_class = BambaModelTester
    all_model_classes = (BambaModel, BambaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": BambaModel,
            "text-generation": BambaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, HybridMambaAttentionDynamicCache)

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
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, hidden_size=64)

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

    def test_initialization(self):
        r"""
        Overriding the test_initialization test as the A_log and D params of the Bamba mixer are initialized differently
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "A_log" in name:
                        A = torch.arange(1, config.mamba_n_heads + 1, dtype=torch.float32)
                        torch.testing.assert_close(param.data, torch.log(A), rtol=1e-5, atol=1e-5)
                    elif "D" in name:
                        D = torch.ones(config.mamba_n_heads, dtype=torch.float32)
                        torch.testing.assert_close(param.data, D, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        Bamba mixer are initialized differently and we tested that in test_initialization
        """
        self.skipTest(reason="Cumbersome and redundant for Bamba")

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Bamba model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = self.model_tester.num_hidden_layers - len(self.model_tester.attn_layer_indices)

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

    @unittest.skip(
        "Bamba requires additionally specifying position_ids, seq_idx, and FlashAttentionKwargs for padding-free training."
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Bamba requires additionally specifying position_ids, seq_idx, and FlashAttentionKwargs for padding-free training."
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @mark.flash_attn_test
    @slow
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_seq_idx_and_fa_kwargs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if 0 not in inputs_dict.get("attention_mask", []) or "attention_mask" not in inputs_dict:
                self.skipTest("Model dummy inputs should contain padding in their attention mask")

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                self.skipTest("Model does not support position_ids")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # ensure left padding, to adapt for some models
                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id
                # Ensure inputs_dict also has labels in it, as their presence/absence can induce
                # dtype conversions. This also lets us compare losses.
                labels = inputs_dict["input_ids"].clone()
                # Mask padding tokens
                labels[~dummy_attention_mask.bool()] = -100
                # Also need to mask the first non-trivial token to match the padding-free batch.
                first_nonneg_idx = (labels >= 0).int().argmax(dim=1)
                labels[torch.arange(labels.size(0), device=labels.device), first_nonneg_idx] = -100
                inputs_dict["labels"] = labels

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                features = [
                    {"input_ids": i[a.bool()].tolist()}
                    for i, a in zip(inputs_dict["input_ids"], inputs_dict["attention_mask"])
                ]

                # add position_ids + fa_kwargs + seq_idx
                data_collator = DataCollatorWithFlattening(
                    return_tensors="pt", return_seq_idx=True, return_flash_attn_kwargs=True
                )
                batch = data_collator(features)
                batch_accelerator = {k: t.to(torch_device) if torch.is_tensor(t) else t for k, t in batch.items()}

                res_padded = model(**inputs_dict)
                res_padfree = model(**batch_accelerator)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                torch.testing.assert_close(logits_padded.argmax(-1), logits_padfree.argmax(-1), rtol=0, atol=0)
                # acceptable numerical instability
                tol = torch.finfo(torch.float16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

                loss_padded = res_padded.loss
                loss_padfree = res_padfree.loss
                torch.testing.assert_close(loss_padded, loss_padfree)


@slow
@require_torch
@require_torch_accelerator
class BambaModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    device_properties: DeviceProperties = (None, None, None)

    @classmethod
    def setUpClass(cls):
        model_id = "ibm-fms/Bamba-9B"
        cls.model = BambaForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # feels a bit forced to have to do this for the generation test
        cls.tokenizer.pad_token_id = cls.model.config.pad_token_id
        cls.tokenizer.padding_side = "left"

        cls.device_properties = get_device_properties()

    def test_simple_generate(self):
        # fmt: off
        expectations = Expectations(
            {
                ("cuda", 8): "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are all having a good time.",
                ("rocm", 9): "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are doing well. I am here",
                ("xpu", 3): "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are all doing well. I am",
            }
        )
        # fmt: on

        self.model.to(torch_device)

        input_ids = self.tokenizer("Hey how are you doing on this lovely evening?", return_tensors="pt")[
            "input_ids"
        ].to(torch_device)
        out = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(out[0, :])
        expected = expectations.get_expectation()
        self.assertEqual(output_sentence, expected)

        # TODO: there are significant differences in the logits across major cuda versions, which shouldn't exist
        if self.device_properties[0] == "cuda" and self.device_properties[1] == 8:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, logits_to_keep=40).logits

            EXPECTED_LOGITS_NO_GRAD = torch.tensor(
                [
                    149., 142., 146., 142., 143., 144., 142., 145.,
                    142., 146., 144., 146., 147., 147., 148., 145.,
                    147., 145., 145., 145., 145., 144., 144., 144.,
                    144., 145., 147., 146., 144., 144., 148., 147.,
                    148., 147., 147., 147., 146., 146., 148., 148.
                ], dtype=torch.bfloat16)  # fmt: skip

            torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD, rtol=1e-3, atol=1)

    @require_deterministic_for_xpu
    def test_simple_batched_generate_with_padding(self):
        # Key 9 for MI300, Key 8 for A100/A10, and Key 7 for T4.
        #
        # Note: Key 9 is currently set for MI300, but may need potential future adjustments for H100s,
        # considering differences in hardware processing and potential deviations in generated text.
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 7): [],
                ("cuda", 8): [
                    "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are doing well. I am here",
                    "!!!<|begin_of_text|>I am late! I need to get to work! I have to get to the",
                ],
                ("rocm", 9): [
                    "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are doing well. I am here",
                    "!!!<|begin_of_text|>I am late! I need to be at the airport in 20 minutes! I",
                ],
                ("xpu", 3): [
                    "<|begin_of_text|>Hey how are you doing on this lovely evening? I hope you are all doing well. I am",
                    "!!!<|begin_of_text|>I am late! I need to get to work! I have to get to the",
                ],
            }
        )
        # fmt: on
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        self.model.to(torch_device)

        inputs = self.tokenizer(
            ["Hey how are you doing on this lovely evening?", "I am late! I need to"],
            padding=True,
            return_tensors="pt",
        ).to(torch_device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=10)
        output_sentences = self.tokenizer.batch_decode(out)
        self.assertEqual(output_sentences[0], EXPECTED_TEXT[0])
        self.assertEqual(output_sentences[1], EXPECTED_TEXT[1])

        # TODO: there are significant differences in the logits across major cuda versions, which shouldn't exist
        if self.device_properties[0] == "cuda" and self.device_properties[1] == 8:
            with torch.no_grad():
                logits = self.model(input_ids=inputs["input_ids"]).logits

            EXPECTED_LOGITS_NO_GRAD_0 = torch.tensor(
                [
                    149., 142., 146., 142., 143., 144., 142., 145.,
                    142., 146., 144., 146., 147., 147., 148., 145.,
                    147., 145., 145., 145., 145., 144., 144., 144.,
                    144., 145., 147., 146., 144., 144., 148., 147.,
                    148., 147., 147., 147., 146., 146., 148., 148.
                ], dtype=torch.bfloat16)  # fmt: skip

            EXPECTED_LOGITS_NO_GRAD_1 = torch.tensor(
                [
                    182., 178., 177., 174., 176., 176., 178., 178.,
                    177., 179., 176., 183., 180., 182., 179., 174.,
                    178., 176., 176., 175., 175., 175., 174., 173.,
                    174., 182., 180., 176., 177., 177., 180., 176.,
                    178., 177., 177., 175., 176., 177., 175., 177.
                ], dtype=torch.bfloat16)  # fmt: skip

            torch.testing.assert_close(logits[0, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_0, rtol=1e-3, atol=1)
            torch.testing.assert_close(logits[1, -1, :40].cpu(), EXPECTED_LOGITS_NO_GRAD_1, rtol=1e-3, atol=1)
