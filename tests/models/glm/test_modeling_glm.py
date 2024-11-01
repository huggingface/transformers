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
"""Testing suite for the PyTorch Glm model."""

import inspect
import tempfile
import unittest

import numpy as np
import pytest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, GlmConfig, is_torch_available
from transformers.testing_utils import (
    is_flaky,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GlmForCausalLM,
        GlmForSequenceClassification,
        GlmForTokenClassification,
        GlmModel,
    )


@require_torch
class GlmModelTester:
    config_class = GlmConfig
    if is_torch_available():
        model_class = GlmModel
        for_causal_lm_class = GlmForCausalLM
        for_sequence_class = GlmForSequenceClassification
        for_token_class = GlmForTokenClassification

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
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        attention_dropout=0.1,
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
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

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
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            head_dim=self.head_dim,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = self.model_class(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = self.for_causal_lm_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = self.for_causal_lm_class(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    # Copied from tests.models.llama.test_modeling_llama.LlamaModelTester.prepare_config_and_inputs_for_common with Llama->Glm
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
class GlmModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (GlmModel, GlmForCausalLM, GlmForSequenceClassification, GlmForTokenClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (GlmForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GlmModel,
            "text-classification": GlmForSequenceClassification,
            "token-classification": GlmForTokenClassification,
            "text-generation": GlmForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = GlmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GlmConfig, hidden_size=37)

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

    def test_Glm_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        print(config)
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Glm_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Glm_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Glm_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = self.model_tester.for_token_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @unittest.skip(reason="Glm uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @is_flaky()
    def test_custom_4d_attention_mask(self):
        """Overwrite the common test to use atol=1e-3 instead of 1e-4. Can still rarely fail, thus flaky."""
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(f"{model_class.__name__} is not guaranteed to work with custom 4D attention masks")
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            if getattr(config, "sliding_window", 0) is not None and getattr(config, "sliding_window", 0) > 0:
                self.skipTest(f"{model_class.__name__} with sliding window attention is not supported by this test")
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                position_ids,
                input_ids_shared_prefix,
                mask_shared_prefix,
                position_ids_shared_prefix,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(input_ids, position_ids=position_ids).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids_shared_prefix,
                attention_mask=mask_shared_prefix,
                position_ids=position_ids_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = torch.nn.functional.softmax(out_last_tokens)
            normalized_1 = torch.nn.functional.softmax(out_shared_prefix_last_tokens)
            print(torch.abs(normalized_0 - normalized_1).max())

            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-3)

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    @slow
    @is_flaky
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        """Overwrite to add flakyness: some cases can sometimes fail"""
        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            # FIXME: we deactivate boolean mask for models using "use_mask_token" in their constructors.
            # These models support masking only in the case `use_mask_token=True`. Otherwise they cannot consume an input mask.
            # This means that the class needs to be instantiated much later, after `use_mask` is set, which means a significant refactor of the code.
            # However masking there is not done at any layers that matters (i.e self-attention), therefore we can safely deactivate it.
            deactivate_mask = "use_mask_token" in inspect.signature(model_class).parameters

            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        has_sdpa = True
                        break
                if not has_sdpa and model_sdpa.config.model_type != "falcon":
                    raise ValueError("The SDPA model should have SDPA attention layers")

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 16 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for output_attentions in [True, False]:
                            can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                            if not (self.has_attentions and can_output_attn) and output_attentions:
                                continue
                            for batch_size in [1, 5]:
                                dummy_input = inputs_dict[model.main_input_name]

                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    dummy_input = dummy_input.to(torch_dtype)

                                dummy_input = dummy_input[:batch_size]
                                if dummy_input.shape[0] != batch_size:
                                    if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                        extension = torch.rand(
                                            batch_size - dummy_input.shape[0],
                                            *dummy_input.shape[1:],
                                            dtype=torch_dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                    else:
                                        extension = torch.randint(
                                            high=5,
                                            size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                            dtype=dummy_input.dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                                if not use_mask:
                                    dummy_attention_mask = None
                                else:
                                    dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                    if dummy_attention_mask is None:
                                        if is_encoder_decoder:
                                            seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                        else:
                                            seqlen = dummy_input.shape[-1]
                                        dummy_attention_mask = (
                                            torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                        )

                                    dummy_attention_mask = dummy_attention_mask[:batch_size]
                                    if dummy_attention_mask.shape[0] != batch_size:
                                        extension = torch.ones(
                                            batch_size - dummy_attention_mask.shape[0],
                                            *dummy_attention_mask.shape[1:],
                                            dtype=dummy_attention_mask.dtype,
                                            device=torch_device,
                                        )
                                        dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                        dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                    dummy_attention_mask[:] = 1
                                    if padding_side == "left":
                                        dummy_attention_mask[-1, :-1] = 1
                                        dummy_attention_mask[-1, -4:] = 0
                                    elif padding_side == "right":
                                        dummy_attention_mask[-1, 1:] = 1
                                        dummy_attention_mask[-1, :3] = 0

                                for enable_kernels in [False, True]:
                                    failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"
                                    if is_encoder_decoder:
                                        decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
                                            :batch_size
                                        ]
                                        if decoder_input_ids.shape[0] != batch_size:
                                            extension = torch.ones(
                                                batch_size - decoder_input_ids.shape[0],
                                                *decoder_input_ids.shape[1:],
                                                dtype=decoder_input_ids.dtype,
                                                device=torch_device,
                                            )
                                            decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                                            decoder_input_ids = decoder_input_ids.to(torch_device)

                                        # TODO: never an `attention_mask` arg here?
                                        processed_inputs = {
                                            model.main_input_name: dummy_input,
                                            "decoder_input_ids": decoder_input_ids,
                                            "decoder_attention_mask": dummy_attention_mask,
                                            "output_hidden_states": True,
                                        }
                                    else:
                                        processed_inputs = {
                                            model.main_input_name: dummy_input,
                                            "output_hidden_states": True,
                                        }

                                        # Otherwise fails for e.g. WhisperEncoderModel
                                        if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                            processed_inputs["attention_mask"] = dummy_attention_mask

                                        if (
                                            self.has_attentions
                                            and "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                                        ):
                                            processed_inputs["output_attentions"] = output_attentions
                                    if not deactivate_mask and (
                                        "bool_masked_pos" in inspect.signature(model_eager.forward).parameters
                                    ):
                                        dummy_mask = torch.ones((self.model_tester.num_masks,))

                                        # In case of additional token (like class) we define a custom `mask_length`
                                        if hasattr(self.model_tester, "mask_length"):
                                            mask_length = self.model_tester.mask_length - dummy_mask.size(0)
                                        else:
                                            mask_length = self.model_tester.seq_length - dummy_mask.size(0)
                                        dummy_mask = torch.cat([dummy_mask, torch.zeros(mask_length)])
                                        dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
                                        processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

                                    if "noise" in inspect.signature(model_eager.forward).parameters:
                                        np.random.seed(2)
                                        num_patches = int(
                                            (self.model_tester.image_size // self.model_tester.patch_size) ** 2
                                        )
                                        noise = np.random.uniform(size=(batch_size, num_patches))
                                        processed_inputs["noise"] = torch.from_numpy(noise)

                                    # TODO: test gradients as well (& for FA2 as well!)
                                    with torch.no_grad():
                                        with torch.backends.cuda.sdp_kernel(
                                            enable_flash=enable_kernels,
                                            enable_math=True,
                                            enable_mem_efficient=enable_kernels,
                                        ):
                                            prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                                            outputs_eager = model_eager(**prepared_inputs)
                                            outputs_sdpa = model_sdpa(**prepared_inputs)

                                    logits_eager = (
                                        outputs_eager.hidden_states[-1]
                                        if not is_encoder_decoder
                                        else outputs_eager.decoder_hidden_states[-1]
                                    )
                                    logits_sdpa = (
                                        outputs_sdpa.hidden_states[-1]
                                        if not is_encoder_decoder
                                        else outputs_sdpa.decoder_hidden_states[-1]
                                    )

                                    if torch_device in ["cpu", "cuda"]:
                                        atol = atols[torch_device, enable_kernels, torch_dtype]
                                        rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                    else:
                                        atol = 1e-7
                                        rtol = 1e-4

                                    # Masked tokens output slightly deviates - we don't mind that.
                                    if use_mask:
                                        if padding_side == "left":
                                            sub_sdpa = logits_sdpa[:-1]
                                            sub_eager = logits_eager[:-1]
                                            if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                                fail_cases.append(
                                                    get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                                )

                                            sub_sdpa = logits_sdpa[-1, :-4]
                                            sub_eager = logits_eager[-1, :-4]
                                            if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                                fail_cases.append(
                                                    get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                                )

                                            # Testing the padding tokens is not really meaningful but anyway
                                            # sub_sdpa = logits_sdpa[-1, -4:]
                                            # sub_eager = logits_eager[-1, -4:]
                                            # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))
                                        elif padding_side == "right":
                                            sub_sdpa = logits_sdpa[:-1]
                                            sub_eager = logits_eager[:-1]
                                            if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                                fail_cases.append(
                                                    get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                                )

                                            sub_sdpa = logits_sdpa[-1, 3:]
                                            sub_eager = logits_eager[-1, 3:]
                                            if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                                fail_cases.append(
                                                    get_mean_reldiff(failcase, sub_sdpa, sub_eager, atol, rtol)
                                                )

                                            # Testing the padding tokens is not really meaningful but anyway
                                            # sub_sdpa = logits_sdpa[-1, :3]
                                            # sub_eager = logits_eager[-1, :3]
                                            # if not torch.allclose(sub_sdpa, sub_eager, atol=atol, rtol=rtol):
                                            #     fail_cases.append(get_mean_reldiff(failcase, sub_sdpa, sub_eager, 4e-2, 4e-2))

                                    else:
                                        if not torch.allclose(logits_sdpa, logits_eager, atol=atol, rtol=rtol):
                                            fail_cases.append(
                                                get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                            )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))


@slow
@require_torch_accelerator
class GlmIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    model_id = "THUDM/glm-4-9b"
    revision = "refs/pr/15"
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def test_model_9b_fp16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_bf16(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, revision=self.revision
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_9b_eager(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_sdpa
    def test_model_9b_sdpa(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_9b_flash_attn(self):
        EXPECTED_TEXTS = [
            "Hello I am doing a project on the history of the internetSolution:\n\nStep 1: Introduction\nThe history of the",
            "Hi today I am going to show you how to make a simple and easy to make a DIY paper flower.",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            revision=self.revision,
        )
        model.to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)
