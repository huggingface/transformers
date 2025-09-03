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
"""Testing suite for the PyTorch MiniMax model."""

import unittest

import pytest

from transformers import MiniMaxConfig, is_torch_available
from transformers.cache_utils import Cache
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MiniMaxForCausalLM,
        MiniMaxForQuestionAnswering,
        MiniMaxForSequenceClassification,
        MiniMaxForTokenClassification,
        MiniMaxModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MiniMaxModelTester(CausalLMModelTester):
    config_class = MiniMaxConfig
    if is_torch_available():
        base_model_class = MiniMaxModel
        causal_lm_class = MiniMaxForCausalLM
        sequence_class = MiniMaxForSequenceClassification
        token_class = MiniMaxForTokenClassification
        question_answering_class = MiniMaxForQuestionAnswering

    def __init__(self, parent, layer_types=None, block_size=3):
        super().__init__(parent)
        self.layer_types = layer_types
        self.block_size = block_size


@require_torch
class MiniMaxModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MiniMaxModel,
            MiniMaxForCausalLM,
            MiniMaxForSequenceClassification,
            MiniMaxForTokenClassification,
            MiniMaxForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MiniMaxModel,
            "text-classification": MiniMaxForSequenceClassification,
            "token-classification": MiniMaxForTokenClassification,
            "text-generation": MiniMaxForCausalLM,
            "question-answering": MiniMaxForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = MiniMaxModelTester

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_local_experts = 8
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MiniMaxForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_local_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))
        use_cache = decoder_past_key_values is not None

        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            expected_shape = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                prompt_length + generated_length,
            )
            for layer_idx, layer_attention in enumerate(iter_attentions):
                if config.layer_types[layer_idx] == "full_attention":
                    self.assertEqual(layer_attention.shape, expected_shape)

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))

        # (batch, head, seq_length, head_features)
        key_value_cache_expected_shape = (
            batch_size,
            config.num_key_value_heads,
            cache_length,
            config.hidden_size // config.num_attention_heads,
        )
        # (batch, head, head_features, head_features)
        linear_cache_expected_shape = (
            batch_size,
            config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
            config.hidden_size // config.num_attention_heads,
        )

        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "full_attention":
                self.assertEqual(decoder_past_key_values[layer_idx][0].shape, key_value_cache_expected_shape)
                self.assertEqual(decoder_past_key_values[layer_idx][1].shape, key_value_cache_expected_shape)
            else:
                self.assertEqual(decoder_past_key_values[layer_idx][0].shape, linear_cache_expected_shape)

    @pytest.mark.generate
    def test_past_key_values_format(self, custom_all_cache_shapes=None):
        """
        Test that the KV cache is formatted correctly.
        """
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            model = model_class(config).to(torch_device)
            model = model.eval()
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            past_kv = outputs["past_key_values"]

            batch_size, seq_length = inputs["input_ids"].shape
            self._check_past_key_values_for_generate(batch_size, past_kv, seq_length, config)

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(reason="MiniMaxCache does not support `crop()` method")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Model needs refactor")
    def test_attention_outputs(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MiniMax is special")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
@require_torch_accelerator
@slow
class MiniMaxIntegrationTest(unittest.TestCase):
    def test_small_model_logits(self):
        model_id = "hf-internal-testing/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)

        with torch.no_grad():
            logits = model(dummy_input).logits

        logits = logits.float()

        expectations = Expectations(
            {
                (None, None): [[1.0312, -0.5156, -0.3262], [-0.1152, 0.4336, 0.2412], [1.2188, -0.5898, -0.0381]],
                ("cuda", 8): [[1.0312, -0.5156, -0.3203], [-0.1201, 0.4375, 0.2402], [1.2188, -0.5898, -0.0396]],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(logits[0, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, :3, :3], expected_slice, atol=1e-3, rtol=1e-3)

    def test_small_model_generation(self):
        model_id = "hf-internal-testing/MiniMax-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MiniMaxForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)
        expected_slice = (
            torch.tensor([[0, 1, 0, 933, 307, 3102, 2457, 1208], [0, 1, 0, 933, 307, 3102, 2457, 1208]])
            .to(torch.int64)
            .to(torch_device)
        )

        outputs = model.generate(dummy_input, max_new_tokens=5, do_sample=False)

        torch.testing.assert_close(outputs, expected_slice, atol=1e-3, rtol=1e-3)
