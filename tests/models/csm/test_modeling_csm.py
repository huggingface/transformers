# coding=utf-8
# Copyright 2024, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ConversationalSpeechModel model."""

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
"""Testing suite for the PyTorch Mllama model."""

import unittest

import pytest
import inspect
import copy
import tempfile
import requests
from parameterized import parameterized

from transformers import (
    CsmConfig,
    CsmBackboneConfig,
    CsmDepthDecoderConfig,
    CsmBackboneModel,
    CsmDepthDecoderModel,
    CsmDepthDecoderForCausalLM,
    CsmForCausalLM,
    is_torch_available,
    is_vision_available,
)
from transformers.cache_utils import Cache
from transformers.models.mllama.configuration_mllama import MllamaTextConfig
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_read_token,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
    require_torch_sdpa,
    is_flaky,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION


if is_torch_available():
    import torch


class CsmModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        batch_size=3,
        seq_length=7,
        is_training=True,
        depth_decoder_config={
            "num_codebooks": 10,
            "backbone_hidden_size": 64,
            "vocab_size": 5,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 10,
        },
        backbone_config={
            "num_codebooks": 10,
            "codebook_vocab_size": 5,
            "vocab_size": 99,
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 10,
            "pad_token_id": 2,
            "codebook_pad_token_id": 2,
        },
    ):
        self.parent = parent
        self.is_training = is_training
        self.ignore_index = ignore_index
        self.depth_decoder_config = depth_decoder_config
        self.backbone_config = backbone_config
        self.seq_length = seq_length
        self.batch_size = batch_size

        self.num_hidden_layers = backbone_config["num_hidden_layers"]
        self.vocab_size = backbone_config["vocab_size"]
        self.hidden_size = backbone_config["hidden_size"]
        self.num_attention_heads = backbone_config["num_attention_heads"]
        self.pad_token_id = self.backbone_config["pad_token_id"] 

    def get_config(self):
        return CsmConfig(
            depth_decoder_config=self.depth_decoder_config,
            backbone_config=self.backbone_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        num_codebooks = self.backbone_config["num_codebooks"]
        text_input_ids = ids_tensor([self.batch_size, self.seq_length], config.backbone_config.vocab_size - 1) + 1
        codebook_input_ids = ids_tensor([self.batch_size, self.seq_length, num_codebooks], config.backbone_config.codebook_vocab_size - 1) + 1
        input_ids = torch.cat([codebook_input_ids, text_input_ids.unsqueeze(-1)], dim=-1)

        attention_mask = input_ids[..., -1].ne(1).to(torch_device)

        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

# @require_torch
# class CsmDepthDecoderModelTest(ModelTesterMixin, unittest.TestCase):
#     all_model_classes = (CsmDepthDecoderForCausalLM,) if is_torch_available() else ()
#     test_pruning = False
#     test_headmasking = False

#     def setUp(self):
#         self.model_tester = CsmDepthDecoderModelTester(self)
#         self.config_tester = ConfigTester(
#             self,
#             config_class=CsmDepthDecoderConfig,
#         )

#     def test_config(self):
#         self.config_tester.run_common_tests()

    # @unittest.skip(reason="CSM depth decoder does not support tieing weights as it has a custom inputs_embeds and custom lm head.")
    # def test_tie_model_weights(self):
    #     pass

    # @unittest.skip(reason="CSM depth decoder does not support tieing weights as it has a custom inputs_embeds and custom lm head.")
    # def test_tied_weights_keys(self):
    #     pass

class CsmForCausalLMTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (CsmForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_resize_embeddings_untied = False
    _is_composite = True

    def setUp(self):
        self.model_tester = CsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CsmConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        """
        Overrides [GenerationTesterMixin._get_logits_processor_kwargs] to restrict to top_k, top_p, and temperature sampling.
        """
        logits_processor_kwargs = {}
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )

        return logits_processor_kwargs

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support assisted decoding.")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support Dola decoding.")
    def test_dola_decoding_sample(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_sample_generate(self):
        pass
    
    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_beam_sample_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support group beam search.")
    def test_group_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support group beam search.")
    def test_group_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support constrained beam search.")
    def test_constrained_beam_search_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support constrained beam search.")
    def test_constrained_beam_search_generate_dict_output(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support contrastive search.")
    def test_contrastive_generate_low_memory(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support prompt lookup decoding.")
    def test_prompt_lookup_decoding_stops_at_eos(self):
        pass

    @pytest.mark.skip(reason="CSM has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="CSM has custom embedding approach (text and audio embeddings).")
    def test_tie_model_weights(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_generate_from_inputs_embeds_1_beam_search(self, _, num_beams):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="CSM does not support beam search.")
    def test_model_parallel_beam_search(self):
        pass

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
            for attr_name in config.to_dict().keys()
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
                cache_position = torch.arange(input_ids.shape[1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]
            attention_mask = inputs_dict.get("attention_mask")
            if attention_mask is None:
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
            pad_size = (input_ids.shape[0], 32, model.config.num_codebooks + 1)
            pad_size_attention_mask = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat(
                (torch.zeros(pad_size_attention_mask, dtype=attention_mask.dtype, device=torch_device), attention_mask),
                dim=1
            )
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)


# class CsmForCausalLMIntegrationTest(unittest.TestCase):