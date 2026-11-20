# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch TrOCR model."""

import unittest

from transformers import TrOCRConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers.models.trocr.modeling_trocr import TrOCRDecoder, TrOCRForCausalLM


@require_torch
class TrOCRStandaloneDecoderModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {
        "batch_size": 13,
        "decoder_seq_length": 7,
        "is_training": False,
        "use_attention_mask": True,
        "use_labels": True,
        "vocab_size": 99,
        "d_model": 16,
        "decoder_layers": 2,
        "decoder_ffn_dim": 32,
        "decoder_attention_heads": 4,
        "eos_token_id": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "decoder_start_token_id": 2,
        "use_cache": False,
        "max_position_embeddings": 30,
        "scope": None,
        "base_model_out_len": 2,
        "decoder_attention_idx": 1,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        # For common tests
        values["seq_length"] = values["decoder_seq_length"]
        values["hidden_size"] = values["d_model"]
        values["num_hidden_layers"] = values["decoder_layers"]
        values["num_attention_heads"] = values["decoder_attention_heads"]
        values["decoder_key_length"] = values["decoder_seq_length"]
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = TrOCRConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            max_position_embeddings=self.max_position_embeddings,
        )

        return (config, input_ids, attention_mask, lm_labels)

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        attention_mask,
        lm_labels,
    ):
        config.use_cache = True
        model = TrOCRDecoder(config=config).to(torch_device).eval()
        input_ids = input_ids[:2]

        input_ids[input_ids == 0] += 1
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        cache = outputs["cache"]

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((2, 1), config.vocab_size - 1) + 1

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, cache=cache)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, next_input_ids.shape[-1] - 1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        assert torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, lm_labels = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class TrOCRStandaloneDecoderModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    model_tester_class = TrOCRStandaloneDecoderModelTester

    all_model_classes = (TrOCRDecoder, TrOCRForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = {"text-generation": TrOCRForCausalLM} if is_torch_available() else {}

    def setUp(self):
        self.model_tester = TrOCRStandaloneDecoderModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TrOCRConfig)

    @unittest.skip(reason="Not yet implemented")
    def test_inputs_embeds(self):
        pass

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decoder_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past(*config_and_inputs)

    @unittest.skip(reason="Decoder cannot keep gradients")
    def test_retain_grad_hidden_states_attentions(self):
        return

    @unittest.skip(reason="The model doesn't support left padding")  # and it's not used enough to be worth fixing :)
    def test_left_padding_compatibility(self):
        pass
