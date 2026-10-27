# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LLaMA model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Lfm2ForCausalLM, Lfm2Model


class Lfm2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Lfm2Model

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class Lfm2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Lfm2ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2ForCausalLM if is_torch_available() else None

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.conv_L_cache)

    def test_packed_documents_match_separate_forwards(self):
        """`seq_idx` (as `DataCollatorWithFlattening(return_seq_idx=True)` produces it) keeps the short-conv window
        inside each packed document, so a packed forward must match the per-document forwards."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["conv"] * config.num_hidden_layers
        model = Lfm2Model(config).to(torch_device).eval()

        doc_1 = torch.arange(1, 9, device=torch_device).unsqueeze(0)
        doc_2 = torch.arange(9, 14, device=torch_device).unsqueeze(0)
        packed_input_ids = torch.cat([doc_1, doc_2], dim=1)
        seq_idx = torch.tensor([[0] * doc_1.shape[1] + [1] * doc_2.shape[1]], dtype=torch.int32, device=torch_device)

        with torch.no_grad():
            packed_out = model(input_ids=packed_input_ids, seq_idx=seq_idx, use_cache=False).last_hidden_state
            unpacked_out = model(input_ids=packed_input_ids, use_cache=False).last_hidden_state
            doc_1_out = model(input_ids=doc_1, use_cache=False).last_hidden_state
            doc_2_out = model(input_ids=doc_2, use_cache=False).last_hidden_state

        torch.testing.assert_close(packed_out[:, : doc_1.shape[1]], doc_1_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(packed_out[:, doc_1.shape[1] :], doc_2_out, rtol=1e-4, atol=1e-4)
        # without the boundaries the second document reads into the first
        self.assertFalse(torch.allclose(unpacked_out[:, doc_1.shape[1] :], doc_2_out, rtol=1e-4, atol=1e-4))

    def test_attention_outputs(self):
        """Lfm2Moe alternates between attention and short-conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])


@require_torch_accelerator
@slow
class Lfm2IntegrationTest(unittest.TestCase):
    pass
