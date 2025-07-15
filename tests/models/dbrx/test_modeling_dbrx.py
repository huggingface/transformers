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
"""Testing suite for the PyTorch DBRX model."""

import unittest

from transformers import DbrxConfig, is_torch_available
from transformers.testing_utils import require_torch, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import DbrxForCausalLM, DbrxModel


class DbrxModelTester(CausalLMModelTester):
    config_class = DbrxConfig
    if is_torch_available():
        base_model_class = DbrxModel
        causal_lm_class = DbrxForCausalLM

    def __init__(
        self,
        parent,
        clip_qkv=8,
        rope_theta=500000,
        attn_config_model_type="",
        moe_jitter_eps=0,
        moe_loss_weight=0.05,
        moe_num_experts=8,
        moe_top_k=4,
        ffn_config_model_type="",
        initializer_range=0.02,
        resid_pdrop=0.0,
        is_decoder=True,
        pad_token_id=0,
    ):
        # Call parent init
        super().__init__(
            parent=parent,
            hidden_dropout_prob=resid_pdrop,
            attention_probs_dropout_prob=resid_pdrop,
            initializer_range=initializer_range,
            pad_token_id=pad_token_id,
            is_decoder=is_decoder,
        )

        # Set DBRX's unusual params
        self.clip_qkv = clip_qkv

        # DBRX takes sub-configurations for the FFN and attention layers, so we need to set that correctly here
        self.ffn_config = {
            "ffn_hidden_size": self.hidden_size,
            "moe_jitter_eps": moe_jitter_eps,
            "moe_loss_weight": moe_loss_weight,
            "moe_num_experts": moe_num_experts,
            "moe_top_k": moe_top_k,
            "model_type": ffn_config_model_type,
            "ffn_act_fn": {"name": self.hidden_act},
        }
        self.attn_config = {
            "clip_qkv": clip_qkv,
            "model_type": attn_config_model_type,
            "rope_theta": rope_theta,
        }

    @property
    def config_args(self):
        return super().config_args + ["ffn_config", "attn_config"]


@require_torch
class DbrxModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (DbrxModel, DbrxForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DbrxModel,
            "text-generation": DbrxForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = DbrxModelTester

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "eitanturok/dbrx-tiny"
        model = DbrxModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Dbrx models have weight tying disabled.")
    def test_tied_weights_keys(self):
        pass

    # Offload does not work with Dbrx models because of the forward of DbrxExperts where we chunk the experts.
    # The issue is that the offloaded weights of the mlp layer are still on meta device (w1_chunked, v1_chunked, w2_chunked)
    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_bin(self):
        pass


@require_torch
class DbrxModelIntegrationTest(unittest.TestCase):
    @slow
    def test_tiny_model_logits(self):
        model = DbrxForCausalLM.from_pretrained("Rocketknight1/dbrx-tiny-random")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        vocab_size = model.vocab_size

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [
                    [-1.6300e-04, 5.0118e-04, 2.5437e-04],
                    [2.0422e-05, 2.7210e-04, -1.5125e-04],
                    [-1.5105e-04, 4.6879e-04, 3.3309e-04],
                ]
            ]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
