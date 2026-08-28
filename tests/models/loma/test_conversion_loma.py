# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers.models.loma.convert_loma_to_hf import (
    convert_matcher_state_dict,
    convert_state_dict,
)
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
class TestLoMaConversion:
    def test_convert_matcher_state_dict(self):
        reference_state_dict = {
            "posenc.Wr.weight": torch.ones(2, 2),
            "input_proj.weight": torch.ones(4, 2),
            "transformers.0.self_attn.Wqkv.weight": torch.ones(6, 2),
            "transformers.0.self_attn.out_proj.bias": torch.ones(2),
            "transformers.0.self_attn.ffn.1.weight": torch.ones(4),
            "transformers.0.cross_attn.to_qk.weight": torch.ones(4, 2),
            "transformers.0.cross_attn.to_v.weight": torch.ones(2, 2),
            "transformers.0.cross_attn.to_out.bias": torch.ones(2),
            "transformers.0.cross_attn.ffn.3.bias": torch.ones(2),
            "log_assignment.0.final_proj.weight": torch.ones(2, 2),
            "log_assignment.1.matchability.bias": torch.ones(1),
            "_descriptor.encoder.vgg.layers.0.weight": torch.ones(2, 2),
        }

        converted_state_dict = convert_state_dict(reference_state_dict, num_hidden_layers=2)

        assert set(converted_state_dict) == {
            "positional_encoder.projector.weight",
            "input_projection.weight",
            "layers.0.self_attention.q_proj.weight",
            "layers.0.self_attention.k_proj.weight",
            "layers.0.self_attention.v_proj.weight",
            "layers.0.self_attention.o_proj.bias",
            "layers.0.self_mlp.layer_norm.weight",
            "layers.0.cross_attention.q_proj.weight",
            "layers.0.cross_attention.k_proj.weight",
            "layers.0.cross_attention.v_proj.weight",
            "layers.0.cross_attention.o_proj.bias",
            "layers.0.cross_mlp.fc2.bias",
            "match_assignment.matchability.bias",
            "descriptor_network.encoder.layers.0.weight",
        }

    def test_convert_descriptor_keys(self):
        """Verify that _descriptor.* keys are correctly renamed to descriptor_network.*."""
        reference_state_dict = {
            "_descriptor.encoder.vgg.layers.0.weight": torch.ones(64, 3, 3, 3),
            "_descriptor.encoder.vgg.layers.1.weight": torch.ones(64),
            "_descriptor.decoder.layers.1.block1.0.weight": torch.ones(256, 128, 1, 1),
            "_descriptor.decoder.layers.1.out_conv.bias": torch.ones(256),
            "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.cls_token": torch.ones(1, 1, 1024),
            "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.patch_embed.proj.weight": torch.ones(1024, 3, 14, 14),
            "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.blocks.0.attn.qkv.weight": torch.ones(3072, 1024),
            "_descriptor.encoder.frozen_dinov2.dinov2_vitl14.blocks.1.ls1.gamma": torch.ones(1024),
        }
        converted = convert_state_dict(reference_state_dict, num_hidden_layers=9)

        assert "descriptor_network.encoder.layers.0.weight" in converted
        assert "descriptor_network.encoder.layers.1.weight" in converted
        assert "descriptor_network.decoder.layers.1.block1.0.weight" in converted
        assert "descriptor_network.auxiliary_backbone.embeddings.cls_token" in converted
        assert "descriptor_network.auxiliary_backbone.embeddings.patch_embeddings.projection.weight" in converted
        assert "descriptor_network.auxiliary_backbone.encoder.layer.0.attention.attention.query.weight" in converted
        assert "descriptor_network.auxiliary_backbone.encoder.layer.0.attention.attention.key.weight" in converted
        assert "descriptor_network.auxiliary_backbone.encoder.layer.0.attention.attention.value.weight" in converted
        assert "descriptor_network.auxiliary_backbone.encoder.layer.1.layer_scale1.lambda1" in converted
        assert converted[
            "descriptor_network.auxiliary_backbone.encoder.layer.0.attention.attention.query.weight"
        ].shape == (1024, 1024)

        # Verify DINOv2 keys are skipped
        assert not any(k for k in converted if "frozen_dinov2" in k)
        # Verify tensors are the same objects (no copy)
        for src_key, dst_key in [
            ("_descriptor.encoder.vgg.layers.0.weight", "descriptor_network.encoder.layers.0.weight"),
            ("_descriptor.decoder.layers.1.out_conv.bias", "descriptor_network.decoder.layers.1.out_conv.bias"),
        ]:
            assert torch.equal(converted[dst_key], reference_state_dict[src_key])

    def test_convert_matcher_state_dict_backward_compat(self):
        """Verify that the deprecated convert_matcher_state_dict still works."""
        reference_state_dict = {
            "posenc.Wr.weight": torch.ones(2, 2),
            "_descriptor.encoder.vgg.layers.0.weight": torch.ones(2, 2),
        }
        result = convert_matcher_state_dict(reference_state_dict, num_hidden_layers=2)
        assert "positional_encoder.projector.weight" in result
        assert "descriptor_network.encoder.layers.0.weight" in result

    def test_qkv_split(self):
        """Verify that fused QKV weights are correctly split into separate q/k/v projections."""
        dim = 4
        # Self attention: Wqkv is [3*dim, dim]
        qkv_weight = torch.arange(3 * dim * dim, dtype=torch.float).reshape(3 * dim, dim)
        reference_state_dict = {
            "transformers.0.self_attn.Wqkv.weight": qkv_weight,
        }
        converted = convert_state_dict(reference_state_dict, num_hidden_layers=1)

        q_weight = converted["layers.0.self_attention.q_proj.weight"]
        k_weight = converted["layers.0.self_attention.k_proj.weight"]
        v_weight = converted["layers.0.self_attention.v_proj.weight"]

        assert q_weight.shape == (dim, dim)
        assert k_weight.shape == (dim, dim)
        assert v_weight.shape == (dim, dim)
        assert torch.equal(q_weight, qkv_weight[:dim])
        assert torch.equal(k_weight, qkv_weight[dim : 2 * dim])
        assert torch.equal(v_weight, qkv_weight[2 * dim :])

    def test_cross_attention_shared_qk(self):
        """Verify that shared to_qk weights are correctly duplicated into q_proj and k_proj."""
        dim = 4
        qk_weight = torch.randn(2 * dim, dim)
        reference_state_dict = {
            "transformers.0.cross_attn.to_qk.weight": qk_weight,
        }
        converted = convert_state_dict(reference_state_dict, num_hidden_layers=1)

        q_weight = converted["layers.0.cross_attention.q_proj.weight"]
        k_weight = converted["layers.0.cross_attention.k_proj.weight"]

        assert q_weight.shape == (dim, dim)
        assert k_weight.shape == (dim, dim)
        # The to_qk is split in half, not duplicated
        assert torch.equal(q_weight, qk_weight[:dim])
        assert torch.equal(k_weight, qk_weight[dim:])
