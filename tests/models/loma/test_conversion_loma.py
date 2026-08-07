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

from transformers import LoMaConfig, LoMaForKeypointMatching
from transformers.models.loma.convert_loma_to_hf import (
    convert_checkpoint,
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
            "transformers.0.cross_attn.to_qk.weight": torch.ones(2, 2),
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
            "transformer_layers.0.self_attention.qkv.weight",
            "transformer_layers.0.self_attention.output.bias",
            "transformer_layers.0.self_attention.mlp.layers.1.weight",
            "transformer_layers.0.cross_attention.query_key.weight",
            "transformer_layers.0.cross_attention.value.weight",
            "transformer_layers.0.cross_attention.output.bias",
            "transformer_layers.0.cross_attention.mlp.layers.3.bias",
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
        assert "descriptor_network.dinov2_encoder.embeddings.cls_token" in converted
        assert "descriptor_network.dinov2_encoder.embeddings.patch_embeddings.projection.weight" in converted
        assert "descriptor_network.dinov2_encoder.encoder.layer.0.attention.attention.query.weight" in converted
        assert "descriptor_network.dinov2_encoder.encoder.layer.0.attention.attention.key.weight" in converted
        assert "descriptor_network.dinov2_encoder.encoder.layer.0.attention.attention.value.weight" in converted
        assert "descriptor_network.dinov2_encoder.encoder.layer.1.layer_scale1.lambda1" in converted
        assert converted[
            "descriptor_network.dinov2_encoder.encoder.layer.0.attention.attention.query.weight"
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

    def test_convert_checkpoint(self, tmp_path):
        model = LoMaForKeypointMatching(LoMaConfig(descriptor_dim=256, num_attention_heads=4))
        reference_state_dict = {}
        for key, tensor in model.state_dict().items():
            if key == "positional_encoder.projector.weight":
                reference_key = "posenc.Wr.weight"
            elif key.startswith("transformer_layers."):
                _, layer_index, attention_type, *suffix_parts = key.split(".")
                suffix = ".".join(suffix_parts)
                if attention_type == "self_attention":
                    replacements = {
                        "qkv": "Wqkv",
                        "output": "out_proj",
                        "mlp.layers.0": "ffn.0",
                        "mlp.layers.1": "ffn.1",
                        "mlp.layers.3": "ffn.3",
                    }
                    reference_attention_type = "self_attn"
                elif attention_type == "cross_attention":
                    replacements = {
                        "query_key": "to_qk",
                        "value": "to_v",
                        "output": "to_out",
                        "mlp.layers.0": "ffn.0",
                        "mlp.layers.1": "ffn.1",
                        "mlp.layers.3": "ffn.3",
                    }
                    reference_attention_type = "cross_attn"
                else:
                    continue
                for source_prefix, destination_prefix in replacements.items():
                    if suffix.startswith(source_prefix + "."):
                        reference_suffix = suffix.replace(source_prefix, destination_prefix, 1)
                        reference_key = f"transformers.{layer_index}.{reference_attention_type}.{reference_suffix}"
                        break
                else:
                    continue
            elif key.startswith("match_assignment."):
                _, source_name, parameter_name = key.split(".")
                source_name = "final_proj" if source_name == "final_projection" else source_name
                reference_key = f"log_assignment.8.{source_name}.{parameter_name}"
            elif key.startswith("descriptor_network.encoder."):
                reference_key = key.replace("descriptor_network.encoder.", "_descriptor.encoder.vgg.", 1)
            elif key.startswith("descriptor_network.decoder."):
                reference_key = key.replace("descriptor_network.", "_descriptor.", 1)
            elif key.startswith("descriptor_network.dinov2_encoder."):
                # Mocking the backward conversion for DINOv2 keys is complex because of QKV split,
                # so we will just skip testing the values of dinov2_encoder in test_convert_checkpoint
                # (we already test it in test_convert_descriptor_keys).
                continue
            else:
                continue
            reference_state_dict[reference_key] = tensor

        checkpoint_path = tmp_path / "loma_b.pt"
        output_dir = tmp_path / "converted"
        torch.save(reference_state_dict, checkpoint_path)
        convert_checkpoint(checkpoint_path, "loma_b", output_dir)

        converted_model = LoMaForKeypointMatching.from_pretrained(output_dir)
        converted_prefixes = (
            "positional_encoder",
            "transformer_layers",
            "match_assignment",
            "descriptor_network",
        )
        for key, tensor in model.state_dict().items():
            if key.startswith(converted_prefixes) and not key.startswith("descriptor_network.dinov2_encoder."):
                assert torch.equal(converted_model.state_dict()[key], tensor), f"Mismatch for key: {key}"
