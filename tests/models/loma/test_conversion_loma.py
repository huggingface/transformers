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
from transformers.models.loma.convert_loma_to_hf import convert_checkpoint, convert_matcher_state_dict
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
            "_descriptor.encoder.layers.0.weight": torch.ones(2, 2),
        }

        converted_state_dict = convert_matcher_state_dict(reference_state_dict, num_hidden_layers=2)

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
        }

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
            else:
                continue
            reference_state_dict[reference_key] = tensor

        checkpoint_path = tmp_path / "loma_b.pt"
        output_dir = tmp_path / "converted"
        torch.save(reference_state_dict, checkpoint_path)
        convert_checkpoint(checkpoint_path, "loma_b", output_dir)

        converted_model = LoMaForKeypointMatching.from_pretrained(output_dir)
        for key, tensor in model.state_dict().items():
            if key.startswith(("positional_encoder", "transformer_layers", "match_assignment")):
                assert torch.equal(converted_model.state_dict()[key], tensor)
