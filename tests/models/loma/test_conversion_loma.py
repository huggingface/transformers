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

from transformers.models.loma.convert_loma_to_hf import convert_matcher_state_dict
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
