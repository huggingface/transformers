# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import hashlib
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import MiniMaxVL01Config, MiniMaxVL01ForConditionalGeneration
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping, get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key


RELEASED_CHECKPOINT_KEY_MANIFEST_SHA256 = "d0caf1b714cdfc4caa65b56cbec591668d77b0e2a5d6cf3b5af169531311c886"


def _released_checkpoint_key_manifest() -> set[str]:
    """Return the exact key manifest from MiniMaxAI/MiniMax-VL-01's checkpoint index."""
    keys = {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        "vision_tower.vision_model.embeddings.class_embedding",
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.embeddings.position_embedding.weight",
        "vision_tower.vision_model.pre_layrnorm.weight",
        "vision_tower.vision_model.pre_layrnorm.bias",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_1.bias",
        "multi_modal_projector.linear_2.weight",
        "multi_modal_projector.linear_2.bias",
        "image_newline",
    }

    for layer_idx in range(80):
        layer_prefix = f"language_model.model.layers.{layer_idx}"
        keys.update(
            {
                f"{layer_prefix}.input_layernorm.weight",
                f"{layer_prefix}.post_attention_layernorm.weight",
                f"{layer_prefix}.block_sparse_moe.gate.weight",
            }
        )
        attention_projections = (
            ("q_proj", "k_proj", "v_proj", "o_proj")
            if (layer_idx + 1) % 8 == 0
            else ("qkv_proj", "output_gate", "norm", "out_proj")
        )
        keys.update(f"{layer_prefix}.self_attn.{projection}.weight" for projection in attention_projections)

        for expert_idx in range(32):
            for projection in ("w1", "w3", "w2"):
                keys.add(f"{layer_prefix}.block_sparse_moe.experts.{expert_idx}.{projection}.weight")

    for layer_idx in range(24):
        layer_prefix = f"vision_tower.vision_model.encoder.layers.{layer_idx}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            keys.update(
                {
                    f"{layer_prefix}.self_attn.{projection}.weight",
                    f"{layer_prefix}.self_attn.{projection}.bias",
                }
            )
        for layer_norm in ("layer_norm1", "layer_norm2"):
            keys.update({f"{layer_prefix}.{layer_norm}.weight", f"{layer_prefix}.{layer_norm}.bias"})
        for projection in ("fc1", "fc2"):
            keys.update(
                {
                    f"{layer_prefix}.mlp.{projection}.weight",
                    f"{layer_prefix}.mlp.{projection}.bias",
                }
            )

    return keys


def _old_text_config(num_hidden_layers: int = 80, num_local_experts: int = 32) -> dict:
    attn_type_list = (
        [int((layer_idx + 1) % 8 == 0) for layer_idx in range(num_hidden_layers)]
        if num_hidden_layers == 80
        else [0, 1]
    )
    text_config = {
        "model_type": "minimax_text_01",
        "attn_type_list": attn_type_list,
        "head_dim": 128 if num_hidden_layers == 80 else 8,
        "hidden_size": 6144 if num_hidden_layers == 80 else 32,
        "intermediate_size": 9216 if num_hidden_layers == 80 else 16,
        "layernorm_full_attention_alpha": 3.5565588200778455,
        "layernorm_full_attention_beta": 1.0,
        "layernorm_linear_attention_alpha": 3.5565588200778455,
        "layernorm_linear_attention_beta": 1.0,
        "layernorm_mlp_alpha": 3.5565588200778455,
        "layernorm_mlp_beta": 1.0,
        "max_position_embeddings": 8192 if num_hidden_layers == 80 else 64,
        "num_attention_heads": 64 if num_hidden_layers == 80 else 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": num_hidden_layers,
        "num_key_value_heads": 8 if num_hidden_layers == 80 else 2,
        "num_local_experts": num_local_experts,
        "postnorm": True,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000_000 if num_hidden_layers == 80 else 10_000,
        "rotary_dim": 64 if num_hidden_layers == 80 else 4,
        "shared_intermediate_size": [0],
        "shared_moe_mode": "sigmoid",
        "vocab_size": 200_064 if num_hidden_layers == 80 else 64,
    }
    if num_hidden_layers == 80:
        text_config.update(
            {
                "architectures": ["MiniMaxText01ForCausalLM"],
                "bos_token_id": None,
                "eos_token_id": None,
            }
        )
    return text_config


def _old_vision_config(full_size: bool = True) -> dict:
    vision_config = {
        "model_type": "clip_vision_model",
        "attention_dropout": 0.0,
        "hidden_act": "gelu",
        "hidden_size": 1024 if full_size else 16,
        "image_size": 336 if full_size else 8,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 4096 if full_size else 32,
        "layer_norm_eps": 1e-5,
        "num_attention_heads": 16 if full_size else 4,
        "num_channels": 3,
        "num_hidden_layers": 24 if full_size else 1,
        "patch_size": 14 if full_size else 4,
        "projection_dim": 6144 if full_size else 32,
    }
    if full_size:
        vision_config.update(
            {
                "auto_map": {"AutoModel": "modeling_clip.CLIPVisionModel"},
                "vocab_size": 32_000,
            }
        )
    return vision_config


def _released_config() -> "MiniMaxVL01Config":
    return MiniMaxVL01Config(
        text_config=_old_text_config(),
        vision_config=_old_vision_config(),
        image_grid_pinpoints=[[336, 672], [672, 336]],
        image_token_index=200_025,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    )


def _tiny_config() -> "MiniMaxVL01Config":
    return MiniMaxVL01Config(
        text_config=_old_text_config(num_hidden_layers=2, num_local_experts=2),
        vision_config=_old_vision_config(full_size=False),
        image_grid_pinpoints=[[8, 8]],
        image_token_index=63,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
    )


def _mapped_key_groups(model, source_keys: set[str]) -> dict[str, list[str]]:
    conversions = get_model_conversion_mapping(model, add_legacy=False)
    renamings = [conversion for conversion in conversions if isinstance(conversion, WeightRenaming)]
    converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]
    mapped_keys = defaultdict(list)
    for source_key in source_keys:
        target_key, _ = rename_source_key(source_key, renamings, converters)
        mapped_keys[target_key].append(source_key)
    return dict(mapped_keys)


def _is_derived_lightning_buffer(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in {"slope_rate", "query_decay", "key_decay", "diagonal_decay"}


def _target_only_keys(model) -> set[str]:
    return {
        key
        for key in model.state_dict()
        if _is_derived_lightning_buffer(key) or key.startswith("model.vision_tower.post_layernorm.")
    }


def _upstream_state_dict(model) -> dict[str, "torch.Tensor"]:
    upstream_state_dict = {}
    for target_key, tensor in model.state_dict().items():
        if target_key in _target_only_keys(model):
            continue

        if target_key.endswith(".mlp.experts.gate_up_proj"):
            gate, up = tensor.chunk(2, dim=1)
            expert_prefix = target_key.removesuffix("gate_up_proj")
            expert_prefix = expert_prefix.replace("model.language_model.", "language_model.model.", 1)
            expert_prefix = expert_prefix.replace(".mlp.", ".block_sparse_moe.")
            for expert_idx, (gate_weight, up_weight) in enumerate(zip(gate, up)):
                upstream_state_dict[f"{expert_prefix}{expert_idx}.w1.weight"] = gate_weight
                upstream_state_dict[f"{expert_prefix}{expert_idx}.w3.weight"] = up_weight
            continue

        if target_key.endswith(".mlp.experts.down_proj"):
            expert_prefix = target_key.removesuffix("down_proj")
            expert_prefix = expert_prefix.replace("model.language_model.", "language_model.model.", 1)
            expert_prefix = expert_prefix.replace(".mlp.", ".block_sparse_moe.")
            for expert_idx, down_weight in enumerate(tensor):
                upstream_state_dict[f"{expert_prefix}{expert_idx}.w2.weight"] = down_weight
            continue

        if target_key.startswith("model.language_model."):
            source_key = target_key.replace("model.language_model.", "language_model.model.", 1)
            source_key = source_key.replace(".mlp.", ".block_sparse_moe.")
        elif target_key.startswith("model.vision_tower."):
            source_key = target_key.replace("model.vision_tower.", "vision_tower.vision_model.", 1)
        elif target_key.startswith("model.multi_modal_projector."):
            source_key = target_key.replace("model.multi_modal_projector.", "multi_modal_projector.", 1)
        elif target_key == "model.image_newline":
            source_key = "image_newline"
        elif target_key.startswith("lm_head."):
            source_key = target_key.replace("lm_head.", "language_model.lm_head.", 1)
        else:
            raise AssertionError(f"No upstream conversion defined for {target_key}")

        upstream_state_dict[source_key] = tensor

    return {key: tensor.detach().cpu().contiguous() for key, tensor in upstream_state_dict.items()}


@require_torch
class MiniMaxVL01ConversionTest(unittest.TestCase):
    def test_root_conversion_mapping(self):
        conversions = get_checkpoint_conversion_mapping("minimax_vl_01")
        self.assertEqual(
            [(conversion.source_patterns[0], conversion.target_patterns[0]) for conversion in conversions],
            [
                (r"^language_model\.lm_head", "lm_head"),
                (r"^language_model\.model\.", "model.language_model."),
                (r"^vision_tower\.", "model.vision_tower."),
                (r"^multi_modal_projector\.", "model.multi_modal_projector."),
                (r"^image_newline$", "model.image_newline"),
            ],
        )

    def test_released_checkpoint_key_coverage(self):
        source_keys = _released_checkpoint_key_manifest()
        manifest_digest = hashlib.sha256("\n".join(sorted(source_keys)).encode()).hexdigest()
        self.assertEqual(len(source_keys), 8_637)
        self.assertEqual(manifest_digest, RELEASED_CHECKPOINT_KEY_MANIFEST_SHA256)

        config = _released_config()
        self.assertEqual(config.text_config.model_type, "minimax_vl_01_text")
        with torch.device("meta"):
            model = MiniMaxVL01ForConditionalGeneration(config)

        mapped_key_groups = _mapped_key_groups(model, source_keys)
        mapped_keys = set(mapped_key_groups)
        target_keys = set(model.state_dict())
        target_only_keys = _target_only_keys(model)

        self.assertEqual(len(mapped_keys), 1_117)
        self.assertEqual(len(target_keys), 1_399)
        self.assertEqual(Counter(map(len, mapped_key_groups.values())), {1: 957, 32: 80, 64: 80})
        gate_up_groups = {
            key: source_group
            for key, source_group in mapped_key_groups.items()
            if key.endswith("experts.gate_up_proj")
        }
        down_groups = {
            key: source_group for key, source_group in mapped_key_groups.items() if key.endswith("experts.down_proj")
        }
        self.assertEqual(len(gate_up_groups), 80)
        self.assertTrue(
            all(
                len(source_group) == 64
                and sum(".w1.weight" in key for key in source_group) == 32
                and sum(".w3.weight" in key for key in source_group) == 32
                for source_group in gate_up_groups.values()
            )
        )
        self.assertEqual(len(down_groups), 80)
        self.assertTrue(
            all(
                len(source_group) == 32 and all(".w2.weight" in key for key in source_group)
                for source_group in down_groups.values()
            )
        )
        self.assertEqual(mapped_keys - target_keys, set())
        self.assertEqual(target_keys - mapped_keys, target_only_keys)
        self.assertEqual(len(target_only_keys), 282)
        self.assertEqual(sum(_is_derived_lightning_buffer(key) for key in target_only_keys), 280)
        self.assertEqual(sum("post_layernorm" in key for key in target_only_keys), 2)

    def test_from_pretrained_converts_tiny_upstream_state_dict(self):
        torch.manual_seed(0)
        config = _tiny_config()
        reference_model = MiniMaxVL01ForConditionalGeneration(config)
        reference_state_dict = reference_model.state_dict()
        upstream_state_dict = _upstream_state_dict(reference_model)

        mapped_keys = set(_mapped_key_groups(reference_model, set(upstream_state_dict)))
        self.assertEqual(mapped_keys, set(reference_state_dict) - _target_only_keys(reference_model))

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            save_file(upstream_state_dict, Path(tmp_dir) / "model.safetensors", metadata={"format": "pt"})
            converted_model, loading_info = MiniMaxVL01ForConditionalGeneration.from_pretrained(
                tmp_dir, output_loading_info=True
            )

        self.assertEqual(set(loading_info["unexpected_keys"]), set())
        self.assertEqual(set(loading_info["mismatched_keys"]), set())
        self.assertEqual(loading_info.get("conversion_errors", {}), {})
        self.assertEqual(set(loading_info["missing_keys"]), set())
        for key, expected in reference_state_dict.items():
            torch.testing.assert_close(converted_model.state_dict()[key], expected)


if __name__ == "__main__":
    unittest.main()
