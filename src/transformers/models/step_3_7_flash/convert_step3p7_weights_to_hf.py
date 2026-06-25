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
"""Convert Step3p7 (Step-3.7-Flash) original checkpoints to HuggingFace format.

Original checkpoint layout (stepfun-ai/Step-3.7-Flash):

  Language model (45 layers):
    model.embed_tokens.weight
    model.norm.weight
    lm_head.weight
    model.layers.{N}.input_layernorm.weight
    model.layers.{N}.post_attention_layernorm.weight
    model.layers.{N}.self_attn.{q,k,v,o,g}_proj.weight       # already split
    model.layers.{N}.mlp.{gate,up,down}_proj.weight           # dense layers (0-2)
    model.layers.{N}.moe.gate.weight                          # MoE router
    model.layers.{N}.moe.router_bias
    model.layers.{N}.moe.gate_proj.weight                     # [n_experts, intermediate, hidden]
    model.layers.{N}.moe.up_proj.weight                       # [n_experts, intermediate, hidden]
    model.layers.{N}.moe.down_proj.weight                     # [n_experts, hidden, intermediate]
    model.layers.{N}.share_expert.{gate,up,down}_proj.weight

  Vision model:
    vision_model.conv1.weight
    vision_model.ln_pre.{weight,bias}
    vision_model.positional_embedding
    vision_model.transformer.resblocks.{N}.attn.in_proj_{weight,bias}   # fused Q/K/V
    vision_model.transformer.resblocks.{N}.attn.out_proj.{weight,bias}
    vision_model.transformer.resblocks.{N}.mlp.c_fc.{weight,bias}
    vision_model.transformer.resblocks.{N}.mlp.c_proj.{weight,bias}
    vision_model.transformer.resblocks.{N}.ln_{1,2}.{weight,bias}
    vision_model.transformer.resblocks.{N}.ls_{1,2}.gamma
    vision_model.vit_downsampler{1,2}.{weight,bias}

  Projector:
    vit_large_projector.weight

Transformations applied:
  1. Key renaming via ordered regex substitutions (all patterns applied in sequence).
  2. Vision in_proj_weight/bias  →  split q/k/v  (chunk dim=0, 3 equal parts).
  3. MoE gate_proj + up_proj     →  fused gate_up_proj  (cat dim=1 → [n_experts, 2*inter, hidden]).
"""

import argparse
import json
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import AutoTokenizer
from transformers.models.step_3_7_flash.configuration_step3p7 import Step3p7Config
from transformers.models.step_3_7_flash.modeling_step3p7 import Step3p7ForConditionalGeneration
from transformers.models.step_3_7_flash.processing_step3 import Step3VLProcessor


# fmt: off
# Ordered list of (pattern, replacement) pairs. ALL patterns are applied in sequence
# (not first-match-wins), so later patterns refine what earlier ones produced.
STATE_DICT_MAPPING = [
    # ── top-level module path remaps ─────────────────────────────────────────
    # vision_model.* → model.vision_model.*
    (r"^vision_model\.",                                r"model.vision_model."),
    # vit_large_projector.* → model.multi_modal_projector.*  (top-level form)
    (r"^vit_large_projector\.",                         r"model.multi_modal_projector."),
    # model.vit_large_projector.* → model.multi_modal_projector.*  (prefixed form)
    (r"^model\.vit_large_projector\.",                  r"model.multi_modal_projector."),
    # multi_modal_projector.* → model.multi_modal_projector.*  (no-op if already renamed)
    (r"^multi_modal_projector\.",                       r"model.multi_modal_projector."),
    # model.* → model.language_model.*  (skip already-remapped sub-modules)
    (r"^model\.(?!(language_model|vision_model|multi_modal_projector)\.)", r"model.language_model."),

    # ── vision encoder internal renames ──────────────────────────────────────
    (r"\.conv1\.weight$",                               r".embeddings.patch_embedding.weight"),
    (r"\.positional_embedding$",                        r".embeddings.position_embedding.weight"),
    (r"\.transformer\.resblocks\.",                     r".encoder.layers."),
    (r"\.transformer\.",                                r".encoder."),
    (r"\.ln_pre\.",                                     r".pre_layernorm."),
    (r"\.ls_1\.gamma$",                                 r".ls_1.scale"),
    (r"\.ls_2\.gamma$",                                 r".ls_2.scale"),
    (r"\.mlp\.c_fc\.",                                  r".mlp.fc1."),
    (r"\.mlp\.c_proj\.",                                r".mlp.fc2."),
    (r"\.attn\.",                                       r".self_attn."),
    (r"\.ln_1\.",                                       r".layer_norm1."),
    (r"\.ln_2\.",                                       r".layer_norm2."),

    # ── vision downsampler ───────────────────────────────────────────────────
    (r"\.vit_downsampler1\.",                           r".downsampler.0."),
    (r"\.vit_downsampler2\.",                           r".downsampler.1."),

    # ── MoE renames (down_proj; gate/up are fused separately in convert_state_dict) ──
    (r"\.moe\.gate\.weight$",                           r".mlp.gate.weight"),
    (r"\.moe\.router_bias$",                            r".mlp.gate.e_score_correction_bias"),
    (r"\.moe\.down_proj\.weight$",                      r".mlp.experts.down_proj"),
    (r"\.share_expert\.",                               r".mlp.shared_experts."),
]
# fmt: on


def remap_key(key: str) -> str:
    """Apply ALL STATE_DICT_MAPPING patterns in order to a single weight key."""
    for pattern, replacement in STATE_DICT_MAPPING:
        key = re.sub(pattern, replacement, key)
    return key


def convert_state_dict(original_state_dict: dict, num_hidden_layers: int | None = None) -> dict:
    """
    Produce the HF-format state dict from the original checkpoint.

    Steps:
      1. Pull out moe.gate_proj / moe.up_proj pairs for later fusion.
      2. Apply remap_key to all remaining keys.
      3. Split vision in_proj_weight/bias into q/k/v (chunk dim=0).
      4. Fuse MoE gate_proj + up_proj → gate_up_proj (cat dim=1).

    If ``num_hidden_layers`` is given, MTP/speculative-decode layers
    (model.layers.{N} for N >= num_hidden_layers) are dropped silently.
    """
    renamed: dict[str, torch.Tensor] = {}
    moe_gate: dict[str, torch.Tensor] = {}  # base_key → gate_proj tensor
    moe_up: dict[str, torch.Tensor] = {}    # base_key → up_proj tensor

    for old_key, tensor in original_state_dict.items():
        if num_hidden_layers is not None:
            m = re.match(r"^model\.layers\.(\d+)\.", old_key)
            if m and int(m.group(1)) >= num_hidden_layers:
                continue
        # ── intercept MoE gate/up before renaming ────────────────────────────
        if old_key.endswith(".moe.gate_proj.weight"):
            base = old_key[: -len(".moe.gate_proj.weight")]
            moe_gate[base] = tensor
            continue
        if old_key.endswith(".moe.up_proj.weight"):
            base = old_key[: -len(".moe.up_proj.weight")]
            moe_up[base] = tensor
            continue

        new_key = remap_key(old_key)

        # ── vision in_proj → q/k/v split ─────────────────────────────────────
        if new_key.endswith(".self_attn.in_proj_weight") and "vision_model" in new_key:
            prefix = new_key[: -len("in_proj_weight")]
            q, k, v = tensor.chunk(3, dim=0)
            renamed[prefix + "q_proj.weight"] = q.contiguous()
            renamed[prefix + "k_proj.weight"] = k.contiguous()
            renamed[prefix + "v_proj.weight"] = v.contiguous()
            continue
        if new_key.endswith(".self_attn.in_proj_bias") and "vision_model" in new_key:
            prefix = new_key[: -len("in_proj_bias")]
            q, k, v = tensor.chunk(3, dim=0)
            renamed[prefix + "q_proj.bias"] = q.contiguous()
            renamed[prefix + "k_proj.bias"] = k.contiguous()
            renamed[prefix + "v_proj.bias"] = v.contiguous()
            continue

        renamed[new_key] = tensor

    # ── fuse MoE gate + up → gate_up_proj ────────────────────────────────────
    # gate_proj.weight shape: [n_experts, intermediate_dim, hidden_dim]
    # up_proj.weight   shape: [n_experts, intermediate_dim, hidden_dim]
    # gate_up_proj     shape: [n_experts, 2*intermediate_dim, hidden_dim]
    if set(moe_gate.keys()) != set(moe_up.keys()):
        only_gate = set(moe_gate) - set(moe_up)
        only_up = set(moe_up) - set(moe_gate)
        raise KeyError(f"Mismatched MoE pairs — only gate: {only_gate}  only up: {only_up}")

    for base, gate in moe_gate.items():
        up = moe_up[base]
        fused = torch.cat([gate, up], dim=1)  # [n_experts, 2*inter, hidden]
        hf_base = remap_key(base)             # e.g. model.language_model.layers.3
        renamed[hf_base + ".mlp.experts.gate_up_proj"] = fused.contiguous()

    return renamed


def _is_hub_repo(path: str) -> bool:
    """Return True if path looks like a Hub repo id (e.g. 'stepfun-ai/Step-3.7-Flash')."""
    return not os.path.exists(path) and "/" in path and not path.startswith("/")


def resolve_input_dir(input_dir: str, cache_dir: str | None = None) -> str:
    """
    If input_dir is a Hub repo id, download the full snapshot and return its local path.
    Otherwise return input_dir unchanged (assumed to be a local directory).
    """
    if _is_hub_repo(input_dir):
        print(f"  {input_dir!r} looks like a Hub repo id — downloading snapshot ...")
        local = snapshot_download(input_dir, cache_dir=cache_dir)
        print(f"  Downloaded to {local}")
        return local
    return input_dir


def load_sharded_state_dict(input_dir: str) -> dict:
    """Load all .safetensors (or .pt/.bin) shards from a directory into one dict."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {input_dir!r}")
    state_dict = {}
    shards = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))
    if not shards:
        shards = sorted(f for f in os.listdir(input_dir) if f.endswith((".pt", ".bin")))
    if not shards:
        raise FileNotFoundError(f"No .safetensors / .pt / .bin files found in {input_dir}")
    for shard in shards:
        path = os.path.join(input_dir, shard)
        print(f"  Loading {shard} ...")
        if shard.endswith(".safetensors"):
            state_dict.update(load_file(path))
        else:
            state_dict.update(torch.load(path, map_location="cpu", weights_only=True))
    return state_dict


def _build_config_from_dict(cfg_dict: dict) -> Step3p7Config:
    """Construct Step3p7Config from a raw JSON dict, handling original-checkpoint quirks."""
    cfg_dict = dict(cfg_dict)  # shallow copy — don't mutate caller's dict
    cfg_dict.pop("model_type", None)
    cfg_dict.pop("architectures", None)
    cfg_dict.pop("auto_map", None)
    # Trim per-layer lists that the original checkpoint extends by
    # num_nextn_predict_layers (3 extra prediction heads we don't model).
    text = cfg_dict.get("text_config", {})
    if isinstance(text, dict):
        text = dict(text)
        cfg_dict["text_config"] = text
        n = text.get("num_hidden_layers")
        extra = text.pop("num_nextn_predict_layers", 0)
        if n is not None and extra:
            for key in ("layer_types", "mlp_layer_types", "swiglu_limits",
                        "swiglu_limits_shared", "partial_rotary_factors",
                        "rope_theta", "use_rope_layers"):
                val = text.get(key)
                if isinstance(val, list) and len(val) == n + extra:
                    text[key] = val[:n]
    return Step3p7Config(**cfg_dict)


def load_config(input_dir: str, config_path: str | None = None) -> Step3p7Config:
    """
    Load Step3p7Config. Tries in order:
      1. ``config_path`` if provided.
      2. ``<input_dir>/config.json``.
    """
    path = config_path or os.path.join(input_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No config.json found at {path!r}. "
            "Pass --config /path/to/config.json explicitly."
        )
    print(f"  Reading config from {path}")
    with open(path) as f:
        cfg_dict = json.load(f)
    return _build_config_from_dict(cfg_dict)


def convert_checkpoint(
    input_dir: str,
    output_dir: str,
    config_path: str | None = None,
    cache_dir: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """
    Convert the original Step3p7 checkpoint to HuggingFace format and save.

    Args:
        input_dir: Local directory OR Hub repo id (e.g. ``stepfun-ai/Step-3.7-Flash``).
        output_dir: Where to write the converted HF model.
        config_path: Path to a config.json to use instead of the one in input_dir.
        cache_dir: Local directory for Hub snapshot cache (only used when input_dir is a Hub repo id).
        push_to_hub: Whether to push the result to the Hub after saving.
        hub_model_id: Hub repo id, e.g. ``itazap/Step-3.7-Flash``.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Resolving input ...")
    input_dir = resolve_input_dir(input_dir, cache_dir=cache_dir)

    print(f"Loading config ...")
    config = load_config(input_dir, config_path)

    print(f"Loading shards from {input_dir} ...")
    original_state_dict = load_sharded_state_dict(input_dir)
    print(f"  Loaded {len(original_state_dict)} tensors.")

    print("Applying key renames and tensor transformations ...")
    new_state_dict = convert_state_dict(original_state_dict, num_hidden_layers=config.text_config.num_hidden_layers)
    print(f"  Produced {len(new_state_dict)} tensors.")
    del original_state_dict

    print("Loading converted weights into model (meta device) ...")
    with torch.device("meta"):
        model = Step3p7ForConditionalGeneration(config)

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False, assign=True)
    del new_state_dict
    if unexpected:
        print(f"  WARNING — unexpected keys (not in model): {unexpected}")
    if missing:
        # lm_head.weight may be absent when tied to embed_tokens; that's fine.
        non_tied = [k for k in missing if k != "lm_head.weight"]
        if non_tied:
            print(f"  WARNING — missing keys (not in checkpoint): {non_tied}")

    print(f"Saving HF model to {output_dir} ...")
    model.save_pretrained(output_dir, safe_serialization=True)
    del model

    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set")
        print(f"Pushing model to Hub: {hub_model_id} ...")
        saved = Step3p7ForConditionalGeneration.from_pretrained(output_dir)
        saved.push_to_hub(hub_model_id)
        print("Done.")


def convert_processor(
    input_dir: str,
    output_dir: str,
    cache_dir: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Copy processor/tokenizer from input_dir to output_dir (and optionally push)."""
    input_dir = resolve_input_dir(input_dir, cache_dir=cache_dir)
    try:
        processor = Step3VLProcessor.from_pretrained(input_dir)
        print("Loaded Step3VLProcessor.")
    except Exception:
        print("Step3VLProcessor not found, falling back to AutoTokenizer.")
        processor = AutoTokenizer.from_pretrained(input_dir)

    processor.save_pretrained(output_dir)

    if push_to_hub and hub_model_id:
        print(f"Pushing processor to Hub: {hub_model_id} ...")
        processor.push_to_hub(hub_model_id)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Step3p7 (Step-3.7-Flash) checkpoint to HuggingFace format."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the original checkpoint directory (safetensors shards + config.json).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write the converted HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the converted model and processor to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub repository id, e.g. 'itazap/Step-3.7-Flash'. Required with --push_to_hub.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Local directory for caching Hub downloads. Defaults to the HuggingFace cache (~/.cache/huggingface).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        dest="config_path",
        help=(
            "Path to a config.json to use for building Step3p7Config. "
            "Defaults to <input_dir>/config.json. "
            "Useful when the original checkpoint has no config.json or an incompatible one."
        ),
    )
    parser.add_argument(
        "--skip_processor",
        action="store_true",
        help="Skip processor/tokenizer conversion.",
    )
    args = parser.parse_args()

    convert_checkpoint(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        cache_dir=args.cache_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    if not args.skip_processor:
        convert_processor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )


if __name__ == "__main__":
    main()
