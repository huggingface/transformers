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
"""Convert an MuseGlimmer checkpoint to a Hugging Face transformers checkpoint.

Usage:
    python src/transformers/models/muse_glimmer/convert_muse_glimmer_weights_to_hf.py \\
        --checkpoint_path        pytorch_weights/llm/mp_1/model_0.pt \\
        --vision_checkpoint_path pytorch_weights/encoder/mp_1/<name>.pt \\
        --tokenizer_path         pytorch_weights/tokenizer/l4_200k_base \\
        --output_path            muse_glimmer-hf-converted
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import processors

from transformers import (
    MuseGlimmerConfig,
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerImageProcessor,
    MuseGlimmerProcessor,
    MuseGlimmerTextConfig,
    MuseGlimmerVideoProcessor,
    MuseGlimmerVisionConfig,
    TokenizersBackend,
)
from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.utils.hub import cached_file


# Canonical home of the chat template. Following the HF convention (cf. the Gemma
# release, which reads ``chat_template.jinja`` from its model repo via ``cached_file``),
# the template's source of truth is the ``chat_template.jinja`` file in the model repo
# on the Hub, not an inline Python string. The converter downloads it from here and
# bakes it into the converted checkpoint.
#
# NOTE: this points at the temporary early-release repo, where the template lives under
# the ``hf/`` subfolder alongside the other HF assets.
# TODO: before release, update to the final public repo and load ``chat_template.jinja``
# from the repo root (drop the ``hf/`` prefix).
MUSE_GLIMMER_MM_CHAT_TEMPLATE = Path(cached_file("someorgtoo/onyx_early_v2", "hf/chat_template.jinja")).read_text(
    encoding="utf-8"
)


NUM_BASE_TOKENS = 200_000
NUM_RESERVED_SPECIAL_TOKENS = 2048
KEEP_SPECIAL_TOKENS: dict[int, str] = {
    0: "<|begin_of_text|>",
    1: "<|end_of_text|>",
    7: "<|eom|>",
    8: "<|eot|>",
    18: "<|finetune_right_pad|>",
    22: "<|start|>",
    23: "<|message|>",
    80: "<|image_start|>",
    81: "<|image_end|>",
    82: "<|vid_start|>",
    83: "<|vid_end|>",
    87: "<|vid_frame_separator|>",
    90: "<|image|>",
    91: "<|video|>",
    92: "<|patch|>",
}

O200K_PATTERN = (
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def _ordered_special_tokens() -> list[str]:
    """Return the 2048-entry ordered table; index i maps to id NUM_BASE_TOKENS + i."""
    return [KEEP_SPECIAL_TOKENS.get(i, f"<|reserved_special_token_{i}|>") for i in range(NUM_RESERVED_SPECIAL_TOKENS)]


def build_config():
    text_config = MuseGlimmerTextConfig(
        vocab_size=202_048,
        hidden_size=6656,
        intermediate_size=19968,
        num_hidden_layers=52,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        hidden_activation="silu",
        # RoPE cache is sized to this; must match the 131072 tokenizer/generation max_length,
        # else sequences >16384 crash on RoPE position indexing. (Was 16_384 - reintroduced the crash on convert.)
        max_position_embeddings=131_072,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        bos_token_id=200_000,
        eos_token_id=200_001,
        sliding_window=2048,
        final_logit_softcapping=20.0,
        rope_parameters={"rope_type": "default", "rope_theta": 500_000.0},
        # The reference config stores 43.7840518911 == 3.87 * sqrt(head_dim); we keep the multiplier
        # actually applied to Q after the QK-norm.
        qk_scale_factor=3.87,
        # == 1 / sqrt(hidden_size / 256)
        output_multiplier=0.19611613513818404,
        post_norm_eps=1e-8,
    )

    vision_config = MuseGlimmerVisionConfig(
        hidden_size=1536,
        num_hidden_layers=50,
        num_attention_heads=16,
        intermediate_size=8960,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=32,
        pos_emb_width=32,
        hidden_act="gelu",
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        max_position_embeddings=32 * 32,
        layer_norm_eps=1e-5,
    )

    return MuseGlimmerConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=200_092,
        video_token_id=200_091,
        out_hidden_size=6144,
        projector_hidden_size=4096,
    )


TEXT_LAYER_RENAMES = {
    "attention.qkv_proj.norm.weight": "input_layernorm.weight",
    "attention.o_proj.weight": "self_attn.o_proj.weight",
    "feed_forward.norm.weight": "pre_feedforward_layernorm.weight",
    "feed_forward.fc2_weight": "mlp.down_proj.weight",
    "post_attn_norm.weight": "post_attention_layernorm.weight",
    "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
}


def convert_text_state_dict(source: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    text_cfg = config.text_config
    n_layers = text_cfg.num_hidden_layers
    q_dim = text_cfg.num_attention_heads * text_cfg.head_dim
    kv_dim = text_cfg.num_key_value_heads * text_cfg.head_dim
    og_dim = q_dim
    ffn_dim = text_cfg.intermediate_size

    out: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": source.pop("tok_embeddings.weight"),
        "model.language_model.norm.weight": source.pop("output.norm.weight"),
        "lm_head.weight": source.pop("output.weight"),
        "model.vision_projection.weight": source.pop("vision_projection.weight"),
    }

    for i in range(n_layers):
        src = f"layers.{i}"
        dst = f"model.language_model.layers.{i}"

        qkv = source.pop(f"{src}.attention.qkv_proj.weight")
        pieces = qkv.split([q_dim, kv_dim, kv_dim, og_dim], dim=0)

        out[f"{dst}.self_attn.q_proj.weight"] = _permute_for_rope(pieces[0], n_heads=text_cfg.num_attention_heads)
        out[f"{dst}.self_attn.k_proj.weight"] = _permute_for_rope(pieces[1], n_heads=text_cfg.num_key_value_heads)
        out[f"{dst}.self_attn.v_proj.weight"] = pieces[2]
        out[f"{dst}.self_attn.gate_proj.weight"] = pieces[3]

        fc1 = source.pop(f"{src}.feed_forward.fc1_weight")
        gate_w, up_w = fc1.split([ffn_dim, ffn_dim], dim=0)
        out[f"{dst}.mlp.gate_proj.weight"] = gate_w
        out[f"{dst}.mlp.up_proj.weight"] = up_w

        for suffix, target_suffix in TEXT_LAYER_RENAMES.items():
            out[f"{dst}.{target_suffix}"] = source.pop(f"{src}.{suffix}")

    leftover = [
        k
        for k in source
        if k.startswith("layers.")
        or k in {"tok_embeddings.weight", "output.weight", "output.norm.weight", "vision_projection.weight"}
    ]
    if leftover:
        raise RuntimeError(f"Unconsumed LLM keys after conversion: {leftover}")

    return out


VISION_TOP_LEVEL_RENAMES = {
    "vision_encoder.positional_embedding_vlm": "model.vision_tower.patch_embedder.position_embedding_table.weight",
    "vision_encoder.conv1_linear.weight": "model.vision_tower.patch_embedder.patch_embedding.weight",
    "vision_encoder.ln_pre.weight": "model.vision_tower.ln_pre.weight",
    "vision_encoder.ln_pre.bias": "model.vision_tower.ln_pre.bias",
    "vision_encoder.ln_post.weight": "model.vision_tower.ln_post.weight",
    "vision_encoder.ln_post.bias": "model.vision_tower.ln_post.bias",
    "vision_adapter.c_fc.weight": "model.vision_adapter.fc1.weight",
    "vision_adapter.c_proj.weight": "model.vision_adapter.fc2.weight",
}

# Per-block renames — targets match the Kimi_K25 vision layout MuseGlimmerVisionEncoderLayer now inherits.
VISION_LAYER_RENAMES = {
    "attn.wv.weight": "attn.v_proj.weight",
    "attn.wv.bias": "attn.v_proj.bias",
    "attn.wo.weight": "attn.proj.weight",
    "attn.wo.bias": "attn.proj.bias",
    "ln_1.weight": "norm1.weight",
    "ln_1.bias": "norm1.bias",
    "ln_2.weight": "norm2.weight",
    "ln_2.bias": "norm2.bias",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
}


def _permute_for_rope(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    if tensor.ndim == 2:
        dim1, dim2 = tensor.shape
        return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    if tensor.ndim == 1:
        (dim1,) = tensor.shape
        return tensor.view(n_heads, dim1 // n_heads // 2, 2).transpose(1, 2).reshape(dim1)
    raise ValueError(f"_permute_for_rope: unexpected tensor shape {tuple(tensor.shape)}")


def convert_vision_state_dict(source: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    if "weights" in source and isinstance(source["weights"], dict):
        source = source["weights"]

    vis_cfg = config.vision_config
    n_heads = vis_cfg.num_attention_heads
    n_layers = vis_cfg.num_hidden_layers

    out: dict[str, torch.Tensor] = {}
    for src_key, dst_key in VISION_TOP_LEVEL_RENAMES.items():
        out[dst_key] = source.pop(src_key)

    for i in range(n_layers):
        src = f"vision_encoder.transformer.resblocks.{i}"
        dst = f"model.vision_tower.layers.{i}"

        for suffix in ("weight", "bias"):
            wq = source.pop(f"{src}.attn.wq.{suffix}")
            wk = source.pop(f"{src}.attn.wk.{suffix}")
            out[f"{dst}.attn.q_proj.{suffix}"] = _permute_for_rope(wq, n_heads)
            out[f"{dst}.attn.k_proj.{suffix}"] = _permute_for_rope(wk, n_heads)

        for src_suffix, dst_suffix in VISION_LAYER_RENAMES.items():
            out[f"{dst}.{dst_suffix}"] = source.pop(f"{src}.{src_suffix}")

    if source:
        raise RuntimeError(f"Unconsumed vision-shard keys after conversion: {sorted(source)[:20]}")

    return out


# HF response_template: the output-side counterpart to the chat template. transformers
# 5.x reads this off the tokenizer (``tokenizer.parse_response``) to turn a generated
# assistant turn back into a structured message (reasoning_content / content /
# tool_calls). Owned by the model team; baked onto the tokenizer so partners get
# ``parse_response()`` for free.
#
# This is the declarative response-template format (the successor to the legacy
# x-regex ``response_schema``): each field is a region delimited by ``open_pattern`` /
# ``close``, parsed by a named content parser. Channel-scoping is structural rather than
# regex-based -- while an explicit region (reasoning / content) is open the parser only
# watches that region's close, so an ``<atem:invoke>`` echoed inside reasoning or the
# final answer is body text, never a spurious tool call.
MUSE_GLIMMER_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    # The chat template's generation prompt is a bare ``<|start|>assistant``; the model
    # then picks a channel (``to=self`` reasoning, ``to=<tool>`` for a call, ``to=user``
    # for the answer) and re-emits ``<|start|>assistant`` for each subsequent channel
    # (harmony-style). ``start_anchor`` is applied only to the PREFIX (the chat prompt),
    # never to the response, so the response's own re-emissions are preserved.
    "start_anchor": "<|start|>assistant",
    "fields": {
        # Private reasoning on the ``to=self`` channel. The model emits at most one
        # ``to=self`` block per turn (verified empirically), so a single string region
        # (matching the standard ``reasoning_content`` contract) is sufficient; no join.
        "reasoning_content": {
            "open_pattern": r"to=self<\|message\|>",
            "close": "<|eom|>",
            "content": "text",
        },
        # Tool calls: one ``<atem:invoke>`` per call, function name in the tag attribute,
        # each ``<atem:parameter>`` parsed as an xml-inline key/value (value as lax JSON
        # so scalars and nested objects/arrays both work). ``repeats`` makes this a list,
        # covering parallel tool calls.
        "tool_calls": {
            "open_pattern": r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">',
            "close": "</atem:invoke>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        # Final answer on the ``to=user`` channel (turn-final). Closes on end-of-turn or
        # end-of-message.
        "content": {
            "open_pattern": r"to=user<\|message\|>",
            "close": ["<|eot|>", "<|eom|>"],
            "content": "text",
        },
    },
}


class MuseGlimmerTokenizerConverter(TikTokenConverter):
    def __init__(self, vocab_file: str):
        super().__init__(vocab_file, pattern=O200K_PATTERN)
        self.additional_special_tokens = _ordered_special_tokens()
        tokenizer_obj = self.converted()

        self.converted_tokenizer = TokenizersBackend(
            tokenizer_object=tokenizer_obj,
            additional_special_tokens=self.additional_special_tokens,
            model_max_length=131072,
        )
        self.converted_tokenizer.bos_token = "<|begin_of_text|>"
        self.converted_tokenizer.eos_token = "<|end_of_text|>"
        self.converted_tokenizer.pad_token = "<|finetune_right_pad|>"
        self.converted_tokenizer.chat_template = MUSE_GLIMMER_MM_CHAT_TEMPLATE
        # Bake the output-side parser so ``tokenizer.parse_response`` can turn generated
        # XML_ATEM output back into a structured message. Persisted into
        # tokenizer_config.json on save (PreTrainedTokenizerBase.save_pretrained).
        self.converted_tokenizer.response_template = MUSE_GLIMMER_RESPONSE_TEMPLATE

        bos_id = self.converted_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        self.converted_tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
            single="<|begin_of_text|> $A",
            pair="<|begin_of_text|>:0 $A:0 <|begin_of_text|>:1 $B:1",
            special_tokens=[("<|begin_of_text|>", bos_id)],
        )


def convert_tokenizer(tokenizer_path: Path, output_dir: Path) -> None:
    converter = MuseGlimmerTokenizerConverter(str(tokenizer_path))
    tokenizer = converter.converted_tokenizer

    expected_ids = {name: NUM_BASE_TOKENS + i for i, name in enumerate(_ordered_special_tokens())}
    for name in ("<|begin_of_text|>", "<|end_of_text|>", "<|eom|>", "<|eot|>", "<|image|>", "<|patch|>", "<|video|>"):
        got = tokenizer.convert_tokens_to_ids(name)
        want = expected_ids[name]
        if got != want:
            raise RuntimeError(f"Special-token id mismatch for {name!r}: got {got}, want {want}")

    processor = MuseGlimmerProcessor(
        image_processor=MuseGlimmerImageProcessor(),
        video_processor=MuseGlimmerVideoProcessor(return_metadata=True),
        tokenizer=tokenizer,
        chat_template=MUSE_GLIMMER_MM_CHAT_TEMPLATE,
    )
    processor.save_pretrained(output_dir)


def write_generation_config(output_dir: Path) -> None:
    # eos is <|end_of_text|> (200001) and <|eot|> (200008, end of turn). <|eom|>
    # (200007) looks like a third stop but isn't one: it separates messages within a
    # turn (it closes the to=self reasoning, and each non-final call in a parallel
    # batch), and the model keeps generating after it to reach the tool call and the
    # final answer. Put it in eos and generation dies right after the reasoning block,
    # before any tool call. The reference decoder likewise keeps <|eom|> off the
    # end-of-generation set and stops on <|end_of_text|>. pad is 200018.
    gen_config = {
        "bos_token_id": 200000,
        "eos_token_id": [200001, 200008],
        "pad_token_id": 200018,
        "max_length": 131072,
        "do_sample": False,
    }
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to the LLM TP=1 shard, e.g. pytorch_weights/llm/mp_1/model_0.pt",
    )
    parser.add_argument(
        "--vision_checkpoint_path",
        type=Path,
        required=True,
        help="Path to the vision-encoder TP=1 shard, e.g. pytorch_weights/encoder/mp_1/<name>.pt",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        required=True,
        help="Path to the tiktoken vocab file, e.g. pytorch_weights/tokenizer/l4_200k_base",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Directory to write the HF checkpoint into.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Dtype for the emitted model weights.",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    config = build_config()

    print(f"Loading LLM shard from {args.checkpoint_path}...")
    llm_state = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)

    state_dict: dict[str, torch.Tensor] = convert_text_state_dict(llm_state, config)

    print(f"Loading vision shard from {args.vision_checkpoint_path}...")
    vision_state = torch.load(args.vision_checkpoint_path, map_location="cpu", weights_only=True)
    state_dict.update(convert_vision_state_dict(vision_state, config))

    print(f"Materialising {MuseGlimmerForConditionalGeneration.__name__} on meta device...")
    with torch.device("meta"):
        model = MuseGlimmerForConditionalGeneration(config)
    model = model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing:
        raise RuntimeError(f"Missing keys after load: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys after load: {unexpected}")
    model = model.to(dtype)

    print(f"Saving model to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_path, safe_serialization=True)

    print("Converting tokenizer...")
    convert_tokenizer(args.tokenizer_path, args.output_path)

    print("Writing generation_config.json (eos = [200001, 200008], drops <|eom|>)...")
    write_generation_config(args.output_path)

    print("Done.")


if __name__ == "__main__":
    main()
