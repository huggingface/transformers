# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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
"""Convert the ZONOS3-Mini release (`Zyphra/ZONOS3-Mini-hf`) to transformers: the LM and the speaker encoder, the
tokenizer, the processor (holding the ZAC-1 audio tokenizer converted by `convert_zac_to_hf.py`), the voices and the
generation config. The codec tensors of the release are not used: the codec is ZAC-1 (`Zyphra/ZAC-1`).

The conversion runs offline: the release stores fused GDN projections, the attention keys and values as one stacked
tensor, the dense feed-forward as a stacked `[2, hidden, dim]` tensor, and reuses `attention.wo` for both mixers.

    python src/transformers/models/zonos3_mini/convert_zonos3_mini_to_hf.py --hf_repo_dir ZONOS3-Mini-hf \
        --audio_tokenizer_dir zac-1-hf --output_dir zonos3-mini-transformers
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from transformers import (
    GenerationConfig,
    TokenizersBackend,
    ZacFeatureExtractor,
    ZacModel,
    Zonos3MiniConfig,
    Zonos3MiniFeatureExtractor,
    Zonos3MiniForConditionalGeneration,
    Zonos3MiniProcessor,
)


# TODO: move to final Zyphra repo
AUDIO_TOKENIZER_REPO = "Zyphra/ZAC-1-hf"

# One `user` message (the text, optionally with the voice clip), optionally followed by the `assistant` target audio to
# train on. The template renders the user text only: the processor lays out the rows (voice, conditions, text, silence).
CHAT_TEMPLATE = """
{%- set training = messages | length == 2 -%}
{%- if messages | length > 2 or messages[0]['role'] != 'user' or (training and messages[1]['role'] != 'assistant') -%}
    {{- raise_exception('ZONOS3-Mini takes one `user` message per conversation, optionally followed by the `assistant` target audio to train on.') -}}
{%- endif -%}
{%- if training -%}
    {%- set target = messages[1]['content'] -%}
    {%- if target is string or target | length != 1 or target[0]['type'] != 'audio' -%}
        {{- raise_exception('The `assistant` turn holds one `audio` item, the target.') -}}
    {%- endif -%}
{%- endif -%}
{%- set content = messages[0]['content'] -%}
{%- if content is string -%}
    {%- set texts = [content] -%}
    {%- set num_audio = 0 -%}
{%- else -%}
    {%- set texts = content | selectattr('type', 'equalto', 'text') | map(attribute='text') | list -%}
    {%- set num_audio = content | selectattr('type', 'equalto', 'audio') | list | length -%}
{%- endif -%}
{%- if texts | length != 1 or num_audio > 1 -%}
    {{- raise_exception('The `user` turn holds one `text` item and at most one `audio` item (the voice).') -}}
{%- endif -%}
{{- texts[0] -}}
"""

ATTENTION_RENAMES = {"wq": "q_proj", "wo": "o_proj", "gater": "g_proj"}
MOE_RENAMES = {
    "router.router_linear.weight": "gate.weight",
    "router.rmsnorm_eda.weight": "gate.norm.weight",
    "router.balancing_biases": "gate.balancing_biases",
    "latent_down.weight": "fc1_latent_proj.weight",
    "latent_up.weight": "fc2_latent_proj.weight",
    "experts.gate_up_proj": "experts.gate_up_proj",
    "experts.down_proj": "experts.down_proj",
}
SPEAKER_RENAMES = [
    (r"^feature_extractor\.head\.0\.", "stem_conv."),
    (r"^feature_extractor\.head\.1\.", "stem_norm."),
    (r"^feature_extractor\.dense3\.inputs_weights$", "stage_weights"),
    (r"^feature_extractor\.mect_block(\d+)\.", lambda m: f"stages.{int(m[1]) - 1}."),
    (r"\.cnn_block\.", ".res_blocks."),
    (r"\.shortcut\.0\.", ".shortcut.convolution."),
    (r"\.shortcut\.1\.", ".shortcut.normalization."),
    (r"\.reduce_conv\.0\.", ".reduce_conv."),
    (r"\.reduce_conv\.1\.", ".reduce_norm."),
    (r"\.reduce_project\.0\.", ".reduce_proj."),
    (r"\.reduce_project\.1\.", ".reduce_layer_norm."),
    (r"\.pos_conv\.pos_conv\.0\.", ".pos_conv.conv."),
    (r"\.pos_conv\.pos_conv\.1\.", ".pos_conv.norm."),
    (r"\.pos_conv\.pos_conv\.3\.", ".pos_conv.proj."),
    (r"\.expansion_project\.0\.", ".expansion_proj."),
    (r"\.intermediate_dense\.token_moe\.", ".intermediate_dense."),
    (r"^encoding_layers\.", "pooling."),
    (r"^bn\.", "norm."),
    (r"^linear\.", "proj."),
]


def build_config(release: dict) -> Zonos3MiniConfig:
    lm, speaker = release["lm"], release["speaker_encoder"]
    if lm["text_rows"] != lm["text_vocab"] + 1:
        raise ValueError("The text column's embedding rows are expected to be the vocabulary plus the padding row.")
    return Zonos3MiniConfig(
        vocab_size=lm["text_rows"],
        hidden_size=lm["dim"],
        intermediate_size=lm["ffn_hidden"],
        num_hidden_layers=lm["n_layers"],
        num_attention_heads=lm["n_heads"],
        num_key_value_heads=lm["n_kv_heads"],
        head_dim=lm["head_dim"],
        rms_norm_eps=lm["norm_eps"],
        max_position_embeddings=lm["max_seqlen"],
        rope_parameters={"rope_type": "default", "rope_theta": lm["rope_theta"]},
        layer_types=[
            "linear_attention" if (i + 1) % lm["gdn_period"] else "full_attention" for i in range(lm["n_layers"])
        ],
        mlp_layer_types=["dense"] + ["sparse"] * (lm["n_layers"] - 1),
        linear_conv_kernel_dim=lm["gdn_kernel"],
        linear_key_head_dim=lm["gdn_head_dim"],
        linear_value_head_dim=lm["gdn_head_dim"],
        linear_num_key_heads=lm["gdn_heads"],
        linear_num_value_heads=lm["gdn_heads"],
        moe_intermediate_size=lm["moe_hidden"],
        moe_latent_size=lm["moe_routed_dim"],
        num_experts=lm["n_experts"],
        num_experts_per_tok=lm["moe_topk"],
        num_codebooks=lm["n_codebooks"],
        codebook_size=lm["codebook_size"],
        audio_eos_token_id=lm["eoa_id"],
        audio_pad_token_id=lm["audio_pad_id"],
        text_pad_token_id=lm["text_vocab"],
        speaker_embedding_size=lm["speaker_embedding_dim"],
        condition_bucket_counts=lm["condition_bucket_counts"],
        final_logit_softcapping=lm["loss_softcap"],
        speaker_encoder_config={
            "num_mel_bins": speaker["num_mel_bins"],
            "in_channels": speaker["in_channels"],
            "hidden_size": speaker["hidden_dim"],
            "num_experts": speaker["num_experts"],
            "bottleneck_size": speaker["bottleneck_dim"],
            "embedding_size": speaker["embed_dim"],
            "stage_block_specs": speaker["stages"],
        },
        dtype="bfloat16",
    )


def convert_lm(state: dict[str, torch.Tensor], config: Zonos3MiniConfig) -> dict[str, torch.Tensor]:
    converted = {}
    num_codebooks = config.num_codebooks
    converted["model.embeddings.embed_audio_tokens.weight"] = torch.cat(
        [state.pop(f"multi_embedder.embedders.{i}.weight") for i in range(num_codebooks)]
    )
    converted["model.embeddings.embed_tokens.weight"] = state.pop(f"multi_embedder.embedders.{num_codebooks}.weight")
    converted["model.embeddings.embed_conditions.weight"] = torch.cat(
        [state.pop(f"condition_embeddings.{i}.weight") for i in range(len(config.condition_bucket_counts))]
    )
    converted["model.embeddings.speaker_proj.weight"] = state.pop("speaker_projection.weight")
    converted["model.embeddings.speaker_proj.bias"] = state.pop("speaker_projection.bias")
    converted["model.depth_queries"] = state.pop("block_attnres_queries")
    converted["model.norm.weight"] = state.pop("out_norm.weight")
    converted["lm_head.weight"] = state.pop("multi_output.weight")

    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    num_v_heads = config.linear_num_value_heads
    for name, value in state.items():
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            raise ValueError(f"Unexpected LM tensor {name!r}.")
        layer, rest = int(match[1]), match[2]
        prefix = f"model.layers.{layer}"
        if rest == "attention_norm.weight":
            converted[f"{prefix}.input_layernorm.weight"] = value
        elif rest == "ffn_norm.weight":
            converted[f"{prefix}.post_attention_layernorm.weight"] = value
        elif rest.startswith("attention.") and config.layer_types[layer] == "linear_attention":
            rest = rest[len("attention.") :]
            gdn = f"{prefix}.linear_attn"
            if rest == "in_proj.weight":
                # rows: [q, k, v | z | a | b] (the release splits `(qkv, z, a, b)`)
                qkv, z, a, b = value.split([2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads])
                converted.update(
                    {
                        f"{gdn}.in_proj_qkv.weight": qkv,
                        f"{gdn}.in_proj_z.weight": z,
                        f"{gdn}.in_proj_a.weight": a,
                        f"{gdn}.in_proj_b.weight": b,
                    }
                )
            elif rest == "conv.weight":
                converted[f"{gdn}.conv1d.weight"] = value
            elif rest == "out_weight":
                converted[f"{gdn}.norm.weight"] = value
            elif rest in ("A_log", "dt_bias"):
                converted[f"{gdn}.{rest}"] = value
            elif rest == "wo.weight":
                converted[f"{gdn}.out_proj.weight"] = value
            else:
                raise ValueError(f"Unexpected GDN tensor {name!r}.")
        elif rest.startswith("attention."):
            rest = rest[len("attention.") :]
            attn = f"{prefix}.self_attn"
            if rest == "wkv.weight":
                converted[f"{attn}.k_proj.weight"], converted[f"{attn}.v_proj.weight"] = value.unbind(0)
            elif rest == "temp":
                converted[f"{attn}.temperature"] = value.reshape(-1)
            elif rest.removesuffix(".weight") in ATTENTION_RENAMES:
                converted[f"{attn}.{ATTENTION_RENAMES[rest.removesuffix('.weight')]}.weight"] = value
            else:
                raise ValueError(f"Unexpected attention tensor {name!r}.")
        elif rest.startswith("feed_forward."):
            rest = rest[len("feed_forward.") :]
            mlp = f"{prefix}.mlp"
            if config.mlp_layer_types[layer] == "dense":
                if rest == "w_in.weight":
                    # the release computes `w_in[0](x) * silu(w_in[1](x))`
                    converted[f"{mlp}.up_proj.weight"], converted[f"{mlp}.gate_proj.weight"] = value.unbind(0)
                elif rest == "w_out.weight":
                    converted[f"{mlp}.down_proj.weight"] = value
                else:
                    raise ValueError(f"Unexpected dense feed-forward tensor {name!r}.")
            elif rest in MOE_RENAMES:
                converted[f"{mlp}.{MOE_RENAMES[rest]}"] = value
            else:
                raise ValueError(f"Unexpected MoE tensor {name!r}.")
        else:
            raise ValueError(f"Unexpected LM tensor {name!r}.")
    return converted


def convert_speaker_encoder(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for name, value in state.items():
        new_name = name
        for pattern, replacement in SPEAKER_RENAMES:
            new_name = re.sub(pattern, replacement, new_name)
        converted[f"speaker_encoder.{new_name}"] = value
    return converted


def convert_weights(hf_repo_dir: str, config: Zonos3MiniConfig) -> Zonos3MiniForConditionalGeneration:
    groups = {"lm": {}, "speaker_encoder": {}}
    with safe_open(os.path.join(hf_repo_dir, "model.safetensors"), framework="pt") as handle:
        for name in handle.keys():
            prefix, _, rest = name.partition(".")
            if prefix in groups:
                groups[prefix][rest] = handle.get_tensor(name)
            elif prefix != "codec":
                raise ValueError(f"Unexpected tensor {name!r}.")
    state_dict = convert_lm(groups["lm"], config)
    state_dict.update(convert_speaker_encoder(groups["speaker_encoder"]))

    model = Zonos3MiniForConditionalGeneration(config).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    print(f"Loaded {len(state_dict)} tensors strictly into Zonos3MiniForConditionalGeneration")
    # the release LM is bf16; the speaker encoder (and the routers' balancing biases) stay fp32
    model.model.to(torch.bfloat16)
    model.lm_head.to(torch.bfloat16)
    for name, buffer in model.named_buffers():
        if name.endswith("balancing_biases"):
            buffer.data = state_dict[name].float()
    return model


def convert_tokenizer(hf_repo_dir: str, output_dir: str) -> TokenizersBackend:
    tokenizer = TokenizersBackend(
        tokenizer_file=os.path.join(hf_repo_dir, "tokenizer.json"),
        bos_token="<|zonos_bos|>",
        eos_token="<|zonos_eos|>",
        pad_token="<|zonos_pad|>",
        unk_token="<|zonos_unk|>",
        padding_side="left",
        model_max_length=32768,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.save_pretrained(output_dir)
    return tokenizer


def convert_voices(hf_repo_dir: str, output_dir: str):
    """One `voices/<Name>.safetensors` per release voice: its reference codes and its speaker embedding."""
    voices_dir = os.path.join(output_dir, "voices")
    os.makedirs(voices_dir, exist_ok=True)
    with open(os.path.join(hf_repo_dir, "voices.json"), encoding="utf-8") as f:
        index = json.load(f)
    with safe_open(os.path.join(hf_repo_dir, "voices.safetensors"), framework="pt") as handle:
        for name in index:
            save_file(
                {
                    "codes": handle.get_tensor(f"{name}.codes").to(torch.int32).contiguous(),
                    "embedding": handle.get_tensor(f"{name}.embedding").float().contiguous(),
                },
                os.path.join(voices_dir, f"{name}.safetensors"),
            )
    shutil.copy(os.path.join(hf_repo_dir, "voices.json"), os.path.join(voices_dir, "voices.json"))
    print(f"Saved {len(index)} voices to {voices_dir}")


def convert(hf_repo_dir: str, output_dir: str, audio_tokenizer_dir: str, audio_tokenizer_repo: str, push_to_hub):
    with open(os.path.join(hf_repo_dir, "config.json"), encoding="utf-8") as f:
        release = json.load(f)
    config = build_config(release)
    model = convert_weights(hf_repo_dir, config)

    generation = release["generation"]
    lm = release["lm"]
    model.generation_config = GenerationConfig(
        do_sample=True,
        temperature=generation["temperature"],
        top_k=generation["topk"],
        top_p=generation["top_p"],
        repetition_penalty=generation["repetition_penalty"],
        repetition_window=generation["repetition_window"],
        repetition_codebooks=generation["repetition_codebooks"],
        max_new_tokens=2000,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
    model.save_pretrained(output_dir)

    tokenizer = convert_tokenizer(hf_repo_dir, output_dir)
    audio_tokenizer = ZacModel.from_pretrained(audio_tokenizer_dir)
    # what the processor's `processor_config.json` points to
    audio_tokenizer.name_or_path = audio_tokenizer_repo
    processor = Zonos3MiniProcessor(
        Zonos3MiniFeatureExtractor(
            sampling_rate=release["sampling_rate"],
            speaker_sampling_rate=release["speaker_encoder"]["sample_rate"],
            num_mel_bins=release["speaker_encoder"]["num_mel_bins"],
            max_seconds=release["speaker_encoder"]["max_seconds"],
            waveform_scale=release["speaker_encoder"]["waveform_scale"],
        ),
        tokenizer,
        ZacFeatureExtractor.from_pretrained(audio_tokenizer_dir),
        audio_tokenizer=audio_tokenizer,
        chat_template=CHAT_TEMPLATE.strip() + "\n",
        languages=generation["languages"],
        markup_tags=generation["markup_tags"],
        condition_names=lm["condition_names"],
        condition_edges=lm["condition_edges"],
        condition_bucket_counts=lm["condition_bucket_counts"],
        silence_frame=generation["silence_frame"],
        max_silence_prefill_frames=generation["silence_prefill_max_frames"],
        num_codebooks=lm["n_codebooks"],
        codebook_size=lm["codebook_size"],
        audio_eos_token_id=lm["eoa_id"],
        audio_pad_token_id=lm["audio_pad_id"],
        text_pad_token_id=lm["text_vocab"],
        reference_max_frames=lm["reference_max_frames"],
        overlap_frames=generation["overlap_frames"],
        decode_context_frames=generation["decode_context_frames"],
        headroom_db=generation["headroom_db"],
        output_limit_dbfs=generation["output_limit_dbfs"],
    )
    processor.save_pretrained(output_dir)
    convert_voices(hf_repo_dir, output_dir)
    print(f"Saved to {output_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf_repo_dir", required=True, help="Local snapshot of `Zyphra/ZONOS3-Mini-hf`.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model and processor.")
    parser.add_argument(
        "--audio_tokenizer_dir", required=True, help="Local ZAC-1 checkpoint converted with `convert_zac_to_hf.py`."
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        default=AUDIO_TOKENIZER_REPO,
        help="Hub repo of the audio tokenizer the saved processor points to.",
    )
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push the converted model to.")
    args = parser.parse_args()
    convert(args.hf_repo_dir, args.output_dir, args.audio_tokenizer_dir, args.audio_tokenizer_repo, args.push_to_hub)
