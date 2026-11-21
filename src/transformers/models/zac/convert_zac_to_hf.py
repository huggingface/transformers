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
"""Convert the ZAC-1 release files (`Zyphra/ZAC-1`, format `zac-1/1`) to a transformers `ZacModel`.

The encoder file holds the encoder, the semantic branch, the quantizers and the Wav2Vec2-BERT teacher; each decoder file
(`keep2_plain` or `full_mr`) holds the decoder stages and the iSTFT head. Their configs are JSON stage specs stored in
the safetensors metadata, and the weight-normed convolutions are stored unfolded, so the conversion runs offline.

    python src/transformers/models/zac/convert_zac_to_hf.py --variant keep2_plain --output_dir zac-1-hf
"""

import argparse
import json
import os
import re

import torch
from safetensors import safe_open

from transformers import ZacConfig, ZacFeatureExtractor, ZacModel


FORMAT = "zac-1/1"
REPO_ID = "Zyphra/ZAC-1"
REVISION = "0744aff6858a33f83e3946f0d26c050659906b97"
ENCODER_FILE = "zac-1-encoder.safetensors"
DECODER_FILES = {
    "keep2_plain": "zac-1-decoder-keep2_plain.safetensors",
    "full_mr": "zac-1-decoder-full_mr.safetensors",
}
TEACHER_LAYERS = 16

# Renames of the transformer layers of `zac/moss.py::MossLayer` (`qkv` is split separately)
LAYER_RENAMES = {
    "norm1": "input_layernorm",
    "norm2": "post_attention_layernorm",
    "attention_out": "self_attn.o_proj",
    "attention_scale": "self_attn_layer_scale.scale",
    "ffn_scale": "mlp_layer_scale.scale",
    "ffn.0": "mlp.fc1",
    "ffn.3": "mlp.fc2",
}


def read_checkpoint(
    path: str, kind: str, variant: str | None = None
) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    """`(metadata, {group: {name: tensor}})` of a `zac-1/1` file (`zac/checkpoint.py::read_checkpoint`)."""
    groups = {"codec": {}, "teacher": {}}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} checkpoint (format {metadata.get('format')!r}).")
        if metadata.get("kind") != kind:
            raise ValueError(f"{path} is a ZAC-1 {metadata.get('kind')!r} checkpoint; expected {kind!r}.")
        if variant is not None and metadata.get("variant") != variant:
            raise ValueError(f"{path} holds the {metadata.get('variant')!r} decoder, not {variant!r}.")
        for name in handle.keys():
            prefix, _, rest = name.partition(".")
            if prefix not in groups or not rest:
                raise ValueError(f"Unexpected tensor {name!r} in {path}.")
            groups[prefix][rest] = handle.get_tensor(name)
    return metadata, groups


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold `weight_g`/`weight_v` and `parametrizations.weight.original0/1` pairs into plain weights (norm over dim 0)."""
    folded = {}
    for name, tensor in state.items():
        for g_suffix, v_suffix in (("weight_g", "weight_v"), ("parametrizations.weight.original0", None)):
            if name.endswith(g_suffix):
                prefix = name[: -len(g_suffix)]
                v_name = prefix + (v_suffix or "parametrizations.weight.original1")
                v = state[v_name]
                folded[prefix + "weight"] = tensor * v / v.norm(dim=tuple(range(1, v.ndim)), keepdim=True)
                break
        else:
            if not name.endswith(("weight_v", "parametrizations.weight.original1")):
                folded[name] = tensor
    return folded


def stage_specs(specs: list[dict], kind: str) -> list[dict]:
    return [spec for spec in specs if spec["type"] == kind]


def build_config(model_config: dict, head_config: dict, teacher_config: dict) -> ZacConfig:
    if head_config.get("type") != "mr_istft":
        raise ValueError(f"Unsupported head type {head_config.get('type')!r}; ZAC-1 decoders use 'mr_istft'.")
    encoder_stages = model_config["encoder_stages"]
    convs = stage_specs(encoder_stages, "S1CausalWNConv1d")
    blocks = stage_specs(encoder_stages, "S1EncoderBlock")
    patches = stage_specs(encoder_stages, "PatchedPretransform")
    encoder_transformers = stage_specs(encoder_stages, "MossProjectedTransformer")
    # decoder stages 0..8: transformers at even indices, x2 un-patches at odd ones (the S1 tail is not used)
    decoder_stages = model_config["decoder_stages"][:9]
    decoder_transformers = decoder_stages[0::2]
    unpatches = decoder_stages[1::2]
    if {spec["patch_size"] for spec in unpatches} != {2}:
        raise ValueError("ZAC-1 decoders un-patch by 2 between transformer stages.")
    stage_layers = head_config.get("stage_layers", {})
    teacher_config = {
        k: v for k, v in teacher_config.items() if k not in ("dtype", "architectures", "transformers_version")
    }
    teacher_config["num_hidden_layers"] = TEACHER_LAYERS
    dual = model_config["dual_encoding"]
    return ZacConfig(
        semantic_model_config=teacher_config,
        sampling_rate=model_config["sample_rate"],
        internal_sampling_rate=model_config["internal_sample_rate"],
        hop_length=model_config["external_samples_per_frame"],
        internal_hop_length=model_config["internal_hop_length"],
        num_quantizers=1 + model_config["residual_codebooks"],
        codebook_size=model_config["residual_codebook_size"],
        codebook_dim=model_config["codebook_dim"],
        hidden_size=encoder_transformers[-1]["output_dimension"],
        semantic_hidden_size=dual["width"],
        semantic_intermediate_size=dual["hidden"],
        semantic_num_layers=dual["depth"],
        semantic_window_ratio=dual["window_ratio"],
        encoder_hidden_size=convs[0]["out_channels"],
        encoder_strides=[spec["stride"] for spec in blocks],
        encoder_patch_sizes=[spec["patch_size"] for spec in patches],
        encoder_input_sizes=[spec["input_dimension"] for spec in encoder_transformers],
        encoder_hidden_sizes=[spec["d_model"] for spec in encoder_transformers],
        encoder_output_sizes=[spec["output_dimension"] for spec in encoder_transformers],
        encoder_num_attention_heads=[spec["num_heads"] for spec in encoder_transformers],
        encoder_num_hidden_layers=[spec["num_layers"] for spec in encoder_transformers],
        encoder_intermediate_sizes=[spec["dim_feedforward"] for spec in encoder_transformers],
        encoder_sliding_windows=[spec["context_frames"] for spec in encoder_transformers],
        decoder_unpatch_size=2,
        decoder_input_sizes=[spec["input_dimension"] for spec in decoder_transformers],
        decoder_hidden_sizes=[spec["d_model"] for spec in decoder_transformers],
        decoder_output_sizes=[spec["output_dimension"] for spec in decoder_transformers[:-1]] + [None],
        decoder_num_attention_heads=[spec["num_heads"] for spec in decoder_transformers],
        decoder_num_hidden_layers=[
            len(stage_layers[str(2 * k)]) if str(2 * k) in stage_layers else spec["num_layers"]
            for k, spec in enumerate(decoder_transformers)
        ],
        decoder_intermediate_sizes=[spec["dim_feedforward"] for spec in decoder_transformers],
        decoder_sliding_windows=[spec["context_frames"] for spec in decoder_transformers],
        head_dim=decoder_transformers[0]["d_model"] // decoder_transformers[0]["num_heads"],
        layer_scale_initial_scale=encoder_transformers[0]["layer_scale"],
        head_num_layers=head_config["depth"],
        head_intermediate_size=head_config["hidden"],
        head_hop_length=head_config["hop"],
        head_n_ffts=head_config["n_ffts"],
        head_mixer_kernel_size=head_config.get("fir_kernel", 127),
        resample_lowpass_filter_width=model_config["boundary_resampler"]["lowpass_filter_width"],
        resample_rolloff=model_config["boundary_resampler"]["rolloff"],
    )


def convert_layer(rest: str, value: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    """A `MossLayer` tensor `layers.{j}.{name}` -> transformers keys under `prefix`."""
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
    layer, name = match.groups()
    layer_prefix = f"{prefix}.layers.{layer}"
    if name == "qkv.weight":
        q, k, v = value.chunk(3, dim=0)
        return {f"{layer_prefix}.self_attn.{n}_proj.weight": t for n, t in zip("qkv", (q, k, v))}
    for old, new in LAYER_RENAMES.items():
        if name == old or name.startswith(old + "."):
            return {f"{layer_prefix}.{new}{name[len(old) :]}": value}
    raise ValueError(f"Unexpected transformer layer tensor {rest!r}.")


def convert_stage(rest: str, value: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    if rest.startswith(("input_proj.", "output_proj.")):
        return {f"{prefix}.{rest}": value}
    return convert_layer(rest, value, prefix)


def convert_convnext(rest: str, value: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    if rest.startswith("dwconv."):
        rest = "dwconv.conv." + rest[len("dwconv.") :]
    return {f"{prefix}.{rest}": value}


def convert_encoder_codec(state: dict[str, torch.Tensor], config: ZacConfig) -> dict[str, torch.Tensor]:
    converted = {}
    num_blocks = len(config.encoder_strides)
    first_stage = 2 + num_blocks  # conv_in, blocks, then (patch, transformer) pairs
    semantic_layers = config.semantic_num_layers
    for name, value in state.items():
        if name == "to_internal.kernel":
            continue
        if match := re.fullmatch(r"encoder\.0\.conv\.(.+)", name):
            converted[f"encoder.conv_in.conv.{match[1]}"] = value
        elif match := re.fullmatch(r"encoder\.(\d+)\.module\.block\.(\d+)\.module\.block\.(\d)\.(.+)", name):
            block, unit, index, rest = int(match[1]) - 1, int(match[2]), int(match[3]), match[4]
            module = {0: "snake1", 1: "conv1.conv", 2: "snake2", 3: "conv2.conv"}[index]
            rest = rest.removeprefix("conv.")
            converted[f"encoder.blocks.{block}.res_unit{unit + 1}.{module}.{rest}"] = value
        elif match := re.fullmatch(r"encoder\.(\d+)\.module\.block\.3\.alpha", name):
            converted[f"encoder.blocks.{int(match[1]) - 1}.snake1.alpha"] = value
        elif match := re.fullmatch(r"encoder\.(\d+)\.module\.block\.4\.conv\.(.+)", name):
            converted[f"encoder.blocks.{int(match[1]) - 1}.conv1.conv.{match[2]}"] = value
        elif match := re.fullmatch(r"encoder\.(\d+)\.(.+)", name):
            index = int(match[1])
            if index < first_stage or (index - first_stage) % 2:
                raise ValueError(f"Unexpected encoder tensor {name!r}.")
            converted.update(convert_stage(match[2], value, f"encoder.stages.{(index - first_stage) // 2}"))
        elif match := re.fullmatch(r"semantic_encoder\.(\d+)\.(.+)", name):
            index, rest = int(match[1]), match[2]
            if index == 0:
                converted[f"semantic_encoder.in_proj.{rest}"] = value
            else:
                converted.update(convert_convnext(rest, value, f"semantic_encoder.layers.{index - 1}"))
        elif match := re.fullmatch(r"semantic_head\.(\d+)\.(.+)", name):
            index, rest = int(match[1]), match[2]
            if index == semantic_layers:
                converted[f"semantic_head.out_proj.{rest}"] = value
            else:
                converted.update(convert_convnext(rest, value, f"semantic_head.layers.{index}"))
        elif name.startswith("quantizer.semantic_quantizer."):
            converted["semantic_quantizer." + name[len("quantizer.semantic_quantizer.") :]] = value
        elif name.startswith("quantizer.quantizer."):
            converted["acoustic_quantizer." + name[len("quantizer.quantizer.") :]] = value
        else:
            raise ValueError(f"Unexpected encoder tensor {name!r}.")
    return converted


def convert_decoder_codec(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for name, value in state.items():
        if name == "to_external.kernel" or name.startswith(("quantizer.", "semantic_head.")):
            continue
        if match := re.fullmatch(r"stages\.(\d+)\.(.+)", name):
            index = int(match[1])
            if index % 2:
                raise ValueError(f"Unexpected decoder tensor {name!r}.")
            converted.update(convert_stage(match[2], value, f"decoder.stages.{index // 2}"))
        elif match := re.fullmatch(r"head\.blocks\.(\d+)\.(.+)", name):
            converted.update(convert_convnext(match[2], value, f"decoder.head.layers.{match[1]}"))
        elif match := re.fullmatch(r"head\.outs\.(\d+)\.(.+)", name):
            converted[f"decoder.head.branches.{match[1]}.linear.{match[2]}"] = value
        elif name.startswith(("head.norm.", "head.mixer.")):
            converted[f"decoder.{name}"] = value
        else:
            raise ValueError(f"Unexpected decoder tensor {name!r}.")
    return converted


def check_shared(encoder_state: dict[str, torch.Tensor], decoder_state: dict[str, torch.Tensor]):
    """The decoder file repeats the quantizers and the semantic head: they must equal the encoder file's."""
    shared = [name for name in decoder_state if name.startswith(("quantizer.", "semantic_head."))]
    for name in shared:
        if not torch.equal(encoder_state[name], decoder_state[name]):
            raise ValueError(f"{name} differs between the encoder and the decoder files.")
    print(f"{len(shared)} quantizer / semantic head tensors match between the encoder and the decoder files")


def check_kernel(stored: torch.Tensor, resampler: torch.nn.Module, name: str):
    if stored.shape != resampler.kernel.shape or not torch.allclose(stored, resampler.kernel, atol=1e-7, rtol=0):
        raise ValueError(f"The stored {name} kernel differs from the recomputed one.")
    print(f"{name} kernel matches (max abs diff {(stored - resampler.kernel).abs().max().item():.2e})")


def convert(input_dir: str | None, revision: str, variant: str, output_dir: str, push_to_hub: str | None):
    if input_dir is None:
        from huggingface_hub import hf_hub_download

        encoder_path = hf_hub_download(REPO_ID, ENCODER_FILE, revision=revision)
        decoder_path = hf_hub_download(REPO_ID, DECODER_FILES[variant], revision=revision)
    else:
        encoder_path = os.path.join(input_dir, ENCODER_FILE)
        decoder_path = os.path.join(input_dir, DECODER_FILES[variant])

    encoder_metadata, encoder_groups = read_checkpoint(encoder_path, "encoder")
    decoder_metadata, decoder_groups = read_checkpoint(decoder_path, "decoder", variant)
    if json.loads(encoder_metadata["model_config"]) != json.loads(decoder_metadata["model_config"]):
        raise ValueError("The encoder and decoder files hold different codec configs.")
    config = build_config(
        json.loads(decoder_metadata["model_config"]),
        json.loads(decoder_metadata["head_config"]),
        json.loads(encoder_metadata["teacher_config"]),
    )

    encoder_state = fold_weight_norm(encoder_groups["codec"])
    decoder_state = fold_weight_norm(decoder_groups["codec"])
    check_shared(encoder_state, decoder_state)

    state_dict = convert_encoder_codec(encoder_state, config)
    state_dict.update(convert_decoder_codec(decoder_state))
    state_dict.update({f"semantic_model.{k}": v.float() for k, v in encoder_groups["teacher"].items()})

    model = ZacModel(config).float().eval()
    check_kernel(encoder_state["to_internal.kernel"], model.encoder_resampler, "44100 -> 44160 Hz")
    check_kernel(decoder_state["to_external.kernel"], model.decoder_resampler, "44160 -> 44100 Hz")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    print(f"Loaded {len(state_dict)} tensors strictly into ZacModel ({variant})")

    feature_extractor = ZacFeatureExtractor(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        lane_frames=config.lane_frames,
        lane_context_frames=config.lane_context_frames,
    )
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    print(f"Saved to {output_dir}; decoder headroom_db = {decoder_metadata['headroom_db']}")
    if push_to_hub:
        model.push_to_hub(push_to_hub)
        feature_extractor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", default=None, help="Local `Zyphra/ZAC-1` snapshot (default: download).")
    parser.add_argument("--revision", default=REVISION, help="`Zyphra/ZAC-1` revision to download.")
    parser.add_argument("--variant", default="keep2_plain", choices=sorted(DECODER_FILES), help="Decoder variant.")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model.")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push the converted model to.")
    args = parser.parse_args()
    convert(args.input_dir, args.revision, args.variant, args.output_dir, args.push_to_hub)
