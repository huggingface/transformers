# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
"""
Converts the original Voxtral TTS checkpoint (`params.json`, `consolidated.safetensors`, `tekken.json` and the
`voice_embedding/*.pt` presets) to the Transformers format.

```bash
python src/transformers/models/voxtral_tts/convert_voxtral_tts_weights_to_hf.py \
    --input_path_or_repo mistralai/Voxtral-4B-TTS-2603 \
    --output_dir ./voxtral-tts-hf
```
"""

import argparse
import json
import os
import re

import torch
from huggingface_hub import list_repo_files
from safetensors import safe_open

from transformers import MistralCommonBackend, VoxtralTtsConfig, VoxtralTtsForTextToSpeech, VoxtralTtsProcessor
from transformers.utils.hub import cached_file


# fmt: off
STATE_DICT_MAPPING = {
    # Language model
    r"^mm_audio_embeddings\.tok_embeddings\.weight$":                         r"model.language_model.embed_tokens.weight",
    r"^norm\.weight$":                                                        r"model.language_model.norm.weight",
    r"^layers\.(\d+)\.attention\.w([qkvo])\.weight$":                         r"model.language_model.layers.\1.self_attn.\2_proj.weight",
    r"^layers\.(\d+)\.attention_norm\.weight$":                               r"model.language_model.layers.\1.input_layernorm.weight",
    r"^layers\.(\d+)\.ffn_norm\.weight$":                                     r"model.language_model.layers.\1.post_attention_layernorm.weight",
    r"^layers\.(\d+)\.feed_forward\.w1\.weight$":                             r"model.language_model.layers.\1.mlp.gate_proj.weight",
    r"^layers\.(\d+)\.feed_forward\.w2\.weight$":                             r"model.language_model.layers.\1.mlp.down_proj.weight",
    r"^layers\.(\d+)\.feed_forward\.w3\.weight$":                             r"model.language_model.layers.\1.mlp.up_proj.weight",
    r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.weight$":  r"model.audio_embeddings.embed_audio_tokens.weight",

    # Acoustic (flow-matching) transformer
    r"^acoustic_transformer\.layers\.(\d+)\.attention\.w([qkvo])\.weight$":   r"acoustic_transformer.layers.\1.self_attn.\2_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.attention_norm\.weight$":         r"acoustic_transformer.layers.\1.input_layernorm.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.ffn_norm\.weight$":               r"acoustic_transformer.layers.\1.post_attention_layernorm.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w1\.weight$":       r"acoustic_transformer.layers.\1.mlp.gate_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w2\.weight$":       r"acoustic_transformer.layers.\1.mlp.down_proj.weight",
    r"^acoustic_transformer\.layers\.(\d+)\.feed_forward\.w3\.weight$":       r"acoustic_transformer.layers.\1.mlp.up_proj.weight",
    r"^acoustic_transformer\.norm\.weight$":                                  r"acoustic_transformer.norm.weight",
    r"^acoustic_transformer\.(llm|time|input)_projection\.weight$":           r"acoustic_transformer.\1_projection.weight",
    r"^acoustic_transformer\.semantic_codebook_output\.weight$":              r"acoustic_transformer.semantic_head.weight",
    r"^acoustic_transformer\.acoustic_codebook_output\.weight$":              r"acoustic_transformer.acoustic_head.weight",

    # Codec decoder (`decoder_blocks` alternate an (up)sampling convolution and a transformer: 2*i -> stage i)
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.conv\.weight$":                r"codec_model.stages.<conv:\1>.conv.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention\.w([qkvo])\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.self_attn.\3_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention\.([qk])_norm\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.self_attn.\3_norm.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention_norm\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.input_layernorm.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.ffn_norm\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.post_attention_layernorm.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.attention_scale$": r"codec_model.stages.<layers:\1>.layers.\2.self_attn_layer_scale",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.ffn_scale$":    r"codec_model.stages.<layers:\1>.layers.\2.mlp_layer_scale",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w1\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.mlp.gate_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w2\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.mlp.down_proj.weight",
    r"^audio_tokenizer\.decoder_blocks\.(\d+)\.layers\.(\d+)\.feed_forward\.w3\.weight$": r"codec_model.stages.<layers:\1>.layers.\2.mlp.up_proj.weight",
    r"^audio_tokenizer\.output_proj\.conv\.weight$":                          r"codec_model.output_proj.weight",
    r"^audio_tokenizer\.quantizer\.semantic_codebook\.(cluster_usage|embedding_sum)$": r"codec_model.quantizer.\1",
}
# fmt: on

# Codec encoder weights are not part of the released checkpoint, but are skipped if present
SKIPPED_KEYS = re.compile(r"^audio_tokenizer\.(input_proj|encoder_blocks)\.")


def map_old_key_to_new(old_key: str) -> str:
    for pattern, replacement in STATE_DICT_MAPPING.items():
        new_key, n_replace = re.subn(pattern, replacement, old_key)
        if n_replace:
            # Decoder block 2*i is the convolution opening stage i, block 2*i + 1 its transformer layers
            new_key = re.sub(r"<conv:(\d+)>", lambda m: str(int(m.group(1)) // 2), new_key)
            new_key = re.sub(r"<layers:(\d+)>", lambda m: str((int(m.group(1)) - 1) // 2), new_key)
            return new_key
    raise ValueError(f"Key {old_key} could not be mapped (check the mapping).")


def permute_for_rope(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """Permutes q/k projections from the interleaved (original) to the half-split (Transformers) RoPE layout."""
    return tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replaces `weight_norm` parametrizations (g=original0, v=original1) by the weight they compute."""
    folded = {}
    for key, tensor in state_dict.items():
        if key.endswith(".parametrizations.weight.original0"):
            prefix = key[: -len(".parametrizations.weight.original0")]
            v = state_dict[f"{prefix}.parametrizations.weight.original1"]
            # Same computation (and dtype) as the parametrization in the original model
            folded[f"{prefix}.weight"] = torch._weight_norm(v, tensor, 0)
        elif not key.endswith(".parametrizations.weight.original1"):
            folded[key] = tensor
    return folded


def convert_config(params: dict) -> VoxtralTtsConfig:
    multimodal = params["multimodal"]
    audio_model_args = multimodal["audio_model_args"]
    acoustic_args = audio_model_args["acoustic_transformer_args"]
    codec_args = multimodal["audio_tokenizer_args"]

    text_config = {
        "vocab_size": params["vocab_size"],
        "hidden_size": params["dim"],
        "intermediate_size": params["hidden_dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "head_dim": params["head_dim"],
        "rms_norm_eps": params["norm_eps"],
        "rope_theta": params["rope_theta"],
        "max_position_embeddings": params.get("max_position_embeddings", params.get("max_seq_len", 128000)),
        "sliding_window": params.get("sliding_window"),
        "tie_word_embeddings": params.get("tied_embeddings", True),
        "bos_token_id": multimodal.get("bos_token_id", 1),
    }
    acoustic_transformer_config = {
        "hidden_size": acoustic_args["dim"],
        "intermediate_size": acoustic_args["hidden_dim"],
        "num_hidden_layers": acoustic_args["n_layers"],
        "num_attention_heads": acoustic_args["n_heads"],
        "num_key_value_heads": acoustic_args["n_kv_heads"],
        "head_dim": acoustic_args["head_dim"],
        "rms_norm_eps": acoustic_args.get("norm_eps", 1e-5),
        "llm_hidden_size": acoustic_args["input_dim"],
        "semantic_codebook_size": audio_model_args["semantic_codebook_size"],
        "acoustic_codebook_size": audio_model_args["acoustic_codebook_size"],
        "num_acoustic_codebooks": audio_model_args["n_acoustic_codebook"],
        # Not in `params.json`: the reference implementation (vllm-omni) defaults to 7 Euler steps
        "num_flow_matching_steps": acoustic_args.get("n_decoding_steps") or 7,
    }

    def str_to_list(value: str) -> list[int]:
        return [int(i) for i in value.split(",")]

    if codec_args["n_kv_heads"] != codec_args["n_heads"] or not codec_args["causal"]:
        raise ValueError("Only causal codecs with as many key/value heads as query heads are supported.")
    if not codec_args.get("half_attn_window_upon_downsampling", True):
        raise ValueError("Only codecs halving the attention window upon downsampling are supported.")
    codec_config = {
        "sampling_rate": codec_args["sampling_rate"],
        "patch_size": codec_args["pretransform_patch_size"],
        "patch_proj_kernel_size": codec_args["patch_proj_kernel_size"],
        "semantic_codebook_size": codec_args["semantic_codebook_size"],
        "semantic_dim": codec_args["semantic_dim"],
        "acoustic_codebook_size": codec_args["acoustic_codebook_size"],
        "acoustic_dim": codec_args["acoustic_dim"],
        "hidden_size": codec_args["dim"],
        "intermediate_size": codec_args["hidden_dim"],
        "num_attention_heads": codec_args["n_heads"],
        "num_key_value_heads": codec_args["n_kv_heads"],
        "head_dim": codec_args["head_dim"],
        "rms_norm_eps": codec_args["norm_eps"],
        "qk_norm_eps": codec_args["qk_norm_eps"],
        "layer_scale_init": codec_args["layer_scale_init"],
        "sliding_window": codec_args["attn_sliding_window_size"],
        "decoder_num_layers": str_to_list(codec_args["decoder_transformer_lengths_str"]),
        "decoder_kernel_sizes": str_to_list(codec_args["decoder_convs_kernels_str"]),
        "decoder_strides": str_to_list(codec_args["decoder_convs_strides_str"]),
    }

    num_special = 2  # [EMPTY_AUDIO], [END_AUDIO]
    codebook_sizes = [audio_model_args["semantic_codebook_size"] + num_special] + [
        audio_model_args["acoustic_codebook_size"] + num_special
    ] * audio_model_args["n_acoustic_codebook"]
    return VoxtralTtsConfig(
        text_config=text_config,
        acoustic_transformer_config=acoustic_transformer_config,
        codec_config=codec_config,
        audio_token_id=audio_model_args["audio_token_id"],
        begin_audio_token_id=audio_model_args["begin_audio_token_id"],
        audio_vocab_size=128 * ((sum(codebook_sizes) + 127) // 128),
        tie_word_embeddings=False,
    )


def convert_state_dict(original_state_dict: dict[str, torch.Tensor], config: VoxtralTtsConfig) -> dict:
    text_config = config.text_config
    new_state_dict = {}
    for old_key, tensor in fold_weight_norm(original_state_dict).items():
        if SKIPPED_KEYS.match(old_key):
            continue
        new_key = map_old_key_to_new(old_key)
        if new_key.startswith("model.language_model.") and re.search(r"self_attn\.[qk]_proj", new_key):
            n_heads = text_config.num_attention_heads if ".q_proj" in new_key else text_config.num_key_value_heads
            tensor = permute_for_rope(tensor, n_heads, n_heads * text_config.head_dim, text_config.hidden_size)
        new_state_dict[new_key] = tensor
    return new_state_dict


def convert_voice_embeddings(input_path_or_repo: str, voices: dict) -> dict[str, torch.Tensor]:
    voice_embeddings = {}
    for voice in sorted(voices, key=voices.get):
        path = cached_file(input_path_or_repo, f"voice_embedding/{voice}.pt")
        voice_embeddings[voice] = torch.load(path, map_location="cpu", weights_only=True).contiguous()
    return voice_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_or_repo", default="mistralai/Voxtral-4B-TTS-2603")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", default=None, help="Optional repo id to push the converted model to.")
    args = parser.parse_args()

    with open(cached_file(args.input_path_or_repo, "params.json"), encoding="utf-8") as f:
        params = json.load(f)
    config = convert_config(params)

    original_state_dict = {}
    with safe_open(cached_file(args.input_path_or_repo, "consolidated.safetensors"), framework="pt") as f:
        for key in f.keys():
            original_state_dict[key] = f.get_tensor(key)
    state_dict = convert_state_dict(original_state_dict, config)
    del original_state_dict

    with torch.device("meta"):
        model = VoxtralTtsForTextToSpeech(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.output_dir)
    del model, state_dict

    tokenizer = MistralCommonBackend.from_pretrained(args.input_path_or_repo)
    voices = params["multimodal"]["audio_tokenizer_args"].get("voice", {})
    if os.path.isdir(args.input_path_or_repo):
        available = {
            os.path.basename(p)[:-3] for p in os.listdir(os.path.join(args.input_path_or_repo, "voice_embedding"))
        }
    else:
        available = {
            os.path.basename(p)[:-3]
            for p in list_repo_files(args.input_path_or_repo)
            if p.startswith("voice_embedding/")
        }
    voices = {voice: idx for voice, idx in voices.items() if voice in available}
    processor = VoxtralTtsProcessor(
        tokenizer=tokenizer, voice_embeddings=convert_voice_embeddings(args.input_path_or_repo, voices)
    )
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub:
        VoxtralTtsForTextToSpeech.from_pretrained(args.output_dir).push_to_hub(args.push_to_hub)
        VoxtralTtsProcessor.from_pretrained(args.output_dir).push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
