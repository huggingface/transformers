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
"""
Convert a Nemotron 3.5 Transcribe checkpoint exported by NeMo (`examples/speechlm2/to_hf.py`: `config.json`,
`llm_backbone/config.json`, `model.safetensors` and the tokenizer files) to the Transformers format.

The weights are converted one tensor at a time and written in shards, as the model does not fit in CPU memory.

    python src/transformers/models/nemotron3_5_transcribe/convert_nemotron3_5_transcribe_to_hf.py \
        --nemo_dir /path/to/nemo_export --output_dir /path/to/output
"""

import argparse
import json
import math
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
    Nemotron3_5TranscribeProcessor,
    NemotronAsrStreamingFeatureExtractor,
)


# Renders the prompt of the original `nemotron3p5` prompt formatter: an empty system turn by default, text and
# audio items of a turn joined by a space, and thinking disabled by default.
CHAT_TEMPLATE = """{%- set enable_thinking = enable_thinking if enable_thinking is defined else false -%}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if not loop.first %}{{ " " }}{% endif -%}
            {%- if item.type == "text" -%}
                {{- item.text -}}
            {%- elif item.type == "audio" -%}
                {{- "<|audio|>" -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if messages[0].role == "system" -%}
    {%- set system_message = render_content(messages[0].content) -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{{- "<|im_start|>system\\n" + system_message + "<|im_end|>\\n" -}}
{%- for message in loop_messages -%}
    {%- set content = render_content(message.content) -%}
    {%- if message.role == "assistant" and "<think>" not in content -%}
        {%- set content = "<think></think>" + content -%}
    {%- endif -%}
    {{- "<|im_start|>" + message.role + "\\n" + content + "<|im_end|>\\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {%- if enable_thinking -%}
        {{- "<|im_start|>assistant\\n<think>\\n" -}}
    {%- else -%}
        {{- "<|im_start|>assistant\\n<think></think>" -}}
    {%- endif -%}
{%- endif -%}"""

# NeMo `TransformerEncoder` key regex -> HF audio encoder key, relative to the encoder prefix. The fused
# `attn.w_qkv` projection is split by `split_fused_qkv`.
ENCODER_MAPPING = {
    r"pre_encode\.proj\.": "embedder.projection.",
    r"embed_norm\.": "input_layer_norm.",
    r"final_norm\.": "layer_norm.",
    r"layers\.(\d+)\.norm1\.": r"layers.\1.layer_norm1.",
    r"layers\.(\d+)\.norm2\.": r"layers.\1.layer_norm2.",
    r"layers\.(\d+)\.attn\.out_proj\.": r"layers.\1.self_attn.o_proj.",
    r"layers\.(\d+)\.attn\.(q_norm|k_norm)\.": r"layers.\1.self_attn.\2.",
    r"layers\.(\d+)\.ffn\.net\.0\.": r"layers.\1.mlp.fc1.",
    r"layers\.(\d+)\.ffn\.net\.3\.": r"layers.\1.mlp.fc2.",
}
ENCODER_PREFIXES = {
    "perception.encoder.asr_encoder.": "model.audio_tower.asr_encoder.",
    "perception.encoder.diarization_model.encoder.": "model.audio_tower.diarization_model.model.audio_tower.",
}

# NeMo key regex -> HF key, first match wins.
STATE_DICT_MAPPING = {
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.encoder_proj\.": "model.audio_tower.diarization_model.model.proj.",
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.subpixel_upsample\.": "model.audio_tower.diarization_model.model.upsampler.conv.",
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.first_hidden_to_hidden\.": "model.audio_tower.diarization_model.classifier.dense.",
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.single_hidden_to_spks\.": "model.audio_tower.diarization_model.classifier.out_proj.",
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.learnable_sil_emb$": "model.audio_tower.diarization_model.silence_embeds",
    r"^perception\.encoder\.asr_norm\.": "model.audio_tower.asr_norm.",
    r"^perception\.encoder\.diar_norm\.": "model.audio_tower.speaker_norm.",
    r"^perception\.proj\.": "model.multi_modal_projector.",
    r"^llm\.model\.embed_tokens\.": "model.language_model.embeddings.",
    r"^llm\.model\.norm\.": "model.language_model.norm_f.",
    r"^llm\.model\.layers\.(\d+)\.mixer\._fp32_params\.": r"model.language_model.layers.\1.mixer.",
    r"^llm\.model\.layers\.(\d+)\.mixer\.experts\.gate_and_up_projs$": r"model.language_model.layers.\1.mixer.experts.up_proj",
    r"^llm\.model\.layers\.(\d+)\.mixer\.experts\.down_projs$": r"model.language_model.layers.\1.mixer.experts.down_proj",
    r"^llm\.model\.layers\.": "model.language_model.layers.",
    r"^llm\.lm_head\.": "lm_head.",
}
# NeMo-Automodel stores the routed experts as `(num_experts, in_features, out_features)`.
TRANSPOSED_KEYS = r"^llm\.model\.layers\.\d+\.mixer\.experts\.(gate_and_up_projs|down_projs)$"
KEYS_TO_DROP = [
    # TransformerEngine metadata, empty tensors.
    r"\._extra_state$",
    # Multi-token prediction head, only used as a draft model for speculative decoding.
    r"^llm\.mtp\.",
    # STFT window and mel filter banks, recomputed by the feature extractor.
    r"^perception\.preprocessor\.",
    r"^perception\.encoder\.diarization_model\.preprocessor\.",
    # Legacy two-branch speaker head, never called in the original forward.
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.hidden_to_spks\.",
]
SHARD_SIZE = 5 * 1024**3
# Defaults of NeMo `ParallelExpertEncoder` for the windows of its online inference and the diarization speaker cache,
# which checkpoint configs usually omit. In encoder frames.
NEMO_ONLINE_INFERENCE_DEFAULTS = {
    "online_inference_length": 500,
    "chunk_left_context": 50,
    "chunk_right_context": 50,
    "diar_fifo_len": 40,
    "diar_spkcache_update_period": 300,
    "diar_spkcache_len": 188,
}
# Defaults of NeMo `TransformerEncoder`, for the options a checkpoint config may omit.
NEMO_ENCODER_DEFAULTS = {
    "subsampling": "feature_stacking",
    "qkv_bias": False,
    "qk_norm": False,
    "pre_block_norm": True,
    "self_attention_model": "rel_pos",
    "rotary_fraction": 1.0,
    "xscaling": False,
    "feat_out": -1,
}


def convert_key(key: str) -> str | None:
    if any(re.search(pattern, key) for pattern in KEYS_TO_DROP):
        return None
    for nemo_prefix, hf_prefix in ENCODER_PREFIXES.items():
        if key.startswith(nemo_prefix):
            suffix = key[len(nemo_prefix) :]
            for pattern, replacement in ENCODER_MAPPING.items():
                if re.match(pattern, suffix):
                    return hf_prefix + re.sub(pattern, replacement, suffix, count=1)
            if re.fullmatch(r"layers\.\d+\.attn\.w_qkv\.weight", suffix):
                return hf_prefix + suffix
            raise ValueError(f"Unexpected NeMo encoder key: {key}")
    for pattern, replacement in STATE_DICT_MAPPING.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key, count=1)
    raise ValueError(f"Unexpected NeMo key: {key}")


def convert_tensors(key, tensor):
    """Yields the HF `(key, tensor)` pairs of one NeMo tensor."""
    hf_key = convert_key(key)
    if hf_key is None:
        return
    match = re.match(r"^(.*layers\.\d+)\.attn\.w_qkv\.weight$", hf_key)
    if match:
        # NeMo `w_qkv` rows are ordered `[query | key | value]`.
        for name, weight in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
            yield f"{match.group(1)}.self_attn.{name}.weight", weight.contiguous()
        return
    if re.match(TRANSPOSED_KEYS, key):
        tensor = tensor.transpose(1, 2).contiguous()
    yield hf_key, tensor


def check_encoder(name: str, encoder_config: dict, **expected):
    """Checks a NeMo `TransformerEncoder` config against the options the HF encoders implement."""
    expected = {
        "subsampling": "feature_stacking",
        "qkv_bias": False,
        "pre_block_norm": True,
        "self_attention_model": "rope",
        "xscaling": False,
        "feat_out": -1,
        **expected,
    }
    for key, value in expected.items():
        # an option missing from the config takes the NeMo default
        actual = encoder_config.get(key, NEMO_ENCODER_DEFAULTS[key])
        if actual != value:
            raise ValueError(f"Unsupported `{name}.{key}={actual}`, expected {value}.")


def convert_config(nemo_dir: Path) -> Nemotron3_5TranscribeConfig:
    nemo_config = json.loads((nemo_dir / "config.json").read_text())
    encoder_config = nemo_config["pe_encoder_config"]
    asr = encoder_config["asr_encoder_cfg"]
    diarization = encoder_config["diarization_model_cfg"]

    expected = {
        "asr_encoder_type": "transformer",
        "asr_normalize_type": "per_feature",
        "diar_normalize_type": "NA",
        "speaker_feature_mode": "thresholded",
        "spk_kernel_scale": 1.0,
    }
    for name, value in expected.items():
        if encoder_config[name] != value:
            raise ValueError(f"Unsupported `pe_encoder_config.{name}={encoder_config[name]}`, expected {value}.")
    diarization_encoder = diarization["encoder"]
    check_encoder("asr_encoder_cfg", asr, qk_norm=True)
    # the diarization audio encoder has no query/key norm and rotates the whole head
    check_encoder("diarization_model_cfg.encoder", diarization_encoder, qk_norm=False, rotary_fraction=1.0)
    if diarization.get("transformer_encoder") is not None:
        raise ValueError("Unsupported `diarization_model_cfg.transformer_encoder`, the Sortformer must have none.")
    if diarization.get("async_streaming", False) or not diarization["sortformer_modules"]["use_learnable_sil_emb"]:
        raise ValueError(
            "Unsupported Sortformer streaming: it must be synchronous with a learnable silence embedding."
        )
    online = {name: encoder_config.get(name, value) for name, value in NEMO_ONLINE_INFERENCE_DEFAULTS.items()}

    asr_config = {
        "num_mel_bins": asr["feat_in"],
        "hidden_size": asr["d_model"],
        "intermediate_size": int(asr["d_model"] * asr["ff_expansion"]),
        "num_hidden_layers": asr["n_layers"],
        "num_attention_heads": asr["n_heads"],
        "subsampling_factor": asr["subsampling_factor"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": asr.get("rope_base", 10000.0),
            "partial_rotary_factor": asr.get("rotary_fraction", 1.0),
        },
    }
    sortformer = diarization["sortformer_modules"]
    diarization_config = {
        "audio_config": {
            "num_mel_bins": diarization_encoder["feat_in"],
            "hidden_size": diarization_encoder["d_model"],
            "intermediate_size": int(diarization_encoder["d_model"] * diarization_encoder["ff_expansion"]),
            "num_hidden_layers": diarization_encoder["n_layers"],
            "num_attention_heads": diarization_encoder["n_heads"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "head_config": {
            "hidden_size": sortformer["tf_d_model"],
            "num_speakers": sortformer["num_spks"],
            "audio_hidden_size": sortformer["fc_d_model"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "streaming_config": {
            "speaker_cache_length": online["diar_spkcache_len"],
            "speaker_cache_silence_frames_per_speaker": sortformer["spkcache_sil_frames_per_spk"],
            "prediction_score_threshold": sortformer["pred_score_threshold"],
            "latest_frames_score_boost": sortformer["scores_boost_latest"],
            "strong_boost_rate": sortformer["strong_boost_rate"],
            "weak_boost_rate": sortformer["weak_boost_rate"],
            "min_positive_scores_rate": sortformer["min_pos_scores_rate"],
            "num_speakers": sortformer["num_spks"],
            "subsampling_factor": diarization_encoder["subsampling_factor"],
        },
        "chunk_length": online["online_inference_length"],
        "chunk_left_context": online["chunk_left_context"],
        "chunk_right_context": online["chunk_right_context"],
        "fifo_length": online["diar_fifo_len"],
        "speaker_cache_update_period": online["diar_spkcache_update_period"],
    }

    text_config = json.loads((nemo_dir / "llm_backbone" / "config.json").read_text())
    # NeMo serializes the infinite upper bound of the Mamba time step as `{"__float__": "Infinity"}`.
    text_config["time_step_limit"] = [
        float(value) if not isinstance(value, dict) else math.inf for value in text_config["time_step_limit"]
    ]
    # the multi-token prediction head is not converted
    text_config["num_nextn_predict_layers"] = 0
    for name in ("architectures", "auto_map", "_name_or_path", "transformers_version"):
        text_config.pop(name, None)

    tokenizer_config = json.loads((nemo_dir / "tokenizer_config.json").read_text())
    tokenizer = json.loads((nemo_dir / "tokenizer.json").read_text())
    audio_token = tokenizer_config["audio_token"]
    audio_token_id = next(token["id"] for token in tokenizer["added_tokens"] if token["content"] == audio_token)

    return Nemotron3_5TranscribeConfig(
        audio_config={
            "asr_config": asr_config,
            "diarization_config": diarization_config,
            "speaker_activity_threshold": encoder_config["speaker_activity_threshold"],
        },
        text_config=text_config,
        audio_token_id=audio_token_id,
        dtype="bfloat16",
    )


def write_weights(nemo_dir: Path, output_dir: Path):
    shard, shard_size, total_size = {}, 0, 0
    shard_names, weight_map = [], {}

    def save_shard():
        name = f"model-{len(shard_names) + 1:05d}.safetensors"
        save_file(shard, output_dir / name, metadata={"format": "pt"})
        shard_names.append(name)
        weight_map.update(dict.fromkeys(shard, name))

    with safe_open(nemo_dir / "model.safetensors", framework="pt") as reader:
        for key in reader.keys():
            for hf_key, tensor in convert_tensors(key, reader.get_tensor(key)):
                size = tensor.numel() * tensor.element_size()
                if shard and shard_size + size > SHARD_SIZE:
                    save_shard()
                    shard, shard_size = {}, 0
                shard[hf_key] = tensor
                shard_size += size
                total_size += size
    save_shard()

    num_shards = len(shard_names)
    final_names = {name: name.replace(".safetensors", f"-of-{num_shards:05d}.safetensors") for name in shard_names}
    for name, final_name in final_names.items():
        (output_dir / name).rename(output_dir / final_name)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {key: final_names[name] for key, name in sorted(weight_map.items())},
    }
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def write_processor(nemo_dir: Path, output_dir: Path) -> Nemotron3_5TranscribeProcessor:
    nemo_config = json.loads((nemo_dir / "config.json").read_text())
    preprocessor = nemo_config["perception"]["preprocessor"]
    sampling_rate = preprocessor["sample_rate"]
    feature_extractor = NemotronAsrStreamingFeatureExtractor(
        feature_size=preprocessor["features"],
        sampling_rate=sampling_rate,
        hop_length=int(preprocessor["window_stride"] * sampling_rate),
        n_fft=preprocessor["n_fft"],
        win_length=int(preprocessor["window_size"] * sampling_rate),
        return_attention_mask=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(nemo_dir, padding_side="left")
    processor = Nemotron3_5TranscribeProcessor(feature_extractor, tokenizer, chat_template=CHAT_TEMPLATE)
    processor.save_pretrained(output_dir)
    return processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_dir", required=True, help="Directory of the NeMo `to_hf.py` export.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_repo_id", default=None)
    args = parser.parse_args()

    nemo_dir, output_dir = Path(args.nemo_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = convert_config(nemo_dir)
    config.architectures = ["Nemotron3_5TranscribeForConditionalGeneration"]
    config.save_pretrained(output_dir)
    write_weights(nemo_dir, output_dir)
    processor = write_processor(nemo_dir, output_dir)
    generation_config = json.loads((nemo_dir / "generation_config.json").read_text())
    GenerationConfig(
        eos_token_id=generation_config["eos_token_id"],
        pad_token_id=generation_config["pad_token_id"],
        do_sample=False,
    ).save_pretrained(output_dir)

    model, loading_info = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
        output_dir, device_map="auto", output_loading_info=True
    )
    for name, keys in loading_info.items():
        if keys:
            raise ValueError(f"{name}: {keys}")

    if args.push_to_repo_id:
        model.push_to_hub(args.push_to_repo_id)
        processor.push_to_hub(args.push_to_repo_id)
        model.generation_config.push_to_hub(args.push_to_repo_id)


if __name__ == "__main__":
    main()
