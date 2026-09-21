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
"""Convert a NeMo Streaming Sortformer checkpoint (`nvidia/Nemotron-3-Diarization-preview`) to Nemotron3Diarization."""

import argparse
import re
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download

from transformers import (
    Nemotron3DiarizationConfig,
    Nemotron3DiarizationEncoderConfig,
    Nemotron3DiarizationForAudioFrameClassification,
    Nemotron3DiarizationProcessor,
    NemotronAsrStreamingFeatureExtractor,
)


# NeMo key regex -> HF key. The fused `attn.w_qkv` projection is split by `split_fused_qkv`.
STATE_DICT_MAPPING = {
    r"^encoder\.pre_encode\.proj\.": "encoder.feature_stacking.projection.",
    r"^encoder\.embed_norm\.": "encoder.input_layer_norm.",
    r"^encoder\.final_norm\.": "encoder.layer_norm.",
    r"^encoder\.layers\.(\d+)\.norm1\.": r"encoder.layers.\1.layer_norm1.",
    r"^encoder\.layers\.(\d+)\.norm2\.": r"encoder.layers.\1.layer_norm2.",
    r"^encoder\.layers\.(\d+)\.attn\.out_proj\.": r"encoder.layers.\1.self_attn.o_proj.",
    r"^encoder\.layers\.(\d+)\.ffn\.net\.0\.": r"encoder.layers.\1.mlp.fc1.",
    r"^encoder\.layers\.(\d+)\.ffn\.net\.3\.": r"encoder.layers.\1.mlp.fc2.",
    r"^sortformer_modules\.encoder_proj\.": "speaker_projection.",
    r"^sortformer_modules\.subpixel_upsample\.": "upsampler.conv.",
    r"^sortformer_modules\.first_hidden_to_hidden\.": "classifier.dense.",
    r"^sortformer_modules\.single_hidden_to_spks\.": "classifier.out_proj.",
    r"^sortformer_modules\.learnable_sil_emb$": "silence_embeds",
}

# Checkpoint tensors that have no counterpart in the HF model.
KEYS_TO_DROP = {
    # Legacy two-branch speaker head (384 -> 8), frozen and never called in the original forward.
    r"^sortformer_modules\.hidden_to_spks\.",
    # STFT window and mel filter bank, recomputed by the feature extractor.
    r"^preprocessor\.featurizer\.",
}

# Model-card inference profile ("Very high latency (offline)", 30.4 s input buffer). The `.nemo` config holds the
# training-time values (fifo 0, chunk 264, update 188, no right context) which are not meant for inference.
OFFLINE_PROFILE = {
    "speaker_cache_length": 264,
    "fifo_length": 40,
    "chunk_length": 340,
    "chunk_right_context": 40,
    "speaker_cache_update_period": 300,
}


def split_fused_qkv(state_dict: dict) -> dict:
    """NeMo `w_qkv` rows are ordered `[query | key | value]`, each `hidden_size` wide."""
    converted = {}
    for key, value in state_dict.items():
        match = re.match(r"^encoder\.layers\.(\d+)\.attn\.w_qkv\.weight$", key)
        if match is None:
            converted[key] = value
            continue
        query, key_weight, value_weight = value.chunk(3, dim=0)
        prefix = f"encoder.layers.{match.group(1)}.self_attn."
        converted[prefix + "q_proj.weight"] = query
        converted[prefix + "k_proj.weight"] = key_weight
        converted[prefix + "v_proj.weight"] = value_weight
    return converted


def convert_state_dict(state_dict: dict) -> dict:
    state_dict = split_fused_qkv(state_dict)
    converted = {}
    for key, value in state_dict.items():
        if any(re.match(pattern, key) for pattern in KEYS_TO_DROP):
            continue
        if key.startswith("encoder.layers.") and ".self_attn." in key:
            converted["model." + key] = value
            continue
        new_key = None
        for pattern, replacement in STATE_DICT_MAPPING.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break
        if new_key is None:
            raise ValueError(f"No mapping for checkpoint key {key!r}")
        if new_key.startswith(("encoder.", "speaker_projection.", "upsampler.", "classifier.")):
            new_key = "model." + new_key
        converted[new_key] = value
    return converted


def build_config(nemo_config: dict) -> Nemotron3DiarizationConfig:
    encoder = nemo_config["encoder"]
    modules = nemo_config["sortformer_modules"]
    if (
        encoder.get("subsampling", "feature_stacking") != "feature_stacking"
        or encoder["self_attention_model"] != "rope"
    ):
        raise ValueError("Only feature-stacking encoders with rotary attention are supported.")
    if not nemo_config.get("high_resolution", False) or not nemo_config.get("streaming_mode", False):
        raise ValueError("Only high-resolution streaming checkpoints are supported.")
    return Nemotron3DiarizationConfig(
        encoder_config=Nemotron3DiarizationEncoderConfig(
            num_mel_bins=encoder["feat_in"],
            subsampling_factor=encoder["subsampling_factor"],
            hidden_size=encoder["d_model"],
            num_hidden_layers=encoder["n_layers"],
            num_attention_heads=encoder["n_heads"],
            intermediate_size=int(encoder.get("ff_expansion", 4.0) * encoder["d_model"]),
            # NeMo's `TransformerEncoder` feed-forward block uses a fixed `nn.GELU()` (exact GELU); it is not configurable.
            hidden_act="gelu",
            max_position_embeddings=encoder.get("pos_emb_max_len", 5000),
            rope_parameters={"rope_type": "default", "rope_theta": encoder.get("rope_base", 10000.0)},
        ),
        speaker_hidden_size=modules["tf_d_model"],
        num_speakers=modules["num_spks"],
        speaker_cache_silence_frames_per_speaker=modules["spkcache_sil_frames_per_spk"],
        prediction_score_threshold=modules["pred_score_threshold"],
        latest_frames_score_boost=modules["scores_boost_latest"],
        strong_boost_rate=modules["strong_boost_rate"],
        weak_boost_rate=modules["weak_boost_rate"],
        min_positive_scores_rate=modules["min_pos_scores_rate"],
        **OFFLINE_PROFILE,
        dtype="float32",
    )


def build_feature_extractor(nemo_config: dict) -> NemotronAsrStreamingFeatureExtractor:
    """
    The NeMo mel front-end of this checkpoint is the one NemotronAsrStreaming already implements: un-normalized
    log-mel features with a `center` switch for streaming chunks.
    """
    preprocessor = nemo_config["preprocessor"]
    sampling_rate = preprocessor["sample_rate"]
    if preprocessor.get("normalize", "per_feature") != "NA":
        raise ValueError("Only un-normalized spectrograms are supported.")
    return NemotronAsrStreamingFeatureExtractor(
        feature_size=preprocessor["features"],
        sampling_rate=sampling_rate,
        hop_length=int(round(preprocessor["window_stride"] * sampling_rate)),
        n_fft=preprocessor["n_fft"],
        win_length=int(round(preprocessor["window_size"] * sampling_rate)),
        preemphasis=preprocessor.get("preemph", 0.97),
    )


def main(nemo_path: str | None, model_id: str, output_dir: str):
    if nemo_path is None:
        nemo_path = hf_hub_download(model_id, f"{model_id.split('/')[-1]}.nemo")
    with tempfile.TemporaryDirectory() as extract_dir, tarfile.open(nemo_path) as archive:
        archive.extractall(extract_dir, filter="data")
        with open(Path(extract_dir) / "model_config.yaml") as f:
            nemo_config = yaml.safe_load(f)
        state_dict = torch.load(Path(extract_dir) / "model_weights.ckpt", map_location="cpu", weights_only=True)

    config = build_config(nemo_config)
    model = Nemotron3DiarizationForAudioFrameClassification(config)
    model.load_state_dict(convert_state_dict(state_dict), strict=True)
    model.save_pretrained(output_dir)
    processor = Nemotron3DiarizationProcessor(
        feature_extractor=build_feature_extractor(nemo_config),
        chunk_length=config.chunk_length,
        chunk_right_context=config.chunk_right_context,
        subsampling_factor=config.encoder_config.subsampling_factor,
    )
    processor.save_pretrained(output_dir)
    print(f"Saved model and processor to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nemo_path", default=None, help="Local `.nemo` file. Downloaded from `--model_id` when unset."
    )
    parser.add_argument("--model_id", default="nvidia/Nemotron-3-Diarization-preview")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.nemo_path, args.model_id, args.output_dir)
