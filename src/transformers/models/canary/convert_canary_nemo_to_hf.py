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

import argparse
import gc
import os
import tarfile

import torch
import yaml

from transformers import (
    CanaryConfig,
    CanaryForConditionalGeneration,
    CanaryProcessor,
    ParakeetFeatureExtractor,
    TokenizersBackend,
)
from transformers.convert_slow_tokenizer import CanaryConverter
from transformers.models.parakeet.convert_nemo_to_hf import convert_encoder_config, convert_key
from transformers.utils.hub import cached_file


NEMO_TO_HF_ENCODER_MAPPING = {
    r"^encoder\.pre_encode\.conv\.": r"encoder.subsampling.layers.",
    r"^encoder\.pre_encode\.out\.": r"encoder.subsampling.linear.",
    r"^encoder\.pos_enc\.": r"encoder.encode_positions.",
    r"^encoder\.layers\.(\d+)\.conv\.batch_norm\.": r"encoder.layers.\1.conv.norm.",
    r"linear_([kv])": r"\1_proj",
    r"linear_out": r"o_proj",
    r"linear_q": r"q_proj",
    r"pos_bias_([uv])": r"bias_\1",
    r"linear_pos": r"relative_k_proj",
}

NEMO_TO_HF_DECODER_MAPPING = {
    r"^transf_decoder\._embedding\.token_embedding\.": r"decoder.embed_tokens.",
    r"^transf_decoder\._embedding\.layer_norm\.": r"decoder.layernorm_embedding.",
    r"^transf_decoder\._decoder\.final_layer_norm\.": r"decoder.layer_norm.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_1\.": r"decoder.layers.\1.self_attn_layer_norm.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_2\.": r"decoder.layers.\1.encoder_attn_layer_norm.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_3\.": r"decoder.layers.\1.final_layer_norm.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.": r"decoder.layers.\1.self_attn.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.": r"decoder.layers.\1.encoder_attn.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_in\.": r"decoder.layers.\1.fc1.",
    r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_out\.": r"decoder.layers.\1.fc2.",
    r"query_net": r"q_proj",
    r"key_net": r"k_proj",
    r"value_net": r"v_proj",
    r"out_projection": r"out_proj",
}


def extract_nemo_archive(nemo_file_path: str, extract_dir: str) -> None:
    """Extract a Canary `.nemo` (tar) archive into `extract_dir`."""
    print(f"Extracting NeMo archive: {nemo_file_path}")
    with tarfile.open(nemo_file_path, "r", encoding="utf-8") as tar:
        # filter="data" sanitizes members (PEP 706) so a malicious archive cannot write outside extract_dir
        tar.extractall(extract_dir, filter="data")


def convert_decoder_config(nemo_config) -> CanaryConfig:
    """Build a [`CanaryConfig`] from the NeMo `transf_decoder` and `head` config blocks."""
    decoder_config = nemo_config["transf_decoder"]["config_dict"]
    head_config = nemo_config["head"]
    encoder_config = convert_encoder_config(nemo_config)

    return CanaryConfig(
        encoder_config=encoder_config.to_dict(),
        vocab_size=head_config["num_classes"],
        d_model=decoder_config["hidden_size"],
        decoder_layers=decoder_config["num_layers"],
        decoder_attention_heads=decoder_config["num_attention_heads"],
        decoder_ffn_dim=decoder_config["inner_size"],
        activation_function=decoder_config["hidden_act"],
        max_target_positions=decoder_config["max_sequence_length"],
        dropout=decoder_config["embedding_dropout"],
        attention_dropout=decoder_config["attn_score_dropout"],
        activation_dropout=decoder_config["ffn_dropout"],
    )


def load_and_convert_state_dict(model_files):
    """Load the NeMo state dict and convert keys to the HF (`model.encoder.*` / `model.decoder.*`) layout."""
    state_dict = torch.load(model_files["model_weights"], map_location="cpu", weights_only=True)
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("preprocessor.") or key.endswith(".position_embedding.pos_enc"):
            # featurizer buffers and the fixed sinusoidal table are recomputed, not loaded
            continue
        if key.startswith("encoder."):
            converted_state_dict["model." + convert_key(key, NEMO_TO_HF_ENCODER_MAPPING)] = value
        elif key.startswith("transf_decoder."):
            converted_state_dict["model." + convert_key(key, NEMO_TO_HF_DECODER_MAPPING)] = value
        elif key == "log_softmax.mlp.layer0.weight":
            converted_state_dict["proj_out.weight"] = value
        elif key == "log_softmax.mlp.layer0.bias":
            converted_state_dict["proj_out.bias"] = value
        else:
            raise ValueError(f"Unhandled NeMo weight key: {key}")
    return converted_state_dict


def write_processor(nemo_config, model_files, output_dir, push_to_repo_id=None):
    tokenizer_object = CanaryConverter(model_files["tokenizer_model_file"]).converted()
    tokenizer = TokenizersBackend(
        tokenizer_object=tokenizer_object,
        clean_up_tokenization_spaces=False,
        bos_token="<|startoftranscript|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    preprocessor = nemo_config["preprocessor"]
    feature_extractor = ParakeetFeatureExtractor(
        feature_size=preprocessor["features"],
        sampling_rate=preprocessor["sample_rate"],
        win_length=int(preprocessor["window_size"] * preprocessor["sample_rate"]),
        hop_length=int(preprocessor["window_stride"] * preprocessor["sample_rate"]),
        n_fft=preprocessor["n_fft"],
    )

    processor = CanaryProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)
    if push_to_repo_id:
        processor.push_to_hub(push_to_repo_id)


def main(hf_repo_id, output_dir, push_to_repo_id=None):
    nemo_filename = f"{hf_repo_id.split('/')[-1]}.nemo"
    filepath = cached_file(hf_repo_id, nemo_filename)
    extract_dir = os.path.dirname(filepath)
    extract_nemo_archive(filepath, extract_dir)

    nemo_config = yaml.load(open(os.path.join(extract_dir, "model_config.yaml"), "r"), Loader=yaml.FullLoader)
    tokenizer_model_name = nemo_config["tokenizer"]["model_path"].split("nemo:")[-1]
    model_files = {
        "model_weights": os.path.join(extract_dir, "model_weights.ckpt"),
        "tokenizer_model_file": os.path.join(extract_dir, tokenizer_model_name),
    }

    write_processor(nemo_config, model_files, output_dir, push_to_repo_id)

    config = convert_decoder_config(nemo_config)
    print(f"Converted config:\n{config}")
    converted_state_dict = load_and_convert_state_dict(model_files)

    print("Loading the checkpoint in a Canary model.")
    with torch.device("meta"):
        model = CanaryForConditionalGeneration(config)
    model.load_state_dict(converted_state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")

    model.generation_config.decoder_start_token_id = config.decoder_start_token_id
    model.generation_config.bos_token_id = config.bos_token_id
    model.generation_config.eos_token_id = config.eos_token_id
    model.generation_config.pad_token_id = config.pad_token_id

    print("Saving the model.")
    model.save_pretrained(output_dir)
    if push_to_repo_id:
        model.push_to_hub(push_to_repo_id)

    del model
    gc.collect()
    print("Reloading the model to check it was saved correctly.")
    CanaryForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16)
    print("Model reloaded successfully.")


"""
Conversion example:
```bash
python src/transformers/models/canary/convert_canary_nemo_to_hf.py \
    --hf_repo_id nvidia/canary-1b-v2 \
    --output_dir OUTPUT_DIR \
    --push_to_repo_id USERNAME/canary-1b-v2-hf
```
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo_id", required=True, help="Model repo on huggingface.co")
    parser.add_argument("--output_dir", required=True, help="Output directory for the HuggingFace model")
    parser.add_argument("--push_to_repo_id", help="Repository ID to push the converted model to on the Hub")
    args = parser.parse_args()
    main(args.hf_repo_id, args.output_dir, args.push_to_repo_id)
