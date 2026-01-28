# Copyright 2025 BosonAI and The HuggingFace Inc. team. All rights reserved.
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
import glob
import io

import torch
import yaml
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    DacFeatureExtractor,
    HiggsAudioV2Config,
    HiggsAudioV2ForConditionalGeneration,
    HiggsAudioV2Processor,
    HiggsAudioV2TokenizerModel,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


torch.serialization.add_safe_globals([io.BytesIO])


def safe_load(path: str) -> dict[str, torch.Tensor]:
    """
    Load only the tensor objects from a checkpoint, skipping any BytesIO
    """
    # Load all safetensor shards
    tensor_dict = {}
    for f in sorted(glob.glob(f"{path}/model-*.safetensors")):
        shard = load_file(f)
        tensor_dict.update(shard)  # merge shard dicts

    return tensor_dict


def convert_old_keys_to_new_keys(original_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted_checkpoint: dict[str, torch.Tensor] = {}

    for old_key, value in original_state_dict.items():
        if not old_key.startswith("audio_decoder_proj."):
            converted_checkpoint[f"model.{old_key}"] = value
        else:
            converted_checkpoint[old_key] = value

    return converted_checkpoint


@torch.no_grad()
def convert_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, push_to_hub=None):
    # load config yaml file
    with open(config_path, "r") as f:
        original_model_config = yaml.safe_load(f)

    original_model_config.pop("audio_encoder_config")
    original_model_config.pop("use_audio_out_self_attention")
    original_model_config.pop("skip_audio_tower")
    original_model_config["text_config"]["rms_norm_eps"] = float(original_model_config["text_config"]["rms_norm_eps"])

    config = HiggsAudioV2Config(**original_model_config)

    # create model
    if not torch.cuda.is_available():
        raise ValueError("Run this script on a machine with a GPU for weight norm layers to be correctly copied.")
    torch_device = "cuda"
    model = HiggsAudioV2ForConditionalGeneration(config).to(torch_device)

    logger.info("Loading original checkpoint ...")

    state_dict = safe_load(checkpoint_path)

    logger.info("Converting model ...")

    new_state_dict = convert_old_keys_to_new_keys(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True, assign=True)  # strict=False)

    if len(unexpected_keys) != 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    if len(missing_keys) != 0:
        raise ValueError(f"missing keys found: {missing_keys}")

    model.generation_config.stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
    model.generation_config.temperature = 0.3
    model.generation_config.top_p = 0.95
    model.generation_config.top_k = 50
    model.generation_config.do_sample = True

    model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model to the hub...")
        model.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=True,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )

    if args.convert_preprocessor:
        processor = HiggsAudioV2Processor(
            DacFeatureExtractor(sampling_rate=24000, hop_length=1),
            AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base"),
            HiggsAudioV2TokenizerModel.from_pretrained("szhengac25/higgs-audio-v2-tokenizer"),
        )
        processor.save_pretrained(args.pytorch_dump_folder_path)

        if args.push_to_hub:
            print("Pushing processor to the hub...")
            processor.push_to_hub(args.push_to_hub)
