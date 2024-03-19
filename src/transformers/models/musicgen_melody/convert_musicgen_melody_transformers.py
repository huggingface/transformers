# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Musicgen Melody checkpoints from the original repository."""
import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import torch
from audiocraft.models import MusicGen

from transformers import (
    AutoTokenizer,
    EncodecModel,
    T5EncoderModel,
)
from transformers.models.musicgen_melody.configuration_musicgen_melody import MusicgenMelodyDecoderConfig
from transformers.models.musicgen_melody.feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
from transformers.models.musicgen_melody.modeling_musicgen_melody import (
    MusicgenMelodyForCausalLM,
    MusicgenMelodyForConditionalGeneration,
)
from transformers.models.musicgen_melody.processing_musicgen_melody import MusicgenMelodyProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]
EXPECTED_ADDITIONAL_KEYS = ["condition_provider.conditioners.self_wav.chroma.spec.window"]


def rename_keys(name):
    if "emb" in name:
        name = name.replace("emb", "model.decoder.embed_tokens")
    if "transformer" in name:
        name = name.replace("transformer", "model.decoder")
    if "cross_attention" in name:
        name = name.replace("cross_attention", "encoder_attn")
    if "linear1" in name:
        name = name.replace("linear1", "fc1")
    if "linear2" in name:
        name = name.replace("linear2", "fc2")
    if "norm1" in name:
        name = name.replace("norm1", "self_attn_layer_norm")
    if "norm_cross" in name:
        name = name.replace("norm_cross", "encoder_attn_layer_norm")
    if "norm2" in name:
        name = name.replace("norm2", "final_layer_norm")
    if "out_norm" in name:
        name = name.replace("out_norm", "model.decoder.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "enc_to_dec_proj")
    if "condition_provider.conditioners.self_wav.output_proj" in name:
        name = name.replace("condition_provider.conditioners.self_wav.output_proj", "audio_enc_to_dec_proj")
    return name


def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> Tuple[Dict, Dict]:
    """Function that takes the fairseq MusicgenMelody state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that for the
    text encoder projection and for the audio encoder projection."""
    keys = list(state_dict.keys())
    enc_dec_proj_state_dict = {}
    audio_enc_to_dec_proj_state_dict = {}
    for key in keys:
        val = state_dict.pop(key)
        key = rename_keys(key)
        if "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        elif "audio_enc_to_dec_proj" in key:
            audio_enc_to_dec_proj_state_dict[key[len("audio_enc_to_dec_proj.") :]] = val
        elif "enc_to_dec_proj" in key:
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            state_dict[key] = val
    return state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict


def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenMelodyDecoderConfig:
    if checkpoint == "facebook/musicgen-melody" or checkpoint == "facebook/musicgen-stereo-melody":
        hidden_size = 1536
        num_hidden_layers = 48
        num_attention_heads = 24
    elif checkpoint == "facebook/musicgen-melody-large" or checkpoint == "facebook/musicgen-stereo-melody-large":
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 32
    else:
        raise ValueError(
            "Checkpoint should be one of `['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, "
            "or `['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` "
            f"for the stereo checkpoints, got {checkpoint}."
        )

    if "stereo" in checkpoint:
        audio_channels = 2
        num_codebooks = 8
    else:
        audio_channels = 1
        num_codebooks = 4

    config = MusicgenMelodyDecoderConfig(
        hidden_size=hidden_size,
        ffn_dim=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_codebooks=num_codebooks,
        audio_channels=audio_channels,
    )
    return config


@torch.no_grad()
def convert_musicgen_melody_checkpoint(
    checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu", test_same_output=False
):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=args.device)
    decoder_config = decoder_config_from_checkpoint(checkpoint)

    decoder_state_dict = fairseq_model.lm.state_dict()
    decoder_state_dict, enc_dec_proj_state_dict, audio_enc_to_dec_proj_state_dict = rename_state_dict(
        decoder_state_dict, hidden_size=decoder_config.hidden_size
    )

    text_encoder = T5EncoderModel.from_pretrained("t5-base")
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    decoder = MusicgenMelodyForCausalLM(decoder_config).eval()

    # load all decoder weights - expect that we'll be missing embeddings and enc-dec projection
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)

    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    for key in unexpected_keys.copy():
        if key in EXPECTED_ADDITIONAL_KEYS:
            unexpected_keys.remove(key)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # init the composite model
    model = MusicgenMelodyForConditionalGeneration(
        text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder
    ).to(args.device)

    # load the pre-trained enc-dec projection (from the decoder state dict)
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)

    # load the pre-trained audio encoder projection (from the decoder state dict)
    model.audio_enc_to_dec_proj.load_state_dict(audio_enc_to_dec_proj_state_dict)

    # check we can do a forward pass
    input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long).reshape(2, -1).to(device)
    decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    output_length = 1 + input_ids.shape[1] + model.config.chroma_length
    if logits.shape != (2 * decoder_config.num_codebooks, output_length, 2048):
        raise ValueError("Incorrect shape for logits")

    # now construct the processor
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    feature_extractor = MusicgenMelodyFeatureExtractor()

    processor = MusicgenMelodyProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048

    # set other default generation config params
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0

    if test_same_output:
        # check same output than original model
        decoder_input_ids = torch.ones_like(decoder_input_ids).to(device) * model.generation_config.pad_token_id
        with torch.no_grad():
            decoder_input_ids = decoder_input_ids[: decoder_config.num_codebooks]
            inputs = processor(text=["gen"], return_tensors="pt", padding=True).to(device)
            logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits

            attributes, prompt_tokens = fairseq_model._prepare_tokens_and_attributes(["gen"], None)
            original_logits = fairseq_model.lm.forward(
                decoder_input_ids.reshape(1, decoder_config.num_codebooks, -1), attributes
            )

            torch.testing.assert_close(
                original_logits.squeeze(2).reshape(decoder_config.num_codebooks, -1),
                logits[:, -1],
                rtol=1e-5,
                atol=5e-5,
            )

    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)

    if repo_id:
        logger.info(f"Pushing model {checkpoint} to {repo_id}")
        model.push_to_hub(repo_id, create_pr=True)
        processor.push_to_hub(repo_id, create_pr=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint",
        default="facebook/musicgen-melody",
        type=str,
        help="Checkpoint size of the Musicgen Melody model you'd like to convert. Can be one of: "
        "`['facebook/musicgen-melody', 'facebook/musicgen-melody-large']` for the mono checkpoints, or "
        "`['facebook/musicgen-stereo-melody', 'facebook/musicgen-stereo-melody-large']` "
        "for the stereo checkpoints.",
    )
    parser.add_argument(
        "--pytorch_dump_folder",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default="musicgen-melody",
        type=str,
        help="Where to upload the converted model on the 🤗 hub.",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )
    parser.add_argument("--test_same_output", default=False, type=bool, help="If `True`, test if same output logits.")

    args = parser.parse_args()
    convert_musicgen_melody_checkpoint(
        args.checkpoint, args.pytorch_dump_folder, args.push_to_hub, args.device, args.test_same_output
    )
