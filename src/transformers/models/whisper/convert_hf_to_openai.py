#!/usr/bin/env python
"""Converts a Whisper model in Hugging Face format to OpenAI format.

This script is based on the following script to do the opposite:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/convert_openai_to_hf.py

Requirements:

```bash
pip install -U openai-whisper
```

Example:

```bash
# Converts the model from Hugging Face to OpenAI format:
python convert_hf_to_openai.py \
    --checkpoint openai/whisper-tiny \
    --whisper_dump_path whisper-tiny-openai.pt
```

```python
>>> # Disabled doctest because it requries the openai-whisper package.
>> import whisper
>> from transformers.models.whisper.convert_hf_to_openai import convert_tfms_to_openai_whisper

>> # Converts the model from Hugging Face to OpenAI format:
>> convert_tfms_to_openai_whisper(
..   "openai/whisper-tiny", "whisper-tiny-openai.pt"
.. )
HF model path: openai/whisper-tiny
OpenAI model path: whisper-tiny-openai.pt
>> # Select an audio file:
>> audio_path = "https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav"

>> # Load the Whisper model in OpenAI format:
>> model = whisper.load_model("whisper-tiny-openai.pt")

>> # Transcribe the audio:
>> prediction = model.transcribe(audio_path)
>> prediction["text"][:70]
' chapter 16. I might have told you of the beginning of this liaison in'
```
"""
# Copyright 2023 Xabier de Zuazo and the Aholab team. All rights reserved.
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

import torch
from torch import nn

from transformers import WhisperConfig, WhisperForConditionalGeneration


# Create the reverse mapping adapting it from the original `WHISPER_MAPPING` in
# the `convert_openai_to_hf.py` script:
REVERSE_WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
}


def reverse_rename_keys(s_dict: dict) -> dict:
    """Renames the keys back from Hugging Face to OpenAI Whisper format.

    By using this function on an HF model's state_dict, we should get the names in the format expected by Whisper.

    Args:
        s_dict (`dict`): A dictionary with keys in Hugging Face format.

    Returns:
        `dict`: The same dictionary but in OpenAI Whisper format.
    """
    keys = list(s_dict.keys())
    for orig_key in keys:
        new_key = orig_key
        for key_r, value_r in REVERSE_WHISPER_MAPPING.items():
            if key_r in orig_key:
                new_key = new_key.replace(key_r, value_r)

        # print(f"{orig_key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(orig_key)
    return s_dict


def make_emb_from_linear(linear: nn.Linear) -> nn.Embedding:
    """Converts a linear layer's weights into an embedding layer.

    The linear layer's `in_features` dimension corresponds to the vocabulary size and its `out_features` dimension
    corresponds to the embedding size.

    Args:
        linear (`nn.Linear`): The linear layer to be converted.

    Returns:
        `nn.Embedding`:
            An embedding layer with weights set to those of the input linear layer.

    """
    vocab_size, emb_size = linear.weight.data.shape
    emb_layer = nn.Embedding(vocab_size, emb_size, _weight=linear.weight.data)
    return emb_layer


def extract_dims_from_hf(config: WhisperConfig) -> dict:
    """Extracts necessary dimensions from Hugging Face's WhisperConfig.

    Extracts necessary dimensions and related configuration data from the Hugging Face model and then restructure it
    for the OpenAI Whisper format.

    Args:
        config (`WhisperConfig`): Configuration of the Hugging Face's model.

    Returns:
        `dict`: The `dims` of the OpenAI Whisper model.
    """
    dims = {
        "n_vocab": config.vocab_size,
        "n_mels": config.num_mel_bins,
        "n_audio_state": config.d_model,
        "n_text_ctx": config.max_target_positions,
        "n_audio_layer": config.encoder_layers,
        "n_audio_head": config.encoder_attention_heads,
        "n_text_layer": config.decoder_layers,
        "n_text_head": config.decoder_attention_heads,
        "n_text_state": config.d_model,
        "n_audio_ctx": config.max_source_positions,
    }
    return dims


def convert_tfms_to_openai_whisper(hf_model_path: str, whisper_dump_path: str):
    """Converts a Whisper model from the Hugging Face to the OpenAI format.

    Takes in the path to a Hugging Face Whisper model, extracts its state_dict, renames keys as needed, and then saves
    the model OpenAI's format.

    Args:
        hf_model_path (`str`):
            Path to the pretrained Whisper model in Hugging Face format.
        whisper_dump_path (`str`):
            Destination path where the converted model in Whisper/OpenAI format will be saved.

    Returns:
        `None`
    """
    print("HF model path:", hf_model_path)
    print("OpenAI model path:", whisper_dump_path)

    # Load the HF model and its state_dict
    model = WhisperForConditionalGeneration.from_pretrained(hf_model_path)
    state_dict = model.state_dict()

    # Use a reverse mapping to rename state_dict keys
    state_dict = reverse_rename_keys(state_dict)

    # Extract configurations and other necessary metadata
    dims = extract_dims_from_hf(model.config)

    # Remove the proj_out weights from state dictionary
    del state_dict["proj_out.weight"]

    # Construct the Whisper checkpoint structure
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    whisper_checkpoint = {"dims": dims, "model_state_dict": state_dict}

    # Save in Whisper's format
    torch.save(whisper_checkpoint, whisper_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path of name of the Hugging Face checkpoint.",  # noqa: E501
    )
    parser.add_argument(
        "--whisper_dump_path",
        type=str,
        help="Path to the output Whisper model.",  # noqa: E501
    )
    args = parser.parse_args()

    convert_tfms_to_openai_whisper(args.checkpoint, args.whisper_dump_path)
