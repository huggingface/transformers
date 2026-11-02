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
Converts YuE2 (https://huggingface.co/m-a-p/YuE2-3B) to a `YuE2ForConditionalGeneration` checkpoint with its
`YuE2Processor`. The processor references the converted acoustic tokenizer (see
`convert_yue2_acoustic_tokenizer_to_hf.py`), which decodes the generated latents.

```bash
python src/transformers/models/yue2/convert_yue2_to_hf.py \
    --checkpoint m-a-p/YuE2-3B \
    --acoustic_tokenizer bezzam/yue2-acoustic-tokenizer \
    --output_dir yue2-3b \
    --push_to_hub bezzam/yue2-3b
```
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from transformers import (
    AddedToken,
    GenerationConfig,
    Qwen2Tokenizer,
    YuE2AcousticTokenizerModel,
    YuE2Config,
    YuE2ForConditionalGeneration,
    YuE2Processor,
)
from transformers.convert_slow_tokenizer import TikTokenConverter


# The special tokens after the 151643 ordinary ones, in id order: those of the original's `qwen.tiktoken` encoding
# (`<abc>` and `</abc>` replace two of its 200 extra tokens), then the tokens opening and closing the semantic tokens,
# which the original leaves unnamed.
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
SPECIAL_TOKENS += [f"<extra_{index}>" for index in range(200)]
SPECIAL_TOKENS[204:206] = ["<abc>", "</abc>"]
SPECIAL_TOKENS += ["<music>", "</music>"]


def convert_config(original_config: dict) -> YuE2Config:
    if original_config.get("latent_type", "vae") != "vae":
        raise ValueError("Only the released latent type (`vae`) is supported.")
    return YuE2Config(
        vocab_size=original_config["vocab_size"],
        hidden_size=original_config["hidden_size"],
        intermediate_size=original_config["intermediate_size"],
        num_hidden_layers=original_config["num_hidden_layers"],
        num_attention_heads=original_config["num_attention_heads"],
        num_key_value_heads=original_config["num_key_value_heads"],
        head_dim=original_config["head_dim"],
        max_position_embeddings=original_config["max_position_embeddings"],
        rms_norm_eps=original_config["rms_norm_eps"],
        tie_word_embeddings=original_config["tie_word_embeddings"],
        rope_parameters={"rope_type": "default", "rope_theta": float(original_config["rope_theta"])},
        latent_dim=original_config["latent_dim"],
        max_latent_frames=original_config["max_latent_frames"],
        timestep_shift=original_config["timestep_shift"],
    )


def convert_generation_config(model_generation_config: dict, yue2_generation_config: dict) -> GenerationConfig:
    """
    The semantic tokens' sampling parameters are the generation config's, the score's are its `abc_generation_config`
    overrides, as the original's `semantic` and `abc` presets.
    """

    def sampling_parameters(preset: dict) -> dict:
        return {
            "temperature": preset["temperature"],
            "top_p": preset["top_p"],
            "top_k": preset["top_k"],
            "repetition_penalty": preset["repetition_penalty"],
            "repetition_penalty_window": preset["penalty_window"],
            "min_new_tokens": preset["min_tokens"],
            "max_new_tokens": preset["max_tokens"],
        }

    if yue2_generation_config["ode_method"] != "midpoint":
        raise ValueError("Only the released midpoint solver is supported.")
    return GenerationConfig(
        do_sample=True,
        **sampling_parameters(yue2_generation_config["semantic"]),
        abc_generation_config=sampling_parameters(yue2_generation_config["abc"]),
        num_flow_steps=yue2_generation_config["ode_steps"],
        bos_token_id=model_generation_config["bos_token_id"],
        pad_token_id=model_generation_config["pad_token_id"],
        eos_token_id=model_generation_config["eos_token_id"],
        return_dict_in_generate=True,
    )


def convert_tokenizer(tiktoken_file: str) -> Qwen2Tokenizer:
    vocab, merges = TikTokenConverter().extract_vocab_merges_from_model(tiktoken_file)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=merges, bos_token="<|endoftext|>", padding_side="left")
    tokenizer.add_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS[1:]], special_tokens=True)
    expected_ids = {token: len(vocab) + index for index, token in enumerate(SPECIAL_TOKENS)}
    actual_ids = dict(zip(SPECIAL_TOKENS, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)))
    if actual_ids != expected_ids:
        raise ValueError("The special tokens did not get the original's ids.")
    return tokenizer


def convert_checkpoint(checkpoint: str, acoustic_tokenizer: str, output_dir: str, push_to_hub: str | None = None):
    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.is_dir():
        checkpoint_dir = Path(
            snapshot_download(checkpoint, allow_patterns=["*.json", "*.safetensors", "qwen.tiktoken"])
        )

    config = convert_config(json.loads((checkpoint_dir / "config.json").read_text()))
    # the original weight names load through the `yue2` entry of `conversion_mapping.py`
    model, loading_info = YuE2ForConditionalGeneration.from_pretrained(
        checkpoint_dir, config=config, dtype=torch.bfloat16, output_loading_info=True
    )
    if loading_info["missing_keys"] or loading_info["unexpected_keys"]:
        raise ValueError(f"The original weights did not load cleanly: {loading_info}")
    model.generation_config = convert_generation_config(
        json.loads((checkpoint_dir / "generation_config.json").read_text()),
        json.loads((checkpoint_dir / "yue2_generation_config.json").read_text()),
    )
    model.save_pretrained(output_dir)
    del model

    tokenizer = convert_tokenizer(str(checkpoint_dir / "qwen.tiktoken"))
    processor = YuE2Processor(
        tokenizer, audio_tokenizer=YuE2AcousticTokenizerModel.from_pretrained(acoustic_tokenizer)
    )
    processor.save_pretrained(output_dir)

    # reload, as a user would
    YuE2ForConditionalGeneration.from_pretrained(output_dir, dtype=torch.bfloat16)
    YuE2Processor.from_pretrained(output_dir)
    print(f"Saved the converted model to {output_dir}")

    if push_to_hub:
        YuE2ForConditionalGeneration.from_pretrained(output_dir).push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)
        print(f"Pushed the converted model to {push_to_hub}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="m-a-p/YuE2-3B", help="Hub id or local directory of the original model."
    )
    parser.add_argument(
        "--acoustic_tokenizer",
        default="bezzam/yue2-acoustic-tokenizer",
        help="Hub id or local directory of the converted acoustic tokenizer, which the processor references.",
    )
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model and processor.")
    parser.add_argument("--push_to_hub", default=None, help="Hub repository to push the converted model to.")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.acoustic_tokenizer, args.output_dir, args.push_to_hub)
