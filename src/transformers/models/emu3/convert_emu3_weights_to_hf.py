# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
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
import json
import os
import re
from typing import Dict, Optional

import requests
import torch
from accelerate import init_empty_weights
from PIL import Image

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Emu3Config,
    Emu3ForConditionalGeneration,
    Emu3ImageProcessor,
    Emu3Processor,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


"""
Sample usage:

```
python src/transformers/models/emu3/convert_emu3_weights_to_hf.py \
    --vq_model_id BAAI/Emu3-VisionTokenizer --llm_model_id BAAI/Emu3-Chat --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import Emu3ForConditionalGeneration, Emu3Processor

model = Emu3ForConditionalGeneration.from_pretrained("/output/path")
processor = Emu3Processor.from_pretrained("/output/path")
```

"""


byte_encoder = bytes_to_unicode()
CHAT_TEMPLATE = "TODO: should be almost same as llava-1.5 vicuna"


# Tiktoken to HF conversion, thanks for Xenova
def token_bytes_to_string(b):
    return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])


# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def bpe(mergeable_ranks: Dict[bytes, int], token: bytes, max_rank: Optional[int] = None):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    return parts


def generate_vocab_and_merges(encoder):
    mergeable_ranks = encoder._mergeable_ranks

    merges = []
    vocab = {}
    for token, rank in mergeable_ranks.items():
        vocab[token_bytes_to_string(token)] = rank

        if len(token) == 1:
            continue
        merged = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(merged) == 2
        merges.append(" ".join(map(token_bytes_to_string, merged)))

    # Also add special tokens
    vocab.update(encoder._special_tokens)
    return vocab, merges


def convert_tiktoken(tokenizer, output_dir):
    encoder = tokenizer.tokenizer
    vocab, merges = generate_vocab_and_merges(encoder)
    added_tokens = [
        {
            "id": id,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for content, id in encoder._special_tokens.items()
    ]

    # https://huggingface.co/Xenova/gpt2/raw/main/tokenizer_config.json
    tokenizer_config_template = {
        "add_prefix_space": False,
        "bos_token": "<|extra_203|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|extra_204|>",
        "pad_token": "<|endoftext|>",
    }
    tokenizer_config_template.update({"tokenizer_class": "GPT2Tokenizer"})
    tokenizer_config_template = dict(sorted(tokenizer_config_template.items(), key=lambda x: x[0]))

    os.makedirs(output_dir, exist_ok=True)

    pre_tokenizer = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }

    # https://huggingface.co/Xenova/gpt2/raw/main/tokenizer.json
    tokenizer_template = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": vocab,
            "merges": merges,
        },
    }

    # Save to files
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as fp:
        json.dump(vocab, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as fp:
        json.dump(tokenizer_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as fp:
        json.dump(tokenizer_config_template, fp, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "special_tokens_map.json"), "w", encoding="utf-8") as fp:
        json.dump(
            {
                "bos_token": "<|extra_203|>",
                "eos_token": "<|extra_204|>",
                "pad_token": "<|endoftext|>",
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as fp:
        fp.write("#version: 0.2\n")
        fp.write("\n".join(merges))


KEYS_TO_MODIFY_MAPPING = {
    "^encoder": "model.vqmodel.encoder",
    "^decoder": "model.vqmodel.decoder",
    "^post_quant_conv": "model.vqmodel.post_quant_conv",
    "^quant_conv": "model.vqmodel.quant_conv",
    "^quantize": "model.vqmodel.quantize",
}


def convert_state_dict_to_hf(old_state_dict, new_state_dict):
    for key, value in old_state_dict.items():
        for old_pattern, new_pattern in KEYS_TO_MODIFY_MAPPING.items():
            key = re.sub(old_pattern, new_pattern, key)

        new_state_dict[key] = value
    return new_state_dict


def convert_model(vq_model_id, llm_model_id, output_dir, test_inference=False):
    os.makedirs(output_dir, exist_ok=True)

    # Convert and save processor
    tokenizer_tiktoken = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
    convert_tiktoken(tokenizer_tiktoken, output_dir)
    tokenizer_converted = AutoTokenizer.from_pretrained(output_dir)

    image_processor = Emu3ImageProcessor.from_pretrained(vq_model_id)
    processor = Emu3Processor(image_processor, tokenizer_converted, chat_template=CHAT_TEMPLATE)
    processor.save_pretrained(output_dir)

    # load models
    model_llm = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        trust_remote_code=True,
    )
    model_vqgan = AutoModel.from_pretrained(vq_model_id, trust_remote_code=True)
    with open(f"{output_dir}/tokenizer.json", "r") as file:
        tokenizer_config = json.load(file)
    vocabulary_map = tokenizer_config["model"]["vocab"]

    config = Emu3Config(
        max_position_embeddings=model_llm.config.max_position_embeddings,
        rope_scaling={"rope_type": "default"},
        vocabulary_map=vocabulary_map,
    )

    with init_empty_weights():
        model = Emu3ForConditionalGeneration(config=config)

    state_dict = {}
    state_dict = convert_state_dict_to_hf(model_llm.state_dict(), state_dict)
    state_dict = convert_state_dict_to_hf(model_vqgan.state_dict(), state_dict)

    model.load_state_dict(state_dict, assign=True, strict=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    if test_inference:
        # Short inference on a few examples to check if generation makes sense
        print("Loading the checkpoint in a Emu3 model...")
        print("*" * 100)
        model = Emu3ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
        processor = Emu3Processor.from_pretrained(output_dir)

        prompt = "I'm very intrigued by this work of art:<image>Please tell me about the artist."
        image = Image.open(
            requests.get(
                "https://uploads4.wikiart.org/images/paul-klee/death-for-the-idea-1915.jpg!Large.jpg", stream=True
            ).raw
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
        length = inputs.input_ids.shape[1]

        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = processor.batch_decode(out[:, length:], skip_special_tokens=True)[0]

        print(f"Generation for single-image: {generated_text}")
        print("*" * 100)

        # Multi-image example
        # prompt = "I used to know a lot about constellations when I was younger, but as I grew older, I forgot most of what I knew. These are the only two constellations that I really remember now.<image><image>I would like for you to tell me about 3 more constellations and give me a little bit of history about the constellation."
        # image = Image.open(
        #     requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        # )
        # image_2 = Image.open(
        #     requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
        # )
        # inputs = processor(images=[image, image_2], text=prompt, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        # length = inputs.input_ids.shape[1]
        # out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        # generated_text = processor.batch_decode(out[:, length:], skip_special_tokens=True)[0]
        # print(f"Generation for multi-image: {generated_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vq_model_id",
        help="Model ID of Emu3 VQ-VAE on the hub",
        default="BAAI/Emu3-VisionTokenizer",
    )
    parser.add_argument(
        "--llm_model_id",
        help="Model ID of Emu3 bacbone LLM on the hub",
        default="BAAI/Emu3-Chat",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Whether to load the model for generation to test it's converted correctly.",
    )
    args = parser.parse_args()
    convert_model(
        vq_model_id=args.vq_model_id,
        llm_model_id=args.llm_model_id,
        output_dir=args.output_dir,
        test_inference=args.test_inference,
    )


if __name__ == "__main__":
    main()
