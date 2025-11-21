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
    Emu3TextConfig,
    GenerationConfig,
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
CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"


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
        if content != "<|extra_0|>"
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

    # add placeholder image token by taking one of the reserved tokens
    reserved_token_id = vocab["<|extra_0|>"]
    vocab["<image>"] = reserved_token_id
    del vocab["<|extra_0|>"]
    added_tokens.append(
        {
            "id": reserved_token_id,
            "content": "<image>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )

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
    "^model": "text_model.model",
    r"lm_head\.weight": "text_model.lm_head.weight",
    r"^text_model\.model\.vqmodel": "vqmodel",
    # rename QKV proj for the VQ-VAE model because we use SiglipAttention
    r"\.q\.": ".q_proj.",
    r"\.k\.": ".k_proj.",
    r"\.v\.": ".v_proj.",
    r"\.proj_out\.": ".out_proj.",
    # move the attention norms outside of attention modules
    r"mid\.attn_1\.norm\.": "mid.attn_norm.",
    r"attn\.0\.norm\.": "attn_norms.0.",
    r"attn\.1\.norm\.": "attn_norms.1.",
    r"attn\.2\.norm\.": "attn_norms.2.",
    r"attn\.3\.norm\.": "attn_norms.3.",
    # isolate down/mid/up into separate classes for readability
    r"\.down\.": ".down_block.down.",
    r"\.up\.": ".up_block.up.",
    r"\.mid\.": ".middle_block.",
}


def convert_state_dict_to_hf(old_state_dict, new_state_dict):
    for key, value in old_state_dict.items():
        # convert conv layers in attn to linear
        if (
            any(key.endswith(name) for name in ["q.weight", "k.weight", "v.weight", "proj_out.weight"])
            and value.ndim == 4
        ):
            value = value.squeeze()

        for old_pattern, new_pattern in KEYS_TO_MODIFY_MAPPING.items():
            key = re.sub(old_pattern, new_pattern, key)

        new_state_dict[key] = value
    return new_state_dict


def convert_model(vq_model_id, llm_model_id, output_dir, hub_model_id=None, test_inference=False):
    os.makedirs(output_dir, exist_ok=True)

    # Convert and save processor
    tokenizer_tiktoken = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
    convert_tiktoken(tokenizer_tiktoken, output_dir)
    extra_special_tokens = extra_special_tokens = {
        "image_token": "<image>",
        "boi_token": "<|image start|>",
        "eoi_token": "<|image end|>",
        "image_wrapper_token": "<|image token|>",
        "eof_token": "<|extra_201|>",
    }
    tokenizer_converted = AutoTokenizer.from_pretrained(output_dir, extra_special_tokens=extra_special_tokens)
    tokenizer_converted.padding_side = "left"

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

    text_config = Emu3TextConfig(
        max_position_embeddings=model_llm.config.max_position_embeddings,
        rope_scaling={"rope_type": "default"},
    )
    config = Emu3Config(text_config=text_config, vocabulary_map=vocabulary_map)

    with init_empty_weights():
        model = Emu3ForConditionalGeneration(config=config)
        model.generation_config = GenerationConfig(
            do_sample=True,
            top_k=2048,
            max_new_tokens=50_000,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    state_dict = {}
    state_dict = convert_state_dict_to_hf(model_llm.state_dict(), state_dict)
    state_dict = convert_state_dict_to_hf(model_vqgan.state_dict(), state_dict)

    model.load_state_dict(state_dict, assign=True, strict=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    if hub_model_id is not None:
        model.push_to_hub(hub_model_id)
        processor.push_to_hub(hub_model_id)

    if test_inference and llm_model_id.endswith("Chat"):
        # Short inference on a few examples to check if generation makes sense
        print("Loading the checkpoint in a Emu3 model...")
        print("*" * 100)
        model = Emu3ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")
        processor = Emu3Processor.from_pretrained(output_dir)

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please tell me about this art work and its artist."},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

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
    elif test_inference and llm_model_id.endswith("Gen"):
        processor = Emu3Processor.from_pretrained(output_dir)
        model = Emu3ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="auto")

        inputs = processor(
            text=[
                "a portrait of young girl. masterpiece, film grained, best quality.",
                "a dog running under the rain",
            ],
            padding=True,
            return_tensors="pt",
            return_for_image_generation=True,
        )
        inputs = inputs.to(device="cuda:0", dtype=torch.bfloat16)

        neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
        neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="pt").to(device="cuda:0")

        image_sizes = inputs.pop("image_sizes")
        HEIGHT, WIDTH = image_sizes[0]
        VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            height, width = HEIGHT, WIDTH
            visual_tokens = VISUAL_TOKENS
            image_token_id = processor.tokenizer.encode("<|image token|>", return_tensors="pt")[0].to(model.device)
            eoi_token_id = processor.tokenizer.encode("<|image end|>", return_tensors="pt")[0]
            eos_token_id = processor.tokenizer.encode("<|extra_204|>", return_tensors="pt")[0]
            pad_token_id = processor.tokenizer.encode("<|endoftext|>", return_tensors="pt")[0]
            eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]
            eof_token_id = processor.tokenizer.encode("<|extra_201|>", return_tensors="pt")[0]

            position = torch.nonzero(input_ids == image_token_id, as_tuple=True)[0][0]
            offset = input_ids.shape[0] - position
            if offset % (width + 1) == 0:
                return (eol_token_id,)
            elif offset == (width + 1) * height + 1:
                return (eof_token_id,)
            elif offset == (width + 1) * height + 2:
                return (eoi_token_id,)
            elif offset == (width + 1) * height + 3:
                return (eos_token_id,)
            elif offset > (width + 1) * height + 3:
                return (pad_token_id,)
            else:
                return visual_tokens

        out = model.generate(
            **inputs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            negative_prompt_ids=neg_inputs.input_ids,
            negative_prompt_attention_mask=neg_inputs.attention_mask,
        )

        image = model.decode_image_tokens(out[:, inputs.input_ids.shape[1] :], height=HEIGHT, width=WIDTH)
        images = processor.postprocess(
            list(image.float()), return_tensors="PIL.Image.Image"
        )  # internally we convert to np but it's not supported in bf16 precision
        for i, image in enumerate(images["pixel_values"]):
            image.save(f"result_{i}.png")


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
        "--hub_model_id",
        help="Model ID in the hub where to push the model.",
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
        hub_model_id=args.hub_model_id,
        test_inference=args.test_inference,
    )


if __name__ == "__main__":
    main()
