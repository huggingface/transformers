# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
import re

import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLImageProcessor,
    InternVLProcessor,
    is_vision_available,
)
from transformers.tokenization_utils import AddedToken


if is_vision_available():
    from transformers.image_utils import load_image
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION = {
    # Vision encoder mapping
    r"vision_model":                                 r"vision_tower",
    r"layers":                                      r"layer",
    r"class_embedding":                             r"cls_token",
    r"position_embedding":                          r"position_embeddings",
    r"patch_embedding":                             r"patch_embeddings.projection",
    r"ls(\d+)":                                     r"lambda_\1",
    r"attn.proj":                                   r"attention.output.dense",
    r"mlp.fc1":                                     r"intermediate.dense",
    r"mlp.fc2":                                     r"output.dense",
    r"norm1":                                       r"layernorm_before",
    r"norm2":                                       r"layernorm_after",

}
ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT = {
    # Vision encoder mapping
    # r"language_model":                                 r"vision_tower",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI = {
    # Vision encoder mapping
    r"mlp1.0":                                 r"multi_modal_projector.layer_norm",
    r"mlp1.1":                                 r"multi_modal_projector.linear_1",
    r"mlp1.3":                                 r"multi_modal_projector.linear_2",
}
# fmt: on

CONTEXT_LENGTH = 8192


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text_vision = "\n".join([key for key in state_dict_keys if key.startswith("vision_model")])
        new_text = old_text_vision
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION.items():
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text_vision.split("\n"), new_text.split("\n")))
        old_text_language = "\n".join([key for key in state_dict_keys if key.startswith("language_model")])
        new_text = old_text_language
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT.items():
            new_text = re.sub(pattern, replacement, new_text)
        output_dict.update(dict(zip(old_text_language.split("\n"), new_text.split("\n"))))
        old_text_multi = "\n".join(
            [
                key
                for key in state_dict_keys
                if not (key.startswith("language_model") or key.startswith("vision_model"))
            ]
        )
        new_text = old_text_multi
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI.items():
            new_text = re.sub(pattern, replacement, new_text)
        output_dict.update(dict(zip(old_text_multi.split("\n"), new_text.split("\n"))))

    return output_dict


def load_original_state_dict(input_base_path):
    model = (
        AutoModel.from_pretrained(
            input_base_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    return model.state_dict()


def get_internvl_config():
    return InternVLConfig()


def write_model(
    model_path,
    input_base_path,
    push_to_hub=False,
):
    os.makedirs(model_path, exist_ok=True)

    config = get_internvl_config()
    config.architectures = ["InternVLForConditionalGeneration"]
    config.save_pretrained(model_path)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    state_dict_old = load_original_state_dict(input_base_path)
    print("Converting model...")
    all_keys = list(state_dict_old.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)
    dim = config.vision_config.hidden_size
    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        if "attn.qkv" in key:
            new_key_query = new_key.replace("attn.qkv", "attention.attention.query")
            state_dict[new_key_query] = state_dict_old[key][:dim]

            new_key_key = new_key.replace("attn.qkv", "attention.attention.key")
            state_dict[new_key_key] = state_dict_old[key][dim : 2 * dim]

            new_key_value = new_key.replace("attn.qkv", "attention.attention.value")
            state_dict[new_key_value] = state_dict_old[key][-dim:]
        else:
            state_dict[new_key] = state_dict_old[key]

    del state_dict_old
    gc.collect()

    print("Loading the checkpoint in a GotOcr2ForConditionalGeneration model.")
    model = InternVLForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    print("model dtype:", model.dtype)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print("Saving the model.")
    model.save_pretrained(model_path)
    # if push_to_hub:
    #     model.push_to_hub("stepfun-ai/GOT-OCR-2.0-hf", use_temp_dir=True)
    # del state_dict, model

    # # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = InternVLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    image_processor = InternVLImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = InternVLProcessor(image_processor=image_processor, tokenizer=tokenizer)
    image = load_image("./000000039769.jpg")
    prompt = "<|im_start|>user\n<image>\nPlease describe the image shortly.<|im_end|>\n<|im_start|>assistant"
    inputs = processor(images=[image], text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    # expected_output = "\\title{\nR"
    # print("Decoded output:", decoded_output)
    # assert decoded_output == expected_output
    # print("Model reloaded successfully.")
    # del model


def write_tokenizer(save_dir: str, push_to_hub: bool = False):
    tokenizer_fast = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer_fast.model_max_length = CONTEXT_LENGTH
    tokenizer_fast.add_special_tokens(
        {
            "additional_special_tokens": [
                AddedToken(
                    "<img>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "</img>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "<IMG_CONTEXT>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "<quad>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "</quad>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "<ref>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "</ref>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "<box>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
                AddedToken(
                    "</box>",
                    rstrip=False,
                    lstrip=False,
                    single_word=False,
                    normalized=False,
                    special=True,
                ),
            ]
        },
        replace_additional_special_tokens=False,
    )
    tokenizer_fast.save_pretrained(save_dir)

    # if push_to_hub:
    #     tokenizer.push_to_hub("stepfun-ai/GOT-OCR-2.0-hf", use_temp_dir=True)


def write_image_processor(save_dir: str, push_to_hub: bool = False):
    image_processor = InternVLImageProcessor(
        do_resize=True,
        size={"height": 448, "width": 448},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        do_center_crop=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    image_processor.save_pretrained(save_dir)
    # if push_to_hub:
    #     image_processor.push_to_hub("stepfun-ai/GOT-OCR-2.0-hf", use_temp_dir=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="OpenGVLab/InternVL2_5-1B-MPO",
        help="Location of original InternVL model",
    )
    parser.add_argument(
        "--output_dir",
        default="InternVLTest",
        help="Location to write HF model and processors",
    )

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    write_tokenizer(
        save_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
    )

    write_image_processor(
        save_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
    )
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
