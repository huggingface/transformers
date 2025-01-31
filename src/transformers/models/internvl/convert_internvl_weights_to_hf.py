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
    is_vision_available,
)
from transformers.tokenization_utils import AddedToken


if is_vision_available():
    pass
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
# fmt: on

CONTEXT_LENGTH = 8192


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join([key for key in state_dict_keys if key.startswith("vision_model")])
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION.items():
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
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
    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
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
    # gc.collect()
    # print("Reloading the model to check if it's saved correctly.")
    # model = InternVLForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    # processor = InternVLProcessor.from_pretrained(model_path)
    # image = load_image(
    #     "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
    # )

    # inputs = processor(image, return_tensors="pt", format=True).to(model.device, dtype=model.dtype)
    # generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
    # decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
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


# def write_image_processor(save_dir: str, push_to_hub: bool = False):
#     image_processor = GotOcr2ImageProcessor(
#         do_resize=True,
#         size={"height": 1024, "width": 1024},
#         do_rescale=True,
#         rescale_factor=1 / 255,
#         do_normalize=True,
#         image_mean=[0.48145466, 0.4578275, 0.40821073],
#         image_std=[0.26862954, 0.26130258, 0.27577711],
#     )

#     image_processor.save_pretrained(save_dir)
#     if push_to_hub:
#         image_processor.push_to_hub("stepfun-ai/GOT-OCR-2.0-hf", use_temp_dir=True)


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

    # write_image_processor(
    #     save_dir=args.output_dir,
    #     push_to_hub=args.push_to_hub,
    # )
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
