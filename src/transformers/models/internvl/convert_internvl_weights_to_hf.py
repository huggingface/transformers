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
from einops import rearrange

from transformers import (
    AutoModel,
    AutoTokenizer,
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLImageProcessor,
    InternVLProcessor,
    LlamaConfig,
    Qwen2Config,
)
from transformers.tokenization_utils import AddedToken


LM_TYPE_CORRESPONDENCE = {
    "OpenGVLab/InternVL2_5-1B-MPO": "qwen2",
    "OpenGVLab/InternVL2_5-2B-MPO": "llama",
    "OpenGVLab/InternVL2_5-4B-MPO": "qwen2",
}
# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION = {
    # Vision encoder mapping
    r"vision_model":                                r"vision_tower",
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

ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_LLAMA = {
    # Vision encoder mapping
    r"tok_embeddings":                              r"embed_tokens",
    r"attention.wo":                                r"self_attn.o_proj",
    r"feed_forward.w1":                             r"mlp.gate_proj",
    r"feed_forward.w2":                             r"mlp.down_proj",
    r"feed_forward.w3":                             r"mlp.up_proj",
    r"attention_norm":                              r"input_layernorm",
    r"ffn_norm":                                    r"post_attention_layernorm",
    r"output":                                      r"lm_head",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI = {
    # Vision encoder mapping
    r"mlp1.0":                                 r"multi_modal_projector.layer_norm",
    r"mlp1.1":                                 r"multi_modal_projector.linear_1",
    r"mlp1.3":                                 r"multi_modal_projector.linear_2",
}


chat_template = (
    "{% for message in messages %}"
        "{{'\n<|im_start|>' + message['role'] + '\n'}}"
        "{% if message['content'] is string %}"
            "{{ message['content'] }}"
        "{% else %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' %}"
                    "{{ '<image>\n' }}"
                "{% elif content['type'] == 'text' %}"
                    "{{ content['text'] }}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
        "{{'<|im_end|>'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{'\n<|im_start|>assistant\n' }}"
    "{% endif %}"
)
# fmt: on

CONTEXT_LENGTH = 8192


def convert_old_keys_to_new_keys(state_dict_keys: dict = None, path: str = None):
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
        if LM_TYPE_CORRESPONDENCE[path] == "llama":
            for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_LLAMA.items():
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


def get_internvl_config(input_base_path):
    base_config = AutoModel.from_pretrained(input_base_path, trust_remote_code=True).config
    llm_config = base_config.llm_config.to_dict()
    vision_config = base_config.vision_config.to_dict()
    vision_config["use_absolute_position_embeddings"] = True
    if LM_TYPE_CORRESPONDENCE[input_base_path] == "qwen2":
        image_token_index = 151667
        language_config_class = Qwen2Config
    else:
        image_token_index = 92546
        language_config_class = LlamaConfig
    # if LM_TYPE_CORRESPONDENCE[input_base_path] == "qwen2":
    #     for config_arg in llm_config.__dict__.keys():
    #         if config_arg not in Qwen2Config.__dict__.keys():
    #             delattr(llm_config, config_arg)
    # elif LM_TYPE_CORRESPONDENCE[input_base_path] == "llama":
    #     for config_arg in llm_config.__dict__.keys():
    #         if config_arg not in LlamaConfig.__dict__.keys():
    #             delattr(llm_config, config_arg)

    # for config_arg in vision_config.__dict__.keys():
    #     if config_arg not in InternVLVisionConfig.__dict__.keys():
    #         delattr(vision_config)

    return InternVLConfig(
        text_config=language_config_class(**llm_config),
        # vision_config=InternVLVisionConfig(),
        image_token_index=image_token_index,
    )


def write_model(
    model_path,
    input_base_path,
    push_to_hub=False,
):
    os.makedirs(model_path, exist_ok=True)

    config = get_internvl_config(input_base_path)
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
    new_keys = convert_old_keys_to_new_keys(all_keys, path=input_base_path)
    lm_dim = config.text_config.hidden_size
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
        elif "attention.wqkv" in key:
            num_key_value_groups = config.text_config.num_attention_heads // config.text_config.num_key_value_heads
            head_dim = config.text_config.head_dim
            wqkv_weights = state_dict_old[key]

            qkv_vecs = rearrange(wqkv_weights, "(h gs d) z -> h gs d z", gs=2 + num_key_value_groups, d=head_dim)
            q_proj = qkv_vecs[:, :num_key_value_groups, ...].reshape(-1, lm_dim).contiguous()
            k_proj = qkv_vecs[:, -2, ...].reshape(-1, lm_dim).contiguous()
            v_proj = qkv_vecs[:, -1, ...].reshape(-1, lm_dim).contiguous()

            new_key_query = new_key.replace("attention.wqkv", "self_attn.q_proj")
            state_dict[new_key_query] = q_proj

            new_key_key = new_key.replace("attention.wqkv", "self_attn.k_proj")
            state_dict[new_key_key] = k_proj

            new_key_value = new_key.replace("attention.wqkv", "self_attn.v_proj")
            state_dict[new_key_value] = v_proj
        else:
            state_dict[new_key] = state_dict_old[key]

    del state_dict_old
    gc.collect()

    print("Loading the checkpoint in a InternVLForConditionalGeneration model.")
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
    processor = InternVLProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                {"type": "text", "text": "Please describe the image shortly."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, torch.bfloat16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded_output = processor.decode(output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)

    expected_output = "The image shows two cats lying on a pink couch. One cat is curled up with its head resting on the couch, while the other is lying on its side with its head on the pink surface. There are two remote controls placed on the couch next to the cats."
    print("Decoded output:", decoded_output)
    assert decoded_output == expected_output
    print("Model reloaded successfully.")
    del model


def write_tokenizer(save_dir: str, push_to_hub: bool = False, path: str = None):
    if LM_TYPE_CORRESPONDENCE[path] == "qwen2":
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
    else:
        tokenizer = AutoTokenizer.from_pretrained("intern_vl_hf_implem/test_fast_tokenizer_llama_gt")
        tokenizer.save_pretrained(save_dir)

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
        do_convert_rgb=True,
    )

    image_processor.save_pretrained(save_dir)
    # if push_to_hub:
    #     image_processor.push_to_hub("stepfun-ai/GOT-OCR-2.0-hf", use_temp_dir=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="OpenGVLab/InternVL2_5-2B-MPO",
        help="Location of original InternVL model",
    )
    parser.add_argument(
        "--output_dir",
        default="InternVLTest-2B",
        help="Location to write HF model and processors",
    )

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    write_tokenizer(
        save_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        path=args.input_dir,
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
