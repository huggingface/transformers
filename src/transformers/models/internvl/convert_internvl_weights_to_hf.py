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
from typing import Literal, Optional

import torch
from einops import rearrange

from transformers import (
    AutoModel,
    AutoTokenizer,
    GenerationConfig,
    GotOcr2ImageProcessorFast,
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLProcessor,
    InternVLVideoProcessor,
    InternVLVisionConfig,
    LlamaConfig,
    Qwen2Config,
)


LM_TYPE_CORRESPONDENCE = {
    "OpenGVLab/InternVL2_5-1B-MPO": "qwen2",
    "OpenGVLab/InternVL2_5-2B-MPO": "llama",
    "OpenGVLab/InternVL2_5-4B-MPO": "qwen2",
    "OpenGVLab/InternVL2_5-8B-MPO": "llama",
    "OpenGVLab/InternVL2_5-26B-MPO": "llama",
    "OpenGVLab/InternVL2_5-38B-MPO": "qwen2",
    "OpenGVLab/InternVL2_5-78B-MPO": "qwen2",
    "OpenGVLab/InternVL3-1B": "qwen2",
    "OpenGVLab/InternVL3-2B": "qwen2",
    "OpenGVLab/InternVL3-8B": "qwen2",
    "OpenGVLab/InternVL3-9B": "llama",
    "OpenGVLab/InternVL3-14B": "qwen2",
    "OpenGVLab/InternVL3-38B": "qwen2",
    "OpenGVLab/InternVL3-78B": "qwen2",
}

UNNECESSARY_CONFIG_KEYS = [ "_name_or_path", "_attn_implementation_autoset", "auto_map", "use_bfloat16", "use_flash_attn", "bias", "laux_allreduce", "moe_coeff_ratio", "moe_intermediate_size", "moe_output_scale", "noisy_gate_policy", "shared_expert_intermediate_size", "use_residual", "use_moe", "use_rts", "use_weighted_residual", "moe_config", "num_experts", "num_routed_experts", "num_shared_experts", "capacity_factor", "eval_capacity_factor", "drop_path_rate"]  # fmt: skip

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING_VISION = {
    # Vision encoder mapping
    r"vision_model":                                r"model.vision_tower",
    r"layers":                                      r"layer",
    r"class_embedding":                             r"cls_token",
    r"position_embedding":                          r"position_embeddings",
    r"patch_embedding":                             r"patch_embeddings.projection",
    r"ls(\d+)":                                     r"lambda_\1",
    r"attn.proj":                                   r"attention.projection_layer",
    r"attn.dropout":                                r"attention.projection_dropout",
    r"attn":                                        r"attention",
    r"norm1":                                       r"layernorm_before",
    r"norm2":                                       r"layernorm_after",

}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_LLAMA = {
    r"language_model.model.":                       r"model.language_model.",
    r"tok_embeddings":                              r"embed_tokens",
    r"attention.wo":                                r"self_attn.o_proj",
    r"feed_forward.w1":                             r"mlp.gate_proj",
    r"feed_forward.w2":                             r"mlp.down_proj",
    r"feed_forward.w3":                             r"mlp.up_proj",
    r"attention_norm":                              r"input_layernorm",
    r"ffn_norm":                                    r"post_attention_layernorm",
    r"language_model.output":                       r"lm_head",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_QWEN2 = {
    # Vision encoder mapping
    r"language_model.model.":                       r"model.language_model.",
    r"language_model.lm_head":                       r"lm_head",
}

ORIGINAL_TO_CONVERTED_KEY_MAPPING_MULTI = {
    # Vision encoder mapping
    r"mlp1.0":                                 r"model.multi_modal_projector.layer_norm",
    r"mlp1.1":                                 r"model.multi_modal_projector.linear_1",
    r"mlp1.3":                                 r"model.multi_modal_projector.linear_2",
}


chat_template = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% if message['content'] is string %}"
            "{{ message['content'] }}"
        "{% else %}"
            "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' %}"
                    "{{ '<IMG_CONTEXT>\n' }}"
                "{% elif content['type'] == 'video' %}"
                    "{{ '<video>\n' }}"
                "{% elif content['type'] == 'text' %}"
                    "{{ content['text'] }}"
                "{% endif %}"
            "{% endfor %}"
        "{% endif %}"
        "{{'<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\n' }}"
    "{% endif %}"
)
# fmt: on

CONTEXT_LENGTH = 8192


def get_lm_type(path: str) -> Literal["qwen2", "llama"]:
    """
    Determine the type of language model (either 'qwen2' or 'llama') based on a given model path.
    """
    if path not in LM_TYPE_CORRESPONDENCE:
        base_config = AutoModel.from_pretrained(path, trust_remote_code=True).config

        lm_arch = base_config.llm_config.architectures[0]

        if lm_arch == "InternLM2ForCausalLM":
            lm_type = "llama"
        elif lm_arch == "Qwen2ForCausalLM":
            lm_type = "qwen2"
        else:
            raise ValueError(
                f"Architecture '{lm_arch}' is not supported. Only 'Qwen2ForCausalLM' and 'InternLM2ForCausalLM' are recognized."
            )
    else:
        lm_type: Literal["qwen2", "llama"] = LM_TYPE_CORRESPONDENCE[path]

    return lm_type


def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None, path: Optional[str] = None):
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
        if get_lm_type(path) == "llama":
            for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_LLAMA.items():
                new_text = re.sub(pattern, replacement, new_text)
        elif LM_TYPE_CORRESPONDENCE[path] == "qwen2":
            for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING_TEXT_QWEN2.items():
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
    model = AutoModel.from_pretrained(
        input_base_path,
        dtype=torch.bfloat16,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval()

    return model.state_dict()


def get_internvl_config(input_base_path):
    base_config = AutoModel.from_pretrained(input_base_path, trust_remote_code=True).config
    llm_config = base_config.llm_config.to_dict()
    vision_config = base_config.vision_config.to_dict()
    vision_config["use_absolute_position_embeddings"] = True
    if get_lm_type(input_base_path) == "qwen2":
        image_token_id = 151667
        language_config_class = Qwen2Config
    else:
        image_token_id = 92546
        language_config_class = LlamaConfig

    llm_config = {k: v for k, v in llm_config.items() if k not in UNNECESSARY_CONFIG_KEYS}
    # Force use_cache to True
    llm_config["use_cache"] = True
    # Force correct eos_token_id for InternVL3
    if "InternVL3" in input_base_path and get_lm_type(input_base_path) == "qwen2":
        llm_config["eos_token_id"] = 151645

    vision_config = {k: v for k, v in vision_config.items() if k not in UNNECESSARY_CONFIG_KEYS}
    if "attention_probs_dropout_prob" in vision_config:
        attention_dropout = vision_config.pop("attention_probs_dropout_prob")
        vision_config["attention_dropout"] = attention_dropout
        vision_config["projection_dropout"] = attention_dropout
    if "qk_normalization" in vision_config:
        use_qk_norm = vision_config.pop("qk_normalization")
        vision_config["use_qk_norm"] = use_qk_norm
    if "qkv_bias" in vision_config:
        attention_bias = vision_config.pop("qkv_bias")
        vision_config["attention_bias"] = attention_bias

    return InternVLConfig(
        text_config=language_config_class(**llm_config),
        vision_config=InternVLVisionConfig(**vision_config),
        image_token_id=image_token_id,
    )


def write_model(
    model_path,
    input_base_path,
    push_to_hub=False,
    hub_dir=None,
):
    os.makedirs(model_path, exist_ok=True)

    config = get_internvl_config(input_base_path)
    config.architectures = ["InternVLForConditionalGeneration"]
    config.save_pretrained(model_path)
    if push_to_hub:
        config.push_to_hub(hub_dir, use_temp_dir=True)
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
            new_key_query = new_key.replace("attention.qkv", "attention.q_proj")
            state_dict[new_key_query] = state_dict_old[key][:dim]

            new_key_key = new_key.replace("attention.qkv", "attention.k_proj")
            state_dict[new_key_key] = state_dict_old[key][dim : 2 * dim]

            new_key_value = new_key.replace("attention.qkv", "attention.v_proj")
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
    if push_to_hub:
        model.push_to_hub(hub_dir, use_temp_dir=True)

    image_processor = GotOcr2ImageProcessorFast.from_pretrained(model_path)
    video_processor = InternVLVideoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = InternVLProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
    processor.save_pretrained(model_path)
    if push_to_hub:
        processor.push_to_hub(hub_dir, use_temp_dir=True)

    # generation config
    if get_lm_type(input_base_path) == "llama":
        print("Saving generation config...")
        # in the original model, eos_token is not the same in the text_config and the generation_config
        # ("</s>" - 2 in the text_config and "<|im_end|>" - 92542 in the generation_config)
        generation_config = GenerationConfig(
            eos_token_id=92542,
        )
        generation_config.save_pretrained(model_path)
        if push_to_hub:
            generation_config.push_to_hub(hub_dir, use_temp_dir=True)

    # del state_dict, model

    # # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = InternVLForConditionalGeneration.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)
    print("Model reloaded successfully.")
    del model


def write_tokenizer(
    save_dir: str, push_to_hub: bool = False, path: Optional[str] = None, hub_dir: Optional[str] = None
):
    if get_lm_type(path) == "qwen2":
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            return_token_type_ids=False,
            extra_special_tokens={
                "start_image_token": "<img>",
                "end_image_token": "</img>",
                "context_image_token": "<IMG_CONTEXT>",
                "video_token": "<video>",
            },
        )
        tokenizer.model_max_length = CONTEXT_LENGTH
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<img>",
                    "</img>",
                    "<IMG_CONTEXT>",
                    "<quad>",
                    "</quad>",
                    "<ref>",
                    "</ref>",
                    "<box>",
                    "</box>",
                ]
            },
            replace_additional_special_tokens=False,
        )
    else:
        # Obtained with:
        # tokenizer_llama_fast = LlamaTokenizerFast.from_pretrained(
        #     "OpenGVLab/InternVL2_5-2B-MPO", pad_token="</s>", legacy=False, from_slow=True
        # )
        # tokenizer_llama_fast._tokenizer.pre_tokenizer.prepend_scheme = "never"
        # Then manually modifying `added_tokens_decoder` indices to match the original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "./intern_vl_hf_implem/tokenizer_internvl_llama_fast",
            return_token_type_ids=False,
            extra_special_tokens={
                "start_image_token": "<img>",
                "end_image_token": "</img>",
                "context_image_token": "<IMG_CONTEXT>",
                "video_token": "<video>",
            },
        )

    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(save_dir)
    if push_to_hub:
        tokenizer.push_to_hub(hub_dir, use_temp_dir=True)


def write_image_processor(save_dir: str, push_to_hub: bool = False, hub_dir: Optional[str] = None):
    image_processor = GotOcr2ImageProcessorFast(
        do_resize=True,
        size={"height": 448, "width": 448},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_convert_rgb=True,
    )

    image_processor.save_pretrained(save_dir)
    if push_to_hub:
        image_processor.push_to_hub(hub_dir, use_temp_dir=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="OpenGVLab/InternVL3-1B",
        help="Location of original InternVL model",
    )
    parser.add_argument(
        "--output_dir",
        default="InternVL3-1B-hf",
        help="Location to write HF model and processors",
    )
    parser.add_argument(
        "--hub_dir",
        default="OpenGVLab/InternVL3-1B-hf",
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
        hub_dir=args.hub_dir,
    )

    write_image_processor(
        save_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_dir=args.hub_dir,
    )
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        push_to_hub=args.push_to_hub,
        hub_dir=args.hub_dir,
    )


if __name__ == "__main__":
    main()
