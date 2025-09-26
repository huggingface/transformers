import json
import os
import re

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from transformers import (
    AutoTokenizer,
    Ernie4_5_VLConfig,
    Ernie4_5_VLImageProcessor,
    Ernie4_5_VLImageProcessorFast,
    Ernie4_5_VLProcessor,
    Ernie4_5_VLVideoProcessor,
    LlamaTokenizer,
)


TIED_MAPPING = {
    "baidu/ERNIE-4.5-VL-28B-A3B-PT": True,
    "baidu/ERNIE-4.5-VL-28B-A3B-Base-PT": True,
    "baidu/ERNIE-4.5-VL-424B-A47B-PT": False,
    "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT": False,
}
SAFETENSOR_INDEX_NAME = "model.safetensors.index.json"

CONFIG_NAME = "config.json"
VALID_VISION_CONFIG_KEYS = [
    "depth",
    "hidden_size",
    "hidden_act",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
]
VALID_TEXT_CONFIG_KEYS = [
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "moe_intermediate_size",
    "moe_k",
    "moe_layer_interval",
    "moe_num_shared_experts",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
    "tie_word_embeddings",
    "use_cache",
    "use_bias",
]
TEXT_TO_VISION_CONFIG_KEYS = [
    "spatial_conv_size",
    "temporal_conv_size",
    "rms_norm_eps",
]
ALL_VISION_CONFIG_KEYS = (
    VALID_VISION_CONFIG_KEYS
    + TEXT_TO_VISION_CONFIG_KEYS
    + ["intermediate_size", "text_hidden_size", "vision_rms_norm_eps"]
)
ALL_TEXT_CONFIG_KEYS = VALID_TEXT_CONFIG_KEYS + [
    "hidden_act",
    "moe_layer_end_index",
    "moe_layer_start_index",
    "moe_num_experts",
    "freq_allocation",
]

TMP_TOKENIZER_DIR = "/tmp/ernie_vl_tokenizer"
ADDED_TOKENS_FILE = "added_tokens.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
DEFAULT_CHAT_TEMPLATE = """
{%- set image_count = namespace(value=0) -%}
{%- set video_count = namespace(value=0) -%}
{{- '<|begin_of_sentence|>' }}
{%- for message in messages -%}
    {%- if message.role in ['system', 'user'] -%}
        {%- if message.role == 'user' -%}
            {{- 'User: ' -}}
        {%- endif -%}
        {%- if message.content is string -%}
            {{- message.content -}}
        {%- else -%}
            {%- for content_item in message.content -%}
                {%- if content_item.type == 'text' -%}
                    {{- content_item.text -}}
                {%- elif content_item.type in ['image_url', 'image'] -%}
                    {%- set image_count.value = image_count.value + 1 -%}
                    Picture {{ image_count.value }}:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>
                {%- elif content_item.type in ['video_url', 'video'] -%}
                    {%- set video_count.value = video_count.value + 1 -%}
                    Video {{ video_count.value }}:<|VIDEO_START|><|VIDEO_PLACEHOLDER|><|VIDEO_END|>
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {%- if message.role == 'system' -%}
            {{- '
            ' -}}
        {%- endif -%}
    {%- elif message.role == 'assistant' -%}
        {%- macro extract_text_content(content_field) -%}
            {%- if content_field is string -%}
                {{- content_field -}}
            {%- elif content_field is iterable and content_field is not string -%}
                {%- set ns = namespace(text_parts=[]) -%}
                {%- set text_parts = [] -%}
                {%- for item in content_field -%}
                    {%- if item.type == 'text' -%}
                        {%- set ns.text_parts = ns.text_parts + [item.text] -%}
                    {%- endif -%}
                {%- endfor -%}
                {{- ns.text_parts | join("") -}}
            {%- else -%}
                {{- '' -}}
            {%- endif -%}
        {%- endmacro -%}
        {%- set reasoning_content = extract_text_content(message.reasoning_content) -%}
        {%- set content = extract_text_content(message.content) -%}
        {%- if '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('
                        ').split('<think>')[-1].lstrip('
                        ') %}
            {%- set content = content.split('</think>')[-1].lstrip('
                        ') %}
        {%- endif %}
        {%- if reasoning_content %}
            {{- '
            ' + 'Assistant: ' + '<think>
            ' + reasoning_content.strip('
                        ') + '
            </think>
            ' + content.lstrip('
            ') }}
        {%- else %}
            {{- '
            ' + 'Assistant: ' + content }}
        {%- endif %}
        {{- '<|end_of_sentence |>' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt is not defined or add_generation_prompt is true %}
    {{- '\nAssistant: ' -}}
    {%- if (enable_thinking is defined and enable_thinking is false) or enable_thinking is not defined %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
    {%- if enable_thinking is defined and enable_thinking is true %}{{- '<think>' }}{%- endif %}
{%- endif %}"""
DEFAULT_TEXT_ADD_TOKENS = [
    "<mask:4>",
    "<mask:5>",
    "<mask:6>",
    "<mask:7>",
]


def load_json(save_dir, filename):
    with open(os.path.join(save_dir, filename), "r") as f:
        return json.load(f)


def write_json(json_object, save_dir, filename):
    with open(os.path.join(save_dir, filename), "w") as f:
        json.dump(json_object, f)


def convert_state_dict_to_hf(state_dict, is_tied=True):
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        key = re.sub("^vision_model", "vision_tower", key)
        key = re.sub("^model", "language_model", key)
        key = re.sub("^language_model.resampler_model", "resampler_model", key)
        key = "model." + key

        if "lm_head" in key and is_tied:
            if is_tied:  # skip tied weights
                pass
            else:
                # avoid any prefix introduced before
                converted_state_dict["lm_head"] = tensor.contiguous()
        # Moe is split into their modalities (text, vision)
        elif "mlp" in key:
            if "moe_statics" in key:
                suffix = "moe_statics.e_score_correction_bias"
                converted_key = key.removesuffix(suffix)
                # splitting param (2, ...) to 2 * (1, ...)
                converted_state_dict[converted_key + "text_moe." + suffix] = tensor[0][None, :].contiguous()
                converted_state_dict[converted_key + "vision_moe." + suffix] = tensor[1][None, :].contiguous()
            elif "gate.weight" in key:
                moe_type = "text_moe"
                if "weight_1" in key:
                    moe_type = "vision_moe"
                suffix = "gate.weight"
                converted_key = key.removesuffix("_1")  # vision
                converted_key = converted_key.removesuffix("gate.weight")
                # previously a `nn.Parameter` which is why we need a transpose for `nn.Linear`
                converted_state_dict[converted_key + f"{moe_type}." + suffix] = tensor.T.contiguous()
            elif ".experts" in key:
                moe_type = "text_moe"
                expert_number = int(re.findall(r"\d+", key)[-1])
                # 128 experts split into 64 each (text, vision)
                if expert_number >= 64:
                    moe_type = "vision_moe"
                    expert_number -= 64
                # avoid subbing the layer idx + experts twice
                prefix = re.findall(r"model.language_model.layers.\d+.mlp.experts.", key)[0]
                converted_key = re.sub(r"\d+", f"{moe_type}.experts.{expert_number}", key.removeprefix(prefix))
                converted_state_dict[re.sub(".experts", "", prefix) + converted_key] = tensor.contiguous()
            else:
                converted_state_dict[key] = tensor.contiguous()
        # Convert sequential to its own module
        elif "spatial_linear" in key or "temporal_linear" in key:
            sequential_number = int(re.findall(r"\d+", key)[-1])

            if sequential_number == 0:
                converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "fc1", key)
            elif sequential_number == 2:
                converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "fc2", key)
            elif sequential_number == 3:
                converted_key = re.sub(r"(?<=\.)\d+(?=\.)", "ln", key)
            else:
                converted_key = key

            converted_state_dict[converted_key] = tensor.contiguous()
        else:
            converted_state_dict[key] = tensor.contiguous()
    return converted_state_dict


def convert_weights(model_path, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # indexing base dict
    index_dict = {"weight_map": {}, "metadata": {"total_size": 0}}

    is_tied = TIED_MAPPING[model_path]
    checkpoint_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors*"])
    for filename in sorted(os.listdir(checkpoint_path)):
        # metadata doesn't change
        if filename == SAFETENSOR_INDEX_NAME:
            original_index = load_json(checkpoint_path, filename)
            index_dict["metadata"] = original_index["metadata"]
        # sharded files are converted 1 by 1
        if filename.endswith(".safetensors"):
            input_file = os.path.join(checkpoint_path, filename)
            output_file = os.path.join(save_dir, filename)

            state_dict = load_file(input_file)
            converted_state_dict = convert_state_dict_to_hf(state_dict, is_tied=is_tied)
            save_file(converted_state_dict, output_file)

            # remap namings in index
            for k in converted_state_dict.keys():
                index_dict["weight_map"][k] = filename

    # save index
    write_json(index_dict, save_dir, SAFETENSOR_INDEX_NAME)


def convert_vision_config_to_hf(vision_config, original_config, original_vision_config):
    # convert vision related stuff
    for key in VALID_VISION_CONFIG_KEYS:
        vision_config[key] = original_vision_config[key]
    vision_config["intermediate_size"] = original_vision_config["hidden_size"] * original_vision_config["mlp_ratio"]

    # convert originally text attributes to vision
    for key in TEXT_TO_VISION_CONFIG_KEYS:
        vision_config[key.replace("conv", "merge")] = original_config[key]
    vision_config["text_hidden_size"] = original_config["hidden_size"]
    vision_config["vision_rms_norm_eps"] = 1e-6

    # delete everything else
    for key in list(vision_config.keys()):
        if key not in ALL_VISION_CONFIG_KEYS:
            del vision_config[key]

    return vision_config


def convert_text_config_to_hf(text_config, original_config):
    # carry directly over
    for key in VALID_TEXT_CONFIG_KEYS:
        text_config[key] = original_config[key]

    # special cases
    text_config["hidden_act"] = "silu"  # default value which is not explicit in their json
    text_config["moe_layer_end_index"] = max(original_config["moe_layer_end_index"])
    text_config["moe_layer_start_index"] = min(original_config["moe_layer_start_index"])
    text_config["moe_num_experts"] = original_config["moe_num_experts"][0]  # the same for both modalities
    text_config["freq_allocation"] = 20  # can also be extracted from mrope

    # delete everything else
    for key in list(text_config.keys()):
        if key not in ALL_TEXT_CONFIG_KEYS:
            del text_config[key]

    return text_config


def convert_config(model_path, save_dir):
    checkpoint_path = snapshot_download(repo_id=model_path, allow_patterns=["*.config"])
    for filename in sorted(os.listdir(checkpoint_path)):
        if filename == CONFIG_NAME:
            hf_config = Ernie4_5_VLConfig()
            original_config = load_json(checkpoint_path, filename)

            # general config
            image_token_id = original_config["im_patch_id"]

            # vision config
            vision_config = hf_config.vision_config.to_dict()
            original_vision_config = original_config["vision_config"]
            vision_config = convert_vision_config_to_hf(vision_config, original_config, original_vision_config)

            # text config
            text_config = hf_config.text_config.to_dict()
            text_config = convert_text_config_to_hf(text_config, original_config)

            # total config
            final_config = Ernie4_5_VLConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=image_token_id,
            )

            final_config.save_pretrained(save_dir)
            break


def convert_tokenizer(original_tokenizer_path, save_dir):
    # same conversion as the moe and base ernie tokenizers
    hf_tok = LlamaTokenizer.from_pretrained(
        original_tokenizer_path,
        pad_token="<unk>",
        cls_token="<|begin_of_sentence|>",
        sep_token="<|end_of_sentence|>",
        mask_token="<mask:1>",
        add_bos_token=False,
        add_prefix_space=False,
        chat_template=DEFAULT_CHAT_TEMPLATE,
        legacy=True,
    )
    hf_tok.model_max_length = 131072
    hf_tok.init_kwargs.pop("auto_map", None)
    # special tokens which we need to map as additional special tokens instead
    hf_tok.init_kwargs.pop("header_start_token", None)
    hf_tok.init_kwargs.pop("header_end_token", None)
    hf_tok.init_kwargs.pop("sys_start_token", None)
    hf_tok.init_kwargs.pop("sys_end_token", None)
    for token in DEFAULT_TEXT_ADD_TOKENS:
        hf_tok.add_tokens([token], special_tokens=True)
    # save slow model
    hf_tok.save_pretrained(TMP_TOKENIZER_DIR)

    # we will exchange the special audio token as we need a dedicated video token
    original_str = "<|AUDIO_PLACEHOLDER|>"
    new_str = "<|VIDEO_PLACEHOLDER|>"

    # overwrite every occurrence of the special tokens with the new string
    added_tokens = load_json(TMP_TOKENIZER_DIR, ADDED_TOKENS_FILE)
    original_id = added_tokens.get(original_str, -1)
    if original_id < 0:
        raise ValueError(f"The requested string '{original_str}' is not a special token.")

    added_tokens.pop(original_str)
    added_tokens[new_str] = original_id
    write_json(added_tokens, TMP_TOKENIZER_DIR, ADDED_TOKENS_FILE)

    special_tokens_map = load_json(TMP_TOKENIZER_DIR, SPECIAL_TOKENS_MAP_FILE)
    for i, token in enumerate(special_tokens_map["additional_special_tokens"]):
        if token == original_str:
            special_tokens_map["additional_special_tokens"][i] = new_str
            break
    write_json(special_tokens_map, TMP_TOKENIZER_DIR, SPECIAL_TOKENS_MAP_FILE)

    tokenizer_config = load_json(TMP_TOKENIZER_DIR, TOKENIZER_CONFIG_FILE)
    for i, token in enumerate(tokenizer_config["additional_special_tokens"]):
        if token == original_str:
            tokenizer_config["additional_special_tokens"][i] = new_str
            break
    tokenizer_config["added_tokens_decoder"][f"{original_id}"]["content"] = new_str
    write_json(tokenizer_config, TMP_TOKENIZER_DIR, TOKENIZER_CONFIG_FILE)

    # reload and save to get correct formatting
    tokenizer = AutoTokenizer.from_pretrained(TMP_TOKENIZER_DIR, from_slow=True)
    tokenizer.save_pretrained(save_dir)


def convert_processor(model_path, save_dir):
    convert_tokenizer(model_path, save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)

    processor = Ernie4_5_VLProcessor(
        # Using the slow processor for now as it changes the default output otherwise
        image_processor=Ernie4_5_VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Ernie4_5_VLVideoProcessor(),
        chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(save_dir)

    """test_1 = Ernie4_5_VLImageProcessor.from_pretrained(save_dir)
    test_2 = Ernie4_5_VLImageProcessorFast.from_pretrained(save_dir)
    test_3 = Ernie4_5_VLImageProcessorFast()

    for attr in [
        #"min_pixels",
        #"max_pixels",
        "size",
        "do_resize",
        "resample",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "image_mean",
        "image_std",
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "do_convert_rgb",
    ]:
        t1 = getattr(test_1, attr, None)
        t2 = getattr(test_2, attr, None)
        t3 = getattr(test_3, attr, None)

        if t1 != t2 or t1 != t3:
            raise ValueError(f"{attr} - {t1} vs {t2} vs {t3}")

    print()"""


"""
convert_weights(
    model_path='baidu/ERNIE-4.5-VL-28B-A3B-PT',
    save_dir='AntonV/ErnieVL',
)
#"""

"""
convert_config(
    model_path='baidu/ERNIE-4.5-VL-28B-A3B-PT',
    save_dir='AntonV/ErnieVL',
)
#"""

"""
convert_tokenizer(
    original_tokenizer_path='baidu/ERNIE-4.5-VL-28B-A3B-PT',
    save_dir='AntonV/ErnieVL',
)
#"""

#"""
convert_processor(
    model_path="baidu/ERNIE-4.5-VL-28B-A3B-PT",
    save_dir="AntonV/ErnieVL",
)
#"""
