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

import gc
import json
import os
import shutil
import warnings

import regex as re
import torch

from transformers import MllamaConfig, MllamaForConditionalGeneration, MllamaImageProcessor, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None


NUM_SHARDS = {
    "90B": 8,
    "11B": 1,
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"text_model.norm.weight":                                                                  r"language_model.model.norm.weight",
    r"text_model.output.weight":                                                                r"language_model.lm_head.weight",
    r"text_model.tok_embeddings":                                                               r"language_model.model.embed_tokens",
    r"text_model.learnable_embedding":                                                          r"language_model.model.learnable_embedding",
    r"text_model.rope.freqs":                                                                   None, # meaning we skip it and don't want it
    # For every cross attention layer, the layer needs to be updated
    r"text_model.cross_attention_layers.(\d+).gate_attn":                                       r"language_model.model.layers.\1.cross_attn_attn_gate",
    r"text_model.cross_attention_layers.(\d+).gate_ffwd":                                       r"language_model.model.layers.\1.cross_attn_mlp_gate",
    # special key, wqkv needs to be split afterwards
    r"text_model.cross_attention_layers.(\d+).attention.wq.weight":                             r"language_model.model.layers.\1.cross_attn.q_proj.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wkv":                                   r"language_model.model.layers.\1.cross_attn.k|v_proj",
    r"text_model.cross_attention_layers.(\d+).attention.wo":                                    r"language_model.model.layers.\1.cross_attn.o_proj",
    r"text_model.cross_attention_layers.(\d+).attention.inner_attention.(q|k)_norm":            r"language_model.model.layers.\1.cross_attn.\2_norm",
    r"text_model.cross_attention_layers.(\d+).attention.wq.layer_norm_weight":                  r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wk.layer_norm_weight":                  r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.fc1.weight":                     r"language_model.model.layers.\1.mlp.up|gate_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.fc2.weight":                     r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.mlp.layer_norm_weight":              r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # self attention layers
    r"text_model.layers.(\d+).attention.wqkv.weight":                                           r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight",
    r"text_model.layers.(\d+).attention.wo":                                                    r"language_model.model.layers.\1.self_attn.o_proj",
    r"text_model.layers.(\d+).attention.wqkv.layer_norm_weight":                                r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.mlp.layer_norm_weight":                              r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.mlp.fc2.":                                           r"language_model.model.layers.\1.mlp.down_proj.",
    r"text_model.layers.(\d+).feed_forward.mlp.fc1.":                                           r"language_model.model.layers.\1.mlp.up|gate_proj.",
    # Vision encoder mapping
    r"vision_model.vision_encoder.conv1._linear":                                               r"vision_model.patch_embedding",
    r'vision_model.vision_projection.':                                                         r"multi_modal_projector.",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wq":    r"vision_model.\1.layers.\2.self_attn.q_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wk":    r"vision_model.\1.layers.\2.self_attn.k_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wv":    r"vision_model.\1.layers.\2.self_attn.v_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wo":    r"vision_model.\1.layers.\2.self_attn.o_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_fc":   r"vision_model.\1.layers.\2.mlp.fc1",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_proj": r"vision_model.\1.layers.\2.mlp.fc2",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_1":       r"vision_model.\1.layers.\2.input_layernorm",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_2":       r"vision_model.\1.layers.\2.post_attention_layernorm",
    r"vision_model.vision_encoder.global_transformer.resblocks.(\d+).(gate_ffn|gate_attn)":     r"vision_model.global_transformer.layers.\1.\2",
    r'vision_model.vision_encoder.ln_(pre|post).(weight|bias)':                                 r'vision_model.vision_encoder.ln_\1.\2',
    r"vision_model.vision_encoder.(?=\w)":                                                      r"vision_model.",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
        This function should be applied only once, on the concatenated keys.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text) # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = {k:v for k,v in zip(old_text.split("\n"), new_text.split("\n"))}
    return output_dict


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor

def process_layer_norm_post_rope(layer_norm_weight, num_shards, dims_per_heads):
    pass # TODO

def write_model(
    model_path,
    input_base_path,
    model_size,
    safe_serialization=True,
    llama_version=1,
    vocab_size=None,
):

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    params = params.get("model", params)

    # language parameters
    n_layers = params["n_layers"] # language model self-attention layers
    n_layers_cross_attention = params["vision_num_cross_attention_layers"]  # language model cross-attention layers; 90B - 20, 11B - 8
    n_heads = params["n_heads"] # 64 for 90b (70b llama), 32 for 11b mllama
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    patch_size = 14
    num_channels = 3
    base = params.get("rope_theta", 10000.0)

    # vision parameters
    n_layers_vision_transformer = 32 # vision model 1st transformer layers
    n_layers_global_transformer = 8 # global transformer vision layers
    n_heads_vision = 16
    n_vision_heads_per_shard = n_heads_vision // num_shards
    vision_hidden_dim = 1280 # width of vision transformers
    vision_dims_per_head = vision_hidden_dim // n_heads_vision
    mlp_ratio = 4 # vision_hidden_dim * mlp_ratio is mlp dim

    vocab_size = vocab_size if vocab_size is not None else 32000
    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_local_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    if num_shards == 1:
        loaded = [torch.load(os.path.join(input_base_path, "consolidated.pth"), map_location="cpu")]
    else:
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]

    param_count = 0
    index_dict = {"weight_map": {}}
    print("1. Converting language model")
    total_layers = len(loaded[0].keys())
    all_keys = list(loaded[0].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    cross_layer_shift = [3, 7, 11, 15, 19, 23, 27, 31]
    attn_layer_shift = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39]
    state_dict = {}
    for idx, key in enumerate(all_keys):
        filename = f"pytorch_model-{idx + 1}-of-{total_layers + 1}.bin"
        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.
        new_key = new_keys[key]
        if "cross_attention" in key and "language_model" in new_key:
            new_key = re.sub(r"layers.(\d+).", lambda _match: f"layers.{cross_layer_shift[int(_match.groups()[0])]}.", new_key)
        elif "text_model.layers" in key and "language_model" in new_key:
            new_key = re.sub(r"layers.(\d+).", lambda _match: f"layers.{attn_layer_shift[int(_match.groups()[0])]}.", new_key)

        if "self_attn.q|k|v|_proj" in new_key and "language_model" in new_key:
            weight = torch.cat([chunk.pop(key).contiguous().clone() for chunk in loaded], dim=0) # TODO maybe a view is needed
            q, k, v = torch.split(weight, [dim, key_value_dim, key_value_dim])
            state_dict[new_key.replace("q|k|v|", "q")] = permute_for_rope(q, n_heads, dim, dim)
            state_dict[new_key.replace("q|k|v|", "k")] = permute_for_rope(k, num_key_value_heads, key_value_dim, dim)
            state_dict[new_key.replace("q|k|v|", "v")] = v
        elif "cross_attn" in key and "q_norm" in key or "k_norm" in key:
            # TODO since rope was permuted, we ought to permute q and k norms
            # depending on whether or not they are sharded as well!
            state_dict[
                new_key  ] = [chunk.pop(key).contiguous().clone() for chunk in loaded][0]
        elif "mlp.up|gate_proj." in new_key:
            weight = torch.cat([chunk.pop(key).contiguous().clone() for chunk in loaded], dim=0)
            gate, up = weight.chunk(2)
            state_dict[new_key.replace("up|gate", "up")] = up
            state_dict[new_key.replace("up|gate", "gate")] = gate
        elif new_key == "vision_model.patch_embedding.weight":
            state_dict[new_key] = torch.cat([chunk.pop(key).contiguous().clone() for chunk in loaded], dim=0).reshape(-1, num_channels, patch_size, patch_size)
        elif "cross_attn.k|v_proj" in new_key and "language_model" in new_key:
            weight = torch.cat([chunk.pop(key).contiguous().clone() for chunk in loaded], dim=0)
            k, v = weight.chunk(2, dim=0)
            state_dict[new_key.replace("k|v", "k")] = k
            state_dict[new_key.replace("k|v", "v")] = v
        elif "layernorm" in new_key:
            state_dict [
                new_key  ]= [chunk.pop(key).contiguous().clone() for chunk in loaded][0]
        elif "_gate" in new_key:
            state_dict  [new_key] =  [chunk.pop(key).contiguous().clone() for chunk in loaded][0][0].view(1)
        elif new_key != "":
            state_dict[
                new_key  ] = torch.cat([chunk.pop(key).contiguous().clone() for chunk in loaded], dim=0)

        # for k, v in state_dict.items():
        #     index_dict["weight_map"][k] = filename
        #     param_count += v.numel()
        # print(f"Saving {filename} in {tmp_model_path}...")
        # torch.save(state_dict, os.path.join(tmp_model_path, filename))
    state_dict["language_model.model.embed_tokens.weight"] = torch.cat(
        [state_dict["language_model.model.embed_tokens.weight"], state_dict.pop("language_model.model.learnable_embedding.weight")], dim=0
    )
    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    vision_config = {
        "n_heads": params["n_heads"],
        "vision_chunk_size": params["vision_chunk_size"],
        "vision_max_num_chunks": params["vision_max_num_chunks"],
        "patch_size": patch_size,
        "projection_dim": params["dim"],  # need to figure out
        "vision_input_dim": 1280, # constant, "self.vision_input_dim = 1280" for CrossAttentionTransformerVision in original code
        "return_intermediate": [3,7,15,23,30],
        "global_vision_layers": 8,  # constant "n_global_layers=8" for VisionEncoder in original code
        "max_num_tiles": 4,  # not used in the modeling code yet, "max_num_tiles=4" for VisionEncoder in original code
        "norm_eps": params["norm_eps"],
        "ffn_dim_multiplier": params["ffn_dim_multiplier"],
        "multiple_of": params["multiple_of"],
    }
    text_config = {
        "vocab_size": params["vocab_size"],
        "n_layers": params["n_layers"],
        "dim": params["dim"],
        "n_heads": params["n_heads"],
        "n_kv_heads": params["n_kv_heads"],
        "max_seq_len": 512,  # in examples
        "ffn_dim_multiplier": params["ffn_dim_multiplier"],
        "norm_eps": params["norm_eps"],
        "rope_theta": params["rope_theta"],
        "use_scaled_rope": params["use_scaled_rope"],
        "vision_num_cross_attention_layers": params["vision_num_cross_attention_layers"],  # should be in vision config?
        "multiple_of": params["multiple_of"],
        "vision_input_dim": 1280, # constant, see "vision_input_dim" in vision config
        "attention_bias":False,
        "tie_word_embeddings":False,
    }
    config = MllamaConfig(vision_config=vision_config, text_config=text_config)
    config.architectures = ["MllamaForConditionalGeneration"]
    config.save_pretrained(tmp_model_path)
    print("Loading the checkpoint in a Llama model.")

    # Make space so we can load the model properly now.
    # del state_dict
    del loaded
    gc.collect()

    with torch.device("meta"):
        model = MllamaForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.save_pretrained(model_path, safe_serialization=safe_serialization)

    mllama_model = MllamaForConditionalGeneration.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    # Avoid saving this as part of the config.
    del mllama_model.config._name_or_path
    mllama_model.config.torch_dtype = torch.float16  # not sure about this.
    print("Saving in the Transformers format.")
    mllama_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)

# TODO: update to new provided code: python + video tokens
class MllamaConverter(TikTokenConverter):
    def __init__(self, vocab_file, num_reserved_special_tokens=256, **kwargs):
        super().__init__(vocab_file, **kwargs)
        tokenizer = self.converted()
        chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = bos_token + content %}"
            "{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        )
        num_reserved_special_tokens = 256
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
            "<|image|>",
        ]
        special_tokens += [
            f"<|reserved_special_token_{i + 2}|>"
            for i in range(num_reserved_special_tokens - len(special_tokens))
        ]
        tokenizer.add_special_tokens(special_tokens)

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            pad_token="<|finetune_right_pad_id|>",
            chat_template=chat_template,
            model_input_names=["input_ids", "attention_mask"],
        )


def write_tokenizer(tokenizer_path: str, save_dir: str):

    converter = MllamaConverter(
        tokenizer_path,
        pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",  # noqa: W605
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)


def write_image_processor(config_path: str, save_dir: str):

    params = read_json(config_path)

    patch_size = params["vision_chunk_size"]
    max_image_tiles = params["vision_max_num_chunks"]

    image_processor = MllamaImageProcessor(
        do_resize=True,
        size={"height": patch_size, "width": patch_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_pad=True,
        max_image_tiles=max_image_tiles,
    )

    image_processor.save_pretrained(save_dir)


write_model(
    model_path="converted-mllama-11b",
    input_base_path="/raid/arthur/mllama-11b",
    safe_serialization=True,
    model_size="11B",
    llama_version=3,
    vocab_size=128256,
)

write_tokenizer(
    tokenizer_path="/home/ubuntu/projects/meta_mllama/weights-11b-biases/tokenizer.model",
    save_dir="/home/ubuntu/projects/new-model-addition-mllama/mllama-11b",
)

write_image_processor(
    config_path="/home/ubuntu/projects/meta_mllama/weights-11b-biases/params.json",
    save_dir="/home/ubuntu/projects/new-model-addition-mllama/mllama-11b",
)
