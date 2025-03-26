import argparse
import gc
import io
import json
import os
import re
from typing import List, Optional

import torch
from tokenizers import AddedToken, processors
from tqdm import tqdm

from transformers import (
    GenerationConfig,
    Llama4Config,
    Llama4ForConditionalGeneration,
    Llama4TextConfig,
    Llama4VisionConfig,
    PreTrainedTokenizerFast,
)
from transformers.integrations.tiktoken import TikTokenConverter


torch.serialization.add_safe_globals([io.BytesIO])
# fmt: off

# layers.29.feed_forward.model.norm.weight
# layers.30.attention.wqkv.layer_model.norm.weight
# Still not sure what to do with those!
# `None` means we drop the key

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # CausalLM keys
    r"output.weight":                                        r"language_model.lm_head.weight",
    r"\nnorm.weight":                                        r"\nlanguage_model.model.norm.weight",
    # Model keys
    r"tok_embeddings.weight":                                r"language_model.model.embed_tokens.weight",
    r"freq_cis":                                             None,
    r"rope.freqs":                                           None,
    r"layers.(\d+).attention_norm.weight":                   r"language_model.model.layers.\1.input_layernorm.weight",
    r"layers.(\d+).attention.wqkv.layer_norm_weight":        r"language_model.model.layers.\1.input_layernorm.weight",
    r"layers.(\d+).feed_forward.norm.weight":                r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"layers.(\d+).attention.wo.weight":                     r"language_model.model.layers.\1.self_attn.o_proj.weight",
    r"layers.(\d+).attention.wqkv.weight":                   r"language_model.model.layers.\1.self_attn.qkv_proj.weight",

    # MoE keys: no simple MLPmodel.
    r"layers.(\d+).feed_forward.experts.moe_w_in_eD_F":      r"language_model.model.layers.\1.feed_forward.experts.gate_proj",       # will be fused with up
    r"layers.(\d+).feed_forward.experts.moe_w_out_eF_D":     r"language_model.model.layers.\1.feed_forward.experts.down_proj",       # expert win
    r"layers.(\d+).feed_forward.experts.moe_w_swiglu_eD_F":  r"language_model.model.layers.\1.feed_forward.experts.up_proj",         # fused with up
    r"layers.(\d+).feed_forward.router_DE":                  r"language_model.model.layers.\1.feed_forward.router.weight",           # used for top
    r"layers.(\d+).feed_forward.w_in_shared_FD":             r"language_model.model.layers.\1.feed_forward.shared_expert.gate_proj", # might need to be fused for efficiency?
    r"layers.(\d+).feed_forward.w_out_shared_DF":            r"language_model.model.layers.\1.feed_forward.shared_expert.down_proj", # might need to be fused for efficiency?
    r"layers.(\d+).feed_forward.w_swiglu_FD":                r"language_model.model.layers.\1.feed_forward.shared_expert.up_proj",   # might need to be fused for efficiency?
    r"layers.(\d+).feed_forward.global_gate_stats_3E":       None,
    # Unused keys in load hooks (explicitly removed)
    r'layers.(\d+).attention.wqkv._extra_state':             None,
    r'layers.(\d+).attention.wo._extra_state':               None,

    # MLP layer variant
    # r"layers.(\d+).feed_forward.w1":                         r"language_model.model.layers.\1.feed_forward.gate_proj",               # might need to be fused for efficiency?
    # r"layers.(\d+).feed_forward.w3":                         r"language_model.model.layers.\1.feed_forward.up_proj",                 # might need to be fused for efficiency?
    r"layers.(\d+).feed_forward.mlp.fc1_weight":             r"language_model.model.layers.\1.feed_forward.gate_up_proj.weight",
    r"layers.(\d+).feed_forward.mlp.fc2_weight":             r"language_model.model.layers.\1.feed_forward.down_proj.weight",
    r"layers.(\d+).feed_forward.mlp.layer_norm.weight":      r"language_model.model.layers.\1.post_attention_layernorm.weight",

    # Vision encoder mapping
    r"vision_embeddings.vision_encoder.conv1._linear":                                            r"vision_model.patch_embedding.linear",
    r'vision_embeddings.vision_adapter.mlp.c_fc':                                                 r"vision_model.vision_adapter.mlp.fc1",
    r'vision_embeddings.vision_adapter.mlp.c_proj':                                               r"vision_model.vision_adapter.mlp.fc2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).attn.wq.(weight|bias)":        r"vision_model.model.layers.\1.self_attn.q_proj.\2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).attn.wk.(weight|bias)":        r"vision_model.model.layers.\1.self_attn.k_proj.\2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).attn.wv.(weight|bias)":        r"vision_model.model.layers.\1.self_attn.v_proj.\2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).attn.wo.(weight|bias)":        r"vision_model.model.layers.\1.self_attn.o_proj.\2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).mlp.c_fc":                     r"vision_model.model.layers.\1.mlp.fc1",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).mlp.c_proj":                   r"vision_model.model.layers.\1.mlp.fc2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).ln_1.(weight|bias)":           r"vision_model.model.layers.\1.input_layernorm.\2",
    r"vision_embeddings.vision_encoder.transformer.resblocks.(\d+).ln_2.(weight|bias)":           r"vision_model.model.layers.\1.post_attention_layernorm.\2",
    # r'vision_embeddings.vision_encoder.ln_(1|2).(weight|bias)':                                   r'vision_model.transformer.vision_encoder.layernorm_\1.\2',
    r'vision_embeddings.vision_encoder.ln_post':                                                  r'vision_model.layernorm_post',
    r'vision_embeddings.vision_encoder.ln_pre':                                                   r'vision_model.layernorm_pre',
    r'vision_embeddings.vision_encoder.class_embedding':                                          r'vision_model.class_embedding',
    r"vision_embeddings.vision_encoder.positional_embedding_vlm":                                 r"vision_model.positional_embedding_vlm",
    r"vision_embeddings.vision_encoder.(?=\w)":                                                   r"vision_model.model.",
    r"vision_projection.weight":                                                                  r"multi_modal_projector.linear_1.weight",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def is_param_same_across_shards(key):
    """
    Return `False` if the parameter is different across checkpoint shards
    and needs to be concatenated.
    """
    patterns = [
        r"language_model.layers.(\d+).(.*)layernorm.weight",
        r"language_model.norm.weight",
        r"router.weight",
        r"feed_forward.global_gate_stats",
        # not all vision weights are sharded, some are repeated
        r"vision_model.class_embedding",
        r"vision_model.positional_embedding_vlm",
        r"vision_embeddings.vision_encoder.positional_embedding_vlm",
        r"vision_model.model.layers.(\d+).self_attn.o_proj.bias",
        r"vision_model.model.layers.(\d+).input_layernorm",
        r"vision_model.model.layers.(\d+).post_attention_layernorm",
        r"vision_model.layernorm_pre",
        r"vision_model.layernorm_post",
        r"vision_model.model.layers.(\d+).mlp.fc2.bias",
        r"norm.weight",
        ]  # fmt: skip
    return any(re.search(pattern, key) for pattern in patterns)


def get_concat_dim(key):
    """
    Return the dimension to concatenate the weights on.
    """
    concat_dim_1 = [
        # language dim 1 sharded weights
        "feed_forward.router.weight",
        "self_attn.o_proj",
        "experts.gate_proj",
        "experts.up_proj",
        "expert.down_proj",
        "feed_forward.down_proj",
        "global_gate_stats",
        # vision dim1 sharded stuff
        "mlp.fc2.weight", # covers all rowparallels across vis
        ]  # fmt: off
    if any(re.search(pattern, key) for pattern in concat_dim_1):
        return 1
    return 0


def compute_intermediate_size(hidden_dim, multiple_of=1024, ffn_dim_multiplier=1.3):
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


# Ignore extra info - h/t Aritra
def safe_load(filename):
    # Can use weights_only because io.BytesIO was registered, but we still need to skip those objects
    shard = torch.load(filename, weights_only=True, map_location="cpu", mmap=True)
    shard = {k: v for k, v in shard.items() if not isinstance(v, io.BytesIO)}
    return shard


def write_model(
    model_path,
    input_base_path,
    num_shards,
    convert_checkpoints,
    safe_serialization=True,
    instruct=False,
):
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)

    params = params.get("model", params)
    torch_dtype = "bfloat16"

    # ------------------------------------------------------------
    # Text model params and config
    # ------------------------------------------------------------

    # params from config
    vocab_size = 202048  # params["vocab_size"] # seems like the lm head is 25256 so padded instead of 202048
    num_layers = params["n_layers"]
    dim = params["dim"]
    num_heads = params["n_heads"]
    rms_norm_eps = params["norm_eps"]
    rope_theta = params["rope_theta"]

    # some constans from original code
    if params["use_scaled_rope"]:
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    else:
        rope_scaling = "default"

    # compute additional params for weight conversion
    num_heads_per_shard = num_heads // num_shards
    dim_per_head = dim // num_heads
    # intermediate_size = compute_intermediate_size(dim, multiple_of=params["multiple_of"])

    num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA

    num_experts = params["moe_args"]["num_experts"]
    interleave_moe_layer_step = params["moe_args"].get("interleave_moe_layer_step", 1)

    bos_token_id = 200000
    eos_token_id = [200001, 200002, 200003] if instruct else 200001
    pad_token_id = 200008

    text_config = Llama4TextConfig(
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
        hidden_size=dim,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        num_hidden_layers=num_layers,
        intermediate_size=8192,
        rope_scaling=rope_scaling,
        num_local_experts=num_experts,
        interleave_moe_layer_step = interleave_moe_layer_step,
        use_qk_norm=params["use_qk_norm"],
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        tie_word_embeddings=False,  # Constant set to False
        torch_dtype=torch_dtype,
    )
    # default vision config frmo params

    vision_params = params["vision_args"]
    vision_dim = vision_params["dim"]
    vision_num_layers = vision_params["n_layers"]
    image_size = vision_params["image_size"]["height"]  # siglip config is outdated
    vision_num_heads = vision_params["n_heads"]

    vision_output_dim = vision_params["output_dim"]

    vision_config = Llama4VisionConfig(
        hidden_act="gelu",
        num_hidden_layers=vision_num_layers,
        image_size=image_size,
        num_attention_heads=vision_num_heads,
        hidden_size=vision_dim,
        vision_output_dim=vision_output_dim,
    )

    config = Llama4Config(text_config=text_config, vision_config=vision_config)
    config.save_pretrained(model_path)

    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    if convert_checkpoints:
        print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
        if num_shards == 1:
            if os.path.exists(os.path.join(input_base_path, "consolidated.00.pth")):
                path = os.path.join(input_base_path, "consolidated.00.pth")
            else:
                path = os.path.join(input_base_path, "consolidated.pth")
            loaded = [safe_load(path)]
        else:
            loaded = [
                safe_load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"))
                for i in tqdm(range(num_shards), desc="Loading shards", unit="shard")
            ]

        all_keys_raw = list(loaded[0].keys())
        repeated_keys = []
        sharded_keys = []
        for _key in all_keys_raw:
            try:
                if (loaded[0][_key] == loaded[1][_key]).all():
                    repeated_keys.append(_key)
                else:
                    sharded_keys.append(_key)
            except Exception as e:
                print(f"Encountered exception {e} for {_key}")
        print("Initializing an empty model")
        with torch.device("meta"):
            model = Llama4ForConditionalGeneration(config)

        print("Converting model...")
        all_keys = list(loaded[0].keys())
        new_keys = convert_old_keys_to_new_keys(all_keys)
        state_dict = {}
        replicated_params = []  # To keep track of replicated weights.
        for key in tqdm(all_keys, desc="Renaming and processing all keys", unit="key"):
            new_key = new_keys[key]
            print(key, new_key)
            # we skip the extra states, which are _io.BytesIO
            # contiguous + clone is fucking costly
            # current_parameter = [chunk.pop(key).contiguous().clone() for chunk in loaded if not isinstance(chunk[key], io.BytesIO)]
            if not is_param_same_across_shards(new_key):
                current_parameter = [chunk.pop(key) for chunk in loaded if not isinstance(chunk[key], io.BytesIO)]
            else:
                print(f"{key} (now {new_key}) is the same across all shards.")
                replicated_params.append((key, new_key))
                current_parameter = [loaded[0].pop(key)] if not isinstance(loaded[0][key], io.BytesIO) else []

            if "running_gate_stats_3E" in key:
                new_keys.pop(new_key)
                continue

            concat_dim = get_concat_dim(new_key)

            # Post-process the current_parameter.
            if "qkv_proj" in new_key:
                queries = []
                keys = []
                values = []
                for param in current_parameter:
                    query, key_, value = param.split(
                        [
                            num_heads * dim_per_head // num_shards,
                            num_key_value_heads * dim_per_head // num_shards,
                            num_key_value_heads * dim_per_head // num_shards,
                        ]
                    )
                    queries.append(query.reshape(num_heads_per_shard, -1, dim))
                    keys.append(key_.reshape(num_key_value_heads // num_shards, -1, dim))
                    values.append(value.reshape(num_key_value_heads // num_shards, -1, dim))

                queries = torch.cat(queries, dim=0).reshape(dim, dim)
                keys = torch.cat(keys, dim=0).reshape(num_key_value_heads * dim_per_head, dim)
                values = torch.cat(values, dim=0).reshape(num_key_value_heads * dim_per_head, dim)
                # queries = permute_for_rope(queries, num_heads, dim, dim)
                # keys = permute_for_rope(keys, num_key_value_heads, num_key_value_heads*dim_per_head, dim)

                q = new_key.replace("qkv", "q")
                tqdm.write(f"Processing: {key.ljust(50)}  ->\t {q}, {queries.shape}")
                state_dict[q] = queries

                k = new_key.replace("qkv", "k")
                tqdm.write(f"Processing: {key.ljust(50)}  ->\t {k}, {keys.shape}")
                state_dict[k] = keys

                v = new_key.replace("qkv", "v")
                tqdm.write(f"Processing: {key.ljust(50)}  ->\t {v}, {values.shape}")
                state_dict[v] = values
            elif re.search(r"(gate|up)_proj", new_key):
                if "gate_up_proj" not in new_key:
                    path = new_key.split(".")
                    gate_key = re.sub(r"(gate|up)_proj", lambda m: "gate_proj", new_key)
                    up_key = re.sub(r"(gate|up)_proj", lambda m: "up_proj", new_key)
                    if gate_key == new_key:
                        state_dict[new_key] = torch.cat(current_parameter, dim=concat_dim)
                    elif new_key == up_key:
                        if "shared" in new_key:
                            gate_proj = state_dict.pop(gate_key)
                            up_proj = torch.cat(current_parameter, dim=concat_dim)
                            state_dict[gate_key] = gate_proj
                            state_dict[new_key] = up_proj
                            # TODO that's kinda low hanging fruit, but shard dim for shared should be
                            # column, then row. TO get hidden // tp * col // tp, gate + up
                            tqdm.write(f"Processing: {key.ljust(50)}  ->\t {gate_key}, {state_dict[gate_key].shape}")
                        else:
                            gate_proj = state_dict.pop(gate_key)
                            gate_proj = [
                                gate_proj.reshape(num_experts, -1, 8, 1024)[:, :, k, :].reshape(num_experts, -1, 1024) for k in range(8)
                            ]
                            gate_proj = torch.cat(gate_proj, dim=-1)

                            up_proj = [k.reshape(num_experts, -1, 8, 1024).reshape(num_experts, -1, 1024) for k in current_parameter]
                            up_proj = torch.cat(up_proj, dim=-1)

                            gate_up_proj = torch.cat((gate_proj, up_proj), dim=-1)
                            new_key = new_key.replace("up_proj", "gate_up_proj")
                            state_dict[new_key] = gate_up_proj.contiguous()
                        tqdm.write(f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}")
                else:
                    state_dict[new_key] = torch.cat(current_parameter, dim=concat_dim)
            elif "down_proj" in new_key:
                current_parameter = torch.cat(current_parameter, dim=concat_dim)
                if "experts" in new_key:
                    p = []
                    for i in range(8):
                        p += [current_parameter.reshape(8, -1, 5120)[i, :, :].view(num_experts, -1, 5120)]
                    current_parameter = torch.cat(p, dim=1)
                state_dict[new_key] = current_parameter.contiguous()
                tqdm.write(f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}")
            elif "router" in new_key:
                current_parameter = torch.cat(current_parameter, dim=concat_dim)
                state_dict[new_key] = current_parameter.transpose(0, 1)
            elif "lm_head" in new_key:
                current_parameter = torch.cat(current_parameter, dim=concat_dim).clone()
                # TODO we need to do better than mean, works for now
                # if (vocab_size - current_parameter.shape[0]) > 0:
                #     mean_embedding = torch.mean(current_parameter, dim=0)[:, None].repeat(vocab_size-current_parameter.shape[0],1)
                #     print(mean_embedding.shape)
                #     current_parameter = torch.cat((current_parameter, mean_embedding), dim=0)
                state_dict[new_key] = current_parameter
                tqdm.write(
                    f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}, concat dim = {concat_dim}"
                )
            elif new_key == "vision_model.patch_embedding.linear.weight":
                current_parameter = torch.cat(current_parameter, dim=concat_dim).clone()
                # We don't reshape the patch embedding as we're using unfolded convolution as well
                state_dict[new_key] = current_parameter  # .reshape(-1, 3, vision_patch_size, vision_patch_size)
            # generic concat for weights/select one for biases
            elif isinstance(current_parameter, list) and len(current_parameter) > 0:
                if not is_param_same_across_shards(new_key):
                    current_parameter = torch.cat(current_parameter, dim=concat_dim)
                    state_dict[new_key] = current_parameter
                    tqdm.write(
                        f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}, concat dim = {concat_dim}"
                    )
                elif is_param_same_across_shards(new_key):
                    state_dict[new_key] = current_parameter[0]
                    tqdm.write(
                        f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}, concat dim = {concat_dim}"
                    )

            elif new_key == "":
                # skip empty keys
                continue
            else:
                # just load the parameter
                state_dict[new_key] = current_parameter
                tqdm.write(
                    f"Processing: {key.ljust(50)}  ->\t {new_key}, {state_dict[new_key].shape}, concat dim = {concat_dim}"
                )
        del loaded
        gc.collect()

        print("Loading the checkpoint in a Llama4 model.")
        state_dict.pop("")
        model.load_state_dict(state_dict, strict=True, assign=True)
        print("Model reloaded successfully. Checking logits...")
        # ipdb.set_trace()
        # zero_out = model.forward(inputs_embeds=torch.zeros((1,743, 4096)))
        # ipdb.set_trace()
        print("Saving the model.")
        model.save_pretrained(model_path, safe_serialization=safe_serialization)
        del state_dict, model

        # Safety check: reload the converted model
        gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    with torch.no_grad():
        # TODO test if we can do `tp_plan="auto"``
        model = Llama4ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
        )
        # ipdb.set_trace()
        model.eval()
        model.generation_config.top_p = 0.9
        model.generation_config.temperature = 0.6
        print("Model reloaded successfully.")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(["Roses are red,"], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=4)
        print(tokenizer.batch_decode(out))
    # generation config
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generation_config.save_pretrained(model_path)


BOS_ADDED_TOKEN = AddedToken(
    "<|begin_of_text|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True
)
EOS_ADDED_TOKEN = AddedToken(
    "<|end_of_text|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True
)
EOT_ADDED_TOKEN = AddedToken("<|eot|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)


def get_reserved_special_tokens(name, count, start_index=0):
    return [f"<|{name}_reserved_special_token_{i}|>" for i in range(start_index, start_index + count)]


# 200005, ..., 200079
LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS = [
    "<|header_start|>",
    "<|header_end|>",
    "<|eom|>",
    "<|eot|>",
    "<|step|>",
    "<|text_post_train_reserved_special_token_0|>",
    "<|text_post_train_reserved_special_token_1|>",
    "<|text_post_train_reserved_special_token_2|>",
    "<|text_post_train_reserved_special_token_3|>",
    "<|text_post_train_reserved_special_token_4|>",
    "<|text_post_train_reserved_special_token_5|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|finetune_right_pad|>",
] + get_reserved_special_tokens(
    "text_post_train", 61, 6
)  # <|text_post_train_reserved_special_token_6|>, ..., <|text_post_train_reserved_special_token_66|>

# 200080, ..., 201133
LLAMA4_VISION_SPECIAL_TOKENS = [
    "<|image_start|>",
    "<|image_end|>",
    "<|vision_reserved_special_token_0|>",
    "<|vision_reserved_special_token_1|>",
    "<|tile_x_separator|>",
    "<|tile_y_separator|>",
    "<|vision_reserved_special_token_2|>",
    "<|vision_reserved_special_token_3|>",
    "<|vision_reserved_special_token_4|>",
    "<|vision_reserved_special_token_5|>",
    "<|image|>",
    "<|vision_reserved_special_token_6|>",
    "<|patch|>",
] + get_reserved_special_tokens(
    "vision", 1041, 7
)  # <|vision_reserved_special_token_7|>, ..., <|vision_reserved_special_token_1047|>

LLAMA4_SPECIAL_TOKENS = LLAMA4_TEXT_POST_TRAIN_SPECIAL_TOKENS + LLAMA4_VISION_SPECIAL_TOKENS

BASIC_SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]


class Llama4Converter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens: List[str],
        pattern: str,
        model_max_length: int = 0,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(vocab_file, pattern=pattern)
        self.additional_special_tokens = special_tokens
        tokenizer = self.converted()
        if chat_template is not None:
            kwargs["chat_template"] = chat_template

        self.converted_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )

        # to check
        # import tiktoken
        # model = tiktoken.Encoding(
        #     name=Path(model_path).name,
        #     pat_str=self.O200K_PATTERN,
        #     mergeable_ranks=mergeable_ranks,
        #     special_tokens=self.special_tokens,
        # )

        instruct = chat_template is not None
        self.update_post_processor(self.converted_tokenizer)
        # finer special_tokens_map.json
        self.converted_tokenizer._bos_token = BOS_ADDED_TOKEN
        self.converted_tokenizer._eos_token = EOT_ADDED_TOKEN if instruct else EOS_ADDED_TOKEN

    # We can't do this while building the tokenizer because we have no easy access to the bos token id
    def update_post_processor(self, tokenizer):
        tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single="<|begin_of_text|> $A",
                    pair="<|begin_of_text|>:0 $A:0 <|begin_of_text|>:1 $B:1",
                    special_tokens=[
                        ("<|begin_of_text|>", tokenizer.convert_tokens_to_ids("<|begin_of_text|>")),
                    ],
                ),
            ]
        )


O200K_PATTERN = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False):
    # TODO: verify chat template
    chat_template = (
        "{% for message in messages %}"
        "{% if loop.index0 == 0 %}"
        "{{ bos_token }}"
        "{% endif %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|image|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|eot|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )
    special_tokens = BASIC_SPECIAL_TOKENS + LLAMA4_SPECIAL_TOKENS
    converter = Llama4Converter(
        vocab_file=tokenizer_path,
        pattern=O200K_PATTERN,
        special_tokens=special_tokens,
        chat_template=chat_template if instruct else None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>" if not instruct else "<|eot|>",
        pad_token="<|finetune_right_pad_id|>",
        model_max_length=131072,
    )
    tokenizer = converter.converted_tokenizer
    tokenizer.save_pretrained(save_dir)

    if instruct:
        print("Saving chat template...")
        chat_template_path = os.path.join(save_dir, "chat_template.json")
        with open(chat_template_path, "w") as f:
            json.dump({"chat_template": chat_template}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/fsx/arthur/Llama-4-17B-Omni-Instruct-Original",
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "--output_dir",
        default="llama4_hf_vision",
        type=str,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--num_shards",
        default=8,
        type=int,
        help="The number of individual shards used for the model. Does not have to be the same as the number of consolidated_xx.pth",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    parser.add_argument(
        "--convert_checkpoints",
        action="store_true",
        help="Whether to convert the original weights (or skip if previously converted)",
    )

    args = parser.parse_args()
    write_tokenizer(
        tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
        save_dir=args.output_dir,
        instruct=args.instruct,
    )

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        num_shards=args.num_shards,
        instruct=args.instruct,
        convert_checkpoints=args.convert_checkpoints,
    )

# torchrun --nproc-per-node=8   .venv/lib/python3.12/site-packages/llama_models/llama4/scripts/text_completion.py --checkpoint_dir   "llama4" --world_size 8
