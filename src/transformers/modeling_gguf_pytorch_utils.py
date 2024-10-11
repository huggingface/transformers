# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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

import re
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from .integrations import (
    GGUF_CONFIG_MAPPING,
    GGUF_TENSOR_MAPPING,
    GGUF_TOKENIZER_MAPPING,
    _gguf_parse_value,
)
from .utils import is_torch_available
from .utils.import_utils import is_gguf_available
from .utils.logging import get_logger


if is_torch_available():
    import torch

logger = get_logger(__name__)


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tensors": GGUF_TENSOR_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["tensors"].keys())


def read_field(reader, field):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    if is_gguf_available() and is_torch_available():
        from gguf import GGUFReader, dequantize
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    model_name = read_field(reader, "general.name")

    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"

    model_size = ""
    # extract the number of params from file name as architectures can differ ;
    # eg. for falcon : `...falcon-7b-...`
    if "falcon" in architecture:
        gguf_file_name = gguf_checkpoint_path.split("/")[-1].lower()
        m = re.search(r"-\d+b-", gguf_file_name)  # regex to catch `-7b-`
        if m is None:
            raise ValueError(
                f"From file name, cannot determine the number of parameters for {architecture} architecture"
            )
        model_size = m.group().strip("-")  # only keeps `7b`

    if architecture + model_size not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture {architecture + model_size} not supported")

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter in GGUF_TO_TRANSFORMERS_MAPPING:
            parameter_renames = GGUF_TO_TRANSFORMERS_MAPPING[parameter]
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # retrieve config vocab_size from tokenizer
    # Pleas refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        tensor_key_mapping = GGUF_TO_TRANSFORMERS_MAPPING["tensors"][architecture + model_size]

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name

            weights = dequantize(tensor.data, tensor.tensor_type)

            if architecture == "llama" and (".attn_k." in name or ".attn_q." in name):
                num_heads = parsed_parameters["config"]["num_attention_heads"]
                num_kv_heads = parsed_parameters["config"]["num_key_value_heads"]
                if ".attn_q." in name:
                    weights = reverse_permute_weights(weights, num_heads, num_heads)
                elif ".attn_k." in name:
                    weights = reverse_permute_weights(weights, num_heads, num_kv_heads)

            if architecture == "qwen2moe":
                if "_exp" in name:
                    split_moe_expert_tensor(weights, parsed_parameters, name, tensor_key_mapping)
                    continue
                if "ffn_gate_inp_shexp" in name:
                    # for compatibility tensor shared_expert_gate must be (1, 2048) dim,
                    # quantized one is (2048)
                    weights = np.expand_dims(weights, axis=0)

            if architecture == "bloom" and "attn_qkv" in name:
                num_heads = parsed_parameters["config"]["n_head"]
                n_embed = parsed_parameters["config"]["hidden_size"]
                if "weight" in name:
                    weights = reverse_reshape_weights(weights, num_heads, n_embed)
                else:
                    weights = reverse_reshape_bias(weights, num_heads, n_embed)

            if architecture == "gpt2":
                if (
                    "attn_qkv.weight" in name
                    or "ffn_down.weight" in name
                    or "ffn_up.weight" in name
                    or "attn_output.weight" in name
                ):
                    # Original transpose implementation
                    # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L2060-L2061
                    weights = weights.T
                if name == "output.weight":
                    # output.weight has conflicts with attn_output.weight in name checking
                    # we have to explicitly check that name is exactly output.weight
                    name = "lm_head.weight"
                    parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))
                    continue

            for tensor_name in tensor_key_mapping:
                if tensor_name in name:
                    name = name.replace(tensor_name, tensor_key_mapping[tensor_name])

            # Use copy to avoid errors with numpy and pytorch
            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters


def reverse_permute_weights(weights: np.ndarray, n_head: int, num_kv_heads: Optional[int] = None) -> np.ndarray:
    # Original permutation implementation
    # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402-L1408
    if num_kv_heads is not None and n_head != num_kv_heads:
        n_head = num_kv_heads

    dim = weights.shape[0] // n_head // 2
    w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
    return w.swapaxes(2, 1).reshape(weights.shape)


def reverse_reshape_weights(weights: np.ndarray, n_head: int, n_embed: int):
    # Original reshape implementation
    # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L972-L985
    q, k, v = np.array_split(weights, 3, axis=0)

    q = q.reshape(n_head, n_embed // n_head, n_embed)
    k = k.reshape(n_head, n_embed // n_head, n_embed)
    v = v.reshape(n_head, n_embed // n_head, n_embed)
    qkv_weights = np.stack([q, k, v], axis=1)

    return qkv_weights.reshape(n_head * 3 * (n_embed // n_head), n_embed)


def reverse_reshape_bias(weights: np.ndarray, n_head: int, n_embed: int):
    # Original reshape implementation
    # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L986-L998
    q_bias, k_bias, v_bias = np.array_split(weights, 3)

    q_bias = q_bias.reshape(n_head, n_embed // n_head)
    k_bias = k_bias.reshape(n_head, n_embed // n_head)
    v_bias = v_bias.reshape(n_head, n_embed // n_head)

    qkv_bias = np.stack([q_bias, k_bias, v_bias], axis=1).flatten()
    return qkv_bias


def split_moe_expert_tensor(
    weights: np.ndarray, parsed_parameters: Dict[str, Dict], name: str, tensor_key_mapping: dict
):
    # Original merge implementation
    # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L1994-L2022
    exp_name = ""
    if "ffn_gate_exps" in name:
        exp_name = "gate_proj"
    elif "ffn_down_exps" in name:
        exp_name = "down_proj"
    elif "ffn_up_exps" in name:
        exp_name = "up_proj"
    else:
        raise ValueError(f"Cannot map expert tensor {name} in Qwen2Moe architecture.")
    for tensor_name in tensor_key_mapping:
        if tensor_name in name:
            name = name.replace(tensor_name, tensor_key_mapping[tensor_name])
    w_counter = parsed_parameters["config"].get("num_experts", 60)
    for i in range(0, w_counter):
        temp_name = name.replace(".weight", f".{i}.{exp_name}.weight")
        exp_weight = weights[i]
        parsed_parameters["tensors"][temp_name] = torch.from_numpy(np.copy(exp_weight))
