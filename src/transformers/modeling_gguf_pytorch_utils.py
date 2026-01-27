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
from typing import NamedTuple

import numpy as np
from tqdm.auto import tqdm

from .integrations import (
    GGUF_CONFIG_DEFAULTS_MAPPING,
    GGUF_CONFIG_MAPPING,
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
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["config"].keys())


class GGUFTensor(NamedTuple):
    weights: np.ndarray
    name: str
    metadata: dict


class TensorProcessor:
    def __init__(self, config=None):
        self.config = config or {}

    def preprocess_name(self, hf_name: str) -> str:
        """
        Preprocesses the tensor name to ease loading the GGUF tensors.
        """
        return hf_name

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map: dict[str, str], suffix: str, qual_name: str, hf_name: str
    ):
        """
        Called when get_gguf_hf_weights_map fails to map a HF parameter
        (tensor) and corresponding GGUF one.

        This is particularly useful to resolve one-to-many
        HF-GGUF mappings sometimes appear in some MoE models.
        """
        pass

    def process(self, weights, name, **kwargs):
        return GGUFTensor(weights, name, {})


class LlamaTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if ".attn_k." in name or ".attn_q." in name:
            num_heads = self.config.get("num_attention_heads")
            num_kv_heads = self.config.get("num_key_value_heads")

            if None in (num_heads, num_kv_heads):
                return GGUFTensor(weights, name, {})
            if ".attn_q." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_heads)
            elif ".attn_k." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_kv_heads)
        return GGUFTensor(weights, name, {})

    def _reverse_permute_weights(
        self, weights: np.ndarray, n_head: int, num_kv_heads: int | None = None
    ) -> np.ndarray:
        # Original permutation implementation
        # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402-L1408
        if num_kv_heads is not None and n_head != num_kv_heads:
            n_head = num_kv_heads

        dim = weights.shape[0] // n_head // 2
        w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
        return w.swapaxes(2, 1).reshape(weights.shape)


class Qwen2MoeTensorProcessor(TensorProcessor):
    HF_EXPERT_RENAME_PATTERN = re.compile(r"mlp.experts.\d+.")
    HF_MOE_W13_PATTERN = re.compile(r"model\.layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj")
    GGUF_MOE_WEIGHTS_PATTERN = re.compile(r"(?P<name>.*\.ffn_(?P<w>gate|down|up)_exps)\.weight$")

    def __init__(self, config=None):
        super().__init__(config=config)

    def preprocess_name(self, hf_name: str) -> str:
        return re.sub(self.HF_EXPERT_RENAME_PATTERN, "mlp.experts.", hf_name)

    def perform_fallback_tensor_mapping(
        self, gguf_to_hf_name_map: dict[str, str], suffix: str, qual_name: str, hf_name: str
    ):
        # Map merged MoE weights (w1 (gate) and w3 (up)) separately.
        if m := re.fullmatch(self.HF_MOE_W13_PATTERN, hf_name):
            full_hf_name = qual_name + hf_name
            gguf_to_hf_name_map[f"blk.{m['bid']}.ffn_gate_exps{suffix}"] = full_hf_name
            gguf_to_hf_name_map[f"blk.{m['bid']}.ffn_up_exps{suffix}"] = full_hf_name

    def process(self, weights, name: str, **kwargs):
        if m := re.fullmatch(self.GGUF_MOE_WEIGHTS_PATTERN, name):
            tensor_key_mapping = kwargs.get("tensor_key_mapping")
            parsed_parameters = kwargs.get("parsed_parameters")
            if tensor_key_mapping:
                self._set_moe_expert_tensor(weights, parsed_parameters, tensor_key_mapping[m["name"]], m["w"])
                return GGUFTensor(weights, None, {})
        if "ffn_gate_inp_shexp" in name:
            # for compatibility tensor shared_expert_gate must be (1, 2048) dim,
            # quantized one is (2048)
            weights = np.expand_dims(weights, axis=0)
        return GGUFTensor(weights, name, {})

    def _set_moe_expert_tensor(self, weights: np.ndarray, parsed_parameters: dict[str, dict], hf_name: str, w: str):
        torch_weights = torch.from_numpy(np.copy(weights))
        if w == "down":
            parsed_parameters["tensors"][hf_name] = torch_weights
        else:
            # Double the size of the second dimension to interleave w1 (gate) and w3 (up)
            # weights per expert (which is the first dimension).
            # w1 (gate) comes first and w3 (up) comes second.
            # ref: https://github.com/vllm-project/vllm/blob/8f8fda261a620234fdeea338f44093d5d8072879/vllm/model_executor/layers/fused_moe/layer.py#L988-L1015
            shape = list(weights.shape)
            shard_dim = 1
            shard_size = shape[shard_dim]
            shape[shard_dim] = shard_size * 2
            if hf_name not in parsed_parameters["tensors"]:
                parsed_parameters["tensors"][hf_name] = torch.zeros(shape, dtype=torch_weights.dtype)
            out: torch.Tensor = parsed_parameters["tensors"][hf_name]
            if w == "gate":
                out = out.narrow(shard_dim, 0, shard_size)
            else:  # w == "up"
                out = out.narrow(shard_dim, shard_size, shard_size)
            out.copy_(torch_weights)


class BloomTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "attn_qkv" in name:
            num_heads = self.config["n_head"]
            n_embed = self.config["hidden_size"]
            if "weight" in name:
                weights = self._reverse_reshape_weights(weights, num_heads, n_embed)
            else:
                weights = self._reverse_reshape_bias(weights, num_heads, n_embed)
        return GGUFTensor(weights, name, {})

    def _reverse_reshape_weights(self, weights: np.ndarray, n_head: int, n_embed: int):
        # Original reshape implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L972-L985
        q, k, v = np.array_split(weights, 3, axis=0)

        q = q.reshape(n_head, n_embed // n_head, n_embed)
        k = k.reshape(n_head, n_embed // n_head, n_embed)
        v = v.reshape(n_head, n_embed // n_head, n_embed)
        qkv_weights = np.stack([q, k, v], axis=1)

        return qkv_weights.reshape(n_head * 3 * (n_embed // n_head), n_embed)

    def _reverse_reshape_bias(self, weights: np.ndarray, n_head: int, n_embed: int):
        # Original reshape implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L986-L998
        q_bias, k_bias, v_bias = np.array_split(weights, 3)

        q_bias = q_bias.reshape(n_head, n_embed // n_head)
        k_bias = k_bias.reshape(n_head, n_embed // n_head)
        v_bias = v_bias.reshape(n_head, n_embed // n_head)

        qkv_bias = np.stack([q_bias, k_bias, v_bias], axis=1).flatten()
        return qkv_bias


class T5TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        bid = None
        for chunk in name.split("."):
            if chunk.isdigit():
                bid = int(chunk)
                break
        return GGUFTensor(weights, name, {"bid": bid})


class GPT2TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        # Original transpose implementation
        # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L2060-L2061
        if (
            "attn_qkv.weight" in name
            or "ffn_down.weight" in name
            or "ffn_up.weight" in name
            or "attn_output.weight" in name
        ):
            weights = weights.T

        # Handle special case for output.weight
        if name == "output.weight":
            # output.weight has conflicts with attn_output.weight in name checking
            # Store the tensor directly and signal to skip further processing
            name = "lm_head.weight"
            parsed_parameters = kwargs.get("parsed_parameters", {})
            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))
            name = None  # Signal to skip further processing
        return GGUFTensor(weights, name, {})


class MambaTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            # for compatibility tensor ssm_conv1d must be (5120, 1, 4]) dim,
            # quantized one is (5120, 4)
            weights = np.expand_dims(weights, axis=1)
        if "ssm_a" in name:
            # Original exponential implementation
            # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L2975-L2977
            weights = np.log(-weights)
        return GGUFTensor(weights, name, {})


class NemotronTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    # ref : https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L4666
    def process(self, weights, name, **kwargs):
        if "norm.weight" in name:
            weights = weights - 1
        return GGUFTensor(weights, name, {})


class Gemma2TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    # ref: https://github.com/ggerganov/llama.cpp/blob/d79d8f39b4da6deca4aea8bf130c6034c482b320/convert_hf_to_gguf.py#L3191
    # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
    def process(self, weights, name, **kwargs):
        if "norm.weight" in name:
            weights = weights - 1
        return GGUFTensor(weights, name, {})


class Lfm2TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "shortconv.conv.weight" in name:
            ## GGUF shape is [hidden_dim, L_cache], HF expects [hidden_dim, 1, L_cache]
            weights = np.expand_dims(weights, axis=1)  ## equivalent to unsqueeze(1)
        return GGUFTensor(weights, name, {})


TENSOR_PROCESSORS = {
    "llama": LlamaTensorProcessor,
    "qwen2moe": Qwen2MoeTensorProcessor,
    "qwen3moe": Qwen2MoeTensorProcessor,
    "bloom": BloomTensorProcessor,
    "t5": T5TensorProcessor,
    "t5encoder": T5TensorProcessor,
    "gpt2": GPT2TensorProcessor,
    "mamba": MambaTensorProcessor,
    "nemotron": NemotronTensorProcessor,
    "gemma2": Gemma2TensorProcessor,
    "gemma3": Gemma2TensorProcessor,
    "lfm2": Lfm2TensorProcessor,
}


def read_field(reader, field):
    if field not in reader.fields:
        return []
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


# modified from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/loader.py#L1115-L1147
def get_gguf_hf_weights_map(
    hf_model,
    processor: TensorProcessor,
    model_type: str | None = None,
    num_layers: int | None = None,
    qual_name: str = "",
):
    """
    GGUF uses this naming convention for their tensors from HF checkpoint:
    `blk.N.BB.weight` and `blk.N.BB.bias`
    where N signifies the block number of a layer, and BB signifies the
    attention/mlp layer components.
    See "Standardized tensor names" in
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
    """
    if is_gguf_available() and is_torch_available():
        from gguf import MODEL_ARCH_NAMES, get_tensor_name_map
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    model_type = hf_model.config.model_type if model_type is None else model_type
    num_layers = hf_model.config.num_hidden_layers if num_layers is None else num_layers
    # hack: ggufs have a different name for cohere
    if model_type == "cohere":
        model_type = "command-r"
    elif model_type == "qwen2_moe":
        model_type = "qwen2moe"
    elif model_type == "qwen3_moe":
        model_type = "qwen3moe"
    elif model_type == "gemma3_text":
        model_type = "gemma3"
    elif model_type == "umt5":
        model_type = "t5"
    arch = None
    for key, value in MODEL_ARCH_NAMES.items():
        if value == model_type:
            arch = key
            break
    if arch is None:
        raise NotImplementedError(
            f"Unknown gguf model_type: {model_type} in gguf-py. "
            "This might because you're using an outdated version of gguf-py package, "
            "you can install `gguf` package from source refer to "
            "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development"
        )
    name_map = get_tensor_name_map(arch, num_layers)

    # Use a dummy conversion to get the mapping, because
    # hf => gguf and gguf => hf mappings are reversed
    gguf_to_hf_name_map = {}
    state_dict = hf_model.state_dict()
    for hf_name in state_dict:
        hf_name = processor.preprocess_name(hf_name)

        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix

        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            processor.perform_fallback_tensor_mapping(gguf_to_hf_name_map, suffix, qual_name, hf_name)
            continue

        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name

    # Some model like Bloom converted from BloomModel instead of BloomForCausalLM
    # Therefore, we need to check submodule as well to get a correct mapping
    if named_children := hf_model.named_children():
        for name, child in named_children:
            sub_map = get_gguf_hf_weights_map(
                child, processor, model_type, num_layers, qual_name=f"{qual_name}{name}."
            )
            # Ignore the keys that are already in the main map to avoid overwriting
            sub_map = {k: v for k, v in sub_map.items() if k not in gguf_to_hf_name_map}
            gguf_to_hf_name_map.update(sub_map)

    return gguf_to_hf_name_map


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `False`):
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
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if model_name and "umt5" in model_name[0].lower():
            updated_architecture = "umt5"
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["UMT5EncoderModel"]
        else:
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
            updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all(tensor.name != "output.weight" for tensor in reader.tensors) or architecture in exceptions
    )

    # Set GGUF-specific default values
    config_defaults = GGUF_CONFIG_DEFAULTS_MAPPING.get(
        updated_architecture, GGUF_CONFIG_DEFAULTS_MAPPING.get(architecture) or {}
    )
    for key, value in config_defaults.items():
        parsed_parameters["config"].setdefault(key, value)

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

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
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

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"]["model_type"] == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    if parsed_parameters["config"]["model_type"] == "lfm2":
        gguf_num_key_value_heads = parsed_parameters["config"]["num_key_value_heads"]
        # LFM2 GGUF checkpoint defines num_key_value_heads as a list of integers .e.g [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0] but we need to set it to the max value for HF
        parsed_parameters["config"]["num_key_value_heads"] = max(gguf_num_key_value_heads)
        ## we already read the correct intermediate_size from the GGUF checkpoint so we need to set block_auto_adjust_ff_dim to False
        parsed_parameters["config"]["block_auto_adjust_ff_dim"] = False

        ## llama.cpp defines the layers that are full-attention by looking at num_key_value_heads
        ## we need to set the full_attn_idxs to the layers that are full-attention
        parsed_parameters["config"]["full_attn_idxs"] = [
            i for i, num_kv_heads in enumerate(gguf_num_key_value_heads) if num_kv_heads > 0
        ]

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
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
        parsed_parameters["tensors"] = {}

        config = parsed_parameters.get("config", {})

        ProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        processor = ProcessorClass(config=config)

        tensor_key_mapping = get_gguf_hf_weights_map(model_to_load, processor)

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            weights = result.weights
            name = result.name

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]

            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
