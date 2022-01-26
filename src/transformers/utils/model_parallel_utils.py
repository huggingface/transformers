# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from math import ceil

from .. import BertPreTrainedModel, GPT2PreTrainedModel, TransfoXLPreTrainedModel


# ie. nn.Linear(3 * dim, dim)
# only defined the models that have fused attention layer.
FUSED_ATTENTION_MAPPING = {
    GPT2PreTrainedModel: {"attn.c_attn": 3, "crossattention.c_attn": 2},
    TransfoXLPreTrainedModel: {"qkv_net": 3},
}

# ie. nn.Linear(out_dim, in_dim) or Conv1D()
# only defined the models that have reversed parameters.
REVERSED_PARAM_MAPPING = {
    GPT2PreTrainedModel: ["attn", "crossattention", "mlp"],
    TransfoXLPreTrainedModel: ["qkv_net"],
}

# All the mapping for tensor parallelism
TENSOR_PARALLEL_MAPPING = {
    BertPreTrainedModel: {
        "column_parallel": ["query", "key", "value", "intermediate.dense"],
        "row_parallel": ["output.dense"],
        "update_attrs": ["num_attention_heads", "all_head_size"],
    },
    GPT2PreTrainedModel: {
        "column_parallel": ["c_attn", "q_attn", "c_fc"],
        "row_parallel": ["c_proj"],
        "update_attrs": ["embed_dim", "split_size", "num_heads"],
    },
}


def get_mapping(model, mapping):
    """
    Helper function to find mapping by model

    Args:
        model (PreTrainedModel): model object
        mapping (Dict): mapping object

    Returns:
        Any: mapping object

    Examples:
        >>> lm_head_model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> get_map(lm_head_model, TENSOR_PARALLEL_MAPPING)
        {
            "column_parallel": ["c_attn", "q_attn", "c_fc"],
            "row_parallel": ["c_proj"],
            "update_attrs": ["embed_dim", "split_size", "num_heads"],
        }
        >>> get_map(lm_head_model, FUSED_ATTENTION_MAPPING)
        {"attn.c_attn": 3, "crossattention.c_attn": 2}

        >>> seq_clf_model = GPT2ForSequenceClassification.from_pretrained("gpt2")
        >>> get_map(seq_clf_model, TENSOR_PARALLEL_MAPPING)
        {
            "column_parallel": ["c_attn", "q_attn", "c_fc"],
            "row_parallel": ["c_proj"],
            "update_attrs": ["embed_dim", "split_size", "num_heads"],
        }
        >>> get_map(seq_clf_model, FUSED_ATTENTION_MAPPING)
        {"attn.c_attn": 3, "crossattention.c_attn": 2}

    """
    for pretrained_model_cls, value in mapping.items():
        if isinstance(model, pretrained_model_cls):
            return value

    return None


def get_tensor_parallel_mapping(model):
    """
    Get tensor parallel mapping by model

    Args:
        model (PreTrainedModel): model object

    Returns:
        Dict: tensor parallel mapping

    Examples:
        >>> lm_head_model = GPT2LMHeadModel.from_pretrained("gpt2")
        >>> get_tensor_parallel_mapping(lm_head_model)
        {
            "column_parallel": ["c_attn", "q_attn", "c_fc"],
            "row_parallel": ["c_proj"],
            "update_attrs": ["embed_dim", "split_size", "num_heads"],
        }
    """
    return get_mapping(model, TENSOR_PARALLEL_MAPPING)


def is_reversed_param(model, param_name):
    """
    Check reversed parameter or not.

    Args:
        model (PreTrainedModel): model object
        param_name (str): the name of parameter (e.g. 'transformer.h.0.attn...')

    Notes:
        ``Conv1D`` of GPT2 and ``qkv_net`` of TransfoXL have reversed parameters

    Returns:
        bool: whether reversed parameter or not.

    Examples:
        >>> is_reversed_param(model, 'transformer.h.0.attn.c_attn')
        True
        >>> is_reversed_param(model, 'transformer.wte')
        False
    """
    mapping = get_mapping(model, REVERSED_PARAM_MAPPING)

    if mapping is not None:
        return any([i in param_name for i in mapping])

    return False


def get_fusion_degree(model, param_name):
    """
    Get fused attention layer degree

    Args:
        model (PreTrainedModel): model object
        param_name (str): the name of parameter (e.g. 'transformer.h.0.attn...')

    Notes:
        The `c_attn` layer in the self-attention layer of GPT2 is size of (dim * 3, dim).
        In this case, the fusion degree is 3.

        The `c_attn` layer in the cross-attention attention layer of GPT2 is size of (dim * 2, dim).
        In this case, the fusion degree is 2.

    Returns:
        int: the fusion degree
    """
    mapping = get_mapping(model, FUSED_ATTENTION_MAPPING)

    if mapping is not None:
        for key, degree in mapping.items():
            if key in param_name:
                return degree
    return 1


def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
            "attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
