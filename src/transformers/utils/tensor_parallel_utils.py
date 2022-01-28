# coding=utf-8
# Copyright 2021 TUNiB Inc and The HuggingFace Inc. team.
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
from .. import (
    BertPreTrainedModel,
    GPT2PreTrainedModel,
    RobertaPreTrainedModel,
    T5PreTrainedModel,
    TransfoXLPreTrainedModel,
)


"""
All the mapping for tensor parallelism.
This mapping is following the follow format.

TENSOR_PARALLEL_MAPPING = {
    PreTrainedModel class: {
        "col": list of column parallel parameters,
        "row": list of row parallel parameters,
        "update": list of attributes to be updated,
        "col_no_replacement": list of column parallel parameters without module replacement (opt)
        "row_no_replacement": list of row parallel parameters without module replacement (opt),
        ...
        could be added more to avoid exceptions.
    }
}

Or if a model A has the same map with the other model B, define like:

TENSOR_PARALLEL_MAPPING = {
    PreTrainedModel class A: PreTrainedModel class B
}

Then, call ``copy_mapping(PreTrainedModel class A)``.
"""

TENSOR_PARALLEL_MAPPING = {
    BertPreTrainedModel: {
        "col": ["query", "key", "value", "intermediate.dense"],
        "row": ["output.dense"],
        "update": ["num_attention_heads", "all_head_size"],
    },
    GPT2PreTrainedModel: {
        "col": ["c_attn", "q_attn", "c_fc"],
        "row": ["c_proj"],
        "update": ["embed_dim", "split_size", "num_heads"],
    },
    T5PreTrainedModel: {
        "col": ["Attention.q", "Attention.k", "Attention.v", "DenseReluDense.wi"],
        "row": ["Attention.o", "DenseReluDense.wo"],
        "row_no_replacement": ["relative_attention_bias"],
        "update": ["d_model", "n_heads", "inner_dim"],
    },
    RobertaPreTrainedModel: BertPreTrainedModel,
}


def copy_mapping(model_cls):
    TENSOR_PARALLEL_MAPPING[model_cls] = TENSOR_PARALLEL_MAPPING[TENSOR_PARALLEL_MAPPING[model_cls]]


# Copy the same mapping.
copy_mapping(RobertaPreTrainedModel)


# ie. nn.Linear(3 * dim, dim) (opt)
FUSED_ATTENTION_MAPPING = {
    GPT2PreTrainedModel: {"attn.c_attn": 3, "crossattention.c_attn": 2},
    TransfoXLPreTrainedModel: {"qkv_net": 3},
}

# ie. nn.Linear(out_dim, in_dim) or Conv1D() (opt)
REVERSED_PARAM_MAPPING = {
    GPT2PreTrainedModel: ["attn", "crossattention", "mlp"],
    TransfoXLPreTrainedModel: ["qkv_net"],
}


def get_mapping(model, mapping):
    """
    Helper function to find

    Args:
        model (PreTrainedModel): model object
        mapping (Dict): map object

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
    e.g. ``Conv1D`` of GPT2 and ``qkv_net`` of TransfoXL have reversed parameters

    Args:
        model (PreTrainedModel): model object
        param_name (str): the name of parameter (e.g. 'transformer.h.0.attn...')

    Returns:
        bool: whether reversed parameter or not.
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
        The `c_attn` layer that has size of (dim * 3, dim) in GPT2.
        In this case, the fusion degree is 3.

    Returns:
        int: the fusion degree
    """
    mapping = get_mapping(model, FUSED_ATTENTION_MAPPING)

    if mapping is not None:
        for key, degree in mapping.items():
            if key in param_name:
                return degree
    return 1
