from .. import BertPreTrainedModel, GPT2PreTrainedModel, T5PreTrainedModel, TransfoXLPreTrainedModel


"""
All the mapping for tensor parallelism.
This mapping is following the follow format.

TENSOR_PARALLEL_MAPPING = {
    PreTrainedModel class: {
        "col": list of column parallel parameters,
        "row": list of row parallel parameters,
        "update": list of attributes to be updated,
        "col_no_replacement": list of column parallel parameters without module replacement (Optional)
        "row_no_replacement": list of row parallel parameters without module replacement (Optional),
        ...
        could be added more to avoid exceptions.
    }
}

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
}

# Optional: fused attention layers like nn.Linear(3 * dim, dim).
FUSED_ATTENTION_MAPPING = {
    GPT2PreTrainedModel: {"attn.c_attn": 3, "crossattention.c_attn": 2},
    TransfoXLPreTrainedModel: {"qkv_net": 3},
}

# Optional: reversed parameters like nn.Linear(out_dim, in_dim) or Conv1D().
REVERSED_PARAM_MAPPING = {
    GPT2PreTrainedModel: ["attn", "crossattention", "mlp"],
    TransfoXLPreTrainedModel: ["qkv_net"],
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
