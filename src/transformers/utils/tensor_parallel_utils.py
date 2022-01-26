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
