from ..utils import is_torch_greater_or_equal


if is_torch_greater_or_equal("2.5"):
    from torch.nn.attention.flex_attention import flex_attention

def flex_attention_forward(module, query, key, value, attention_mask, output_attentions=False, **_kwargs):

    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        enable_gqa=True,
        scale=module.scaling,
        return_lse=True,
    )
    return attn_output, attention_weights
