from ..utils import is_torch_greater_or_equal


if is_torch_greater_or_equal("2.5"):
    from torch.nn.attention.flex_attention import flex_attention


def flex_attention_forward(module, query, key, value, attention_mask, output_attentions=False, **_kwargs):
    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def causal_mod(score, b, h, q_idx, kv_idx):
        if causal_mask is not None:
            score += causal_mask[b][0][q_idx][kv_idx]
        return score

    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        score_mod=causal_mod,
        enable_gqa=True,
        scale=module.scaling,
        return_lse=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attention_weights
