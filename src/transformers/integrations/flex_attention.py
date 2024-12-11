import torch

from ..utils import is_torch_greater_or_equal


if is_torch_greater_or_equal("2.5"):
    from torch.nn.attention.flex_attention import flex_attention

def flex_attention_forward(config, query, key, value, mask, output_attentions=False, **_kwargs):
    def tanh_softcap(score, b, h, q_idx, kv_idx):
        soft_cap = config.attn_logit_softcapping
        score = soft_cap * torch.tanh(score / soft_cap)
        if mask is not None:
            return score + mask[b][0][q_idx][kv_idx]
        return score

    attn_output = flex_attention(
        query,
        key,
        value,
        score_mod=tanh_softcap,
        enable_gqa=True,
        scale=config.scaling,
        return_lse=output_attentions,
    )
    if not output_attentions:
        return attn_output, None
    else:
        return attn_output[0], attn_output[1]
