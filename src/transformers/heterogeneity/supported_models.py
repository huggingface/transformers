from collections.abc import Callable

from transformers.heterogeneity.heterogeneous_modeling_spec import HeterogeneousModelingSpec
from transformers.heterogeneity.skip_utils import ReturnEntry, get_skip_replacement


MODEL_TO_SPEC_FACTORY: dict[str, Callable[[], HeterogeneousModelingSpec]] = {
    "llama": lambda: llama(),
    "gpt_oss": lambda: gpt_oss(),
}


def llama() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=LlamaDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "self_attn": get_skip_replacement(
                    LlamaAttention,
                    [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None],
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "mlp": get_skip_replacement(LlamaMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)),
            },
        },
    )


def gpt_oss() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention,
        GptOssDecoderLayer,
        GptOssMLP,
        GptOssRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=GptOssDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    GptOssRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "self_attn": get_skip_replacement(
                    GptOssAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    GptOssRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "mlp": get_skip_replacement(
                    GptOssMLP, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
        },
    )
