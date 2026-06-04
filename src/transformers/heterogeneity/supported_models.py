from collections.abc import Callable

from transformers.heterogeneity.heterogeneous_modeling_spec import HeterogeneousModelingSpec
from transformers.heterogeneity.skip_utils import ReturnEntry, get_skip_replacement


MODEL_TO_SPEC_FACTORY: dict[str, Callable[[], HeterogeneousModelingSpec]] = {
    "gpt_oss": lambda: gpt_oss(),
    "llama": lambda: llama(),
    "llama4": lambda: llama4(),
    "nemotron_h": lambda: nemotron_h(),
}

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


def llama4() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.llama4.modeling_llama4 import (
        Llama4TextAttention,
        Llama4TextDecoderLayer,
        Llama4TextMLP,
        Llama4TextMoe,
        Llama4TextRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=Llama4TextDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    Llama4TextRMSNorm, ReturnEntry(arg_name="x", transform=lambda x: x)
                ),
                "self_attn": get_skip_replacement(
                    Llama4TextAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    Llama4TextRMSNorm, ReturnEntry(arg_name="x", transform=lambda x: x)
                ),
                "feed_forward": get_skip_replacement(
                    Llama4TextMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)
                ),
                ("feed_forward", Llama4TextMoe): get_skip_replacement(
                    Llama4TextMoe, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                ),
            },
        },
    )


def nemotron_h() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHAttention,
        NemotronHBlock,
        NemotronHMamba2Mixer,
        NemotronHMoE,
        NemotronHRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=NemotronHBlock,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "mixer": {
                "norm": get_skip_replacement(
                    NemotronHRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                ("mixer", NemotronHAttention): get_skip_replacement(
                    NemotronHAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
                ("mixer", NemotronHMoE): get_skip_replacement(
                    NemotronHMoE, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                ),
                ("mixer", NemotronHMamba2Mixer): get_skip_replacement(
                    NemotronHMamba2Mixer, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                ),
            },
        },
    )
