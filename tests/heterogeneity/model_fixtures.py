"""Model-specific definitions for heterogeneity tests.

Contains all architecture-dependent knowledge in one place:
- Skip-aware reference layer classes (for building ground-truth models)
- Model/layer class mappings (for _hetero_context setup)
- Skip descriptors (for configuring the heterogeneity mechanism)

When adding a new architecture, all updates go here.
"""

from dataclasses import dataclass

import torch

from transformers.heterogeneity import ReturnEntry, get_skip_replacement
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssMLP,
    GptOssPreTrainedModel,
    GptOssRMSNorm,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4PreTrainedModel,
    Llama4TextAttention,
    Llama4TextDecoderLayer,
    Llama4TextMLP,
    Llama4TextMoe,
    Llama4TextRMSNorm,
)
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHBlock,
    NemotronHMamba2Mixer,
    NemotronHMoE,
    NemotronHPreTrainedModel,
    NemotronHRMSNorm,
)


# ──────────────────────────────────────────────────────────────────────
# Skip-aware reference layer classes
# ──────────────────────────────────────────────────────────────────────
# These represent what native skip support would look like if each model
# implemented it directly: skipped modules are set to None, and forward
# simply skips them. Used as ground truth — independent of the generic
# heterogeneity code.
#
# The forward methods are copied from the original layer classes and
# wrapped with skip guards.


class SkipAwareLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        layer_config = config.get_full_layer_config(layer_idx)
        super().__init__(layer_config, layer_idx)
        if getattr(layer_config, "skip_attention", False):
            self.input_layernorm = None
            self.self_attn = None
        if getattr(layer_config, "skip_mlp", False):
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn is not None:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if self.post_attention_layernorm is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.mlp is not None:
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


class SkipAwareGptOssDecoderLayer(GptOssDecoderLayer):
    def __init__(self, config, layer_idx):
        layer_config = config.get_full_layer_config(layer_idx)
        super().__init__(layer_config, layer_idx)
        if getattr(layer_config, "skip_attention", False):
            self.input_layernorm = None
            self.self_attn = None
        if getattr(layer_config, "skip_mlp", False):
            self.post_attention_layernorm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn is not None:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if self.post_attention_layernorm is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.mlp is not None:
            hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
            hidden_states = residual + hidden_states

        return hidden_states


class SkipAwareLlama4TextDecoderLayer(Llama4TextDecoderLayer):
    def __init__(self, config, layer_idx):
        layer_config = config.get_full_layer_config(layer_idx)
        super().__init__(layer_config, layer_idx)
        if getattr(layer_config, "skip_attention", False):
            self.input_layernorm = None
            self.self_attn = None
        if getattr(layer_config, "skip_mlp", False):
            self.post_attention_layernorm = None
            self.feed_forward = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        # Self Attention
        residual = hidden_states

        if self.input_layernorm is not None:
            hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn is not None:
            attention_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        if self.post_attention_layernorm is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.feed_forward is not None:
            hidden_states = self.feed_forward(hidden_states)
            if self.is_moe_layer:
                hidden_states, _ = hidden_states
            hidden_states = residual + hidden_states.view(residual.shape)

        return hidden_states


class SkipAwareNemotronHBlock(NemotronHBlock):
    def __init__(self, config, layer_idx):
        layer_config = config.get_full_layer_config(layer_idx)
        super().__init__(layer_config, layer_idx)
        if getattr(layer_config, "skip_mixer", False):
            self.norm = None
            self.mixer = None

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ):
        residual = hidden_states

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.mixer is not None:
            if self.block_type == "mamba":
                hidden_states = self.mixer(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
            elif self.block_type == "attention":
                hidden_states, _ = self.mixer(
                    hidden_states=hidden_states,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    **kwargs,
                )
            else:
                hidden_states = self.mixer(hidden_states)

            hidden_states = residual + hidden_states

        return hidden_states


# ──────────────────────────────────────────────────────────────────────
# Per-model fixture registry
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelFixture:
    pretrained_cls: type  # PreTrainedModel subclass — patched with _layer_cls / _skip_descriptors
    layer_cls: type  # layer class set as _layer_cls on the model
    ref_layer_cls: type  # skip-aware reference layer for building ground-truth models
    skip_descriptors: dict  # skip type → {attr: replacement_cls}


MODEL_FIXTURES = {
    "llama": ModelFixture(
        pretrained_cls=LlamaPreTrainedModel,
        layer_cls=LlamaDecoderLayer,
        ref_layer_cls=SkipAwareLlamaDecoderLayer,
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "self_attn": get_skip_replacement(
                    LlamaAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda x: x)
                ),
                "mlp": get_skip_replacement(LlamaMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)),
            },
        },
    ),
    "gpt_oss": ModelFixture(
        pretrained_cls=GptOssPreTrainedModel,
        layer_cls=GptOssDecoderLayer,
        ref_layer_cls=SkipAwareGptOssDecoderLayer,
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
    ),
    "llama4": ModelFixture(
        pretrained_cls=Llama4PreTrainedModel,
        layer_cls=Llama4TextDecoderLayer,
        ref_layer_cls=SkipAwareLlama4TextDecoderLayer,
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
    ),
    "nemotron_h": ModelFixture(
        pretrained_cls=NemotronHPreTrainedModel,
        layer_cls=NemotronHBlock,
        ref_layer_cls=SkipAwareNemotronHBlock,
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
    ),
}
