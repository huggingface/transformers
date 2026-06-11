"""Model-specific fixtures for heterogeneity tests.

Contains architecture-dependent test metadata:
- Skip-aware reference layer classes (for building ground-truth models)
- Model class mappings (for _hetero_context setup)
- References to production heterogeneous modeling spec factories

When adding a new architecture, update the production spec in
transformers.heterogeneity.supported_models and add only the test-only
reference layer mapping here.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from transformers.heterogeneity import HeterogeneousModelingSpec
from transformers.heterogeneity.supported_models import MODEL_TO_SPEC_FACTORY
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssDecoderLayer,
    GptOssPreTrainedModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4PreTrainedModel,
    Llama4TextDecoderLayer,
)
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHBlock,
    NemotronHPreTrainedModel,
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
        layer_config = config.per_layer_config[layer_idx]
        super().__init__(layer_config, layer_idx)
        if "attention" in layer_config.skip:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" in layer_config.skip:
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
        layer_config = config.per_layer_config[layer_idx]
        super().__init__(layer_config, layer_idx)
        if "attention" in layer_config.skip:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" in layer_config.skip:
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
        layer_config = config.per_layer_config[layer_idx]
        super().__init__(layer_config, layer_idx)
        if "attention" in layer_config.skip:
            self.input_layernorm = None
            self.self_attn = None
        if "mlp" in layer_config.skip:
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
        layer_config = config.per_layer_config[layer_idx]
        super().__init__(layer_config, layer_idx)
        if "mixer" in layer_config.skip:
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
    pretrained_cls: type  # PreTrainedModel subclass patched with the production heterogeneous modeling spec
    ref_layer_cls: type  # skip-aware reference layer for building ground-truth models
    spec_factory: Callable[[], HeterogeneousModelingSpec]


MODEL_FIXTURES = {
    "llama": ModelFixture(
        pretrained_cls=LlamaPreTrainedModel,
        ref_layer_cls=SkipAwareLlamaDecoderLayer,
        spec_factory=MODEL_TO_SPEC_FACTORY["llama"],
    ),
    "gpt_oss": ModelFixture(
        pretrained_cls=GptOssPreTrainedModel,
        ref_layer_cls=SkipAwareGptOssDecoderLayer,
        spec_factory=MODEL_TO_SPEC_FACTORY["gpt_oss"],
    ),
    "llama4": ModelFixture(
        pretrained_cls=Llama4PreTrainedModel,
        ref_layer_cls=SkipAwareLlama4TextDecoderLayer,
        spec_factory=MODEL_TO_SPEC_FACTORY["llama4"],
    ),
    "nemotron_h": ModelFixture(
        pretrained_cls=NemotronHPreTrainedModel,
        ref_layer_cls=SkipAwareNemotronHBlock,
        spec_factory=MODEL_TO_SPEC_FACTORY["nemotron_h"],
    ),
}
