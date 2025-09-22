import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import check_model_inputs
from ..vit.modeling_vit import ViTAttention, ViTEncoder, ViTSelfAttention
from ..vitdet.configuration_vitdet import VitDetConfig
from ..vitdet.modeling_vitdet import (
    VitDetBackbone,
    VitDetDropPath,
    VitDetEmbeddings,
    VitDetMlp,
    VitDetPreTrainedModel,
)


logger = logging.get_logger(__name__)


class LwDetrViTConfig(VitDetConfig):
    model_type = "lw_detr_vit"
    attribute_map = {
        "attention_probs_dropout_prob": "dropout_prob",
        "decoder_self_attention_heads": "num_attention_heads",
    }

    def __init__(
        self,
        image_size=256,
        use_cae: bool = True,
        cae_init_values: float = 0.1,
        num_windows=16,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        del self.residual_block_indices

        self.use_cae = use_cae
        self.cae_init_values = cae_init_values
        if num_windows % math.sqrt(num_windows) != 0:
            raise ValueError(
                f"`num_windows` has to be a perfect square, where num_windows % math.sqrt(num_windows) != 0, but got {num_windows}."
            )
        if image_size / num_windows % math.sqrt(num_windows) != 0:
            raise ValueError(
                f"`image_size` has to be divisible by `num_windows`, where image_size / num_windows % math.sqrt(num_windows) != 0,but got {image_size} and {num_windows}."
            )
        self.num_windows = num_windows
        self.num_windows_side = int(math.sqrt(num_windows))
        self.drop_path_rates = [x.item() for x in np.linspace(0, self.drop_path_rate, self.num_hidden_layers)]


class LwDetrViTSelfAttention(ViTSelfAttention):
    def __init__(self, config: LwDetrViTConfig):
        super().__init__(config)
        del self.key
        if config.use_cae:
            self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)


class LwDetrViTAttention(ViTAttention):
    def __init__(self, config: LwDetrViTConfig):
        """
        Args:
            config (`LwDetrViTConfig`):
                Model configuration.
        """
        super().__init__(config)
        self.attention = LwDetrViTSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, head_mask, **kwargs)
        output = self.output(self_attn_output)
        return output


class LwDetrViTDropPath(VitDetDropPath):
    pass


class LwDetrViTMlp(VitDetMlp):
    pass


class LwDetrViTLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: LwDetrViTConfig,
        layer_idx,
    ) -> None:
        super().__init__()

        dim = config.hidden_size
        self.attention = LwDetrViTAttention(config)
        self.intermediate = LwDetrViTMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.use_cae = config.use_cae
        if self.use_cae:
            self.gamma_1 = nn.Parameter(torch.Tensor(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(torch.Tensor(dim), requires_grad=True)

        drop_path_rate = config.drop_path_rates[layer_idx]
        self.drop_path = LwDetrViTDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.window = layer_idx in config.window_block_indices
        self.num_windows = config.num_windows
        self.num_windows_side = int(math.sqrt(self.num_windows))
        self.use_cae = config.use_cae

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        batch_size, seq_len, channels = hidden_states.shape
        hidden_states_norm = self.layernorm_before(hidden_states)

        if not self.window:
            hidden_states_norm = hidden_states_norm.reshape(
                batch_size // self.num_windows, self.num_windows * seq_len, channels
            )
            if head_mask is not None:
                head_mask = head_mask.reshape(batch_size // self.num_windows, self.num_windows * seq_len)

        attention_output = self.attention(hidden_states_norm, head_mask, **kwargs)

        if self.use_cae:
            attention_output = attention_output * self.gamma_1

        if not self.window:
            attention_output = attention_output.reshape(batch_size, seq_len, channels)
            if head_mask is not None:
                head_mask = head_mask.reshape(batch_size, seq_len)

        # first residual connection
        hidden_states = hidden_states + self.drop_path(attention_output)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        if self.use_cae:
            layer_output = layer_output * self.gamma_2

        hidden_states = hidden_states + self.drop_path(layer_output)

        return hidden_states


class LwDetrViTEncoder(ViTEncoder):
    def __init__(self, config: LwDetrViTConfig) -> None:
        super().__init__(config)
        self.layer = nn.ModuleList([LwDetrViTLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        list_hidden_states = []
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, layer_head_mask, **kwargs)
            list_hidden_states.append(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=tuple(list_hidden_states))


class LwDetrViTEmbeddings(VitDetEmbeddings):
    pass


class LwDetrViTPreTrainedModel(VitDetPreTrainedModel):
    config: LwDetrViTConfig
    base_model_prefix = "lw_detr_vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LwDetrViTEmbeddings", "LwDetrViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ["LwDetrViTLayer", "LwDetrViTEncoder"],
        "attentions": LwDetrViTSelfAttention,
    }

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, LwDetrViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
        elif isinstance(module, LwDetrViTLayer):
            if module.use_cae:
                nn.init.constant_(module.gamma_1, self.config.cae_init_values)
                nn.init.constant_(module.gamma_2, self.config.cae_init_values)


class LwDetrViTBackbone(VitDetBackbone):
    @auto_docstring
    @can_return_tuple
    @check_model_inputs
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import LwDetrViTConfig, LwDetrViTBackbone
        >>> import torch

        >>> config = LwDetrViTConfig()
        >>> model = LwDetrViTBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""
        embedding_output = self.embeddings(pixel_values, **kwargs)

        batch_size, channels, height, width = embedding_output.shape
        # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
        hidden_states = embedding_output.permute(0, 2, 3, 1)

        window_height = height // self.config.num_windows_side
        window_width = width // self.config.num_windows_side
        # (batch_size, height, width, channels) -> (batch_size*16, window_height*window_width, channels)
        hidden_states = (
            hidden_states.reshape(
                batch_size,
                self.config.num_windows_side,
                window_height,
                self.config.num_windows_side,
                window_width,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size * 16, window_height * window_width, channels)
        )

        encoder_outputs = self.encoder(hidden_states, **kwargs)
        hidden_states = encoder_outputs.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = (
                    hidden_state.reshape(
                        batch_size,
                        self.config.num_windows_side,
                        self.config.num_windows_side,
                        window_height,
                        window_width,
                        channels,
                    )
                    .permute(0, 5, 1, 3, 2, 4)
                    .reshape(batch_size, channels, height, width)
                )
                feature_maps += (hidden_state,)

        return BackboneOutput(feature_maps=feature_maps)


__all__ = ["LwDetrViTConfig", "LwDetrViTPreTrainedModel", "LwDetrViTBackbone"]
