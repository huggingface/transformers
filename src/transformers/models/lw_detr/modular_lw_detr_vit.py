import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from ..vitdet.configuration_vitdet import VitDetConfig
from ..vitdet.modeling_vitdet import (
    VitDetAttention,
    VitDetBackbone,
    VitDetDropPath,
    VitDetEncoder,
    VitDetLayer,
    VitDetMlp,
    VitDetPreTrainedModel,
)


logger = logging.get_logger(__name__)


class LwDetrVitConfig(VitDetConfig):
    model_type = "lw_detr_vit"

    def __init__(
        self,
        use_cae: bool = True,
        cae_init_values: float = 0.1,
        num_windows = 16,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        del self.residual_block_indices

        self.use_cae = use_cae
        self.cae_init_values = cae_init_values
        if num_windows % math.sqrt(num_windows) != 0:
            raise ValueError(f"`num_windows` has to be a perfect square, but got {num_windows}.")
        self.num_windows = num_windows


class LwDetrVitAttention(VitDetAttention):
    def __init__(self, config: LwDetrVitConfig, input_size=None):
        """
        Args:
            config (`LwDetrVitConfig`):
                Model configuration.
            input_size (`tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__(config, input_size)
        del self.qkv

        self.proj = nn.Linear(dim, dim)

        self.use_cae = config.use_cae
        if self.use_cae:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)

    def forward(self, hidden_state, output_attentions=False, mask=None):
        batch_size, N, _ = hidden_state.shape  # N = H * W

        if self.use_cae:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(hidden_state, self.qkv.weight, qkv_bias)
        else:
            qkv = self.qkv(hidden_state)

        qkv = qkv.reshape(batch_size, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)

        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)
        if mask is not None:
            attention_scores.masked_fill_(
                mask.reshape(batch_size, 1, 1, N).expand_as(attention_scores), torch.finfo(attention_scores.dtype).min
            )
        attention_probs = attention_scores.softmax(dim=-1)

        hidden_state = attention_probs @ values
        hidden_state = hidden_state.permute(0, 2, 1, 3)
        hidden_state = hidden_state.reshape(batch_size, N, -1)
        hidden_state = self.proj(hidden_state)

        if output_attentions:
            attention_probs = attention_probs.reshape(
                batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1]
            )
            outputs = (hidden_state, attention_probs)
        else:
            outputs = (hidden_state,)

        return outputs


class LwDetrVitDropPath(VitDetDropPath):
    pass


class LwDetrVitMlp(VitDetMlp):
    pass


class LwDetrVitLayer(VitDetLayer):
    def __init__(
        self,
        config: LwDetrVitConfig,
        drop_path_rate: float = 0,
        window: bool = False,
    ) -> None:
        VitDetLayer.__init__()

        dim = config.hidden_size

        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attention = LwDetrVitAttention(config)

        self.drop_path = LwDetrVitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = LwDetrVitMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))

        self.window = window
        self.num_windows = config.num_windows
        self.use_cae = config.use_cae

        if self.use_cae:
            init_values = config.cae_init_values
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        del self.use_residual_block
        del self.residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        batch_size, seq_len, channels = hidden_states.shape

        shortcut = hidden_states

        hidden_states = self.norm1(hidden_states)

        if not self.window:
            hidden_states = hidden_states.reshape(batch_size // self.num_windows, self.num_windows * seq_len, channels)
            if head_mask is not None:
                head_mask = head_mask.reshape(batch_size // self.num_windows, self.num_windows * seq_len)

        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.use_cae:
            hidden_states = hidden_states * self.gamma_1

        if not self.window:
            hidden_states = hidden_states.reshape(batch_size, seq_len, channels)
            if head_mask is not None:
                head_mask = head_mask.reshape(batch_size, seq_len)

        # first residual connection
        hidden_states = shortcut + self.drop_path(hidden_states)

        shortcut = hidden_states

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.use_cae:
            hidden_states = hidden_states * self.gamma_2

        hidden_states = shortcut + self.drop_path(hidden_states)

        outputs = (hidden_states,) + outputs

        return outputs


class LwDetrVitEncoder(VitDetEncoder):
    def __init__(self, config: LwDetrVitConfig) -> None:
        VitDetEncoder().__init__()
        self.config = config
        depth = config.num_hidden_layers

        # stochastic depth decay rule
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth, device="cpu")]

        layers = []
        for i in range(depth):
            layers.append(
                LwDetrVitLayer(
                    config,
                    drop_path_rate=drop_path_rate[i],
                    window=i in config.window_block_indices,
                )
            )

        self.layer = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size, channels, height, width = hidden_states.shape
        # (batch_size, channels, height, width) -> (batch_size, height, width, channels)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        assert (height % 4 == 0) and (width % 4 == 0)  # TODO: remove this
        num_windows_side = int(math.sqrt(self.config.num_windows))
        window_height = height // num_windows_side
        window_width = width // num_windows_side

        # (batch_size, height, width, channels) -> (batch_size*16, window_height*window_width, channels)
        windowed_hidden_states = (
            hidden_states.reshape(batch_size, 4, window_height, 4, window_width, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch_size * 16, window_height * window_width, channels)
        )

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(windowed_hidden_states, layer_head_mask, output_attentions)

            windowed_hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                hidden_states = (
                    windowed_hidden_states.reshape(
                        batch_size, num_windows_side, num_windows_side, window_height, window_width, channels
                    )
                    .permute(0, 5, 1, 3, 2, 4)
                    .reshape(batch_size, channels, height, width)
                )
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=windowed_hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class LwDetrVitPreTrainedModel(VitDetPreTrainedModel):
    pass


class LwDetrVitBackbone(VitDetBackbone):
    pass


__all__ = ["LwDetrVitConfig", "LwDetrVitPreTrainedModel", "LwDetrVitBackbone"]
