from typing import Optional, Union

import torch
from torch import nn

from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersBackbone,
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersEncoder,
    Dinov2WithRegistersLayer,
    Dinov2WithRegistersPatchEmbeddings,
    Dinov2WithRegistersPreTrainedModel,
)


class RfDetrDinov2WithRegistersConfig(Dinov2WithRegistersConfig):
    model_type = "rf_detr_dinov2_with_registers"
    def __init__(self, num_windows: int = 4, **super_kwargs):
        super().__init__(**super_kwargs)

        self.num_windows = num_windows
        window_block_indexes = set(range(self._out_indices[-1] + 1))
        window_block_indexes.difference_update(self._out_indices)
        window_block_indexes = list(window_block_indexes)
        self.window_block_indexes = window_block_indexes


class RfDetrDinov2WithRegistersPatchEmbeddings(Dinov2WithRegistersPatchEmbeddings):
    pass


class RfDetrDinov2WithRegistersEmbeddings(Dinov2WithRegistersEmbeddings):
    def __init__(self, config: RfDetrDinov2WithRegistersConfig):
        super().__init__(config)
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
            if config.num_register_tokens > 0
            else None
        )

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if self.config.num_windows > 1:
            # reshape for windows
            num_h_patches = height // self.config.patch_size
            num_w_patches = width // self.config.patch_size
            cls_token_with_pos_embed = embeddings[:, :1]
            pixel_tokens_with_pos_embed = embeddings[:, 1:]
            pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(
                batch_size, num_h_patches, num_w_patches, -1
            )
            num_w_patches_per_window = num_w_patches // self.config.num_windows
            num_h_patches_per_window = num_h_patches // self.config.num_windows
            num_windows = self.config.num_windows
            windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
                batch_size, num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1
            )
            windowed_pixel_tokens = windowed_pixel_tokens.permute(0, 1, 3, 2, 4, 5)
            windowed_pixel_tokens = windowed_pixel_tokens.reshape(
                batch_size * num_windows**2, num_h_patches_per_window * num_w_patches_per_window, -1
            )
            windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)
            embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)

        # add register tokens
        embeddings = (
            torch.cat(
                (embeddings[:, :1], self.register_tokens.expand(embeddings.shape[0], -1, -1), embeddings[:, 1:]), dim=1
            )
            if self.config.num_register_tokens > 0
            else embeddings
        )

        embeddings = self.dropout(embeddings)

        return embeddings


class RfDetrDinov2WithRegistersLayer(Dinov2WithRegistersLayer):
    def __init__(self, config: RfDetrDinov2WithRegistersConfig):
        super().__init__(config)
        self.num_windows = config.num_windows

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        remove_windows: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        assert head_mask is None, "head_mask is not supported for windowed attention"
        shortcut = hidden_states
        if remove_windows:
            # reshape x to remove windows
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(
            hidden_states_norm,
            head_mask,
        )

        if remove_windows:
            # reshape x to add windows back
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows**2
            # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
            self_attention_output = self_attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)

        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + shortcut

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class RfDetrDinov2WithRegistersEncoder(Dinov2WithRegistersEncoder):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):
            remove_windows = i not in self.config.window_block_indexes
            hidden_states = layer_module(hidden_states, remove_windows=remove_windows)
            if all_hidden_states:
                all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
        )

class RfDetrDinov2WithRegistersPreTrainedModel(Dinov2WithRegistersPreTrainedModel):
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
        elif isinstance(module, RfDetrDinov2WithRegistersEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            module.mask_token.data.zero_()
            if module.config.num_register_tokens > 0:
                module.register_tokens.data.zero_()
        elif isinstance(module, RfDetrDinov2WithRegistersLayerScale):  # noqa: F821
            module.lambda1.data.fill_(self.config.layerscale_value)


class RfDetrDinov2WithRegistersBackbone(Dinov2WithRegistersBackbone):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:
        Returns:

        Examples:


        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-with-registers-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)

        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size

                    if self.config.num_windows > 1:
                        # undo windowing
                        num_windows_squared = self.config.num_windows**2
                        B, HW, C = hidden_state.shape
                        num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows
                        hidden_state = hidden_state.reshape(B // num_windows_squared, num_windows_squared * HW, C)
                        hidden_state = hidden_state.view(
                            B // num_windows_squared,
                            self.config.num_windows,
                            self.config.num_windows,
                            num_h_patches_per_window,
                            num_w_patches_per_window,
                            C,
                        )
                        hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
        )


__all__ = [
    "RfDetrDinov2WithRegistersConfig",
    "RfDetrDinov2WithRegistersBackbone",
]
