from numbers import Number
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ..auto import CONFIG_MAPPING
from ..deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrEncoder,
    DeformableDetrForObjectDetection,
    DeformableDetrModel,
    DeformableDetrPreTrainedModel,
)
from ..dinov2_with_registers.configuration_dinov2_with_registers import Dinov2WithRegistersConfig
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersBackbone,
    Dinov2WithRegistersEmbeddings,
    Dinov2WithRegistersEncoder,
    Dinov2WithRegistersLayer,
)
from ..vitdet.modeling_vitdet import VitDetLayerNorm


class RFDetrConfig(PretrainedConfig):
    model_type = "rf_detr"
    sub_configs = {"backbone_config": Dinov2WithRegistersConfig}

    def __init__(
        self,
        backbone_config=None,
        num_windows: int = 4,
        window_block_indexes=None,
        out_feature_indexes: List[int] = [2, 5, 8, 11],
        scale_factors: List[Number[2.0, 1.0, 0.5, 0.25]] = [1.0],
        layer_norm: bool = False,
        rms_norm: bool = False,
        **kwargs,
    ):
        self.out_feature_indexes = out_feature_indexes

        if isinstance(backbone_config, dict):
            backbone_config["out_indices"] = out_feature_indexes
            backbone_config["model_type"] = (
                backbone_config["model_type"] if "model_type" in backbone_config else "dinov2_with_registers"
            )
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)
        elif backbone_config is None:
            backbone_config = CONFIG_MAPPING["dinov2_with_registers"](out_indices=out_feature_indexes)
        self.backbone_config = backbone_config
        self.backbone_config.num_windows = num_windows
        self.backbone_config.window_block_indexes = (
            list(range(backbone_config.num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        )

        self.scale_factors = [1.0] if scale_factors is None else scale_factors
        assert self.scale_factors > 0, "scale_factors must be a list of at least one element"
        assert sorted(self.scale_factors) == self.scale_factors, "scale_factors must be sorted"
        assert all(scale in [2.0, 1.0, 0.5, 0.25] for scale in self.scale_factors), (
            "scale_factors must be a consecutive list subset of [2.0, 1.0, 0.5, 0.25]"
        )

        self.layer_norm = layer_norm
        self.rms_norm = rms_norm
        super().__init__(**kwargs)


class RFDetrEmbeddings(Dinov2WithRegistersEmbeddings):
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


class RFDetrLayer(Dinov2WithRegistersLayer):
    def __init__(self, config):
        super(Dinov2WithRegistersLayer).__init__(config)

        self.num_windows = config.num_windows

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        run_full_attention: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        assert head_mask is None, "head_mask is not supported for windowed attention"
        assert not output_attentions, "output_attentions is not supported for windowed attention"
        shortcut = hidden_states
        if run_full_attention:
            # reshape x to remove windows
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows**2
            hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            # reshape x to add windows back
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows**2
            # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)

        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = self.drop_path(attention_output) + shortcut

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class RFDetrEncoder(Dinov2WithRegistersEncoder):
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

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > int(self.config.out_features[-1][5:]):  # TODO check this
                # early stop if we have reached the last output feature
                break

            run_full_attention = i not in self.config.window_block_indexes

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    run_full_attention,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class RFDetrBackbone(Dinov2WithRegistersBackbone):
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

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

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


class RFDetrLayerNorm(VitDetLayerNorm):
    pass


class ConvX(nn.Module):
    """Conv-bn module"""

    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act="relu"):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = ACT2FN[act]

    def forward(self, x):
        """forward"""
        out = self.act(self.bn(self.conv(x)))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act="silu"):
        """ch_in, ch_out, shortcut, groups, kernels, expand"""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RFDetrC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act="silu"):
        """ch_in, ch_out, number, shortcut, groups, expansion"""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act) for _ in range(n))

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RFDetrMultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        config: RFDetrConfig,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super().__init__()

        self.scale_factors = config.scale_factors
        in_channels = [config.backbone_config.hidden_size] * len(config.out_feature_indexes)

        stages_sampling = []
        stages = []

        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                            RFDetrLayerNorm(in_dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        ]
                    )
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimention of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        ]
                    )
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(in_dim, in_dim, 3, 2, layer_norm=config.layer_norm),
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            layers = [
                RFDetrC2f(in_dim, out_channels, num_blocks, layer_norm=config.layer_norm),
                RFDetrLayerNorm(out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(feat_fuse))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class RFDetrDecoderLayer(DeformableDetrDecoderLayer):
    pass


class RFDetrDecoder(DeformableDetrDecoder):
    pass


class RFDetrPreTrainedModel(DeformableDetrPreTrainedModel):
    pass


class RFDetrDecoder(DeformableDetrDecoder):
    pass


class RFDetrEncoder(DeformableDetrEncoder):
    pass


class RFDetrModel(DeformableDetrModel):
    pass


class RFDetrForObjectDetection(DeformableDetrForObjectDetection):
    pass
