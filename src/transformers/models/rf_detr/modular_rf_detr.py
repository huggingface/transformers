# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import Tensor, nn
from torch.nn import functional as F

from ...activations import ACT2FN
from ...backbone_utils import BackboneConfigMixin, consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging, torch_int
from ...utils.generic import ModelOutput, TransformersKwargs, can_return_tuple
from ..clip.modeling_clip import CLIPMLP
from ..convnext.modeling_convnext import ConvNextLayer
from ..dinov2.configuration_dinov2 import Dinov2Config
from ..dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Embeddings,
    Dinov2Encoder,
    Dinov2Layer,
    Dinov2PreTrainedModel,
)
from ..lw_detr.configuration_lw_detr import LwDetrConfig
from ..lw_detr.modeling_lw_detr import (
    LwDetrC2FLayer,
    LwDetrConvEncoder,
    LwDetrConvNormLayer,
    LwDetrForObjectDetection,
    LwDetrLayerNorm,
    LwDetrModel,
    LwDetrObjectDetectionOutput,
    LwDetrPreTrainedModel,
    refine_bboxes,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="stevenbucaille/rf-detr-base")
@strict
class RfDetrDinov2Config(Dinov2Config):
    r"""
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    drop_path_rate (`float`, *optional*, defaults to 0.0):
        Stochastic depth rate per sample (when applied in the main path of residual layers).
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps in case the model is used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the feature maps to 4D tensors of shape `(batch_size, d_model, height, width)` in
        case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
        seq_len, d_model)`.
    use_mask_token (`bool`, *optional*, defaults to `True`):
        Whether to use mask_token in embeddings.
    num_windows (`int`, *optional*, defaults to 4):
        Number of windows to use for windowed attention. If 1, no windowed attention is used.

    Example:

    ```python
    >>> from transformers import RfDetrDinov2Config, RfDetrDinov2Backbone

    >>> # Initializing a RfDetrDinov2 base style configuration
    >>> configuration = RfDetrDinov2Config()

    >>> # Initializing a model (with random weights) from the base style configuration
    >>> model = RfDetrDinov2Backbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr_dinov2"

    num_windows: int = 4

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        window_block_indexes = set(range(self._out_indices[-1] + 1))
        window_block_indexes.difference_update(self._out_indices)
        self.window_block_indexes = list(window_block_indexes)
        BackboneConfigMixin.__post_init__(**kwargs)


@auto_docstring(checkpoint="stevenbucaille/rf-detr-base")
@strict
class RfDetrConfig(LwDetrConfig):
    r"""
    hidden_expansion (`float`, *optional*, defaults to 0.5):
        Expansion factor for hidden dimensions in the projector layers.
    c2f_num_blocks (`int`, *optional*, defaults to 3):
        Number of blocks in the C2F layer.
    activation_function (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function in the projector. Supported values are `"silu"`, `"relu"`, `"gelu"`.
    decoder_n_points (`int`, *optional*, defaults to 4):
        The number of sampled keys in each feature level for each attention head in the decoder.
    decoder_layers (`int`, *optional*, defaults to 3):
        Number of decoder layers in the transformer.
    decoder_self_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the decoder self-attention.
    decoder_cross_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the decoder cross-attention.
    decoder_activation_function (`str`, *optional*, defaults to `"relu"`):
        The non-linear activation function in the decoder. Supported values are `"relu"`, `"silu"`, `"gelu"`.
    num_queries (`int`, *optional*, defaults to 300):
        Number of object queries, i.e. detection slots. This is the maximal number of objects
        [`RfDetrModel`] can detect in a single image.
    group_detr (`int`, *optional*, defaults to 13):
        Number of groups for Group DETR attention mechanism, which helps reduce computational complexity.
    disable_custom_kernels (`bool`, *optional*, defaults to `True`):
        Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
        kernels are not supported by PyTorch ONNX export.
    class_loss_coefficient (`float`, *optional*, defaults to 1):
        Relative weight of the classification loss in the Hungarian matching cost.
    dice_loss_coefficient (`float`, *optional*, defaults to 1):
        Relative weight of the DICE/F-1 loss in the object detection loss.
    bbox_loss_coefficient (`float`, *optional*, defaults to 5):
        Relative weight of the L1 bounding box loss in the object detection loss.
    giou_loss_coefficient (`float`, *optional*, defaults to 2):
        Relative weight of the generalized IoU loss in the object detection loss.
    num_feature_levels (`int`, *optional*, defaults to 1):
        Number of feature levels used in the multiscale deformable attention.
    mask_loss_coefficient (`float`, *optional*, defaults to 1):
        Relative weight of the Focal loss in the instance segmentation mask loss.
    mask_point_sample_ratio (`int`, *optional*, defaults to 16):
        The ratio of points to sample for the mask loss calculation.
    mask_downsample_ratio (`int`, *optional*, defaults to 4):
        The downsample ratio for the segmentation masks compared to the input image resolution.
    mask_class_loss_coefficient (`float`, *optional*, defaults to 5.0):
        Relative weight of the Focal loss in the instance segmentation loss.
    mask_dice_loss_coefficient (`float`, *optional*, defaults to 5.0):
        Relative weight of the DICE/F-1 loss in the instance segmentation loss.
    segmentation_head_activation_function (`str`, *optional*, defaults to `"gelu"`):
        The non-linear activation function in the segmentation head. Supported values are `"relu"`, `"silu"`, `"gelu"`.

    Examples:

    ```python
    >>> from transformers import RfDetrConfig, RfDetrModel

    >>> # Initializing a LW-DETR stevenbucaille/RfDetr_small_60e_coco style configuration
    >>> configuration = RfDetrConfig()

    >>> # Initializing a model (with random weights) from the stevenbucaille/RfDetr_small_60e_coco style configuration
    >>> model = RfDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"

    layer_norm_eps: float = 1e-5
    dropout: float = 0.1
    num_feature_levels: int = 1
    mask_loss_coefficient: int | float = 1
    mask_point_sample_ratio: int = 16
    mask_downsample_ratio: int = 4
    mask_class_loss_coefficient: int | float = 5.0
    mask_dice_loss_coefficient: int | float = 5.0
    segmentation_head_activation_function: str = "gelu"
    intermediate_size: int = 1024

    batch_norm_eps = AttributeError("batch_norm_eps is replaced by layer_norm_eps in RfDetrConfig.")
    projector_scale_factors = AttributeError("RfDetr only uses a single scale layer instead of multiple scale layers.")

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="rf_detr_dinov2",
            default_config_kwargs={
                "num_attention_heads": 6,
                "out_features": ["stage2", "stage5", "stage8", "stage11"],
                "hidden_size": 384,
                "num_register_tokens": 0,
                "image_size": 518,
            },
            **kwargs,
        )
        PreTrainedConfig.__post_init__(**kwargs)

    def validate_architecture(self):
        raise AttributeError(
            "validate_architecture is not used in RfDetrConfig because it does not rely on multiple scale layers."
        )


class RfDetrDinov2Embeddings(Dinov2Embeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and interpolation at torch.float32 precision.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            # Difference from Dinov2, we use align_corners=False and antialias=True
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def window_partition(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Splits each image's patch-token grid into num_windows^2 local windows,
        replicates the [CLS] token per window, and returns window-local token sequences
        """
        batch_size = embeddings.shape[0]
        num_windows = self.config.num_windows
        patch_size = self.patch_size
        num_height_patches = height // patch_size
        num_width_patches = width // patch_size
        num_width_patches_per_window = num_width_patches // num_windows
        num_height_patches_per_window = num_height_patches // num_windows

        # Split the embeddings into the [CLS] token and the pixel tokens
        cls_token_with_pos_embed = embeddings[:, :1]
        pixel_tokens_with_pos_embed = embeddings[:, 1:]
        pixel_tokens_with_pos_embed = pixel_tokens_with_pos_embed.view(
            batch_size, num_height_patches, num_width_patches, -1
        )

        # Reshape the pixel tokens into windowed pixel tokens
        windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(
            batch_size, num_windows, num_width_patches_per_window, num_windows, num_height_patches_per_window, -1
        )
        windowed_pixel_tokens = windowed_pixel_tokens.transpose(2, 3)
        windowed_pixel_tokens = windowed_pixel_tokens.reshape(
            batch_size * num_windows**2, num_height_patches_per_window * num_width_patches_per_window, -1
        )

        # Repeat the [CLS] token per window
        windowed_cls_token_with_pos_embed = cls_token_with_pos_embed.repeat(num_windows**2, 1, 1)

        # Concatenate the [CLS] token with the windowed pixel tokens to get the final embeddings
        embeddings = torch.cat((windowed_cls_token_with_pos_embed, windowed_pixel_tokens), dim=1)
        return embeddings

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None and self.use_mask_token:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        # Difference from Dinov2, we use window partitioning
        if self.config.num_windows > 1:
            embeddings = self.window_partition(embeddings, height, width)
        embeddings = self.dropout(embeddings)

        return embeddings


class RfDetrDinov2Layer(Dinov2Layer):
    def __init__(self, config: RfDetrDinov2Config, layer_idx: int):
        super().__init__(config)
        self.num_windows = config.num_windows
        self.global_attention = layer_idx not in config.window_block_indexes

    def window_unpartition_before_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        For layers configured to use global attention, merges the window-batched sequences back
        into one sequence per image so attention can be computed across all windows jointly.
        """
        batch_size, seq_len, channels = hidden_states.shape
        num_windows_squared = self.num_windows**2
        hidden_states = hidden_states.view(batch_size // num_windows_squared, num_windows_squared * seq_len, channels)
        return hidden_states

    def window_partition_after_attention(
        self, hidden_state_shape: tuple[int, int, int], self_attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        After global attention, reshapes the output sequence back into window-batched
        form so the model can continue in the same windowed pipeline.
        """

        batch_size, seq_len, channels = hidden_state_shape
        num_windows_squared = self.num_windows**2
        self_attention_output = self_attention_output.view(
            batch_size * num_windows_squared, seq_len // num_windows_squared, channels
        )
        return self_attention_output

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        # Difference from Dinov2, when the layer is not a window block, we need to unpartition the hidden states before the attention
        if self.global_attention:
            hidden_states = self.window_unpartition_before_attention(hidden_states)

        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)

        # And reverse the operation after the attention
        if self.global_attention:
            self_attention_output = self.window_partition_after_attention(hidden_states.shape, self_attention_output)

        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + residual
        residual = hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)

        # second residual connection
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class RfDetrDinov2PreTrainedModel(Dinov2PreTrainedModel):
    pass


class RfDetrDinov2Encoder(Dinov2Encoder):
    def __init__(self, config: RfDetrDinov2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([RfDetrDinov2Layer(config, i) for i in range(config.num_hidden_layers)])


class RfDetrDinov2Backbone(Dinov2Backbone):
    def window_unpartition(self, hidden_state: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Reassembles windowed patch tokens into their original 2D patch layout (image-level grid structure)
        before converting backbone hidden states into spatial feature maps.
        """
        num_windows = self.config.num_windows
        patch_size = self.config.patch_size
        num_h_patches = height // patch_size
        num_w_patches = width // patch_size
        hidden_batch_size, seq_len, channels = hidden_state.shape
        num_windows_squared = num_windows**2
        num_h_patches_per_window = num_h_patches // num_windows
        num_w_patches_per_window = num_w_patches // num_windows

        # Reshape the hidden states into the original sequence length
        hidden_state = hidden_state.reshape(
            hidden_batch_size // num_windows_squared, num_windows_squared * seq_len, channels
        )
        hidden_state = hidden_state.view(
            hidden_batch_size // num_windows_squared,
            num_windows,
            num_windows,
            num_h_patches_per_window,
            num_w_patches_per_window,
            channels,
        )
        hidden_state = hidden_state.transpose(2, 3)
        return hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""
        # Like Dinov2, we need to output the hidden states to extract the layers for the stages
        kwargs["output_hidden_states"] = True

        embedding_output = self.embeddings(pixel_values)
        output: BaseModelOutput = self.encoder(embedding_output, **kwargs)
        hidden_states = output.hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    num_h_patches = height // self.config.patch_size
                    num_w_patches = width // self.config.patch_size

                    # Difference from Dinov2, when the layer is not a window block, we need to unpartition the hidden states before reshaping
                    if self.config.num_windows > 1:
                        hidden_state = self.window_unpartition(hidden_state, height, width)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_maps += (hidden_state,)

        return BackboneOutput(
            feature_maps=tuple(feature_maps),
            hidden_states=hidden_states,
            attentions=output.attentions,
        )


class RfDetrLayerNorm(LwDetrLayerNorm):
    pass


class RfDetrConvNormLayer(LwDetrConvNormLayer):
    def __init__(
        self,
        config: RfDetrConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str | None = None,
    ):
        super().__init__(
            config,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation,
        )
        self.norm = RfDetrLayerNorm(out_channels, data_format="channels_first", eps=config.layer_norm_eps)


class RfDetrC2FLayer(LwDetrC2FLayer):
    pass


class RfDetrScaleProjector(nn.Module):
    def __init__(self, config: RfDetrConfig):
        super().__init__()
        projector_input_dim: int = config.backbone_config.hidden_size * len(config.backbone_config.out_indices)
        self.projector_layer = RfDetrC2FLayer(config, projector_input_dim)
        self.layer_norm = RfDetrLayerNorm(config.d_model, data_format="channels_first")

    def forward(self, hidden_states: tuple[torch.Tensor]) -> torch.Tensor:
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = self.projector_layer(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class RfDetrConvEncoder(LwDetrConvEncoder):
    def __init__(self, config: RfDetrConfig):
        super().__init__(config)
        self.backbone = RfDetrDinov2Backbone(config.backbone_config)
        self.projector = RfDetrScaleProjector(config)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.backbone(pixel_values).feature_maps
        features = self.projector(features)
        mask = nn.functional.interpolate(pixel_mask[None].float(), size=features.shape[-2:]).to(torch.bool)[0]
        return features, mask


class RfDetrPreTrainedModel(LwDetrPreTrainedModel):
    # Roboflow checkpoints use bare keys with no top-level prefix
    _checkpoint_conversion_prefix_free = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if hasattr(module, "segmentation_bias") and isinstance(module.segmentation_bias, nn.Parameter):
            nn.init.constant_(module.segmentation_bias, 0.0)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for outputs of the RfDetr backbone-decoder model.
    """
)
class RfDetrModelOutput(ModelOutput):
    r"""
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, d_model)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    backbone_features (list of `torch.FloatTensor` of shape `(batch_size, config.num_channels, config.image_size, config.image_size)`):
        Features from the backbone.
    """

    last_hidden_state: torch.FloatTensor | None = None
    init_reference_points: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_class: torch.FloatTensor | None = None
    enc_outputs_coord_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    backbone_features: list[torch.Tensor] = None


class RfDetrModel(LwDetrModel):
    def __init__(self, config: RfDetrConfig):
        super().__init__(config)
        self.d_model = config.d_model

    def generate_topk_proposals(
        self, group_id: int, object_query_embedding: Tensor, output_proposals: Tensor, invalid_mask: Tensor, topk: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generates and selects the top-k object query embeddings and bounding box proposals for a specific query group.

        The pipeline proceeds as follows:

        1. Project and normalize the base query embeddings for the specific query group.
        2. Predict classification scores and bounding box refinements for the current query features.
        3. Apply the predicted deltas to the initial proposals to obtain refined spatial coordinates.
        4. Identify the indices of the highest-confidence predictions and gather the refined coordinates for these
           top-k candidates (detached to prevent gradient flow back to the proposal generation stage).
        5. Gather the associated query features to be used as starting points for the decoder stage.
        """
        # Step 1.
        object_query = self.enc_output[group_id](object_query_embedding)
        object_query = self.enc_output_norm[group_id](object_query)

        # Step 2.
        enc_outputs_class_proposals = self.enc_out_class_embed[group_id](object_query)
        enc_outputs_class_proposals = enc_outputs_class_proposals.masked_fill(invalid_mask, float("-inf"))
        delta_bbox = self.enc_out_bbox_embed[group_id](object_query)

        # Step 3.
        enc_outputs_coord = refine_bboxes(output_proposals, delta_bbox)

        # Step 4.
        topk_proposals = torch.topk(enc_outputs_class_proposals.max(-1)[0], topk, dim=1)[1]
        topk_coords_logits_undetach = torch.gather(
            enc_outputs_coord,
            1,
            topk_proposals.unsqueeze(-1).expand(-1, -1, 4),
        )
        topk_coords_logits = topk_coords_logits_undetach.detach()

        # Step 5.
        object_query_undetach = torch.gather(
            object_query, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        )
        return object_query_undetach, topk_coords_logits, topk_coords_logits_undetach

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
    Forward pass of the RF-DETR model. The pipeline proceeds as follows:

        1. Generate an initial set of object query embeddings and spatial location proposals from the
           backbone's flattened output.
        2. Initialize storage for refined encoder-stage predictions (accommodating multi-group query
           structures) and iteratively refine object queries and their coordinates for each query group
           to capture the highest-confidence candidates from the encoder stage.
        3. Initialize learnable query features and spatial reference points (restricting to the primary
           group during inference for efficiency).
        4. Project the base reference points across the batch, refine them with the predicted coordinate
           refinements (shifting attention to the discovered object locations before decoding), and expand
           the target query features to match the batch dimensions.
        5. Pass the refined queries and updated reference points through the transformer decoder to
            aggregate detailed spatial context from the multi-scale features.
    """
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RfDetrModelOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RfDetrModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/rfdetr_small_60e_coco")
        >>> model = RfDetrModel.from_pretrained("stevenbucaille/rfdetr_small_60e_coco")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 200, 256]
        ```"""
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), dtype=torch.long, device=device)

        features, mask = self.backbone(pixel_values, pixel_mask)

        source_flatten = features.flatten(2).transpose(1, 2)
        mask_flatten = mask.flatten(1)
        spatial_shapes_list = [features.shape[2:]]
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = self.get_valid_ratio(mask, dtype=source_flatten.dtype).unsqueeze(1)

        # Step 1.
        object_query_embedding, output_proposals, invalid_mask = self.gen_encoder_output_proposals(
            source_flatten, ~mask_flatten, spatial_shapes_list
        )

        # Step 2.
        group_detr = self.group_detr if self.training else 1
        topk = self.num_queries
        topk_coords_logits = torch.empty(
            (batch_size, topk * group_detr, 4), device=self.device, dtype=output_proposals.dtype
        )
        enc_outputs_coord_logits = torch.empty(
            (batch_size, topk * group_detr, 4), device=self.device, dtype=output_proposals.dtype
        )
        enc_outputs_class = torch.empty(
            (batch_size, topk * group_detr, self.config.d_model), device=self.device, dtype=output_proposals.dtype
        )
        for group_id in range(group_detr):
            object_query_undetach, group_topk_coords_logits, topk_coords_logits_undetach = (
                self.generate_topk_proposals(group_id, object_query_embedding, output_proposals, invalid_mask, topk)
            )
            topk_coords_logits[:, group_id * topk : (group_id + 1) * topk] = group_topk_coords_logits
            enc_outputs_coord_logits[:, group_id * topk : (group_id + 1) * topk] = topk_coords_logits_undetach
            enc_outputs_class[:, group_id * topk : (group_id + 1) * topk] = object_query_undetach

        # Step 3.
        if self.training:
            reference_points = self.reference_point_embed.weight
            query_feat = self.query_feat.weight
        else:
            reference_points = self.reference_point_embed.weight[: self.num_queries]
            query_feat = self.query_feat.weight[: self.num_queries]

        # Step 4.
        reference_points = reference_points.unsqueeze(0).expand(batch_size, -1, -1)
        two_stage_len = enc_outputs_coord_logits.shape[-2]
        reference_points_two_stage_subset = reference_points[..., :two_stage_len, :]
        reference_points_subset = reference_points[..., two_stage_len:, :]
        reference_points_two_stage_subset = refine_bboxes(topk_coords_logits, reference_points_two_stage_subset)
        reference_points = torch.cat([reference_points_two_stage_subset, reference_points_subset], dim=-2)
        init_reference_points = reference_points
        target = query_feat.unsqueeze(0).expand(batch_size, -1, -1)

        # Step 5.
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=mask_flatten,
            **kwargs,
        )

        return RfDetrModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            backbone_features=features,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class RfDetrObjectDetectionOutput(LwDetrObjectDetectionOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~DeformableDetrProcessor.post_process_object_detection`] to retrieve the
        unnormalized bounding boxes.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, d_model)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_class (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Predicted bounding boxes scores where the top `config.two_stage_num_proposals` scoring bounding boxes are
        picked as region proposals in the first stage. Output of bounding box binary classification (i.e.
        foreground and background).
    enc_outputs_coord_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 4)`, *optional*, returned when `config.with_box_refine=True` and `config.two_stage=True`):
        Logits of predicted bounding boxes coordinates in the first stage.
    backbone_features (list of `torch.FloatTensor` of shape `(batch_size, config.num_channels, config.image_size, config.image_size)`):
        Features from the backbone.
    """

    backbone_features: list[torch.Tensor] = None


class RfDetrForObjectDetection(LwDetrForObjectDetection):
    def predict_encoder_class_logits(self, enc_outputs_class: torch.Tensor) -> Tensor:
        """
        Predicts classification logits from encoder hidden states for each query group.
        """
        enc_outputs_class_list = enc_outputs_class.split(self.config.num_queries, dim=1)
        group_detr = self.config.group_detr if self.training else 1
        pred_class = [
            self.model.enc_out_class_embed[group_index](enc_outputs_class_list[group_index])
            for group_index in range(group_detr)
        ]
        return torch.cat(pred_class, dim=1)

    def predict_class_and_boxes(
        self, hidden_states: torch.Tensor, reference_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts classification logits and refined bounding boxes from transformer hidden states and reference points.
        """
        logits = self.class_embed(hidden_states)
        boxes_delta = self.bbox_embed(hidden_states)
        boxes = refine_bboxes(reference_points, boxes_delta)
        return logits, boxes

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
    The forward pass proceeds as follows:

        1. Process the visual input through the base RF-DETR model to obtain the transformer's last hidden state and
           the final sequence of reference points.
        2. First stage: Generate classification logits from the encoder's proposed object query embeddings.
        3. Second stage: Predict the final classification labels and refined bounding boxes using the decoder's last hidden state
           and the most recent reference points.

    """
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RfDetrObjectDetectionOutput:
        r"""
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, RfDetrForObjectDetection
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("stevenbucaille/rf-detr-base")
        >>> model = RfDetrForObjectDetection.from_pretrained("stevenbucaille/rf-detr-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[
        ...     0
        ... ]
        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected cat with confidence 0.8 at location [16.5, 52.84, 318.25, 470.78]
        Detected cat with confidence 0.789 at location [342.19, 24.3, 640.02, 372.25]
        Detected remote with confidence 0.633 at location [40.79, 72.78, 176.76, 117.25]
        ```"""
        # Step 1.
        outputs = self.model(pixel_values, pixel_mask=pixel_mask, **kwargs)

        last_hidden_states = outputs.last_hidden_state
        intermediate_reference_points = outputs.intermediate_reference_points

        # Step 2.
        enc_outputs_class_logits = self.predict_encoder_class_logits(outputs.enc_outputs_class)

        # Step 3.
        logits, pred_boxes = self.predict_class_and_boxes(last_hidden_states, intermediate_reference_points[-1])

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                outputs_class, outputs_coord = self.predict_class_and_boxes(
                    outputs.intermediate_hidden_states, intermediate_reference_points
                )

            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                self.config,
                outputs_class,
                outputs_coord,
                enc_outputs_class_logits,
                outputs.enc_outputs_coord_logits,
            )

        return RfDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_class=enc_outputs_class_logits,
            enc_outputs_coord_logits=outputs.enc_outputs_coord_logits,
            backbone_features=outputs.backbone_features,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`RfDetrForInstanceSegmentation`].
    """
)
class RfDetrInstanceSegmentationOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
        Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
        bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
        scale-invariant IoU loss.
    loss_dict (`Dict`, *optional*):
        A dictionary containing the individual losses. Useful for logging.
    logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        Classification logits (including no-object) for all queries.
    pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
        Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
        values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
        possible padding). You can use [`~DeformableDetrProcessor.post_process_object_detection`] to retrieve the
        unnormalized bounding boxes.
    pred_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height/4, width/4)`):
        Segmentation masks logits for all queries. See also
        [`~DetrImageProcessor.post_process_semantic_segmentation`] or
        [`~DetrImageProcessor.post_process_instance_segmentation`]
        [`~DetrImageProcessor.post_process_panoptic_segmentation`] to evaluate semantic, instance and panoptic
        segmentation masks respectively.
    auxiliary_outputs (`list[Dict]`, *optional*):
        Optional, only returned when auxiliary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
        and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
        `pred_boxes`) for each decoder layer.
    init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
        Initial reference points sent through the Transformer decoder.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, d_model)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the decoder of the model.
    intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, d_model)`):
        Stacked intermediate hidden states (output of each layer of the decoder).
    intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
        Stacked intermediate reference points (reference points of each layer of the decoder).
    enc_outputs_mask_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, width, height)`, *optional*):
        Mask logits from the encoder for all queries.
    """

    loss: torch.FloatTensor | None = None
    loss_dict: dict | None = None
    logits: torch.FloatTensor | None = None
    pred_boxes: torch.FloatTensor | None = None
    pred_masks: torch.FloatTensor = None
    auxiliary_outputs: list[dict] | None = None
    init_reference_points: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    intermediate_hidden_states: torch.FloatTensor | None = None
    intermediate_reference_points: torch.FloatTensor | None = None
    enc_outputs_mask_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


class RfDetrSegmentationBlock(ConvNextLayer):
    def __init__(self, config: RfDetrConfig):
        dim = config.d_model
        super().__init__(config)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.layernorm = RfDetrLayerNorm(dim, eps=1e-6)
        self.pointwise_conv = nn.Linear(dim, dim)
        self.act = ACT2FN[config.segmentation_head_activation_function]
        del self.dwconv
        del self.pwconv1
        del self.pwconv2
        del self.layer_scale_parameter
        del self.drop_path

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.pointwise_conv(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        hidden_states = hidden_states + residual
        return hidden_states


class RfDetrSegmentationMLP(CLIPMLP):
    def __init__(self, config: RfDetrConfig):
        super().__init__(config)
        self.activation_fn = ACT2FN[config.segmentation_head_activation_function]
        self.fc1 = nn.Linear(config.d_model, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.d_model)


class RfDetrSegmentationMLPBlock(nn.Module):
    def __init__(self, config: RfDetrConfig):
        super().__init__()
        dim = config.d_model
        self.norm = nn.LayerNorm(dim)
        self.mlp = RfDetrSegmentationMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class RfDetrForInstanceSegmentation(RfDetrPreTrainedModel):
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # We can't initialize the model on meta device as some weights are modified during the initialization
    _no_split_modules = None

    def __init__(self, config: RfDetrConfig):
        super().__init__(config)

        self.model = RfDetrForObjectDetection(config)

        num_blocks = config.decoder_layers
        self.downsample_ratio = config.mask_downsample_ratio
        self.blocks = nn.ModuleList([RfDetrSegmentationBlock(config) for _ in range(num_blocks)])
        self.spatial_features_proj = nn.Conv2d(config.d_model, config.d_model, kernel_size=1)

        self.query_features_block = RfDetrSegmentationMLPBlock(config)
        self.query_features_proj = nn.Linear(config.d_model, config.d_model)

        self.segmentation_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.post_init()

    def get_mask_logits(self, query_features: Tensor, spatial_features: Tensor) -> Tensor:
        """
        Compute the per-query mask logits.
        Query features are projected to the same dimension as the spatial features and then multiplied
        with the spatial features to get the mask logits.
        The mask logits are then reshaped to the spatial dimensions and broadcast with a segmentation bias
        parameter.

        Args:
            query_features (`torch.Tensor`): Query features of shape (batch_size, num_queries, d_model).
            spatial_features (`torch.Tensor`): Spatial features of shape (batch_size, hidden_dim, height, width).

        Returns:
            `torch.Tensor`: Mask logits of shape (batch_size, num_queries, height, width).
        """
        batch_size, num_queries, _ = query_features.shape
        height, width = spatial_features.shape[2], spatial_features.shape[3]

        query_features = self.query_features_block(query_features)
        query_features = self.query_features_proj(query_features)
        mask_logits = torch.matmul(query_features, spatial_features.flatten(2))
        mask_logits = mask_logits.view(batch_size, num_queries, height, width)
        mask_logits = mask_logits + self.segmentation_bias
        return mask_logits

    def segmentation_head(
        self, spatial_features, list_query_features, image_size: torch.Size, skip_blocks: bool = False
    ) -> list[torch.Tensor] | torch.Tensor:
        """
        Compute mask logits from spatial features and query features.

        Args:
            spatial_features: Multi-scale spatial features of shape
                (batch_size, num_channels, feature_height, feature_width).
            list_query_features: When `skip_blocks` is False, a list of query feature tensors of shape
                (batch_size, num_queries, d_model) for each decoder layer. When `skip_blocks` is True,
                a single tensor of shape (batch_size, num_queries, d_model).
            image_size: Original image spatial dimensions (height, width).
            skip_blocks: If True, skip the convolutional blocks and compute mask logits directly.

        Returns:
            When `skip_blocks` is False: list of mask logit tensors of shape
            (batch_size, num_queries, mask_height, mask_width), where mask size is image size divided
            by `downsample_ratio`. When `skip_blocks` is True: a single such tensor.
        """
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

        if not skip_blocks:
            list_mask_logits = []
            for block, query_features in zip(self.blocks, list_query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                mask_logits = self.get_mask_logits(query_features, spatial_features_proj)
                list_mask_logits.append(mask_logits)
        else:
            list_mask_logits = self.get_mask_logits(list_query_features, spatial_features)

        return list_mask_logits

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
    Forward pass of the RF-DETR model for instance segmentation. The pipeline proceeds as follows:

        1. Process the visual input through the base RF-DETR model to obtain multi-scale spatial features,
           query embeddings, and their transformation history.
        2. Generate classification logits and initial segmentation masks from the encoder's proposed
           object query embeddings (first stage).
        3. Predict the final classification labels and refined bounding boxes using the decoder's last
           hidden state (second stage).
        4. Pass the high-resolution spatial features and query hidden states through the segmentation
           head to produce the final, detailed instance masks.
    """
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        pixel_mask: torch.LongTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> dict[str, torch.Tensor]:
        r"""
        labels (`list[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        """
        image_size = pixel_values.shape[-2:]

        # Step 1.
        outputs = self.model.model(pixel_values, pixel_mask=pixel_mask, **kwargs)

        spatial_features = outputs.backbone_features
        last_hidden_states = outputs.last_hidden_state
        intermediate_reference_points = outputs.intermediate_reference_points
        enc_outputs_class = outputs.enc_outputs_class

        # Step 2.
        enc_outputs_class_logits = self.model.predict_encoder_class_logits(enc_outputs_class)
        enc_outputs_masks = self.segmentation_head(spatial_features, enc_outputs_class, image_size, skip_blocks=True)

        # Step 3.
        logits, pred_boxes = self.model.predict_class_and_boxes(last_hidden_states, intermediate_reference_points[-1])

        # Step 4.
        outputs_masks = self.segmentation_head(spatial_features, outputs.intermediate_hidden_states, image_size)
        pred_masks = outputs_masks[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            outputs_class, outputs_coord = None, None
            if self.config.auxiliary_loss:
                outputs_class, outputs_coord = self.model.predict_class_and_boxes(
                    outputs.intermediate_hidden_states, intermediate_reference_points
                )
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                logits,
                labels,
                self.device,
                pred_boxes,
                pred_masks,
                self.config,
                outputs_class,
                outputs_coord,
                outputs_masks,
                enc_outputs_class_logits,
                outputs.enc_outputs_coord_logits,
                enc_outputs_masks,
            )

        return RfDetrInstanceSegmentationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_masks=pred_masks,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
            enc_outputs_mask_logits=enc_outputs_masks,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


__all__ = [
    "RfDetrConfig",
    "RfDetrModel",
    "RfDetrForObjectDetection",
    "RfDetrForInstanceSegmentation",
    "RfDetrPreTrainedModel",
    "RfDetrDinov2Config",
    "RfDetrDinov2Backbone",
    "RfDetrDinov2PreTrainedModel",
]
