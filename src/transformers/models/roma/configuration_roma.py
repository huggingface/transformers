# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="Parskatt/roma_outdoor")
@strict
class RomaConfig(PreTrainedConfig):
    r"""
    backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*):
        The configuration of the coarse-feature backbone, a DINOv2 vision transformer. If unset, a `Dinov2Config`
        with the ViT-L/14 settings used by RoMa is created.
    cnn_feature_dims (`list[int]`, *optional*, defaults to `[64, 128, 256, 512]`):
        The number of channels of the fine VGG-19 CNN feature pyramid at scales 1, 2, 4 and 8 respectively.
    gp_dim (`int`, *optional*, defaults to 512):
        The dimensionality of the Gaussian-Process (GP) coordinate embeddings produced at the coarse scale.
    coarse_feature_dim (`int`, *optional*, defaults to 512):
        The dimensionality the coarse features are projected to before being concatenated with the GP embeddings
        and fed to the transformer coordinate decoder. Must equal the first entry of `proj_out_dims`.
    proj_out_dims (`list[int]`, *optional*, defaults to `[512, 512, 256, 64, 9]`):
        The output channels of the per-scale projection layers (scales 16, 8, 4, 2, 1).
    anchor_resolution (`int`, *optional*, defaults to 64):
        Square root of the number of anchor coordinates predicted by the classification-based coarse decoder. The
        decoder predicts `anchor_resolution ** 2 + 1` logits per location.
    num_decoder_layers (`int`, *optional*, defaults to 5):
        The number of transformer blocks in the coarse coordinate decoder.
    decoder_num_attention_heads (`int`, *optional*, defaults to 8):
        The number of attention heads in each transformer block of the coarse coordinate decoder.
    kernel_temperature (`float`, *optional*, defaults to 0.2):
        Temperature of the cosine kernel used by the Gaussian-Process module.
    refiner_hidden_dims (`list[int]`, *optional*, defaults to `[1384, 1144, 576, 144, 24]`):
        The number of channels of each of the five coarse-to-fine `ConvRefiner` blocks (scales 16, 8, 4, 2, 1).
        These are the padded channel counts used by `roma_model_pad`, which allow every RoMa checkpoint
        (official, MatchAnything, MINIMA) to load into the same module tree.
    refiner_displacement_emb_dims (`list[int]`, *optional*, defaults to `[128, 64, 32, 16, 6]`):
        The dimensionality of the displacement embedding of each `ConvRefiner` block.
    refiner_local_corr_radius (`list[int]`, *optional*, defaults to `[7, 3, 2, 0, 0]`):
        The local-correlation radius of each `ConvRefiner` block. A value of `0` disables local correlation for
        that block.
    refiner_kernel_size (`int`, *optional*, defaults to 5):
        The kernel size of the convolutions inside each `ConvRefiner` block.
    refiner_hidden_blocks (`int`, *optional*, defaults to 8):
        The number of residual hidden blocks inside each `ConvRefiner` block.
    symmetric (`bool`, *optional*, defaults to `True`):
        Whether to run the decoder symmetrically (A->B and B->A) and return a bidirectional warp.
    upsample_predictions (`bool`, *optional*, defaults to `False`):
        Whether to run the high-resolution refinement pass. When `True`, `RomaForKeypointMatching.forward` also
        expects `pixel_values_upsampled` (the higher-resolution pair produced by the image processor with
        `do_upsample=True`) and returns the dense warp/certainty at that resolution.
    attenuate_certainty (`bool`, *optional*, defaults to `True`):
        Whether to attenuate the fine certainty using the (negative) coarse certainty, as done by RoMa at inference.
    sample_threshold (`float`, *optional*, defaults to 0.05):
        Certainty threshold above which a dense match is considered fully certain when sampling sparse matches.
    num_samples (`int`, *optional*, defaults to 5000):
        The default number of sparse matches sampled from the dense warp.
    batch_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the batch-normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated-normal initializer for the (non-backbone) weight matrices.

    Examples:
        ```python
        >>> from transformers import RomaConfig, RomaForKeypointMatching

        >>> # Initializing a RoMa configuration
        >>> configuration = RomaConfig()

        >>> # Initializing a model from the RoMa configuration
        >>> model = RomaForKeypointMatching(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "roma"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    cnn_feature_dims: list[int] | None = None
    gp_dim: int = 512
    coarse_feature_dim: int = 512
    proj_out_dims: list[int] | None = None
    anchor_resolution: int = 64
    num_decoder_layers: int = 5
    decoder_num_attention_heads: int = 8
    kernel_temperature: float = 0.2
    refiner_hidden_dims: list[int] | None = None
    refiner_displacement_emb_dims: list[int] | None = None
    refiner_local_corr_radius: list[int] | None = None
    refiner_kernel_size: int = 5
    refiner_hidden_blocks: int = 8
    symmetric: bool = True
    upsample_predictions: bool = False
    attenuate_certainty: bool = True
    sample_threshold: float = 0.05
    num_samples: int = 5000
    batch_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.cnn_feature_dims is None:
            self.cnn_feature_dims = [64, 128, 256, 512]
        if self.proj_out_dims is None:
            self.proj_out_dims = [512, 512, 256, 64, 9]
        if self.refiner_hidden_dims is None:
            self.refiner_hidden_dims = [1384, 1144, 576, 144, 24]
        if self.refiner_displacement_emb_dims is None:
            self.refiner_displacement_emb_dims = [128, 64, 32, 16, 6]
        if self.refiner_local_corr_radius is None:
            self.refiner_local_corr_radius = [7, 3, 2, 0, 0]

        default_config_kwargs = {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "mlp_ratio": 4,
            "image_size": 518,
            "patch_size": 14,
            "layerscale_value": 1.0,
            "out_indices": [-1],
            "apply_layernorm": True,
            "reshape_hidden_states": True,
        }
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="dinov2",
            default_config_kwargs=default_config_kwargs,
            **kwargs,
        )

        # The coordinate decoder consumes the concatenation of the GP embeddings and the projected coarse features.
        self.decoder_hidden_size = self.gp_dim + self.coarse_feature_dim
        # The classification decoder predicts a probability over `anchor_resolution ** 2` anchors plus one extra logit.
        self.num_anchors = self.anchor_resolution**2 + 1

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        for name, value in (
            ("refiner_hidden_dims", self.refiner_hidden_dims),
            ("refiner_displacement_emb_dims", self.refiner_displacement_emb_dims),
            ("refiner_local_corr_radius", self.refiner_local_corr_radius),
            ("proj_out_dims", self.proj_out_dims),
        ):
            if len(value) != 5:
                raise ValueError(f"`{name}` must have 5 entries (one per coarse-to-fine scale), got {len(value)}.")
        if self.coarse_feature_dim != self.proj_out_dims[0]:
            raise ValueError(
                f"`coarse_feature_dim` ({self.coarse_feature_dim}) must equal the first entry of `proj_out_dims` "
                f"({self.proj_out_dims[0]}), as the coarse projection feeds the transformer decoder."
            )


__all__ = ["RomaConfig"]
