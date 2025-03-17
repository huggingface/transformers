# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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
import torch
from torch import nn

from ...activations import ACT2FN

from ...utils import logging
from ..mistral.modeling_mistral import MistralRMSNorm
from ..llava.configuration_llava import LlavaConfig
from ..llava.modeling_llava import LlavaForConditionalGeneration, LlavaMultiModalProjector
from ..pixtral.image_processing_pixtral import PixtralImageProcessor
from ..pixtral.image_processing_pixtral_fast import PixtralImageProcessorFast
from ..pixtral.processing_pixtral import PixtralProcessor


logger = logging.get_logger(__name__)


class Mistral3Config(LlavaConfig):
    
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        multimodal_projector_bias=True,
        spatial_merge_size=2,
        **kwargs,
    ):
        super().__init__(vision_config, text_config)
        self.spatial_merge_size = spatial_merge_size


class Mistral3ImageProcessor(PixtralImageProcessor):
    pass


class Mistral3ImageProcessorFast(PixtralImageProcessorFast):
    pass


class Mistral3Processor(PixtralProcessor):
    pass


class Mistral3RMSNorm(MistralRMSNorm):
    pass


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()

        vision_encoder_dim = config.vision_args.hidden_size
        spatial_merge_size = config.spatial_merge_size

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = nn.Linear(mlp_input_dim, vision_encoder_dim, bias=False)

    def forward(self, x: torch.Tensor, image_sizes: list[tuple[int, int]]) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == len(x)

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """

        sub_grids = get_sub_grids(
            x=x,
            image_sizes=image_sizes,
            spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor: list[torch.Tensor] = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted_tensor.append(grid.view(-1, n_patches).t(
            ))  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(
            permuted_tensor, dim=0
        )  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(
            2, 0, 1)[None, :, :, :]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(image_grid,
                                               kernel_size=sub_grid_size,
                                               stride=sub_grid_size)
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size,
            -1)  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids

class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.norm = Mistral3RMSNorm(config.vision_config)
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Mistral3ForConditionalGeneration(LlavaForConditionalGeneration):
    pass


__all__ = [
    "Mistral3PreTrainedModel",  # noqa
    "Mistral3ForConditionalGeneration",
    "Mistral3Config",
    "Mistral3ImageProcessor",
    "Mistral3ImageProcessorFast",
    "Mistral3Processor",
]
