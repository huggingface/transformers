# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""PyTorch SuperPoint model."""

from dataclasses import dataclass

import torch
from torch import nn

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
)
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig

from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)


logger = logging.get_logger(__name__)


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Applies non-maximum suppression on scores."""
    if nms_radius < 0:
        raise ValueError("Expected positive values for nms_radius")

    def max_pool(x):
        return nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


@auto_docstring(
    custom_intro="""
    Base class for outputs of image point description models. Due to the nature of keypoint detection, the number of
    keypoints is not fixed and can vary from image to image, which makes batching non-trivial. In the batch of images,
    the maximum number of keypoints is set as the dimension of the keypoints, scores and descriptors tensors. The mask
    tensor is used to indicate which values in the keypoints, scores and descriptors tensors are keypoint information
    and which are padding.
    """
)
@dataclass
class SuperPointKeypointDescriptionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Loss computed during training.
    keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
        Relative (x, y) coordinates of predicted keypoints in a given image.
    scores (`torch.FloatTensor` of shape `(batch_size, num_keypoints)`):
        Scores of predicted keypoints.
    descriptors (`torch.FloatTensor` of shape `(batch_size, num_keypoints, descriptor_size)`):
        Descriptors of predicted keypoints.
    mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
        Mask indicating which values in keypoints, scores and descriptors are keypoint information.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
    when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
        (also called feature maps) of the model at the output of each stage.
    """

    loss: torch.FloatTensor | None = None
    keypoints: torch.IntTensor | None = None
    scores: torch.FloatTensor | None = None
    descriptors: torch.FloatTensor | None = None
    mask: torch.BoolTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


class SuperPointConvBlock(nn.Module):
    def __init__(
        self, config: SuperPointConfig, in_channels: int, out_channels: int, add_pooling: bool = False
    ) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_b = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if add_pooling else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.conv_a(hidden_states))
        hidden_states = self.relu(self.conv_b(hidden_states))
        if self.pool is not None:
            hidden_states = self.pool(hidden_states)
        return hidden_states


class SuperPointEncoder(nn.Module):
    """
    SuperPoint encoder module. It is made of 4 convolutional layers with ReLU activation and max pooling, reducing the
     dimensionality of the image.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        # SuperPoint uses 1 channel images
        self.input_dim = 1

        conv_blocks = []
        conv_blocks.append(
            SuperPointConvBlock(config, self.input_dim, config.encoder_hidden_sizes[0], add_pooling=True)
        )
        for i in range(1, len(config.encoder_hidden_sizes) - 1):
            conv_blocks.append(
                SuperPointConvBlock(
                    config, config.encoder_hidden_sizes[i - 1], config.encoder_hidden_sizes[i], add_pooling=True
                )
            )
        conv_blocks.append(
            SuperPointConvBlock(
                config, config.encoder_hidden_sizes[-2], config.encoder_hidden_sizes[-1], add_pooling=False
            )
        )
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(
        self,
        input,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
    ) -> tuple | BaseModelOutputWithNoAttention:
        all_hidden_states = () if output_hidden_states else None

        for conv_block in self.conv_blocks:
            input = conv_block(input)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (input,)
        output = input
        if not return_dict:
            return tuple(v for v in [output, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=output,
            hidden_states=all_hidden_states,
        )


class SuperPointInterestPointDecoder(nn.Module):
    """
    The SuperPointInterestPointDecoder uses the output of the SuperPointEncoder to compute the keypoint with scores.
    The scores are first computed by a convolutional layer, then a softmax is applied to get a probability distribution
    over the 65 possible keypoint classes. The keypoints are then extracted from the scores by thresholding and
    non-maximum suppression. Post-processing is then applied to remove keypoints too close to the image borders as well
    as to keep only the k keypoints with highest score.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.keypoint_threshold = config.keypoint_threshold
        self.max_keypoints = config.max_keypoints
        self.nms_radius = config.nms_radius
        self.border_removal_distance = config.border_removal_distance

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_score_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],
            config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_score_b = nn.Conv2d(
            config.decoder_hidden_size, config.keypoint_decoder_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self._get_pixel_scores(encoded)
        keypoints, scores, mask = self._extract_keypoints(scores)
        return keypoints, scores, mask

    def _get_pixel_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output, compute the scores for each pixel of the image."""
        scores = self.relu(self.conv_score_a(encoded))
        scores = self.conv_score_b(scores)
        scores = nn.functional.softmax(scores, 1)[:, :-1]
        batch_size, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)
        scores = simple_nms(scores, self.nms_radius)
        return scores

    def _extract_keypoints(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract keypoints from the score map using a statically-shaped pipeline
        compatible with ``torch.export``.

        Instead of ``torch.nonzero`` and boolean-index masking (both of which
        produce data-dependent shapes), we apply threshold and border
        constraints as additive ``-inf`` penalties and delegate selection to
        ``torch.topk``, which always returns exactly ``k`` entries.

        Args:
            scores: ``(batch_size, height, width)`` pixel score map after NMS.

        Returns:
            keypoints: ``(batch_size, k, 2)`` float tensor of ``(x, y)``
                keypoint coordinates in pixel space.
            topk_scores: ``(batch_size, k)`` float tensor of keypoint scores.
            mask: ``(batch_size, k)`` bool tensor; ``True`` where the entry is
                a real keypoint rather than padding.
        """
        batch_size, height, width = scores.shape

        border = self.border_removal_distance

        # Build row / column index grids for the border check without any
        # boolean indexing (keeps shapes static).
        rows = torch.arange(height, device=scores.device).view(1, height, 1)
        cols = torch.arange(width, device=scores.device).view(1, 1, width)

        # Replicate the original border logic exactly:
        # the legacy code called remove_keypoints_from_borders with
        # ``height * 8`` and ``width * 8``, so we reproduce that arithmetic.
        valid = (
            (scores > self.keypoint_threshold)
            & (rows >= border)
            & (rows < height * 8 - border)
            & (cols >= border)
            & (cols < width * 8 - border)
        )

        # Replace invalid positions with -inf so they sink to the end of topk.
        masked_scores = scores.masked_fill(~valid, float("-inf"))

        # Determine k — must be a concrete integer for export.
        # When max_keypoints is -1 ("no limit") we use every pixel; this path
        # is valid in eager mode but cannot be exported because k is
        # data-dependent via height * width.  Users who need export must set
        # max_keypoints to a positive value.
        k = self.max_keypoints if self.max_keypoints > 0 else height * width

        # torch.topk always returns exactly k entries → static output shape.
        topk_scores, topk_indices = torch.topk(masked_scores.reshape(batch_size, height * width), k=k, dim=1)

        # Convert flat indices to (x, y) pixel coordinates (pure arithmetic,
        # no data-dependent shapes).
        keypoints_y = topk_indices // width  # (B, k)
        keypoints_x = topk_indices % width  # (B, k)
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1).to(scores.dtype)  # (B, k, 2)

        # A position is a real keypoint if and only if its score is finite.
        # Cast to int (0/1) to match the original mask dtype expected by callers.
        mask = (topk_scores > float("-inf")).to(torch.int)  # (B, k) int

        return keypoints, topk_scores, mask


class SuperPointDescriptorDecoder(nn.Module):
    """
    The SuperPointDescriptorDecoder uses the outputs of both the SuperPointEncoder and the
    SuperPointInterestPointDecoder to compute the descriptors at the keypoints locations.

    The descriptors are first computed by a convolutional layer, then normalized to have a norm of 1. The descriptors
    are then interpolated at the keypoints locations.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_descriptor_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],
            config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_descriptor_b = nn.Conv2d(
            config.decoder_hidden_size,
            config.descriptor_decoder_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, encoded: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Compute descriptors for all keypoints in a batch.

        Args:
            encoded: ``(batch_size, channels, height, width)`` encoder feature map.
            keypoints: ``(batch_size, num_keypoints, 2)`` keypoint coordinates in
                ``(x, y)`` pixel space of the full-resolution score map.

        Returns:
            descriptors: ``(batch_size, num_keypoints, descriptor_dim)``
        """
        descriptors = self.conv_descriptor_b(self.relu(self.conv_descriptor_a(encoded)))
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        descriptors = self._sample_descriptors(keypoints, descriptors, 8)
        # (batch_size, descriptor_dim, num_keypoints) -> (batch_size, num_keypoints, descriptor_dim)
        descriptors = descriptors.transpose(1, 2)
        return descriptors

    @staticmethod
    def _sample_descriptors(keypoints: torch.Tensor, descriptors: torch.Tensor, scale: int = 8) -> torch.Tensor:
        """
        Interpolate descriptors at keypoint locations.

        Args:
            keypoints: ``(batch_size, num_keypoints, 2)`` in ``(x, y)`` pixel
                space of the full-resolution image.
            descriptors: ``(batch_size, num_channels, height, width)`` feature
                map at ``1/scale`` resolution.
            scale: downsampling factor between the full image and the feature map.

        Returns:
            descriptors: ``(batch_size, num_channels, num_keypoints)``
        """
        batch_size, num_channels, height, width = descriptors.shape
        keypoints = keypoints - scale / 2 + 0.5
        divisor = torch.tensor([[(width * scale - scale / 2 - 0.5), (height * scale - scale / 2 - 0.5)]])
        divisor = divisor.to(keypoints)
        keypoints = keypoints / divisor
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        kwargs = {"align_corners": True}
        # (batch_size, num_keypoints, 2) -> (batch_size, 1, num_keypoints, 2) for grid_sample
        keypoints = keypoints.view(batch_size, 1, -1, 2)
        descriptors = nn.functional.grid_sample(descriptors, keypoints, mode="bilinear", **kwargs)
        # (batch_size, num_channels, 1, num_keypoints) -> (batch_size, num_channels, num_keypoints)
        descriptors = descriptors.reshape(batch_size, num_channels, -1)
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors


@auto_docstring
class SuperPointPreTrainedModel(PreTrainedModel):
    config: SuperPointConfig
    base_model_prefix = "superpoint"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = False

    def extract_one_channel_pixel_values(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Assuming pixel_values has shape (batch_size, 3, height, width), and that all channels values are the same,
        extract the first channel value to get a tensor of shape (batch_size, 1, height, width) for SuperPoint. This is
        a workaround for the issue discussed in :
        https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

        Args:
            pixel_values: torch.FloatTensor of shape (batch_size, 3, height, width)

        Returns:
            pixel_values: torch.FloatTensor of shape (batch_size, 1, height, width)

        """
        return pixel_values[:, 0, :, :][:, None, :, :]


@auto_docstring(
    custom_intro="""
    SuperPoint model outputting keypoints and descriptors.
    """
)
class SuperPointForKeypointDetection(SuperPointPreTrainedModel):
    """
    SuperPoint model. It consists of a SuperPointEncoder, a SuperPointInterestPointDecoder and a
    SuperPointDescriptorDecoder. SuperPoint was proposed in `SuperPoint: Self-Supervised Interest Point Detection and
    Description <https://huggingface.co/papers/1712.07629>`__ by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. It
    is a fully convolutional neural network that extracts keypoints and descriptors from an image. It is trained in a
    self-supervised manner, using a combination of a photometric loss and a loss based on the homographic adaptation of
    keypoints. It is made of a convolutional encoder and two decoders: one for keypoints and one for descriptors.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__(config)

        self.config = config

        self.encoder = SuperPointEncoder(config)
        self.keypoint_decoder = SuperPointInterestPointDecoder(config)
        self.descriptor_decoder = SuperPointDescriptorDecoder(config)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | SuperPointKeypointDescriptionOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SuperPointForKeypointDetection
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        >>> model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        loss = None
        if labels is not None:
            raise ValueError("SuperPoint does not support training for now.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        pixel_values = self.extract_one_channel_pixel_values(pixel_values)

        batch_size, _, height, width = pixel_values.shape

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # Process the entire batch in one pass through the keypoint decoder.
        # Returns statically-shaped tensors: (B, k, 2), (B, k), (B, k) —
        # no Python loops over batch elements, no data-dependent shapes.
        keypoints, scores, mask = self.keypoint_decoder(last_hidden_state)

        # Compute descriptors for the full batch at once: (B, k, descriptor_dim).
        descriptors = self.descriptor_decoder(last_hidden_state, keypoints)

        # Convert keypoint pixel coordinates to relative (x, y) in [0, 1].
        keypoints = keypoints / torch.tensor([width, height], device=keypoints.device)

        hidden_states = encoder_outputs[1] if output_hidden_states else None
        if not return_dict:
            return tuple(v for v in [loss, keypoints, scores, descriptors, mask, hidden_states] if v is not None)

        return SuperPointKeypointDescriptionOutput(
            loss=loss,
            keypoints=keypoints,
            scores=scores,
            descriptors=descriptors,
            mask=mask,
            hidden_states=hidden_states,
        )


__all__ = ["SuperPointForKeypointDetection", "SuperPointPreTrainedModel"]
