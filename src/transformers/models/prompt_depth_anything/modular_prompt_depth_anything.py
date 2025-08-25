# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.models.depth_anything.configuration_depth_anything import DepthAnythingConfig
from transformers.models.depth_anything.modeling_depth_anything import (
    DepthAnythingDepthEstimationHead,
    DepthAnythingFeatureFusionLayer,
    DepthAnythingFeatureFusionStage,
    DepthAnythingForDepthEstimation,
    DepthAnythingNeck,
    DepthAnythingReassembleStage,
)
from transformers.utils.generic import torch_int

from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring


class PromptDepthAnythingConfig(DepthAnythingConfig):
    model_type = "prompt_depth_anything"


class PromptDepthAnythingLayer(nn.Module):
    def __init__(self, config: PromptDepthAnythingConfig):
        super().__init__()
        self.convolution1 = nn.Conv2d(
            1,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.activation1 = nn.ReLU()

        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.activation2 = nn.ReLU()

        self.convolution3 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, prompt_depth: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution1(prompt_depth)
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution3(hidden_state)
        return hidden_state


class PromptDepthAnythingFeatureFusionLayer(DepthAnythingFeatureFusionLayer):
    def __init__(self, config: PromptDepthAnythingConfig):
        super().__init__(config)
        self.prompt_depth_layer = PromptDepthAnythingLayer(config)

    def forward(self, hidden_state, residual=None, size=None, prompt_depth=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=hidden_state.shape[2:], mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)

        if prompt_depth is not None:
            prompt_depth = nn.functional.interpolate(
                prompt_depth, size=hidden_state.shape[2:], mode="bilinear", align_corners=False
            )
            res = self.prompt_depth_layer(prompt_depth)
            hidden_state = hidden_state + res

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class PromptDepthAnythingFeatureFusionStage(DepthAnythingFeatureFusionStage):
    def forward(self, hidden_states, size=None, prompt_depth=None):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        fused_hidden_state = None

        for idx, (hidden_state, layer) in enumerate(zip(hidden_states, self.layers)):
            size = hidden_states[idx + 1].shape[2:] if idx != (len(hidden_states) - 1) else None

            if fused_hidden_state is None:
                # first layer only uses the last hidden_state
                fused_hidden_state = layer(hidden_state, size=size, prompt_depth=prompt_depth)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size, prompt_depth=prompt_depth)

            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class PromptDepthAnythingDepthEstimationHead(DepthAnythingDepthEstimationHead):
    def forward(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        hidden_states = hidden_states[-1]

        predicted_depth = self.conv1(hidden_states)
        target_height = torch_int(patch_height * self.patch_size)
        target_width = torch_int(patch_width * self.patch_size)
        predicted_depth = nn.functional.interpolate(
            predicted_depth,
            (target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )
        predicted_depth = self.conv2(predicted_depth)
        predicted_depth = self.activation1(predicted_depth)
        predicted_depth = self.conv3(predicted_depth)
        predicted_depth = self.activation2(predicted_depth)
        # (batch_size, 1, height, width) -> (batch_size, height, width), which
        # keeps the same behavior as Depth Anything v1 & v2
        predicted_depth = predicted_depth.squeeze(dim=1)

        return predicted_depth


@auto_docstring
class PromptDepthAnythingPreTrainedModel(PreTrainedModel):
    config: PromptDepthAnythingConfig
    base_model_prefix = "prompt_depth_anything"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True


class PromptDepthAnythingReassembleLayer(nn.Module):
    def __init__(self, config: PromptDepthAnythingConfig, channels: int, factor: int):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=config.reassemble_hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            stride = torch_int(1 / factor)
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)

        return hidden_state


class PromptDepthAnythingReassembleStage(DepthAnythingReassembleStage):
    pass


class PromptDepthAnythingNeck(DepthAnythingNeck):
    def forward(
        self,
        hidden_states: list[torch.Tensor],
        patch_height: Optional[int] = None,
        patch_width: Optional[int] = None,
        prompt_depth: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features, prompt_depth=prompt_depth)

        return output


@auto_docstring(
    custom_intro="""
    Prompt Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """
)
class PromptDepthAnythingForDepthEstimation(DepthAnythingForDepthEstimation):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        prompt_depth: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        prompt_depth (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`, *optional*):
            Prompt depth is the sparse or low-resolution depth obtained from multi-view geometry or a
            low-resolution depth sensor. It generally has shape (height, width), where height
            and width can be smaller than those of the images. It is optional and can be None, which means no prompt depth
            will be used. If it is None, the output will be a monocular relative depth.
            The values are recommended to be in meters, but this is not necessary.

        Example:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/image.jpg?raw=true"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
        >>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

        >>> prompt_depth_url = "https://github.com/DepthAnything/PromptDA/blob/main/assets/example_images/arkit_depth.png?raw=true"
        >>> prompt_depth = Image.open(requests.get(prompt_depth_url, stream=True).raw)

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt", prompt_depth=prompt_depth)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 1000.
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint16")) # mm
        ```
        """
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        if prompt_depth is not None:
            # normalize prompt depth
            batch_size = prompt_depth.shape[0]
            depth_min = torch.min(prompt_depth.reshape(batch_size, -1), dim=1).values
            depth_max = torch.max(prompt_depth.reshape(batch_size, -1), dim=1).values
            depth_min, depth_max = depth_min.view(batch_size, 1, 1, 1), depth_max.view(batch_size, 1, 1, 1)
            prompt_depth = (prompt_depth - depth_min) / (depth_max - depth_min)
            # normalize done

        hidden_states = self.neck(hidden_states, patch_height, patch_width, prompt_depth=prompt_depth)

        predicted_depth = self.head(hidden_states, patch_height, patch_width)
        if prompt_depth is not None:
            # denormalize predicted depth
            depth_min = depth_min.squeeze(1).to(predicted_depth.device)
            depth_max = depth_max.squeeze(1).to(predicted_depth.device)
            predicted_depth = predicted_depth * (depth_max - depth_min) + depth_min
            # denormalize done

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = [
    "PromptDepthAnythingConfig",
    "PromptDepthAnythingForDepthEstimation",
    "PromptDepthAnythingPreTrainedModel",
]
