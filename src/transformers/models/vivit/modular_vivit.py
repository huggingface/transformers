# Copyright 2023 Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViViT model - modular file inheriting transformer core from ViT."""

from collections.abc import Iterable

import torch
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..vit.modeling_vit import (
    PreTrainedModel,
    ViTAttention,
    ViTEmbeddings,
    ViTLayer,
    ViTMLP,
    ViTModel,
    ViTPooler,
    ViTPreTrainedModel,
)
from .configuration_vivit import VivitConfig


logger = logging.get_logger(__name__)


class VivitTubeletEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_frames, num_channels, height, width)` into the initial
    `hidden_states` (tubelet embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer encoder.

    The seq_length equals (num_frames // tubelet_size[0]) * (height // tubelet_size[1]) * (width // tubelet_size[2]).
    """

    def __init__(self, config: VivitConfig):
        super().__init__()
        tubelet_size = config.tubelet_size
        image_size = config.image_size
        image_size = image_size if isinstance(image_size, Iterable) else (image_size, image_size)

        self.num_patches = (
            (config.num_frames // tubelet_size[0])
            * (image_size[0] // tubelet_size[1])
            * (image_size[1] // tubelet_size[2])
        )
        self.image_size = image_size
        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # transpose (batch_size, num_channels, num_frames, height, width) for Conv3d
        pixel_values = pixel_values.transpose(1, 2)
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class VivitEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, position and tubelet patch embeddings for video input.
    """

    def __init__(self, config: VivitConfig):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = VivitTubeletEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # patch_size is the spatial (height, width) part of the tubelet for pos encoding interpolation
        self.patch_size = config.tubelet_size[1:]
        del self.mask_token

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        super().interpolate_pos_encoding(embeddings, height, width)
        # patch_size is a 2-tuple (height, width) for the spatial tubelet dimensions
        new_height = height // self.patch_size[0]  # noqa: F841
        new_width = width // self.patch_size[1]  # noqa: F841

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class VivitAttention(ViTAttention):
    pass


class VivitMLP(ViTMLP):
    pass


class VivitLayer(ViTLayer):
    pass


class VivitPooler(ViTPooler):
    pass


@auto_docstring
class VivitPreTrainedModel(ViTPreTrainedModel):
    config: VivitConfig
    base_model_prefix = "vivit"
    input_modalities = ("video",)
    _no_split_modules = ["VivitEmbeddings", "VivitLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, VivitEmbeddings):
            init.zeros_(module.cls_token)
            init.zeros_(module.position_embeddings)


@auto_docstring
class VivitModel(ViTModel):
    def __init__(self, config: VivitConfig, add_pooling_layer: bool = True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.embeddings = VivitEmbeddings(config)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import av
        >>> import numpy as np

        >>> from transformers import VivitImageProcessor, VivitModel
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`list[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`list[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 32 frames
        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container=container, indices=indices)

        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        >>> model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")

        >>> # prepare video for the model
        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 3137, 768]
        ```"""

        embedding_output = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )
        hidden_states = embedding_output
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, **kwargs)
        sequence_output = self.layernorm(hidden_states)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)


@auto_docstring(
    custom_intro="""
        ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for Kinetics-400.

        <Tip>

            Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
            setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
            position embeddings to the higher resolution.

        </Tip>
    """
)
class VivitForVideoClassification(VivitPreTrainedModel):
    def __init__(self, config: VivitConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vivit = VivitModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> import av
        >>> import numpy as np
        >>> import torch

        >>> from transformers import VivitImageProcessor, VivitForVideoClassification
        >>> from huggingface_hub import hf_hub_download

        >>> np.random.seed(0)


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`list[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


        >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        ...     '''
        ...     Sample a given number of frame indices from the video.
        ...     Args:
        ...         clip_len (`int`): Total number of frames to sample.
        ...         frame_sample_rate (`int`): Sample every n-th frame.
        ...         seg_len (`int`): Maximum allowed index of sample's last frame.
        ...     Returns:
        ...         indices (`list[int]`): List of sampled frame indices
        ...     '''
        ...     converted_len = int(clip_len * frame_sample_rate)
        ...     end_idx = np.random.randint(converted_len, seg_len)
        ...     start_idx = end_idx - converted_len
        ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
        ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        ...     return indices


        >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
        >>> file_path = hf_hub_download(
        ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
        ... )
        >>> container = av.open(file_path)

        >>> # sample 32 frames
        >>> indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container=container, indices=indices)

        >>> image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        >>> model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

        >>> inputs = image_processor(list(video), return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     logits = outputs.logits

        >>> # model predicts one of the 400 Kinetics-400 classes
        >>> predicted_label = logits.argmax(-1).item()
        >>> print(model.config.id2label[predicted_label])
        LABEL_116
        ```"""

        outputs: BaseModelOutput = self.vivit(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["VivitModel", "VivitPreTrainedModel", "VivitForVideoClassification"]
