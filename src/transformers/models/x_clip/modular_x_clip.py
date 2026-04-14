# Copyright 2022 Microsoft Research and The HuggingFace Team. All rights reserved.
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
"""PyTorch X-CLIP model."""

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ...utils.output_capturing import OutputRecorder
from ..altclip.modeling_altclip import AltCLIPEncoder, AltCLIPEncoderLayer
from ..beit.modeling_beit import BeitDropPath
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPModel,
    CLIPOutput,
    CLIPPreTrainedModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    eager_attention_forward,
    image_text_contrastive_loss,
)
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig


class XCLIPOutput(CLIPOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for video-text similarity.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
        The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`XCLIPTextModel`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`XCLIPVisionModel`].
    logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
        The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
        similarity scores.
    video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The video embeddings obtained by applying the projection layer to the pooled output of
        [`XCLIPVisionModel`].
    mit_output (`BaseModelOutputWithPooling`):
        The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """

    logits_per_video: torch.FloatTensor | None = None
    video_embeds: torch.FloatTensor | None = None
    mit_output: BaseModelOutputWithPooling = None
    logits_per_image = AttributeError()
    image_embeds = AttributeError()

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k]
            if k not in ["text_model_output", "vision_model_output", "mit_output"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class XCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__(config)


class XCLIPTextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: XCLIPTextConfig):
        super().__init__(config)


class XCLIPAttention(CLIPAttention):
    def __init__(self, config: XCLIPVisionConfig | XCLIPTextConfig):
        super().__init__(config)


class XCLIPMLP(CLIPMLP):
    def __init__(self, config: XCLIPVisionConfig | XCLIPTextConfig):
        super().__init__(config)


class XCLIPEncoderLayer(AltCLIPEncoderLayer):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.self_attn = XCLIPAttention(config)
        self.mlp = XCLIPMLP(config)


class XCLIPDropPath(BeitDropPath):
    pass


class XCLIPVisionEncoderLayer(CLIPEncoderLayer):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.self_attn = XCLIPAttention(config)
        self.mlp = XCLIPMLP(config)
        self.num_frames = config.num_frames
        self.message_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.message_ln = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.message_attn = XCLIPAttention(config)
        self.drop_path = XCLIPDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, torch.Tensor | None]:
        batch_time, seq_length, hidden_size = hidden_states.size()
        batch_size = batch_time // self.num_frames
        msg_token = self.message_fc(hidden_states[:, 0, :])
        msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)

        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token), **kwargs)[0])
        # add dummy sequence dimension
        msg_token = msg_token.view(-1, 1, hidden_size)

        hidden_states = torch.cat([hidden_states, msg_token], dim=1)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        hidden_states = hidden_states[:, :seq_length, :]

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring
class XCLIPPreTrainedModel(CLIPPreTrainedModel):
    config: XCLIPConfig
    base_model_prefix = "x_clip"
    _can_record_outputs = {
        "hidden_states": [XCLIPEncoderLayer, XCLIPVisionEncoderLayer],
        "attentions": OutputRecorder(XCLIPAttention, layer_name="self_attn", index=1),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, XCLIPTextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=factor * 0.02)
            init.normal_(module.position_embedding.weight, mean=0.0, std=factor * 0.02)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, XCLIPVisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, XCLIPAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, XCLIPMLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, XCLIPModel):
            init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )
            init.normal_(module.prompts_visual_projection, mean=0.0, std=module.vision_embed_dim**-0.5 * factor)
        elif isinstance(module, XCLIPMultiframeIntegrationTransformer):
            init.normal_(module.position_embedding, std=factor)

        if isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=factor)
            if module.bias is not None:
                init.zeros_(module.bias)


class XCLIPEncoder(AltCLIPEncoder):
    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.layers = nn.ModuleList([XCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class XCLIPVisionEncoder(CLIPEncoder):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([XCLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class XCLIPTextModel(CLIPTextModel, XCLIPPreTrainedModel):
    config: XCLIPTextConfig

    def __init__(self, config: XCLIPTextConfig):
        super().__init__(config)
        self.embeddings = XCLIPTextEmbeddings(config)
        self.encoder = XCLIPEncoder(config)
        self.eos_token_id = 2  # Force legacy behaviour

    def forward(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, XCLIPTextModel

        >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return super().forward(**super_kwargs)


class XCLIPVisionModel(CLIPVisionModel, XCLIPPreTrainedModel):
    config: XCLIPVisionConfig

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__(config)
        # TODO: fix typos across all models and add in conversion mapping
        del self.pre_layrnorm
        embed_dim = config.hidden_size

        self.embeddings = XCLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = XCLIPVisionEncoder(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, XCLIPVisionModel
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

        >>> # sample 16 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

        >>> batch_size, num_frames, num_channels, height, width = pixel_values.shape
        >>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        >>> outputs = model(pixel_values)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layernorm(hidden_states)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()

        self.position_embedding = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        self.encoder = XCLIPEncoder(config)

    def forward(
        self,
        hidden_states,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutput:
        residual = hidden_states

        # add position embeddings
        hidden_states = hidden_states + self.position_embedding

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            **kwargs,
        )
        last_hidden_state = encoder_outputs[0]

        last_hidden_state = last_hidden_state.type(hidden_states.dtype) + residual

        pooled_output = last_hidden_state.mean(dim=1, keepdim=False)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class XCLIPCrossAttention(CLIPAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        del self.dropout
        del self.out_proj

        self.num_heads = config.prompt_num_attention_heads
        self.embed_dim = config.projection_dim

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, False)

        self.attn_drop = config.prompt_attention_dropout
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        batch_size, query_seq_len, hidden_size = queries.shape
        batch_size, key_seq_len, hidden_size = keys.shape

        query_shape = (batch_size, query_seq_len, -1, self.head_dim)
        key_shape = (batch_size, key_seq_len, -1, self.head_dim)

        queries = self.q_proj(queries).view(*query_shape).transpose(1, 2)
        keys = self.k_proj(keys).view(*key_shape).transpose(1, 2)
        values = self.v_proj(values).view(*key_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask=None,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.attn_drop,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, query_seq_len, -1).contiguous()
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


class PromptGeneratorLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.projection_dim
        self.cross_attn = XCLIPCrossAttention(config)
        self.norm1 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            ACT2FN[config.prompt_hidden_act],
            nn.Dropout(config.prompt_attention_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, hidden_states, visual):
        hidden_states = hidden_states + self.cross_attn(self.norm1(hidden_states), visual, visual)
        hidden_states = hidden_states + self.mlp(self.norm3(hidden_states))
        return hidden_states


class XCLIPPromptGenerator(nn.Module):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.projection_dim
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.vision_config.layer_norm_eps)
        self.decoder = nn.ModuleList([PromptGeneratorLayer(config) for _ in range(config.prompt_layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * config.prompt_alpha)

    def forward(self, text, visual):
        visual = self.layernorm(visual)
        for layer in self.decoder:
            text = layer(text, visual)

        return self.alpha * text


class XCLIPModel(CLIPModel, XCLIPPreTrainedModel):
    config: XCLIPConfig

    def __init__(self, config: XCLIPConfig):
        super().__init__(config)
        vision_config = self.config.vision_config
        text_config = self.config.text_config

        self.text_model = XCLIPTextModel(text_config)
        self.vision_model = XCLIPVisionModel(vision_config)

        self.prompts_visual_layernorm = nn.LayerNorm(self.vision_embed_dim, eps=config.vision_config.layer_norm_eps)
        self.prompts_visual_projection = nn.Parameter(torch.randn(self.vision_embed_dim, self.projection_dim))
        mit_config = copy.copy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)
        self.prompts_generator = XCLIPPromptGenerator(config)

    def get_text_features(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, AutoModel

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        return super().get_text_features(**super_kwargs)

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, AutoModel
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

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = processor(videos=list(video), return_tensors="pt")

        >>> video_features = model.get_video_features(**inputs)
        ```"""
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        video_outputs: BaseModelOutputWithPooling = self.vision_model(pixel_values=pixel_values, **kwargs)
        video_embeds = video_outputs.pooler_output
        video_embeds = self.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)
        mit_outputs: BaseModelOutputWithPooling = self.mit(cls_features, **kwargs)
        video_outputs.pooler_output = mit_outputs.pooler_output

        return video_outputs

    def get_image_features(self):
        raise AttributeError("XCLIP doesn't support images")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | XCLIPOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import av
        >>> import torch
        >>> import numpy as np

        >>> from transformers import AutoProcessor, AutoModel
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

        >>> # sample 8 frames
        >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        >>> video = read_video_pyav(container, indices)

        >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = processor(
        ...     text=["playing sports", "eating spaghetti", "go shopping"],
        ...     videos=list(video),
        ...     return_tensors="pt",
        ...     padding=True,
        ... )

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
        >>> probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
        >>> print(probs)
        tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])
        ```"""
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.mit(
            cls_features,
            **kwargs,
        )
        video_embeds = mit_outputs[1]

        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.prompts_visual_projection
        img_features = img_features.view(batch_size, num_frames, -1, video_embeds.shape[-1])
        img_features = img_features.mean(dim=1, keepdim=False)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        text_embeds = text_embeds + self.prompts_generator(text_embeds, img_features)

        # normalized features
        video_embeds = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_video = torch.einsum("bd,bkd->bk", video_embeds, logit_scale * text_embeds)
        logits_per_text = logits_per_video.T

        loss = None
        if return_loss:
            loss = image_text_contrastive_loss(logits_per_text)

        return XCLIPOutput(
            loss=loss,
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
            mit_output=mit_outputs,
        )


__all__ = ["XCLIPModel", "XCLIPPreTrainedModel", "XCLIPTextModel", "XCLIPVisionModel"]
