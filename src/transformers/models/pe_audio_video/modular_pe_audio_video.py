from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import check_model_inputs
from ..auto import AutoModel
from ..clip.modeling_clip import CLIPOutput
from ..pe_audio.modeling_pe_audio import (
    PEAudioAttention,
    PEAudioContrastiveHead,
    PEAudioDecoderLayer,
    PEAudioResnetBlock1d,
)
from ..pe_audio.modeling_pe_audio import PEAudioEncoderEmbeddings
from ..qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer, Qwen3Model, Qwen3RMSNorm
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..auto import CONFIG_MAPPING, AutoConfig
from ...configuration_utils import PretrainedConfig

class PEAudioVideoEncoderConfig(Qwen3Config):
    sub_configs = {"audio_config": AutoConfig, "video_config": AutoConfig}

    def __init__(
        self,
        audio_config=None,
        video_config=None,
        hidden_size=1792,
        intermediate_size=4800,
        num_hidden_layers=6,
        num_attention_heads=14,
        num_key_value_heads=14,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=10000,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_theta=20000,
        rope_scaling=None,
        attention_bias=False,
        max_window_layers=28,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        layer_types=None,
        tie_word_embeddings=False,
        vocab_size=None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            vocab_size=vocab_size,
            layer_types=layer_types,
            tie_word_embeddings=tie_word_embeddings,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            **kwargs,
        )

        if isinstance(audio_config, dict):
            audio_config["model_type"] = audio_config.get("model_type", "pe_audio_encoder")
            audio_config = CONFIG_MAPPING[audio_config["model_type"]](**audio_config)
        elif audio_config is None:
            audio_config = CONFIG_MAPPING["pe_audio_encoder"]()
            # TODO: add log

        self.audio_config = audio_config

        if isinstance(video_config, dict):
            video_config["model_type"] = video_config.get("model_type", "pe_video_encoder")
            video_config = CONFIG_MAPPING[video_config["model_type"]](**video_config)
        elif video_config is None:
            video_config = CONFIG_MAPPING["pe_video_encoder"]()
            # TODO: add log

        self.video_config = video_config


class PEAudioVideoConfig(PretrainedConfig):
    model_type = "pe_audio"
    sub_configs = {"text_config": AutoConfig, "audio_video_config": PEAudioVideoEncoderConfig}

    def __init__(
        self,
        text_config=None,
        audio_video_config=None,
        projection_dim=1024,
        nth_text_layer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "modernbert")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["modernbert"](
                hidden_size=1024,
                intermediate_size=2624,
                num_hidden_layers=28,
                num_attention_heads=16,
                # classifier_pooling="mean",
            )
            # TODO: add log

        if isinstance(audio_video_config, dict):
            audio_video_config = PEAudioVideoEncoderConfig.from_dict(audio_video_config)
        elif audio_video_config is None:
            audio_video_config = PEAudioVideoEncoderConfig()
            # TODO: add log

        self.text_config = text_config
        self.audio_video_config = audio_video_config

        self.projection_dim = projection_dim
        self.nth_text_layer = nth_text_layer

        self.logit_scale_init_value = 1.0
        self.logit_bias_init_value = 0.0


class PEAudioVideoAlignModalities(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        btc: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.btc = btc
        self.conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        if self.normalize:
            self.layer_norm = nn.LayerNorm(self.out_channels)

    def get_sizes(self, seq, mask):
        if mask is not None:
            sizes = mask.sum(-1)
        else:
            sizes = torch.full((seq.size(0),), seq.size(-1), device=seq.device)
        if sizes.dim() > 1:
            sizes = sizes.squeeze(1)
        return sizes.long()

    def interpolate(self, tgt, tgt_sizes, src_sizes) -> torch.Tensor:
        result = torch.zeros(tgt.size(0), tgt.size(1), src_sizes.max(), device=tgt.device)
        for i, (tgt_row, tgt_size, src_size) in enumerate(zip(tgt, tgt_sizes, src_sizes)):
            tgt_row = tgt_row[:, :tgt_size]
            interpolated = F.interpolate(tgt_row[None], size=src_size, mode="nearest")
            result[i, :, :src_size] = interpolated[0]
        return result

    def forward(self, src, src_mask, tgt, tgt_mask):
        # BxTxC -> BxCxT
        src = src.transpose(1, 2)
        tgt = tgt.transpose(1, 2)

        tgt = self.conv(tgt)

        src_sizes = self.get_sizes(src, src_mask)
        tgt_sizes = self.get_sizes(tgt, tgt_mask)
        if all(src_sizes == tgt_sizes):
            upsampled = tgt
        else:
            upsampled = self.interpolate(tgt, tgt_sizes, src_sizes)

        upsampled = upsampled.permute(0, 2, 1)  # BxCxT -> BxTxC
        if self.normalize:
            upsampled = self.layer_norm(upsampled)
        return upsampled, src_mask


class PEAudioVideoContrastiveHead(PEAudioContrastiveHead): ...


# TODO: not sure about the typing for text_model_output
class PEAudioVideoOutput(CLIPOutput): ...


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`PEAudioVideoEncoder`].
    """
)
class PEAudioVideoEncoderOutput(BaseModelOutputWithPooling):
    audio_features: Optional[torch.FloatTensor] = None
    video_features: Optional[torch.FloatTensor] = None
    outputs_mask: Optional[tuple[torch.FloatTensor]] = None


class PEAudioVideoEncoderEmbeddings(PEAudioEncoderEmbeddings): ...


class PEAudioVideoEncoderAttention(Qwen3Attention): ...


class PEAudioVideoEncoderLayer(Qwen3DecoderLayer): ...


class PEAudioVideoRMSNorm(Qwen3RMSNorm): ...


@auto_docstring
class PEAudioVideoPretrainedModel(PreTrainedModel):
    config: PEAudioVideoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PEAudioVideoEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": PEAudioVideoEncoderLayer,
        "attentions": PEAudioVideoEncoderAttention,
    }


@auto_docstring(
    custom_intro="""
    The PEAudioVideo Encoder model.
    """
)
class PEAudioVideoEncoder(PEAudioVideoPretrainedModel):
    def __init__(self, config: PEAudioVideoEncoderConfig):
        super().__init__(config)
        self.audio_encoder = AutoModel.from_config(config.audio_config)
        self.video_encoder = AutoModel.from_config(config.video_config)
        self.modality_aligner = PEAudioVideoAlignModalities(
            config.audio_config.hidden_size, config.video_config.hidden_size, normalize=True, btc=True
        )
        self.concat_modality_proj = nn.Linear(
            config.audio_config.hidden_size + config.video_config.hidden_size,
            config.hidden_size,
        )
        self.data_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.embeddings = PEAudioVideoEncoderEmbeddings(config)
        self.layers = nn.ModuleList(
            [PEAudioVideoEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PEAudioVideoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def get_audio_video_features(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        audio_output = self.audio_encoder(input_values, padding_mask=padding_mask)
        video_output = self.video_encoder(pixel_values_videos, padding_mask_videos=padding_mask_videos)

        video, video_padding_mask = self.modality_aligner(
            audio_output.last_hidden_state,
            padding_mask,
            video_output.last_hidden_state,
            padding_mask_videos,
        )

        inputs_embeds = torch.cat([audio_output.last_hidden_state, video], dim=-1)
        inputs_embeds = self.concat_modality_proj(inputs_embeds)
        inputs_embeds = self.data_proj(inputs_embeds)

        return inputs_embeds, video_padding_mask

    @can_return_tuple
    @check_model_inputs
    def forward(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        inputs_embeds, attention_mask = self.get_audio_video_features(
            input_values,
            pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )

        inputs_embeds, attention_mask = self.embeddings(inputs_embeds, attention_mask=attention_mask)

        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = self.rope_embeddings(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states[:, 1:],
            pooler_output=hidden_states[:, 0],
        )



# class PEAudioVideoEncoder(PEAudioVideoPretrainedModel):
#     config_class = PEAudioVideoEncoderConfig
#     base_model_prefix = "audio_video_encoder"

#     def __init__(self, config: PEAudioVideoEncoderConfig):
#         super().__init__(config)
#         self.audio_encoder = AutoModel.from_config(config.audio_config)
#         self.video_encoder = AutoModel.from_config(config.video_config)

#         self.modality_aligner = PEAudioVideoAlignModalities(
#             self.config.hidden_size, self.config.hidden_size, normalize=True, btc=True
#         )
#         self.concat_modality_proj = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
#         self.data_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#         self.transformer = PEAudioVideoTransformer(config)

#         self.post_init()

#     @can_return_tuple
#     @check_model_inputs
#     def forward(
#         self,
#         input_values: torch.Tensor,
#         pixel_values_videos: torch.Tensor,
#         padding_mask: Optional[torch.Tensor] = None,
#         padding_mask_videos: Optional[torch.Tensor] = None,
#         **kwargs: Unpack[TransformersKwargs],
#     ) -> PEAudioVideoEncoderOutput:
#         audio_output = self.audio_encoder(input_values, padding_mask=padding_mask)
#         video_output = self.video_encoder(pixel_values_videos, padding_mask_videos=padding_mask_videos)

#         video, video_padding_mask = self.modality_aligner(
#             audio_output.last_hidden_state,
#             padding_mask,
#             video_output.last_hidden_state,
#             padding_mask_videos,
#         )

#         x = torch.cat([audio_output.last_hidden_state, video], dim=-1)
#         x = self.concat_modality_proj(x)
#         x = self.data_proj(x)

#         av_output = self.transformer(
#             x,
#             attention_mask=video_padding_mask,
#             **kwargs,
#         )

#         return PEAudioVideoEncoderOutput(
#             last_hidden_state=av_output.last_hidden_state,
#             pooler_output=av_output.pooler_output,
#             hidden_states=av_output.hidden_states,
#             attentions=av_output.attentions,
#             audio_features=audio_output.last_hidden_state,
#             video_features=video_output.last_hidden_state,
#             outputs_mask=video_padding_mask,
#         )


@dataclass
class PEAudioVideoTextOutput(ModelOutput):
    audio_video_loss: Optional[torch.FloatTensor] = None
    text_audio_loss: Optional[torch.FloatTensor] = None
    text_audio_video_loss: Optional[torch.FloatTensor] = None
    text_video_loss: Optional[torch.FloatTensor] = None
    # embeddings
    audio_embeds: Optional[torch.FloatTensor] = None
    audio_video_embeds: Optional[torch.FloatTensor] = None
    video_embeds: Optional[torch.FloatTensor] = None
    audio_text_embeds: Optional[torch.FloatTensor] = None
    audio_video_text_embeds: Optional[torch.FloatTensor] = None
    video_text_embeds: Optional[torch.FloatTensor] = None
    # model outputs
    audio_model_output: Optional[BaseModelOutputWithPooling] = None
    audio_video_model_output: Optional[BaseModelOutputWithPooling] = None
    text_model_output: Optional[BaseModelOutputWithPooling] = None
    video_model_output: Optional[BaseModelOutputWithPooling] = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(self[k] if not k.endswith("model_output") else getattr(self, k).to_tuple() for k in self.keys())


class PEAudioVideoModel(PEAudioVideoPretrainedModel):
    def __init__(self, config: PEAudioVideoConfig):
        super().__init__(config)
        self.text_model = AutoModel.from_config(config.text_config)
        self.audio_video_encoder = PEAudioVideoEncoder(config.audio_video_config)

        self.text_head_audio = PEAudioVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.text_head_video = PEAudioVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)
        self.text_head_audio_video = PEAudioVideoContrastiveHead(config.text_config.hidden_size, config.projection_dim)

        self.audio_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.audio_config.hidden_size, config.projection_dim
        )
        self.video_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.video_config.hidden_size, config.projection_dim
        )
        self.audio_video_head = PEAudioVideoContrastiveHead(
            config.audio_video_config.hidden_size, config.projection_dim
        )

        self.logit_scale = nn.Parameter(torch.tensor([config.logit_scale_init_value]).log())
        self.logit_bias = nn.Parameter(torch.tensor([config.logit_bias_init_value]))

        self.post_init()

    def _get_text_output(self, input_ids, attention_mask):
        nth_layer = self.config.nth_text_layer
        output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=nth_layer is not None
        )
        if nth_layer is None:
            text_model_output = output.last_hidden_state
        else:
            text_model_output = output.hidden_states[nth_layer]

        return BaseModelOutputWithPooling(last_hidden_state=text_model_output, pooler_output=text_model_output[:, 0])

    def _maybe_compute_loss(
        self, embeds1: Optional[torch.Tensor], embeds2: Optional[torch.Tensor], return_loss: bool
    ) -> Optional[torch.Tensor]:
        if return_loss and embeds1 is not None and embeds2 is not None:
            logits = embeds1 @ embeds2.t()
            logits = logits * self.logit_scale + self.logit_bias
            labels = torch.eye(embeds1.size(0), device=embeds1.device)
            return -F.logsigmoid(labels * logits).sum() / embeds1.size(0)
        return None

    def get_video_features(
        self, pixel_values_videos: torch.Tensor, padding_mask_videos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, channels, height, width)`):
                The input video frames tensor.
            padding_mask_videos (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Mask indicating non-padded frames in the video input.
        Returns:
            video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the video embedding
                obtained by applying the projection layer to the pooled output of the video encoder.

        Example:
            ```python
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained("facebook/pe-av-large")
            processor = AutoProcessor.from_pretrained("facebook/pe-av-large")

            inputs = processor(
                videos=["<path to video file>"],
                padding=True,
                return_tensors="pt",
            )

            with torch.inference_mode():
                video_features = model.get_video_features(**inputs)
            ```
        """

        return self.video_head(
            self.audio_video_encoder.video_encoder(
                pixel_values_videos, padding_mask_videos=padding_mask_videos
            ).pooler_output
        )

    def get_audio_features(
        self, input_values: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input audio waveform tensor.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):

        Returns:
            audio_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the audio embedding
                obtained by applying the projection layer to the pooled output of the audio encoder.

        Example:
            ```python
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained("facebook/pe-av-large")
            processor = AutoProcessor.from_pretrained("facebook/pe-av-large")

            inputs = processor(
                audio=["<path to audio file>"],
                padding=True,
                return_tensors="pt",
            )

            with torch.inference_mode():
                audio_features = model.get_audio_features(**inputs)
            ```
        """

        return self.audio_head(
            self.audio_video_encoder.audio_encoder(input_values, padding_mask=padding_mask).pooler_output
        )

    def get_audio_video_features(
        self,
        input_values: torch.Tensor,
        pixel_values_videos: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input audio waveform tensor.
            pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, channels, height, width)`):
                The input video frames tensor.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask indicating non-padded elements in the audio input.
            padding_mask_videos (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Mask indicating non-padded frames in the video input.

        Returns:
            audio_video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the audio-video embedding
                obtained by applying the projection layer to the pooled output of the audio-video encoder.

        Provides a single embedding representing both the audio and video

        Example:
            ```python
            from transformers import AutoModel, AutoProcessor

            model = AutoModel.from_pretrained("facebook/pe-av-large")
            processor = AutoProcessor.from_pretrained("facebook/pe-av-large")

            inputs = processor(
                audio=["<path to audio file>"],
                videos=["<path to video file>"],
                padding=True,
                return_tensors="pt",
            )

            with torch.inference_mode():
                audio_video_features = model.get_audio_video_features(**inputs)
            ```
        """
        output = self.audio_video_encoder(
            input_values=input_values,
            pixel_values_videos=pixel_values_videos,
            padding_mask=padding_mask,
            padding_mask_videos=padding_mask_videos,
        )
        return self.audio_video_head(output.pooler_output)

    def get_audio_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input token ids for the text encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask indicating non-padded elements in the input for the text encoder.

        Returns:
            audio_text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the audio-text embedding
                obtained by applying the projection layer to the pooled output of the text encoder

        This embedding is suitable for retrieving audios from a text description, but if you want to specifically
        retrieve video from text, you should use `get_video_text_features` instead.

        ```python
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained("facebook/pe-av-large")
        processor = AutoProcessor.from_pretrained("facebook/pe-av-large")

        inputs = processor(text=["<text>"], return_tensors="pt", padding=True)

        with torch.inference_mode():
            audio_text_features = model.get_audio_text_features(**inputs)
        ```
        """
        return self.text_head_audio(self._get_text_output(input_ids, attention_mask).pooler_output)

    def get_video_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input token ids for the text encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask indicating non-padded elements in the input for the text encoder.

        Returns:
            video_text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the video-text embedding
                obtained by applying the projection layer to the pooled output of the text encoder

        This embedding is suitable for retrieving videos from a text description, but if you want to specifically
        retrieve audio from text, you should use `get_audio_text_features` instead.

        ```python
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained("facebook/pe-av-large")
        processor = AutoProcessor.from_pretrained("facebook/pe-av-large")

        inputs = processor(text=["<text>"], return_tensors="pt", padding=True)

        with torch.inference_mode():
            video_text_features = model.get_video_text_features(**inputs)
        ```
        """
        return self.text_head_video(self._get_text_output(input_ids, attention_mask).pooler_output)

    def get_audio_video_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input token ids for the text encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask indicating non-padded elements in the input for the text encoder.

        Returns:
            audio_video_text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`): the audio-video text
                embedding obtained by applying the projection layer to the pooled output of the text encoder

        This is a good general purpose text embedding for, but if you want to specifically retrieve audio from
        a text description, you should use `get_audio_text_features` instead (and similarly `get_video_text_features` for video).

        ```python
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained("facebook/pe-av-large)
        processor = AutoProcessor.from_pretrained("facebook/pe-av-large)

        inputs = processor(text=["<text>"], return_tensors="pt", padding=True)

        with torch.inference_mode():
            audio_video_text_features = model.get_audio_video_text_features(**inputs)
        ```
        """
        return self.text_head_audio_video(self._get_text_output(input_ids, attention_mask).pooler_output)

    @can_return_tuple
    def forward(
        self,
        input_ids,
        pixel_values_videos,
        input_values,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask_videos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_loss=False,
    ) -> PEAudioVideoTextOutput:

        # text embeddings
        audio_text_embeds = video_text_embeds = audio_video_text_embeds = None
        # media embeddings (audio, video, audio_video)
        audio_embeds = video_embeds = audio_video_embeds = None
        # model outputs
        audio_outputs = video_outputs = audio_video_outputs = text_outputs = None

        # Compute model outputs and embeddings for each modality
        if input_ids is not None:
            text_outputs = self._get_text_output(input_ids, attention_mask)
        if input_values is not None and pixel_values_videos is not None:
            # If we compute audio/video outputs, then extract the intermediate audio and video outputs
            audio_video_outputs = self.audio_video_encoder(
                input_values, pixel_values_videos, padding_mask=padding_mask, padding_mask_videos=padding_mask_videos
            )
            # Get intermediate audio and video outputs
            audio_outputs = self.audio_video_encoder.audio_encoder(input_values, padding_mask=padding_mask)
            video_outputs = self.audio_video_encoder.video_encoder(
                pixel_values_videos, padding_mask_videos=padding_mask_videos
            )

            audio_embeds = self.audio_head(audio_outputs.pooler_output)
            video_embeds = self.video_head(video_outputs.pooler_output)
            audio_video_embeds = self.audio_video_head(audio_video_outputs.pooler_output)
            if text_outputs is not None:
                # Compute the corresponding text embeddings
                audio_text_embeds = self.text_head_audio(text_outputs.pooler_output)
                video_text_embeds = self.text_head_video(text_outputs.pooler_output)
                audio_video_text_embeds = self.text_head_audio_video(text_outputs.pooler_output)
        else:
            if pixel_values_videos is not None:
                video_outputs = self.audio_video_encoder.video_encoder(
                    pixel_values_videos, padding_mask_videos=padding_mask_videos
                )
                video_embeds = self.video_head(video_outputs.pooler_output)
                if text_outputs is not None:
                    video_text_embeds = self.text_head_video(text_outputs.pooler_output)
            elif input_values is not None:
                audio_outputs = self.audio_video_encoder.audio_encoder(input_values, padding_mask=padding_mask)
                audio_embeds = self.audio_head(audio_outputs.pooler_output)
                if text_outputs is not None:
                    audio_text_embeds = self.text_head_audio(text_outputs.pooler_output)
            elif text_outputs is not None:
                # If text is supplied, but no audio or video, use audio_video_text as the default embedding
                audio_video_text_embeds = self.text_head_audio_video(text_outputs.pooler_output)

        return PEAudioVideoTextOutput(
            audio_video_loss=self._maybe_compute_loss(audio_embeds, video_embeds, return_loss),
            text_audio_loss=self._maybe_compute_loss(audio_text_embeds, audio_embeds, return_loss),
            text_audio_video_loss=self._maybe_compute_loss(audio_video_text_embeds, audio_video_embeds, return_loss),
            text_video_loss=self._maybe_compute_loss(video_text_embeds, video_embeds, return_loss),
            audio_embeds=audio_embeds,
            audio_video_embeds=audio_video_embeds,
            video_embeds=video_embeds,
            audio_text_embeds=audio_text_embeds,
            audio_video_text_embeds=audio_video_text_embeds,
            video_text_embeds=video_text_embeds,
            audio_model_output=audio_outputs,
            audio_video_model_output=audio_video_outputs,
            text_model_output=text_outputs,
            video_model_output=video_outputs,
        )


__all__ = [
    "PEAudioVideoModel",
    "PEAudioVideoEncoder",
    "PEAudioVideoEncoderConfig",
    "PEAudioVideoConfig",
]
