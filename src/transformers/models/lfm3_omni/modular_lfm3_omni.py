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
"""LFM3-Omni: the LFM3-Audio model with the LFM3-VL vision tower."""

import re

import torch
from huggingface_hub.dataclasses import strict

from ...audio_utils import AudioInput
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...feature_extraction_utils import BatchFeature
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import (
    AudioKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.chat_template_utils import _get_template_variables
from ...utils.generic import accepts_precomputed_kwargs
from ...video_utils import make_batched_videos
from ..auto import AutoConfig, AutoModel
from ..lfm3_audio.configuration_lfm3_audio import (
    Lfm3AudioAudioConfig,
    Lfm3AudioConfig,
    Lfm3AudioDepthDecoderConfig,
)
from ..lfm3_audio.modeling_lfm3_audio import (
    Lfm3AudioAudioModel,
    Lfm3AudioAudioModelOutput,
    Lfm3AudioCausalLMOutputWithPast,
    Lfm3AudioDepthDecoderForMaskedLM,
    Lfm3AudioDepthDecoderModel,
    Lfm3AudioEncoderCausalConvPaddingCache,
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioModel,
    Lfm3AudioModelOutputWithPast,
    Lfm3AudioMultiModalProjector,
    Lfm3AudioPreTrainedModel,
)
from ..lfm3_audio.processing_lfm3_audio import Lfm3AudioProcessor
from ..lfm3_vl.modeling_lfm3_vl import Lfm3VlMultiModalProjector


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B")
@strict
class Lfm3OmniAudioConfig(Lfm3AudioAudioConfig):
    model_type = "lfm3_omni_audio"


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B")
@strict
class Lfm3OmniDepthDecoderConfig(Lfm3AudioDepthDecoderConfig):
    model_type = "lfm3_omni_depth_decoder"


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B")
@strict
class Lfm3OmniConfig(Lfm3AudioConfig):
    r"""
    audio_config (`Lfm3AudioAudioConfig` or `dict`, *optional*):
        Speech encoder config. `None`: no speech encoder nor projector, no speech input.
    depth_decoder_config (`Lfm3AudioDepthDecoderConfig` or `dict`, *optional*):
        Depth decoder config (bidirectional LFM3 model over the codebooks of a speech frame). `None`: no depth
        decoder nor speech frame embeddings, no speech frames in or out.
    num_codebooks (`int`, *optional*, defaults to 12):
        The number of codebooks of a speech frame.
    codebook_size (`int`, *optional*, defaults to 2017):
        The size of each codebook, the codec's codes plus the end-of-audio code.
    audio_eos_code (`int`, *optional*, defaults to 2016):
        The end-of-audio code. A frame holding it in every codebook ends a speech segment.
    audio_out_token_id (`int`, *optional*, defaults to 125032):
        The id of the placeholder token standing for one output speech frame.
    audio_start_token_id (`int`, *optional*, defaults to 125012):
        The id of `<|audio_start|>`, which switches generation to speech until an end-of-audio frame.
    text_end_token_id (`int`, *optional*, defaults to 125014):
        The id of `<|text_end|>`, which ends the text of an interleaved answer.
    mixed_start_token_id (`int`, *optional*, defaults to 125015):
        The id of `<|mixed_start|>`, which switches generation to interleaved text and speech.
    mixed_end_token_id (`int`, *optional*, defaults to 125016):
        The id of `<|mixed_end|>`, which opens the closing text of a live transcription after its last audio.
    interleaved_n_text (`int`, *optional*, defaults to 6):
        The number of text tokens of a block of an interleaved answer.
    interleaved_n_audio (`int`, *optional*, defaults to 12):
        Speech frames per block of an interleaved answer; also encoder tokens per block of a live transcription.
    maskgit_schedule (`list[int]`, *optional*):
        Codebooks revealed per depth decoder pass, summing to `num_codebooks`. Defaults to three equal passes.
    projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The activation of the projector from the speech encoder to the language model.
    text_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the text loss in the training loss.
    audio_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the speech loss in the training loss.
    vision_config (`Lfm3VlVisionConfig` or `dict`, *optional*):
        The configuration of the vision tower.
    image_token_id (`int`, *optional*, defaults to 125030):
        The id of the placeholder token standing for one projected image token.
    video_token_id (`int`, *optional*, defaults to 125029):
        The id of the placeholder token standing for one projected video token.
    projector_hidden_size (`int`, *optional*, defaults to 4096):
        The hidden size of the vision projector.
    projector_bias (`bool`, *optional*, defaults to `True`):
        Whether the vision projector's linear layers have a bias.
    """

    model_type = "lfm3_omni"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
        "audio_config": SubConfigSpec(config_class=Lfm3OmniAudioConfig, optional=True),
        "depth_decoder_config": SubConfigSpec(config_class=Lfm3OmniDepthDecoderConfig, optional=True),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3_vl_vision"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    projector_hidden_size: int = 4096
    projector_bias: bool = True


class Lfm3OmniEncoderCausalConvPaddingCache(Lfm3AudioEncoderCausalConvPaddingCache):
    pass


class Lfm3OmniAudioModelOutput(Lfm3AudioAudioModelOutput):
    pass


class Lfm3OmniModelOutputWithPast(Lfm3AudioModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor` of shape `(num_audio_tokens, hidden_size)`, *optional*):
        The projected speech encoder outputs scattered into the `audio_token_id` placeholders.
    padding_cache (`Lfm3OmniEncoderCausalConvPaddingCache`, *optional*):
        Subsampling state of the speech encoder, for the next chunk.
    encoder_cache (`Cache`, *optional*):
        Cache of the speech encoder, for the next chunk.
    image_outputs (`BaseModelOutputWithPooling`, *optional*):
        The vision tower outputs of the images, with the projected features as `pooler_output`.
    video_outputs (`BaseModelOutputWithPooling`, *optional*):
        The vision tower outputs of the videos, with the projected features as `pooler_output`.
    """

    image_outputs: BaseModelOutputWithPooling | None = None
    video_outputs: BaseModelOutputWithPooling | None = None


class Lfm3OmniCausalLMOutputWithPast(Lfm3AudioCausalLMOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor` of shape `(num_audio_tokens, hidden_size)`, *optional*):
        The projected speech encoder outputs scattered into the `audio_token_id` placeholders.
    text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        The language modeling loss of the text tokens.
    audio_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `audio_labels` is provided):
        The depth decoder loss of the speech frames.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, kept_length, hidden_size)`):
        The language model's hidden states behind `logits`, sliced like them.
    padding_cache (`Lfm3OmniEncoderCausalConvPaddingCache`, *optional*):
        Subsampling state of the speech encoder, for the next chunk.
    encoder_cache (`Cache`, *optional*):
        Cache of the speech encoder, for the next chunk.
    image_outputs (`BaseModelOutputWithPooling`, *optional*):
        The vision tower outputs of the images, with the projected features as `pooler_output`.
    video_outputs (`BaseModelOutputWithPooling`, *optional*):
        The vision tower outputs of the videos, with the projected features as `pooler_output`.
    """

    image_outputs: BaseModelOutputWithPooling | None = None
    video_outputs: BaseModelOutputWithPooling | None = None


class Lfm3OmniPreTrainedModel(Lfm3AudioPreTrainedModel):
    input_modalities = ("image", "video", "audio", "text")


class Lfm3OmniAudioModel(Lfm3AudioAudioModel):
    pass


class Lfm3OmniDepthDecoderModel(Lfm3AudioDepthDecoderModel):
    pass


class Lfm3OmniDepthDecoderForMaskedLM(Lfm3AudioDepthDecoderForMaskedLM):
    pass


class Lfm3OmniVisionProjector(Lfm3VlMultiModalProjector):
    pass


class Lfm3OmniAudioProjector(Lfm3AudioMultiModalProjector):
    pass


class Lfm3OmniModel(Lfm3AudioModel):
    def __init__(self, config: Lfm3OmniConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        # the checkpoint's names: the vision projector is `multi_modal_projector`, the speech one `audio_projector`
        self.multi_modal_projector = Lfm3OmniVisionProjector(config)
        self.audio_projector = Lfm3OmniAudioProjector(config) if config.audio_config is not None else None
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        padding_cache: Lfm3OmniEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`):
            Log-mel features from [`Lfm3AudioFeatureExtractor`], clips of all samples stacked and right-padded.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask of the valid frames: 1 valid, 0 padding.
        padding_cache (`Lfm3OmniEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.

        Returns:
            [`Lfm3OmniAudioModelOutput`], `pooler_output`: projected valid tokens of all clips, flattened.
        """
        encoder_outputs = self.audio_model(
            input_features,
            input_features_mask=input_features_mask,
            padding_cache=padding_cache,
            cache=encoder_cache,
            use_cache=encoder_cache is not None,
            return_dict=True,
            **kwargs,
        )
        encoded = encoder_outputs.last_hidden_state
        if encoder_outputs.attention_mask is not None:
            encoded = encoded[encoder_outputs.attention_mask.bool().to(encoded.device)]

        encoder_outputs.pooler_output = self.audio_projector(encoded.reshape(-1, encoded.shape[-1]))
        return encoder_outputs

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        image_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_outputs.pooler_output = torch.split(image_features, split_sizes)
        return image_outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains video last hidden states from the vision tower and apply multimodal projection."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        padding_cache: Lfm3OmniEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniModelOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the prompt, one row per `audio_out_token_id` placeholder.
        padding_cache (`Lfm3OmniEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds")
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )
        for modality, token_id in (("image", self.config.image_token_id), ("video", self.config.video_token_id)):
            if mm_encoder_outputs.get(modality) is not None:
                features = torch.cat(mm_encoder_outputs[modality].pooler_output).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                mask = self.get_placeholder_mask(input_ids, inputs_embeds, token_id, features.shape[0])
                inputs_embeds = inputs_embeds.masked_scatter(mask, features)
        audio_embeds = None
        if input_features is not None:
            if self.audio_model is None:
                raise ValueError("`input_features` need the speech encoder, which `audio_config=None` dropped")
            audio_embeds = self.get_audio_features(
                input_features, input_features_mask, padding_cache, encoder_cache, return_dict=True
            ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_token_id, audio_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        if audio_input_ids is not None:
            if self.embed_audio_tokens is None:
                raise ValueError(
                    "`audio_input_ids` need the speech frame embeddings, which `depth_decoder_config=None` dropped"
                )
            audio_out_embeds = self.get_audio_input_embeds(audio_input_ids).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            audio_out_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_out_token_id, audio_out_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_out_mask, audio_out_embeds)
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Lfm3OmniModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
            padding_cache=padding_cache,
            encoder_cache=encoder_cache,
            image_outputs=mm_encoder_outputs.get("image"),
            video_outputs=mm_encoder_outputs.get("video"),
        )


@auto_docstring(
    custom_intro="""
    The LFM3-Omni model: an LFM3 language model that reads speech through the Liquid speech encoder, images and
    videos through the LFM3-VL vision tower, and writes speech as codebook frames through a depth decoder, alongside
    text.
    """
)
class Lfm3OmniForConditionalGeneration(Lfm3AudioForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        audio_labels_mask: torch.BoolTensor | None = None,
        padding_cache: Lfm3OmniEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniCausalLMOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the sequence, one row per `audio_out_token_id` placeholder.
        audio_labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames to learn, one row per `audio_out_token_id` placeholder.
        audio_labels_mask (`torch.BoolTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of `audio_labels` to learn; all of them by default.
        padding_cache (`Lfm3OmniEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_input_ids=audio_input_ids,
            padding_cache=padding_cache,
            encoder_cache=encoder_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = outputs[0]
        kept_hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(kept_hidden_states)

        loss = text_loss = audio_loss = None
        if labels is not None:
            # mean over learned tokens, zero without any
            text_count = (labels[..., 1:] != -100).sum()
            kwargs.setdefault("num_items_in_batch", text_count.clamp(min=1))
            text_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        if audio_labels is not None:
            if self.depth_decoder is None:
                raise ValueError("`audio_labels` need the depth decoder, which `depth_decoder_config` disables")
            if input_ids is None or audio_input_ids is None:
                raise ValueError("`audio_labels` need `input_ids` and `audio_input_ids` to locate the speech frames")

            audio_loss, audio_count = self.audio_loss(
                input_ids, hidden_states, audio_input_ids, audio_labels, audio_labels_mask
            )

        if text_loss is not None and audio_loss is not None:
            # mean over learned text tokens and speech frames, weighted per modality
            device = text_loss.device
            text_weight = self.config.text_loss_weight * text_count.to(device)
            audio_weight = self.config.audio_loss_weight * audio_count.to(device)
            loss = (text_weight * text_loss + audio_weight * audio_loss.to(device)) / (
                text_weight + audio_weight + 1e-6
            )

        return Lfm3OmniCausalLMOutputWithPast(
            loss=loss if loss is not None else text_loss if text_loss is not None else audio_loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
            last_hidden_state=kept_hidden_states,
            padding_cache=outputs.padding_cache,
            encoder_cache=outputs.encoder_cache,
            image_outputs=outputs.image_outputs,
            video_outputs=outputs.video_outputs,
        )


class Lfm3OmniVideosKwargs(VideosKwargs, total=False):
    """
    use_audio_in_video (`bool`, *optional*, defaults to `False`):
        Whether each video comes with its soundtrack: the first audio clip after the video in the text (the one
        `apply_chat_template(load_audio_from_video=True)` puts right behind the video) is interleaved after the
        video's frame groups instead of standing in its own media span.
    soundtrack_tokens (`list[int]`, *optional*):
        The encoder tokens of each video's soundtrack, set by the processor under `use_audio_in_video`.
    """

    use_audio_in_video: bool
    soundtrack_tokens: list[int]


class Lfm3OmniAudioKwargs(AudioKwargs, total=False):
    """
    soundtracks (`list[int]`, *optional*):
        The clips the videos take as their soundtracks, set by the processor under `use_audio_in_video`.
    """

    soundtracks: list[int]


class Lfm3OmniProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Lfm3OmniAudioKwargs
    videos_kwargs: Lfm3OmniVideosKwargs


class Lfm3OmniProcessor(Lfm3AudioProcessor):
    r"""
    The LFM3-Audio processor with the LFM3-VL image and video processors. Every image, input clip, video and video
    frame group is expanded in place and wrapped in the vision span tokens.
    """

    valid_processor_kwargs = Lfm3OmniProcessorKwargs
    videos_kwargs = {"return_metadata": True, "use_audio_in_video": False}

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        video_processor=None,
        codec_feature_extractor=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token: str = "<|audio|>",
        audio_out_token: str = "<|audio_out|>",
        audio_start_token: str = "<|audio_start|>",
        image_token: str = "<|image|>",
        video_token: str = "<|video|>",
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        num_audio_tokens_per_chunk: int = 12,
        max_audio_chunks_per_context: int = 31,
        **kwargs,
    ):
        r"""
        image_processor ([`Lfm3VlImageProcessor`]):
            The image patch processor.
        feature_extractor ([`Lfm3AudioFeatureExtractor`]):
            The feature extractor computing the log-mel spectrograms of the input clips.
        tokenizer ([`PreTrainedTokenizerFast`]):
            The LFM3 tokenizer.
        video_processor ([`Lfm3VlVideoProcessor`]):
            The video frame sampler and patch processor.
        codec_feature_extractor ([`LiquidCodecFeatureExtractor`]):
            The codec feature extractor, used to encode `audio_out` waveforms.
        audio_tokenizer ([`LiquidCodecModel`], *optional*):
            The codec, used to encode `audio_out` waveforms and to decode generated speech.
        chat_template (`str`, *optional*):
            A Jinja template to turn conversations into prompts.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The placeholder token of one speech encoder frame.
        audio_out_token (`str`, *optional*, defaults to `"<|audio_out|>"`):
            The placeholder token of one codec frame.
        audio_start_token (`str`, *optional*, defaults to `"<|audio_start|>"`):
            The token opening a speech segment.
        image_token (`str`, *optional*, defaults to `"<|image|>"`):
            The placeholder token of one projected image token.
        video_token (`str`, *optional*, defaults to `"<|video|>"`):
            The placeholder token of one projected video token.
        vision_start_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            The token opening an image, an input clip, a video and each of its timestamped frame groups.
        vision_end_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            The token closing them.
        num_audio_tokens_per_chunk (`int`, *optional*, defaults to 12):
            The number of speech encoder frames in one chunk of a live transcription.
        max_audio_chunks_per_context (`int`, *optional*, defaults to 31):
            The maximum number of chunks in one live transcription context (30 seconds).
        """
        self.audio_token = getattr(tokenizer, "audio_token", audio_token)
        self.audio_out_token = getattr(tokenizer, "audio_out_token", audio_out_token)
        self.audio_start_token = getattr(tokenizer, "audio_start_token", audio_start_token)
        self.image_token = getattr(tokenizer, "image_token", image_token)
        self.video_token = getattr(tokenizer, "video_token", video_token)
        self.vision_start_token = getattr(tokenizer, "vision_start_token", vision_start_token)
        self.vision_end_token = getattr(tokenizer, "vision_end_token", vision_end_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_out_token_id = tokenizer.convert_tokens_to_ids(self.audio_out_token)
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_start_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.num_audio_tokens_per_chunk = num_audio_tokens_per_chunk
        self.max_audio_chunks_per_context = max_audio_chunks_per_context
        ProcessorMixin.__init__(
            self,
            image_processor,
            feature_extractor,
            tokenizer,
            video_processor,
            codec_feature_extractor,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    def _wrap(self, placeholders: str) -> str:
        return self.vision_start_token + placeholders + self.vision_end_token

    def _strip_spans(self, prompt: str) -> tuple[str, list[dict]]:
        # a chat template may already wrap a placeholder in the span pair; the expansion adds exactly one pair
        wrapped = {
            self.vision_start_token + token + self.vision_end_token: token
            for token in (self.image_token, self.video_token, self.audio_token)
        }
        return self._rewrite_with_offsets(
            prompt, "|".join(map(re.escape, wrapped)), lambda match: wrapped[match.group()], "span"
        )

    @staticmethod
    def _rewrite_with_offsets(prompt: str, pattern: str, replace, kind: str) -> tuple[str, list[dict]]:
        """Rewrites `prompt` like `ProcessorMixin.get_text_with_replacements`, with the same offset records."""
        pieces, row, last, offset = [], [], 0, 0
        for match in re.finditer(pattern, prompt):
            start, end = match.span()
            replacement = replace(match)
            pieces.extend((prompt[last:start], replacement))
            row.append(
                {
                    "type": kind,
                    "span": (start, end),
                    "new_span": (start + offset, start + offset + len(replacement)),
                    "text": match.group(),
                    "replacement": replacement,
                }
            )
            offset += len(replacement) - (end - start)
            last = end
        pieces.append(prompt[last:])
        return "".join(pieces), row

    @staticmethod
    def _compose_replacement_offsets(prompts: list[str], first: list[list[dict]], second) -> list[list[dict]]:
        """The records of two successive rewrites (the caller's text A to B, then B to C) as one rewrite of A: an entry
        of the second that covers entries of the first absorbs them, so every entry spans the caller's text."""
        second = second or [[] for _ in prompts]
        composed = []
        for prompt, row_first, row_second in zip(prompts, first, second):

            def to_first(position, row_first=row_first):
                return position - sum(
                    len(r["replacement"]) - (r["span"][1] - r["span"][0])
                    for r in row_first
                    if r["new_span"][1] <= position
                )

            mapped = []
            for entry in row_second:
                start, end = to_first(entry["span"][0]), to_first(entry["span"][1])
                mapped.append({**entry, "span": (start, end), "text": prompt[start:end]})
            kept = [
                r
                for r in row_first
                if not any(m["span"][0] <= r["span"][0] and r["span"][1] <= m["span"][1] for m in mapped)
            ]
            row, offset = [], 0
            for entry in sorted(kept + mapped, key=lambda r: r["span"]):
                start, end = entry["span"]
                row.append({**entry, "new_span": (start + offset, start + offset + len(entry["replacement"]))})
                offset += len(entry["replacement"]) - (end - start)
            composed.append(row)
        return composed

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        generation_mode=None,
        load_audio_from_video=False,
        **kwargs,
    ):
        r"""
        [`~ProcessorMixin.apply_chat_template`] with the output mode of each answer and the videos' soundtracks.

        Args:
            generation_mode (`str` or `list[str]`, *optional*):
                Marker opening the assistant answer: `"text"` (none), `"audio"` (`<|audio_start|>`) or `"interleaved"`
                (`<|mixed_start|>`). One for all conversations, or one each. Requires `add_generation_prompt=True`.
            load_audio_from_video (`bool`, *optional*, defaults to `False`):
                Whether each video's soundtrack becomes the audio part right behind it, which `use_audio_in_video`
                pairs with the video; standalone clips are kept (the shared implementation drops them).
        """
        if load_audio_from_video and kwargs.get("tokenize"):
            load_audio_from_video = False
            conversation = self._with_soundtrack_parts(conversation)
        if generation_mode is not None:
            if not kwargs.get("add_generation_prompt"):
                raise ValueError(
                    "`generation_mode` opens the assistant answer: it requires `add_generation_prompt=True`."
                )
            chat_template = self._resolve_chat_template(chat_template)
            if "generation_mode" not in _get_template_variables(chat_template):
                raise ValueError("This chat template does not support `generation_mode`.")
            is_batched = isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
            conversations = conversation if is_batched else [conversation]
            if isinstance(generation_mode, str):
                generation_mode = [generation_mode] * len(conversations)
            elif not is_batched or len(generation_mode) != len(conversations):
                raise ValueError(
                    f"Got {len(generation_mode)} generation modes for {len(conversations)} conversations: pass one "
                    "mode, or one per conversation."
                )
            for mode in generation_mode:
                if mode not in ("text", "audio", "interleaved"):
                    raise ValueError(f'`generation_mode` must be "text", "audio" or "interleaved", got {mode!r}.')
            # a list can not be a template kwarg: the mode goes on each conversation's last message
            conversations = [
                [*messages[:-1], {**messages[-1], "generation_mode": mode}]
                for messages, mode in zip((getattr(chat, "messages", chat) for chat in conversations), generation_mode)
            ]
            conversation = conversations if is_batched else conversations[0]
        return ProcessorMixin.apply_chat_template(
            self, conversation, chat_template=chat_template, load_audio_from_video=load_audio_from_video, **kwargs
        )

    @staticmethod
    def _with_soundtrack_parts(conversation):
        if not conversation:
            return conversation
        batched = isinstance(conversation[0], (list, tuple))
        rewritten = []
        for messages in conversation if batched else [conversation]:
            rewritten.append([])
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    parts, track_added = [], False
                    for part in content:
                        # a sourceless audio part right behind a video is that video's soundtrack slot
                        if track_added and part.get("type") == "audio" and not part.keys() & {"audio", "url", "path"}:
                            track_added = False
                            continue
                        parts.append(part)
                        source = next((part[key] for key in ("video", "url", "path") if key in part), None)
                        track_added = part.get("type") == "video" and source is not None
                        if track_added:
                            parts.append({"type": "audio", "audio": source})
                    message = {**message, "content": parts}
                rewritten[-1].append(message)
        return rewritten if batched else rewritten[0]

    def _soundtracks(self, prompts: list[str]) -> list[int]:
        """`use_audio_in_video`: the clip each video takes as its soundtrack, the first audio placeholder after it."""
        soundtracks, first_clip = [], 0
        for prompt in prompts:
            audio = [match.start() for match in re.finditer(re.escape(self.audio_token), prompt)]
            taken = set()
            for video in re.finditer(re.escape(self.video_token), prompt):
                clip = next(
                    (i for i, position in enumerate(audio) if position >= video.end() and i not in taken), None
                )
                if clip is None:
                    raise ValueError("`use_audio_in_video` needs an audio clip after each video")
                taken.add(clip)
                soundtracks.append(first_clip + clip)
            first_clip += len(audio)
        return soundtracks

    def _num_audio_tokens(self, num_samples: int) -> int:
        extractor = self.feature_extractor
        num_frames = 1 + (max(num_samples - extractor.n_fft, 0) + extractor.hop_length - 1) // extractor.hop_length
        return -(num_frames // -extractor.encoder_stride)

    @auto_docstring
    def __call__(
        self,
        text=None,
        audio: AudioInput | None = None,
        audio_out: AudioInput | None = None,
        images=None,
        videos=None,
        is_last_audio_chunk: bool | None = None,
        output_labels: bool | None = None,
        **kwargs,
    ):
        r"""
        audio_out (`AudioInput` or `torch.LongTensor`, or a list of them, *optional*):
            Speech segments, one per `audio_out_token` in the text: 24 kHz waveforms (encoded by the codec, end-of-speech
            frame appended) or codes `(num_codebooks, num_frames)`. Returned as `audio_input_ids`
            `(num_frames, num_codebooks)`.
        videos (`VideoInput`, *optional*):
            One clip per `video_token`. With `use_audio_in_video=True` each video takes the first audio clip after it
            as its soundtrack, interleaved after each timestamped frame group; otherwise every audio clip is its own
            media span.
        is_last_audio_chunk (`bool`, *optional*):
            Live transcription: whether `audio` is the last chunk of the stream. Returned as is.
        output_labels (`bool`, *optional*):
            Whether to return `labels` (`input_ids` with placeholders, `audio_start_token` and padding at `-100`) and,
            with `audio_out`, `audio_labels` (copy of `audio_input_ids`). Requires `return_tensors="pt"`.
        """
        caller, offsets = None, None
        if text is not None:
            single = isinstance(text, str)
            caller = [text] if single else list(text)
            stripped = [self._strip_spans(prompt) for prompt in caller]
            prompts, offsets = [prompt for prompt, _ in stripped], [row for _, row in stripped]
            text = prompts[0] if single else prompts
            merged = self._merge_kwargs(
                self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
            )
            if videos is not None and merged["videos_kwargs"]["use_audio_in_video"]:
                soundtracks = self._soundtracks(prompts)
                if audio is None:
                    raise ValueError("`use_audio_in_video` needs an audio clip after each video")
                # the video expansion splits its soundtrack's encoder tokens across the frame groups, so it needs
                # their count; the soundtrack's own placeholder then expands to nothing
                _, _, _, audio = self.prepare_inputs_layout(audio=audio, **kwargs)
                kwargs = {
                    **kwargs,
                    "videos_kwargs": {
                        **kwargs.get("videos_kwargs", {}),
                        "soundtrack_tokens": [self._num_audio_tokens(len(audio[clip])) for clip in soundtracks],
                    },
                    "audio_kwargs": {**kwargs.get("audio_kwargs", {}), "soundtracks": soundtracks},
                }

        audio_input_ids = None
        if audio_out is not None:
            if text is None:
                raise ValueError("`audio_out` requires `text`, which holds its placeholders.")
            audio_input_ids = self._prepare_audio_out(audio_out)
            self._audio_out_replacements = [self.audio_out_token * len(frames) for frames in audio_input_ids]

        try:
            outputs = ProcessorMixin.__call__(self, text=text, audio=audio, images=images, videos=videos, **kwargs)
        finally:
            self._audio_out_replacements = []

        if offsets is not None and "text_replacement_offsets" in outputs:
            # the expansion saw the prompts after the span pairs were stripped
            outputs["text_replacement_offsets"] = self._compose_replacement_offsets(
                caller, offsets, outputs["text_replacement_offsets"]
            )

        if audio_input_ids is not None:
            audio_input_ids = torch.cat(audio_input_ids)
            self._check_audio_out_not_truncated(outputs["input_ids"], num_frames=len(audio_input_ids))
            merged_kwargs = self._merge_kwargs(tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs)
            return_tensors = merged_kwargs["text_kwargs"].get("return_tensors")
            outputs["audio_input_ids"] = BatchFeature({"ids": audio_input_ids}, tensor_type=return_tensors)["ids"]

        if is_last_audio_chunk is not None:
            outputs["is_last_audio_chunk"] = is_last_audio_chunk

        if output_labels:
            if not torch.is_tensor(outputs["input_ids"]):
                raise ValueError('`output_labels` requires `return_tensors="pt"`.')
            labels = outputs["input_ids"].clone()
            for token_id in (self.audio_token_id, self.audio_out_token_id, self.audio_start_token_id):
                labels[labels == token_id] = -100
            for token_id in (self.image_token_id, self.video_token_id):
                labels[labels == token_id] = -100
            labels[outputs["attention_mask"] == 0] = -100
            outputs["labels"] = labels
            if audio_input_ids is not None:
                outputs["audio_labels"] = outputs["audio_input_ids"].clone()

        return outputs

    def _process_videos(self, videos, use_audio_in_video=False, soundtrack_tokens=(), **kwargs):
        # the placeholder timestamps need the metadata; `__call__` still drops it from the output unless asked for
        video_inputs = self.video_processor(videos, **{**kwargs, "return_metadata": True})
        soundtrack_tokens = list(soundtrack_tokens) or [None] * len(make_batched_videos(videos))
        replacements = [
            self.replace_video_token(video_inputs, index, soundtrack_tokens=tokens)
            for index, tokens in enumerate(soundtrack_tokens)
        ]
        return video_inputs, replacements

    def _process_audio(self, audio, soundtracks=(), **kwargs):
        kwargs.pop("load_audio_backend", None)
        audio_inputs = self.feature_extractor(audio, **kwargs)
        # a soundtrack's tokens follow its video's frame groups instead
        replacements = [
            "" if index in soundtracks else self.replace_audio_token(audio_inputs, index)
            for index in range(len(audio_inputs["input_features_mask"]))
        ]
        return audio_inputs, replacements

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        count = image_inputs["image_grid_thw"][image_idx].prod() // self.image_processor.merge_size**2
        return self._wrap(self.image_token * int(count))

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        # one token per `encoder_stride` frames, rounded up
        num_frames = int(audio_inputs["input_features_mask"][audio_idx].sum())
        return self._wrap(self.audio_token * -(num_frames // -self.feature_extractor.encoder_stride))

    def replace_video_token(
        self, video_inputs: dict, video_idx: int, soundtrack_tokens: int | None = None, **kwargs
    ) -> str:
        """One timestamped, wrapped group of placeholders per temporal patch, wrapped again as a whole; with a
        soundtrack, its encoder tokens follow the groups, split at each group's first frame."""
        grid = video_inputs["video_grid_thw"][video_idx]
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None:
            logger.warning_once(
                "LFM3-Omni requires frame timestamps to construct prompts, but the `fps` of the input video could not "
                "be inferred. Defaulting to `fps=24`; pass `video_metadata` for accurate timestamps."
            )
        fps = 24 if metadata.fps is None else metadata.fps
        num_groups, patch_tokens = int(grid[0]), int(grid[1:].prod() // self.video_processor.merge_size**2)
        temporal = self.video_processor.temporal_patch_size
        timestamps = self._calculate_timestamps(metadata.frames_indices, fps, temporal)[:num_groups]
        bounds = [0] * (num_groups + 1)
        if soundtrack_tokens is not None:
            starts = [index / fps for index in list(metadata.frames_indices)[::temporal]][:num_groups]
            extractor = self.feature_extractor
            seconds_per_token = extractor.hop_length * extractor.encoder_stride / extractor.sampling_rate
            bounds = [0] + [min(soundtrack_tokens, round(start / seconds_per_token)) for start in starts[1:]]
            bounds = [max(bounds[: i + 1]) for i in range(len(bounds))] + [soundtrack_tokens]
        groups = ""
        for index, timestamp in enumerate(timestamps):
            groups += f"<{timestamp:.1f} seconds>" + self._wrap(self.video_token * patch_tokens)
            groups += self.audio_token * (bounds[index + 1] - bounds[index])
        return self._wrap(groups)

    @staticmethod
    def _calculate_timestamps(indices, video_fps: float, temporal_patch_size: int) -> list[float]:
        indices = list(indices)
        if pad := -len(indices) % temporal_patch_size:
            indices.extend([indices[-1]] * pad)
        return [
            (indices[i] + indices[i + temporal_patch_size - 1]) / (2 * video_fps)
            for i in range(0, len(indices), temporal_patch_size)
        ]

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.feature_extractor.model_input_names
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
                + ["audio_input_ids"]
            )
        )


__all__ = [
    "Lfm3OmniProcessor",
    "Lfm3OmniConfig",
    "Lfm3OmniAudioConfig",
    "Lfm3OmniAudioModel",
    "Lfm3OmniDepthDecoderConfig",
    "Lfm3OmniDepthDecoderModel",
    "Lfm3OmniDepthDecoderForMaskedLM",
    "Lfm3OmniPreTrainedModel",
    "Lfm3OmniModel",
    "Lfm3OmniForConditionalGeneration",
]
