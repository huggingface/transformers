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

import numpy as np
import torch
from huggingface_hub.dataclasses import strict

from ...audio_utils import AudioInput
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import accepts_precomputed_kwargs
from ...video_utils import make_batched_videos
from ..auto import AutoConfig, AutoModel
from ..lfm3_audio.configuration_lfm3_audio import Lfm3AudioConfig, Lfm3AudioEncoderConfig
from ..lfm3_audio.modeling_lfm3_audio import (
    Lfm3AudioCausalLMOutputWithPast,
    Lfm3AudioForConditionalGeneration,
    Lfm3AudioModel,
    Lfm3AudioModelOutputWithPast,
    Lfm3AudioPreTrainedModel,
)
from ..lfm3_audio.processing_lfm3_audio import Lfm3AudioFeatureExtractorKwargs, Lfm3AudioProcessor
from ..lfm3_vl.modeling_lfm3_vl import Lfm3VlMultiModalProjector


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B-Preview")
@strict
class Lfm3OmniEncoderConfig(Lfm3AudioEncoderConfig):
    pass


@auto_docstring(checkpoint="LiquidAI/LFM3-Omni-2B-Preview")
@strict
class Lfm3OmniConfig(Lfm3AudioConfig):
    r"""
    audio_encoder_config (`Lfm3OmniEncoderConfig` or `dict`, *optional*):
        The configuration of the speech encoder.
    depth_config (`Lfm3Config` or `dict`, *optional*):
        The configuration of the depth decoder, a bidirectional LFM3 model over the codebook axis of one speech
        frame. Defaults to the 3-layer, 1024-wide model of LFM3-Audio.
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
    text_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the text loss in the training loss.
    audio_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the speech loss in the training loss.
    projector_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon of the layer normalization applied to the vision features before projection.
    """

    model_type = "lfm3_omni"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
        "audio_encoder_config": SubConfigSpec(config_class=Lfm3OmniEncoderConfig),
        "depth_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="lfm3",
            init_kwargs={
                "vocab_size": 1,
                "pad_token_id": None,
                "bos_token_id": None,
                "eos_token_id": None,
                "hidden_size": 1024,
                "intermediate_size": 2816,
                "num_hidden_layers": 3,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 32,
                "max_position_embeddings": 12,
                "layer_types": ["full_attention"] * 3,
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
                },
                "sliding_window": None,
                "use_sandwich_norm": False,
                "use_cache": False,
            },
        ),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3_vl_vision"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6

    _token_id_fields = (
        "audio_token_id",
        "audio_out_token_id",
        "audio_start_token_id",
        "text_end_token_id",
        "mixed_start_token_id",
        "image_token_id",
        "video_token_id",
    )


class Lfm3OmniModelOutputWithPast(Lfm3AudioModelOutputWithPast):
    pass


class Lfm3OmniCausalLMOutputWithPast(Lfm3AudioCausalLMOutputWithPast):
    pass


class Lfm3OmniPreTrainedModel(Lfm3AudioPreTrainedModel):
    input_modalities = ("image", "video", "audio", "text")


class Lfm3OmniMultiModalProjector(Lfm3VlMultiModalProjector):
    pass


class Lfm3OmniModel(Lfm3AudioModel):
    def __init__(self, config: Lfm3OmniConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Lfm3OmniMultiModalProjector(config)
        self.post_init()

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
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3OmniModelOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the prompt, one row per `audio_out_token_id` placeholder. A speech
            placeholder without its features or codes is refused, except in a compiled or captured forward, which
            skips the check.
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
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be supplied together")
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
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring(
    custom_intro="""
    The LFM3-Omni model: an LFM3 language model that reads speech through the Liquid speech encoder, images and
    videos through the LFM3-VL vision tower, and writes speech as codebook frames through a depth decoder, alongside
    text.
    """
)
class Lfm3OmniForConditionalGeneration(Lfm3AudioForConditionalGeneration):
    @property
    def _placeholder_token_ids(self) -> list[int]:
        return [
            self.config.audio_token_id,
            self.config.audio_out_token_id,
            self.config.image_token_id,
            self.config.video_token_id,
        ]

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
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
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
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the sequence, one row per `audio_out_token_id` placeholder. A speech
            placeholder without its features or codes is refused, except in a compiled or captured forward, which
            skips the check.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        audio_labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames to learn, one row per `audio_out_token_id` placeholder; rows of `-100` are
            ignored. The depth decoder predicts a random subset of each frame's codebooks from the others and the
            hidden state preceding the frame.
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        # the vision inputs are named for `generate`, which pre-encodes them; the speech head forwards extra kwargs
        kwargs.update(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_encoder_outputs=mm_encoder_outputs,
        )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            labels=labels,
            audio_labels=audio_labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
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


class Lfm3OmniAudioProcessorKwargs(Lfm3AudioFeatureExtractorKwargs):
    """
    sampling_rate (`int`, *optional*):
        The sampling rate of the clips, which must be the extractor's own; it is checked, never resampled.
    return_tensors (`str` or [`~utils.TensorType`], *optional*):
        The tensor type of the returned features.
    load_audio_backend (`str`, *optional*):
        The decoding backend `apply_chat_template` loads audio with; the extractor itself ignores it.
    soundtracks (`list[int]`, *optional*):
        The clips the videos take as their soundtracks, set by the processor under `use_audio_in_video`.
    """

    soundtracks: list[int]


class Lfm3OmniProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Lfm3OmniAudioProcessorKwargs
    videos_kwargs: Lfm3OmniVideosKwargs
    _defaults = {"videos_kwargs": {"return_metadata": True, "use_audio_in_video": False}}


class Lfm3OmniProcessor(Lfm3AudioProcessor):
    r"""
    The LFM3-Audio processor with the LFM3-VL image and video processors. Every image, input clip, video and video
    frame group is expanded in place and wrapped in the vision span tokens.
    """

    valid_processor_kwargs = Lfm3OmniProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        video_processor=None,
        audio_tokenizer=None,
        chat_template=None,
        audio_token: str = "<|audio|>",
        audio_out_token: str = "<|audio_out|>",
        image_token: str = "<|image|>",
        video_token: str = "<|video|>",
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        **kwargs,
    ):
        r"""
        image_processor ([`Qwen2VLImageProcessor`]):
            The image patch processor.
        feature_extractor ([`Lfm3AudioFeatureExtractor`]):
            The feature extractor computing the log-mel spectrograms.
        tokenizer ([`PreTrainedTokenizerFast`]):
            The LFM3 tokenizer.
        video_processor ([`Qwen3VLVideoProcessor`]):
            The video frame sampler and patch processor.
        audio_tokenizer ([`LiquidCodecModel`], *optional*):
            The codec turning 24 kHz output clips into speech frames.
        chat_template (`str`, *optional*):
            A Jinja template to turn conversations into prompts.
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            The placeholder token of one encoded frame of an input clip.
        audio_out_token (`str`, *optional*, defaults to `"<|audio_out|>"`):
            The placeholder token of one speech frame of the prompt.
        image_token (`str`, *optional*, defaults to `"<|image|>"`):
            The placeholder token of one projected image token.
        video_token (`str`, *optional*, defaults to `"<|video|>"`):
            The placeholder token of one projected video token.
        vision_start_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            The token opening an image, an input clip, a video and each of its timestamped frame groups.
        vision_end_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            The token closing them.
        """
        self.audio_token = getattr(tokenizer, "audio_token", audio_token)
        self.audio_out_token = getattr(tokenizer, "audio_out_token", audio_out_token)
        self.image_token = getattr(tokenizer, "image_token", image_token)
        self.video_token = getattr(tokenizer, "video_token", video_token)
        self.vision_start_token = getattr(tokenizer, "vision_start_token", vision_start_token)
        self.vision_end_token = getattr(tokenizer, "vision_end_token", vision_end_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_out_token_id = tokenizer.convert_tokens_to_ids(self.audio_out_token)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        ProcessorMixin.__init__(
            self,
            image_processor,
            feature_extractor,
            tokenizer,
            video_processor,
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

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        tools=None,
        documents=None,
        add_generation_prompt=False,
        continue_final_message=False,
        return_assistant_tokens_mask=False,
        tokenize=False,
        return_tensors=None,
        return_dict=False,
        load_audio_from_video=False,
        processor_kwargs=None,
        **kwargs,
    ):
        """With `load_audio_from_video=True`, each video's soundtrack becomes the audio part right behind it, which
        `use_audio_in_video` pairs with the video; standalone clips are kept (the shared implementation drops them)."""
        if load_audio_from_video and tokenize:
            load_audio_from_video = False
            conversation = self._with_soundtrack_parts(conversation)
        return ProcessorMixin.apply_chat_template(
            self,
            conversation,
            chat_template,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenize=tokenize,
            return_tensors=return_tensors,
            return_dict=return_dict,
            load_audio_from_video=load_audio_from_video,
            processor_kwargs=processor_kwargs,
            **kwargs,
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

    @auto_docstring
    def __call__(
        self,
        text=None,
        audio: AudioInput | None = None,
        audio_out: AudioInput | None = None,
        images=None,
        videos=None,
        **kwargs,
    ):
        r"""
        audio_out (`np.ndarray`, `torch.Tensor` or a list of them, *optional*):
            24 kHz clips, one per `audio_out_token` in the text, encoded with the codec into `audio_out_codes`; the
            placeholder is repeated once per frame plus once for the end-of-speech frame that closes every clip.
        videos (`VideoInput`, *optional*):
            One clip per `video_token`. With `use_audio_in_video=True` each video takes the first audio clip after it
            as its soundtrack, interleaved after each timestamped frame group; otherwise every audio clip is its own
            media span.
        """
        for name, media in (("audio", audio), ("audio_out", audio_out), ("images", images), ("videos", videos)):
            if isinstance(media, (list, tuple)) and len(media) == 0:
                raise ValueError(f"`{name}` must hold at least one item; pass `{name}=None` for prompts without one")
        if text is None:
            if audio_out is not None:
                raise ValueError("`audio_out` needs the text holding its placeholders")
            return ProcessorMixin.__call__(self, audio=audio, images=images, videos=videos, **kwargs)
        single = isinstance(text, str)
        caller = [text] if single else list(text)
        stripped = [self._strip_spans(prompt) for prompt in caller]
        prompts, offsets = [prompt for prompt, _ in stripped], [row for _, row in stripped]
        merged = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        audio_out_codes = None
        if audio_out is not None:
            prompts, audio_out_codes, audio_out_offsets = self._expand_audio_out(prompts, audio_out)
            offsets = self._compose_replacement_offsets(caller, offsets, audio_out_offsets)
        if videos is not None and merged["videos_kwargs"]["use_audio_in_video"]:
            soundtracks = self._soundtracks(prompts)
            if audio is None:
                raise ValueError("`use_audio_in_video` needs an audio clip after each video")
            # the video expansion splits its soundtrack's encoder tokens across the frame groups, so it needs their
            # count; the soundtrack's own placeholder then expands to nothing
            _, _, _, audio = self.prepare_inputs_layout(audio=audio, **kwargs)
            extractor = self.feature_extractor
            soundtrack_tokens = [
                extractor.num_encoder_tokens(extractor.num_frames(len(audio[clip]))) for clip in soundtracks
            ]
            kwargs = {
                **kwargs,
                "videos_kwargs": {**kwargs.get("videos_kwargs", {}), "soundtrack_tokens": soundtrack_tokens},
                "audio_kwargs": {**kwargs.get("audio_kwargs", {}), "soundtracks": soundtracks},
            }
        outputs = ProcessorMixin.__call__(
            self, text=prompts[0] if single else prompts, audio=audio, images=images, videos=videos, **kwargs
        )
        if "text_replacement_offsets" in outputs:
            # the expansion saw the prompts after the span pairs were stripped and the speech placeholders repeated
            outputs["text_replacement_offsets"] = self._compose_replacement_offsets(
                caller, offsets, outputs["text_replacement_offsets"]
            )
        if audio_out_codes is not None:
            self._check_audio_out_placeholders(outputs["input_ids"], audio_out_codes.shape[0])
            outputs["audio_out_codes"] = self._as_return_tensors(
                audio_out_codes, merged["text_kwargs"].get("return_tensors")
            )
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
        lengths = np.asarray(audio_inputs["feature_attention_mask"]).sum(-1).tolist()
        audio_inputs["num_audio_tokens"] = [
            self.feature_extractor.num_encoder_tokens(int(length)) for length in lengths
        ]
        # a soundtrack's tokens follow its video's frame groups instead
        replacements = [
            "" if index in soundtracks else self.replace_audio_token(audio_inputs, index)
            for index in range(len(lengths))
        ]
        return audio_inputs, replacements

    def replace_image_token(self, image_inputs, image_idx, **kwargs):
        count = image_inputs["image_grid_thw"][image_idx].prod() // self.image_processor.merge_size**2
        return self._wrap(self.image_token * int(count))

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self._wrap(self.audio_token * audio_inputs["num_audio_tokens"][audio_idx])

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
                + ["audio_out_codes"]
            )
        )


__all__ = [
    "Lfm3OmniProcessor",
    "Lfm3OmniConfig",
    "Lfm3OmniEncoderConfig",
    "Lfm3OmniModel",
    "Lfm3OmniPreTrainedModel",
    "Lfm3OmniForConditionalGeneration",
]
