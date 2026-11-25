# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...audio_utils import AudioInput
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import MultiModalData, ProcessorMixin, Unpack
from ...utils import TransformersKwargs, auto_docstring, torch_compilable_check
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..audioflamingo3.configuration_audioflamingo3 import AudioFlamingo3EncoderConfig
from ..auto import AutoConfig, AutoModel
from ..minicpmv2_6.configuration_minicpmv2_6 import MiniCPMV2_6Config, MiniCPMV2_6VisionConfig
from ..minicpmv2_6.modeling_minicpmv2_6 import (
    MiniCPMV2_6ForConditionalGeneration,
    MiniCPMV2_6Model,
    MiniCPMV2_6PreTrainedModel,
    MiniCPMV2_6VisionTransformer,
)
from ..minicpmv2_6.processing_minicpmv2_6 import MiniCPMV2_6Processor
from ..nemotron_asr_streaming.modeling_nemotron_asr_streaming import chunked_limited_mask_function
from ..qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention, Qwen2AudioEncoder, Qwen2AudioEncoderLayer


@auto_docstring(checkpoint="openbmb/MiniCPM-o-2_6")
@strict
class MiniCPMO2_6VisionConfig(MiniCPMV2_6VisionConfig):
    model_type = "minicpmo2_6_vision"


@auto_docstring(checkpoint="openbmb/MiniCPM-o-2_6")
@strict
class MiniCPMO2_6AudioConfig(AudioFlamingo3EncoderConfig):
    r"""
    max_source_positions (`int`, *optional*, defaults to 1500):
        The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
    chunk_size (`int`, *optional*, defaults to 50):
        Number of encoder frames (50 per second of audio) per attention chunk. Each frame attends to the frames of its
        own chunk and of all earlier chunks.

    Example:

    ```python
    >>> from transformers import MiniCPMO2_6AudioConfig, MiniCPMO2_6AudioEncoder

    >>> configuration = MiniCPMO2_6AudioConfig()
    >>> model = MiniCPMO2_6AudioEncoder(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmo2_6_audio"
    base_config_key = "audio_config"

    num_mel_bins: int = 80
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_size: int = 1024
    chunk_size: int = 50


@auto_docstring(checkpoint="openbmb/MiniCPM-o-2_6")
@strict
class MiniCPMO2_6Config(MiniCPMV2_6Config):
    r"""
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of learned latent queries of the resampler, i.e. the number of tokens each image slice becomes.
    resampler_n_heads (`int`, *optional*, defaults to 28):
        Number of attention heads of the resampler's cross-attention.
    image_token_id (`int`, *optional*, defaults to 128244):
        Token id of the image placeholder (`<unk>`), repeated `resampler_n_latents` times per image slice.
    audio_token_id (`int`, *optional*, defaults to 151686):
        Token id of the audio placeholder (`<|audio|>`), repeated once per audio token (25 per second of audio).

    Example:

    ```python
    >>> from transformers import MiniCPMO2_6Config, MiniCPMO2_6ForConditionalGeneration

    >>> configuration = MiniCPMO2_6Config()
    >>> model = MiniCPMO2_6ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "minicpmo2_6"
    sub_configs_defaults = {
        "vision_config": MiniCPMO2_6VisionConfig,
        "audio_config": MiniCPMO2_6AudioConfig,
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="qwen2",
            init_kwargs={
                "vocab_size": 151700,
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
                "bos_token_id": 151643,
                "eos_token_id": 151645,
            },
        ),
    }

    audio_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151686


class MiniCPMO2_6PreTrainedModel(MiniCPMV2_6PreTrainedModel):
    input_modalities = ("image", "audio", "text")
    _no_split_modules = ["MiniCPMO2_6EncoderLayer", "MiniCPMO2_6Resampler", "MiniCPMO2_6AudioEncoderLayer"]
    # The checkpoint also holds the speech decoder (`tts.*`), which this model doesn't include
    _keys_to_ignore_on_load_unexpected = [r"^tts\."]


class MiniCPMO2_6VisionTransformer(MiniCPMV2_6VisionTransformer):
    pass


class MiniCPMO2_6AudioAttention(Qwen2AudioAttention):
    pass


class MiniCPMO2_6AudioEncoderLayer(Qwen2AudioEncoderLayer):
    def __init__(self, config: MiniCPMO2_6AudioConfig):
        super().__init__(config)
        self.self_attn = MiniCPMO2_6AudioAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )


@auto_docstring(
    custom_intro="""
    The audio encoder of MiniCPM-o 2.6, without any head or projection on top.
    """
)
class MiniCPMO2_6AudioEncoder(Qwen2AudioEncoder):
    """
    Whisper encoder over up to 30 s of audio per row; each frame attends to its own 1 s chunk and to all earlier ones.
    """

    config: MiniCPMO2_6AudioConfig
    # The chunked attention mask can't be expressed with flash attention's padding-only masks
    _supports_flash_attn = False
    _can_record_outputs = {
        "hidden_states": MiniCPMO2_6AudioEncoderLayer,
        "attentions": MiniCPMO2_6AudioAttention,
    }

    def __init__(self, config: MiniCPMO2_6AudioConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([MiniCPMO2_6AudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        del self.avg_pooler

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, num_frames)`):
            Log-mel features of up to 30 s of audio per row, as returned by [`WhisperFeatureExtractor`].
        attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`):
            Which mel frames are real (`1`) rather than padding (`0`).
        """
        # Padding frames and their conv outputs are zeroed, so each row is encoded as if it ended at its last frame
        frame_mask = attention_mask[:, None, :].to(input_features.dtype)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features * frame_mask)) * frame_mask
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds)).permute(0, 2, 1)
        hidden_states = inputs_embeds + self.embed_positions.weight[: inputs_embeds.shape[1]]
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        lengths, _ = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=torch.arange(hidden_states.shape[1], device=hidden_states.device) < lengths[:, None],
            and_mask_function=chunked_limited_mask_function(left_ctx=-1, right_ctx=self.config.chunk_size - 1),
        )
        for encoder_layer in self.layers:
            if self.training and torch.rand([]) < self.layerdrop:
                continue
            hidden_states = encoder_layer(hidden_states, attention_mask, **kwargs)

        return BaseModelOutputWithPooling(last_hidden_state=self.layer_norm(hidden_states))


class MiniCPMO2_6AudioProjector(nn.Module):
    """
    Projects the audio encoder's frames to the language model's width, then averages each pair of frames (25 audio
    tokens per second).
    """

    def __init__(self, config: MiniCPMO2_6Config):
        super().__init__()
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_2(self.act(self.linear_1(audio_features)))
        return nn.functional.avg_pool1d(hidden_states.transpose(1, 2), kernel_size=2).transpose(1, 2)


@auto_docstring(
    custom_intro="""
    The MiniCPM-o 2.6 model (vision tower and resampler, audio encoder and projector, language model), without a
    language modeling head.
    """
)
class MiniCPMO2_6Model(MiniCPMV2_6Model):
    def __init__(self, config: MiniCPMO2_6Config):
        super().__init__(config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.audio_projector = MiniCPMO2_6AudioProjector(config)

    @can_return_tuple
    @auto_docstring(custom_intro="Encodes audio windows and projects them to 25 tokens per second of audio.")
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_windows, num_mel_bins, num_frames)`):
            Log-mel features of every audio of the batch, split into windows of up to 30 s.
        input_features_mask (`torch.Tensor` of shape `(num_windows, num_frames)`):
            Which mel frames are real (`1`) rather than padding (`0`).
        """
        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype), attention_mask=input_features_mask, return_dict=True, **kwargs
        )
        audio_embeds = self.audio_projector(audio_outputs.last_hidden_state)
        _, num_tokens = self.audio_tower._get_feat_extract_output_lengths(input_features_mask.sum(-1))
        valid_mask = torch.arange(audio_embeds.shape[1], device=audio_embeds.device) < num_tokens[:, None]
        audio_outputs.pooler_output = audio_embeds[valid_mask]
        return audio_outputs

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, features: torch.Tensor, token_id: int
    ) -> torch.Tensor:
        """
        Obtains the placeholder mask of `token_id` from `input_ids` or `inputs_embeds`, and checks that the placeholder
        token count matches the number of `features`.
        """
        if input_ids is None:
            special_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mask = special_mask.all(-1)
        else:
            special_mask = input_ids == token_id
        n_tokens = special_mask.sum()
        special_mask = special_mask.unsqueeze(-1).to(inputs_embeds.device)
        torch_compilable_check(
            n_tokens * inputs_embeds.shape[-1] == features.numel(),
            f"Multimodal features and tokens do not match, tokens: {n_tokens}, features: {features.numel() // inputs_embeds.shape[-1]}",
        )
        return special_mask

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.
        input_features_mask (`torch.Tensor` of shape `(num_windows, num_frames)`, *optional*):
            Which mel frames of `input_features` are real (`1`) rather than padding (`0`).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = dict(mm_encoder_outputs) if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                patch_position_ids=patch_position_ids,
                return_dict=True,
                **kwargs,
            )
        if mm_encoder_outputs.get("audio") is None and input_features is not None:
            mm_encoder_outputs["audio"] = self.get_audio_features(
                input_features=input_features, input_features_mask=input_features_mask, return_dict=True, **kwargs
            )

        for modality, token_id in (("image", self.config.image_token_id), ("audio", self.config.audio_token_id)):
            if mm_encoder_outputs.get(modality) is not None:
                features = mm_encoder_outputs[modality].pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
                mask = self.get_placeholder_mask(input_ids, inputs_embeds, features, token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask, features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return MultimodalBaseModelOutputWithPast(
            **outputs, image_outputs=mm_encoder_outputs.get("image"), audio_outputs=mm_encoder_outputs.get("audio")
        )


@auto_docstring(
    custom_intro="""
    The MiniCPM-o 2.6 model (vision tower and resampler, audio encoder and projector, language model) with a language
    modeling head.
    """
)
class MiniCPMO2_6ForConditionalGeneration(MiniCPMV2_6ForConditionalGeneration):
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_windows, num_mel_bins, num_frames)`):
            Log-mel features of every audio of the batch, split into windows of up to 30 s.
        input_features_mask (`torch.Tensor` of shape `(num_windows, num_frames)`):
            Which mel frames are real (`1`) rather than padding (`0`).
        """
        return self.model.get_audio_features(
            input_features=input_features, input_features_mask=input_features_mask, **kwargs
        )

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_attention_mask: torch.Tensor | None = None,
        patch_position_ids: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(num_slices, max_patches)`, *optional*):
            Which patches of `pixel_values` are real (`True`) rather than padding.
        patch_position_ids (`torch.LongTensor` of shape `(num_slices, max_patches, 2)`, *optional*):
            `(row, col)` of each patch in its slice's patch grid.
        input_features_mask (`torch.Tensor` of shape `(num_windows, num_frames)`, *optional*):
            Which mel frames of `input_features` are real (`1`) rather than padding (`0`).

        Example:

        ```python
        >>> from transformers import AutoModelForMultimodalLM, AutoProcessor

        >>> model = AutoModelForMultimodalLM.from_pretrained("openbmb/MiniCPM-o-2_6", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-o-2_6")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"},
        ...             {"type": "text", "text": "Transcribe this audio."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> generated = model.generate(**inputs, max_new_tokens=32)
        >>> processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            patch_position_ids=patch_position_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = slice_logits_to_keep(outputs[0], logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        return MultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
            audio_outputs=outputs.audio_outputs,
        )


@auto_docstring
class MiniCPMO2_6Processor(MiniCPMV2_6Processor):
    audio_kwargs = {"sampling_rate": 16000, "return_attention_mask": True}

    def __init__(
        self,
        image_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        image_token: str = "<unk>",
        image_start_token: str = "<image>",
        image_end_token: str = "</image>",
        slice_start_token: str = "<slice>",
        slice_end_token: str = "</slice>",
        num_image_tokens: int = 64,
        audio_token: str = "<|audio|>",
        audio_start_token: str = "<|audio_start|>",
        audio_end_token: str = "<|audio_end|>",
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<unk>"`):
            Marks an image in the text, and is repeated `num_image_tokens` times per image slice.
        image_start_token (`str`, *optional*, defaults to `"<image>"`):
            Opens the image tokens of the overview image.
        image_end_token (`str`, *optional*, defaults to `"</image>"`):
            Closes the image tokens of the overview image.
        slice_start_token (`str`, *optional*, defaults to `"<slice>"`):
            Opens the image tokens of one slice.
        slice_end_token (`str`, *optional*, defaults to `"</slice>"`):
            Closes the image tokens of one slice.
        num_image_tokens (`int`, *optional*, defaults to 64):
            Number of image tokens per slice (the resampler's number of latents).
        audio_token (`str`, *optional*, defaults to `"<|audio|>"`):
            Marks an audio in the text, and is repeated once per audio token (25 per second of audio).
        audio_start_token (`str`, *optional*, defaults to `"<|audio_start|>"`):
            Opens the audio tokens of each second of audio.
        audio_end_token (`str`, *optional*, defaults to `"<|audio_end|>"`):
            Closes the audio tokens of each second of audio.
        """
        self.image_token = image_token
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        self.slice_start_token = slice_start_token
        self.slice_end_token = slice_end_token
        self.num_image_tokens = num_image_tokens
        self.audio_token = audio_token
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        ProcessorMixin.__init__(
            self, image_processor, feature_extractor, tokenizer, chat_template=chat_template, **kwargs
        )
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

    def _process_audio(self, audio: AudioInput, **kwargs):
        # Audio longer than 30 s is split into 30 s windows, one row of `input_features` each
        window = self.feature_extractor.n_samples
        windows = [waveform[start : start + window] for waveform in audio for start in range(0, len(waveform), window)]
        # Always zero-padded to 30 s, as the checkpoint's features were computed
        audio_inputs = self.feature_extractor(windows, **{**kwargs, "padding": "max_length"})
        audio_inputs["input_features_mask"] = audio_inputs.pop("attention_mask")
        audio_inputs["num_audio_tokens"] = self._get_num_multimodal_tokens(
            audio_lengths=[len(waveform) for waveform in audio]
        )["num_audio_tokens"]
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        # Each second of audio gets its own <|audio_start|>...<|audio_end|>, as the checkpoint was trained
        num_tokens, tokens_per_second = audio_inputs["num_audio_tokens"][audio_idx], 25
        return "".join(
            self.audio_start_token
            + self.audio_token * min(tokens_per_second, num_tokens - start)
            + self.audio_end_token
            for start in range(0, num_tokens, tokens_per_second)
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, audio_lengths=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            audio_lengths (`list[int]`, *optional*):
                The lengths of audio inputs in number of samples.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        multimodal_data = {}
        if image_sizes is not None:
            images_kwargs = self._merge_kwargs(**kwargs)["images_kwargs"]
            max_slice_nums = images_kwargs.get("max_slice_nums") or self.image_processor.max_slice_nums
            scale_resolution = images_kwargs.get("scale_resolution") or self.image_processor.scale_resolution
            slice_mode = images_kwargs.get("slice_mode", self.image_processor.slice_mode)

            num_image_patches = []
            for image_size in image_sizes:
                grid = self.image_processor.get_sliced_grid(image_size, max_slice_nums, scale_resolution)
                num_image_patches.append(1 + grid[0] * grid[1] if slice_mode and grid is not None else 1)
            num_image_tokens = [self.num_image_tokens * num_patches for num_patches in num_image_patches]
            multimodal_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if audio_lengths is not None:
            # 100 mel frames per second, halved by the encoder's strided conv and by the projector's pooling
            num_frames = -(-np.array(audio_lengths) // self.feature_extractor.hop_length)
            multimodal_data["num_audio_tokens"] = (((num_frames - 1) // 2 + 1 - 2) // 2 + 1).tolist()

        return MultiModalData(**multimodal_data)

    @property
    def model_input_names(self) -> list[str]:
        return super().model_input_names + ["input_features_mask"]

    @property
    def unused_input_names(self) -> list[str]:
        return ["grids", "num_audio_tokens"]


__all__ = [
    "MiniCPMO2_6Config",
    "MiniCPMO2_6VisionConfig",
    "MiniCPMO2_6AudioConfig",
    "MiniCPMO2_6Processor",
    "MiniCPMO2_6PreTrainedModel",
    "MiniCPMO2_6VisionTransformer",
    "MiniCPMO2_6AudioEncoder",
    "MiniCPMO2_6Model",
    "MiniCPMO2_6ForConditionalGeneration",
]
