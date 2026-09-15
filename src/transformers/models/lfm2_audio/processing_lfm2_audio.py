# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import copy

from ...audio_utils import AudioInput, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available
from ...utils.import_utils import requires
from ..auto import AutoTokenizer
from ..parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
from .configuration_lfm2_audio import Lfm2AudioConfig


if is_torch_available():
    import torch

    from ..mimi.modeling_mimi import MimiModel


TEXT_MODALITY = 1
AUDIO_INPUT_MODALITY = 2
DEFAULT_AUDIO_TOKEN = "<|reserved_123|>"


class Lfm2AudioAudioKwargs(AudioKwargs, total=False):
    device: str | torch.device | None


class Lfm2AudioProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: Lfm2AudioAudioKwargs
    _defaults = {}


Lfm2AudioProcessorKwargs.__annotations__["audio_kwargs"] = Lfm2AudioAudioKwargs


DEFAULT_CHAT_TEMPLATE = r"""{{- bos_token -}}
{%- set ns = namespace(system_prompt="") -%}
{%- if messages and messages[0]["role"] == "system" -%}
    {%- set system_content = messages[0]["content"] -%}
    {%- if system_content is string -%}
        {%- set ns.system_prompt = system_content -%}
    {%- else -%}
        {%- for part in system_content -%}
            {%- if part["type"] == "text" -%}
                {%- set ns.system_prompt = ns.system_prompt + part["text"] -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message["role"] + "\n" -}}
    {%- if message["content"] is string -%}
        {{- message["content"] -}}
    {%- else -%}
        {%- for part in message["content"] -%}
            {%- if part["type"] == "audio" -%}
                {{- "<|reserved_123|>" -}}
            {%- elif part["type"] == "text" -%}
                {{- part["text"] -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- "<|im_end|>\n" -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}"""


@requires(backends=("torch", "librosa"))
@auto_docstring
class Lfm2AudioProcessor(ProcessorMixin):
    r"""
    Constructs an LFM2-Audio processor that combines a [`ParakeetFeatureExtractor`] and a tokenizer.

    Args:
        feature_extractor (`ParakeetFeatureExtractor`):
            Log-mel frontend used for audio prompts.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer used by the LFM2 backbone.
        chat_template (`str`, *optional*):
            Jinja template used to turn structured conversations into model prompts.
        audio_token (`str`, *optional*, defaults to `"<|reserved_123|>"`):
            Placeholder token replaced with FastConformer features.
        decoder_model_id (`str`, *optional*):
            Checkpoint containing the LFM audio detokenizer. When omitted, [`~Lfm2AudioProcessor.from_pretrained`]
            uses the processor checkpoint itself and looks in its `audio_detokenizer` subfolder.
        audio_codec_model_id (`str`, *optional*, defaults to `"kyutai/mimi"`):
            Mimi checkpoint used lazily as a fallback by [`~Lfm2AudioProcessor.decode_audio`].
    """

    valid_processor_kwargs = Lfm2AudioProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token=DEFAULT_AUDIO_TOKEN,
        decoder_model_id=None,
        decoder_subfolder="audio_detokenizer",
        audio_codec_model_id="kyutai/mimi",
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<|reserved_123|>"`):
            Placeholder token replaced with FastConformer features.
        decoder_model_id (`str`, *optional*):
            Checkpoint containing the LFM audio detokenizer.
        decoder_subfolder (`str`, *optional*, defaults to `"audio_detokenizer"`):
            Subfolder containing the detokenizer configuration and weights.
        audio_codec_model_id (`str`, *optional*, defaults to `"kyutai/mimi"`):
            Mimi checkpoint used lazily when an LFM detokenizer is unavailable.
        """
        if chat_template is None or audio_token not in chat_template:
            chat_template = DEFAULT_CHAT_TEMPLATE.replace(DEFAULT_AUDIO_TOKEN, audio_token)
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.decoder_model_id = None if decoder_model_id is None else str(decoder_model_id)
        self.decoder_subfolder = decoder_subfolder
        self.audio_codec_model_id = audio_codec_model_id
        self._detokenizer = None
        self._detokenizer_unavailable = False
        self._audio_codec = None
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        decoder_model_id = kwargs.pop("decoder_model_id", None)
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if decoder_model_id is not None:
            processor.decoder_model_id = str(decoder_model_id)
        elif processor.decoder_model_id is None:
            processor.decoder_model_id = str(pretrained_model_name_or_path)
        return processor

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        """Build the frontend from `config.json` for checkpoints released before processor files existed."""
        try:
            return super()._get_arguments_from_pretrained(
                pretrained_model_name_or_path,
                processor_dict=processor_dict,
                **copy.deepcopy(kwargs),
            )
        except (OSError, ValueError):
            # `LiquidAI/LFM2.5-Audio-1.5B` predates native Transformers support and therefore has no
            # preprocessor config. Its complete frontend configuration is embedded in `config.json`.
            pass

        load_kwargs = copy.deepcopy(kwargs)
        subfolder = load_kwargs.pop("subfolder", "")
        config = Lfm2AudioConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **load_kwargs,
        )
        frontend = config.preprocessor_config
        feature_extractor = ParakeetFeatureExtractor(
            feature_size=frontend.features,
            sampling_rate=frontend.sample_rate,
            hop_length=round(frontend.window_stride * frontend.sample_rate),
            n_fft=frontend.n_fft,
            win_length=round(frontend.window_size * frontend.sample_rate),
            preemphasis=0.97,
            padding_value=frontend.pad_value,
        )
        tokenizer_kwargs = copy.deepcopy(kwargs)
        tokenizer_kwargs.pop("subfolder", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            **tokenizer_kwargs,
        )
        return [feature_extractor, tokenizer]

    def validate_inputs(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(text=text, audio=audio)
        if audio is not None:
            if text is None:
                raise ValueError("Text containing an audio placeholder is required when `audio` is provided.")
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            if num_audio_tokens != len(audio):
                raise ValueError(
                    f"Found {num_audio_tokens} audio placeholders in text but received {len(audio)} audio samples."
                )

    def _process_audio(self, audio: AudioInput, **kwargs):
        kwargs["return_attention_mask"] = True
        audio_inputs = self.feature_extractor(audio, **kwargs)
        feature_attention_mask = audio_inputs.pop("attention_mask")
        feature_lengths = feature_attention_mask.sum(-1)
        # Liquid Audio's inference path uses the complete centered-STFT output, including the terminal frame that
        # the NeMo frontend zeroes beyond its reported length. That extra frame matters whenever the reported length
        # is divisible by FastConformer's 8x subsampling factor: omitting it removes one audio placeholder and can
        # make ASR generation collapse to empty text.
        feature_lengths = (feature_lengths + 1).clamp_max(feature_attention_mask.shape[-1])
        frame_indices = torch.arange(feature_attention_mask.shape[-1], device=feature_attention_mask.device)
        feature_attention_mask = frame_indices[None] < feature_lengths[:, None]
        audio_inputs["input_features_attention_mask"] = feature_attention_mask
        audio_inputs["num_audio_tokens"] = (feature_lengths + 7) // 8
        replacements = [self.replace_audio_token(audio_inputs, idx) for idx in range(len(audio))]
        return audio_inputs, replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["num_audio_tokens"][audio_idx])

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[Lfm2AudioProcessorKwargs],
    ) -> BatchFeature:
        common_kwargs = dict(kwargs.pop("common_kwargs", {}))
        if "return_tensors" not in kwargs:
            common_kwargs.setdefault("return_tensors", "pt")
        kwargs["common_kwargs"] = common_kwargs

        text_kwargs = dict(kwargs.pop("text_kwargs", {}))
        if "add_special_tokens" not in kwargs:
            text_kwargs.setdefault("add_special_tokens", False)
        if "padding" not in kwargs:
            text_kwargs.setdefault("padding", True)
        text_kwargs["return_mm_token_type_ids"] = True
        kwargs["text_kwargs"] = text_kwargs

        audio_kwargs = dict(kwargs.pop("audio_kwargs", {}))
        if "sampling_rate" not in kwargs:
            audio_kwargs.setdefault("sampling_rate", self.feature_extractor.sampling_rate)
        kwargs["audio_kwargs"] = audio_kwargs
        outputs = super().__call__(text=text, audio=audio, **kwargs)

        if "input_ids" in outputs:
            mm_token_type_ids = outputs.pop("mm_token_type_ids")
            modality_ids = torch.full_like(outputs["input_ids"], TEXT_MODALITY)
            modality_ids = modality_ids.masked_fill(mm_token_type_ids == 3, AUDIO_INPUT_MODALITY)
            if "attention_mask" in outputs:
                modality_ids = modality_ids.masked_fill(~outputs["attention_mask"].bool(), 0)
            outputs["modality_ids"] = modality_ids
        return outputs

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_audio_tokens"]

    @property
    def model_input_names(self) -> list[str]:
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names + ["input_features", "input_features_attention_mask", "modality_ids"]
            )
        )

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        prompt: str = "Perform ASR.",
        device: str | torch.device | None = None,
        **kwargs: Unpack[Lfm2AudioProcessorKwargs],
    ) -> BatchFeature:
        """Prepare an ASR request, optionally running the log-mel frontend on `device`."""
        processor_kwargs = dict(kwargs)
        if device is not None:
            audio_kwargs = dict(processor_kwargs.get("audio_kwargs", {}))
            audio_kwargs["device"] = device
            processor_kwargs["audio_kwargs"] = audio_kwargs
        audio_items = list(make_list_of_audio_chat_template(audio))
        conversations = [
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": item} if isinstance(item, str) else {"type": "audio", "audio": item}
                    ],
                },
            ]
            for item in audio_items
        ]
        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs=processor_kwargs,
        )

    def apply_text_to_speech_request(
        self,
        text: str | list[str],
        prompt: str = "Perform TTS.",
        **kwargs: Unpack[Lfm2AudioProcessorKwargs],
    ) -> BatchFeature:
        texts = [text] if isinstance(text, str) else list(text)
        conversations = [
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "text", "text": item}]},
            ]
            for item in texts
        ]
        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs=dict(kwargs),
        )

    def decode_audio(
        self,
        audio_codes: torch.LongTensor,
        audio_codec: MimiModel | None = None,
        device: str | torch.device | None = None,
    ) -> torch.FloatTensor:
        """Decode generated codebooks with the bundled LFM detokenizer or a native Mimi model."""
        if audio_codes.ndim == 2:
            audio_codes = audio_codes.unsqueeze(0)
        if audio_codes.ndim != 3:
            raise ValueError("`audio_codes` must have shape `(batch_size, codebooks, timesteps)`.")
        if audio_codes.shape[-1] and torch.all(audio_codes[..., -1] == 2048):
            audio_codes = audio_codes[..., :-1]
        if audio_codes.shape[-1] == 0:
            raise ValueError("No decodable audio tokens were provided.")
        if torch.any((audio_codes < 0) | (audio_codes >= 2048)):
            raise ValueError("Mimi audio tokens must be in the range [0, 2047].")

        if device is None:
            device = audio_codes.device

        if audio_codec is None and self.decoder_model_id is not None:
            if self._detokenizer is None and not self._detokenizer_unavailable:
                from .modeling_lfm2_audio import Lfm2AudioDetokenizer

                try:
                    self._detokenizer = Lfm2AudioDetokenizer.from_pretrained(
                        self.decoder_model_id,
                        subfolder=self.decoder_subfolder,
                        dtype=torch.float32,
                    ).eval()
                except OSError:
                    self._detokenizer_unavailable = True
            if self._detokenizer is not None:
                detokenizer = self._detokenizer.to(device)
                with torch.no_grad():
                    return detokenizer(audio_codes.to(device))

        if audio_codec is None:
            if self._audio_codec is None:
                self._audio_codec = MimiModel.from_pretrained(self.audio_codec_model_id).eval()
            audio_codec = self._audio_codec
        audio_codec = audio_codec.to(device)
        with torch.no_grad():
            decoded = audio_codec.decode(audio_codes.to(device), return_dict=True).audio_values
        if decoded.ndim == 3 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        return decoded

    @property
    def output_sampling_rate(self) -> int:
        return 24_000


__all__ = ["Lfm2AudioProcessor"]
