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

from ...audio_utils import AudioInput, make_list_of_audio
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

# `language_mapping` indexes the language embedding table from 1, leaving row 0 for the language-agnostic mode.
LANGUAGE_AGNOSTIC = "auto"
LANGUAGE_AGNOSTIC_ID = 0


class OmniASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
        "text_kwargs": {
            "padding": True,
            "padding_side": "right",
            "add_special_tokens": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class OmniASRProcessor(ProcessorMixin):
    valid_processor_kwargs = OmniASRProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        language_mapping=None,
        group_tokens=None,
        audio_token_id=None,
        language_token_id=None,
        language_embedding_token_id=None,
        bos_token_id=None,
        conv_kernel=None,
        conv_stride=None,
    ):
        r"""
        language_mapping (`dict[str, int]`, *optional*):
            Mapping from a language code (e.g. `"eng_Latn"`) to its index in the model's language embedding table.
            Only the LLM variant is language-conditioned, so only its checkpoints carry a mapping; when it is
            `None`, `language` cannot be resolved and no `language_ids` are produced.
        group_tokens (`bool`, *optional*):
            Whether [`~OmniASRProcessor.decode`] collapses runs of identical tokens. This is what CTC decoding
            requires, and what the autoregressive LLM variant must not do. Defaults to `True` for the CTC variant
            and `False` for the LLM variant.
        audio_token_id (`int`, *optional*):
            Id of the placeholder token that stands for one speech encoder frame in `input_ids`, i.e.
            [`OmniASRConfig.audio_token_id`]. Only the LLM variant is prompted with the audio, so only its
            checkpoints carry the ids and the convolution geometry below.
        language_token_id (`int`, *optional*):
            Id of the LID marker token that opens the language slot of the prompt, i.e.
            [`OmniASRConfig.language_token_id`].
        language_embedding_token_id (`int`, *optional*):
            Id of the placeholder token that stands for the language embedding in `input_ids`, i.e.
            [`OmniASRConfig.language_embedding_token_id`].
        bos_token_id (`int`, *optional*):
            Id of the token that closes the prompt, and from which the transcription is decoded. Defaults to the
            tokenizer's `bos_token_id`.
        conv_kernel (`list[int]`, *optional*):
            Kernel size of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASREncoderConfig.conv_kernel`]. Needed to count the frames an audio input is subsampled to, and
            therefore how many audio placeholders its prompt holds.
        conv_stride (`list[int]`, *optional*):
            Stride of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASREncoderConfig.conv_stride`].
        """
        super().__init__(feature_extractor, tokenizer)
        self.language_mapping = language_mapping
        if group_tokens is None:
            # Checkpoints converted before `group_tokens` was explicit are recognised by carrying a mapping.
            group_tokens = language_mapping is None
        self.group_tokens = group_tokens
        self.audio_token_id = audio_token_id
        self.language_token_id = language_token_id
        self.language_embedding_token_id = language_embedding_token_id
        self.bos_token_id = bos_token_id if bos_token_id is not None else tokenizer.bos_token_id
        # Lists rather than tuples, so that saving and reloading the processor round-trips to an equal object.
        self.conv_kernel = list(conv_kernel) if conv_kernel is not None else None
        self.conv_stride = list(conv_stride) if conv_stride is not None else None

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        language: str | list[str] = "auto",
        sampling_rate: int | None = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ):
        """
        Processes audio input and optionally text/language for OmniASR models.

        For the CTC variant, pass `audio` (and optionally `text` for training labels).
        For the LLM variant, pass `audio` and `language` (e.g., `["eng_Latn"]`).

        Args:
            audio (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*):
                The audio input, passed to the feature extractor.
            text (`str`, `list[str]`, *optional*):
                Text input, passed to the tokenizer (used for training labels).
            language (`str` or `list[str]`, *optional*, defaults to `"auto"`):
                Language code(s) for the LLM variant (e.g. `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`), resolved
                into the `language_ids` model input via `language_mapping`. Either a single code applied to the
                whole batch, or one per audio. `"auto"` selects the model's language-agnostic mode; naming the
                language explicitly gives better transcription quality. Ignored by the CTC variant, which is not
                language-conditioned.
            sampling_rate (`int`, *optional*):
                The sampling rate of the audio input. Will warn if not provided.

        Returns:
            [`BatchFeature`]: For the CTC variant, `input_values` and its `attention_mask`. For the LLM variant, the
            decoder prompt as `input_ids` and its `attention_mask`, alongside `input_values`, `padding_mask` (the
            mask over the raw samples) and `language_ids`. `labels` is added whenever `text` is given.
        """
        audio = make_list_of_audio(audio)

        output_kwargs = self._merge_kwargs(
            OmniASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if sampling_rate is None:
            logger.warning_once(
                f"You've provided audio without specifying the sampling rate. It will be assumed to be {output_kwargs['audio_kwargs']['sampling_rate']}, which can result in silent errors."
            )
        elif sampling_rate != output_kwargs["audio_kwargs"]["sampling_rate"]:
            raise ValueError(
                f"The sampling rate of the audio ({sampling_rate}) does not match the sampling rate of the processor ({output_kwargs['audio_kwargs']['sampling_rate']}). Please provide resampled the audio to the expected sampling rate."
            )

        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        # Only the LLM variant is language-conditioned, and only its checkpoints ship a mapping.
        if self.language_mapping is not None:
            # The speech encoder gets its own mask over the raw samples, named `padding_mask` as in every other
            # model that takes `input_values`, so that `attention_mask` is free to cover the decoder prompt.
            padding_mask = inputs.pop("attention_mask", None)
            if padding_mask is not None:
                audio_lengths = padding_mask.sum(-1)
                inputs["padding_mask"] = padding_mask
            else:
                audio_lengths = torch.full((len(audio),), inputs["input_values"].shape[-1], dtype=torch.long)
            inputs["language_ids"] = self._resolve_language_ids(language, len(audio))
            inputs["input_ids"], inputs["attention_mask"] = self._build_prompt(audio_lengths)
        elif language != LANGUAGE_AGNOSTIC:
            logger.warning_once(
                f"`language={language!r}` is ignored: this processor has no `language_mapping`, so the model it "
                "belongs to is not language-conditioned."
            )

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            labels = encodings["input_ids"]
            # Mask padding positions with -100 so the CTC loss ignores them.
            # (pad_token_id=0 satisfies labels >= 0, which would otherwise inflate target_lengths.)
            if "attention_mask" in encodings:
                labels[encodings["attention_mask"] == 0] = -100
            inputs["labels"] = labels

        return inputs

    def decode(self, *args, **kwargs):
        # CTC decoding collapses runs of identical tokens; the autoregressive LLM variant must keep them.
        kwargs.setdefault("group_tokens", self.group_tokens)
        return self.tokenizer.decode(*args, **kwargs)

    def _resolve_language_ids(self, language: str | list[str], batch_size: int) -> "torch.LongTensor":
        if not is_torch_available():
            raise ImportError("Resolving `language` into `language_ids` requires PyTorch. Please install PyTorch.")
        if isinstance(language, str):
            language = [language] * batch_size
        if len(language) == 1 and batch_size > 1:
            # A single language code is broadcast to every audio sample in the batch.
            language = list(language) * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Received {len(language)} `language` entries for {batch_size} audio input(s).")

        language_ids = []
        for lang in language:
            key = lang.lower()
            if key == LANGUAGE_AGNOSTIC:
                language_ids.append(LANGUAGE_AGNOSTIC_ID)
            elif key in self.language_mapping:
                language_ids.append(self.language_mapping[key])
            else:
                raise ValueError(
                    f"Unknown `language={lang!r}`. Pass {LANGUAGE_AGNOSTIC!r} for the language-agnostic mode, or one "
                    f"of the {len(self.language_mapping)} codes in `language_mapping`, e.g. "
                    f"{sorted(self.language_mapping)[:5]}."
                )
        return torch.tensor(language_ids, dtype=torch.long)

    def _get_num_audio_tokens(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        """
        Number of speech encoder frames each audio length is subsampled to, i.e. how many audio placeholders its
        prompt holds. Mirrors `OmniASRPreTrainedModel._get_feat_extract_output_lengths`.
        """
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            audio_lengths = torch.div(audio_lengths - kernel, stride, rounding_mode="floor") + 1
        return audio_lengths

    def _build_prompt(self, audio_lengths: "torch.Tensor") -> tuple["torch.LongTensor", "torch.LongTensor"]:
        """
        Build the decoder prompt `audio | lid_marker | language | bos` of each audio input, as `input_ids` holding
        one audio placeholder per speech encoder frame.

        The prompts are left-padded, so that every sequence ends with `bos` -- decoding continues from there for the
        whole batch -- and the distance between the audio and the markers does not depend on how much the batch was
        padded. The [original implementation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/model.py#L1051)
        instead packs the markers directly after each sample's own last frame, and tracks the lengths alongside.
        """
        missing = [
            name
            for name, value in [
                ("audio_token_id", self.audio_token_id),
                ("language_token_id", self.language_token_id),
                ("language_embedding_token_id", self.language_embedding_token_id),
                ("bos_token_id", self.bos_token_id),
                ("conv_kernel", self.conv_kernel),
                ("conv_stride", self.conv_stride),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} cannot build the decoder prompt without {missing}, which are saved "
                "alongside the LLM variant's checkpoints. A checkpoint converted before the prompt was built here "
                "has to be converted again."
            )

        markers = [self.language_token_id, self.language_embedding_token_id, self.bos_token_id]
        num_audio_tokens = self._get_num_audio_tokens(audio_lengths).tolist()
        max_length = max(num_audio_tokens) + len(markers)

        input_ids = torch.full((len(num_audio_tokens), max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(num_audio_tokens), max_length), dtype=torch.long)
        for idx, num_frames in enumerate(num_audio_tokens):
            prompt = [self.audio_token_id] * num_frames + markers
            input_ids[idx, max_length - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            attention_mask[idx, max_length - len(prompt) :] = 1

        return input_ids, attention_mask

    @property
    def model_input_names(self):
        if self.language_mapping is None:
            # CTC variant: the audio is the whole input, and `attention_mask` masks its padding.
            return self.feature_extractor.model_input_names + ["labels"]
        # LLM variant: the audio fills the placeholders of a decoder prompt, so it carries its own mask.
        return ["input_values", "padding_mask", "input_ids", "attention_mask", "language_ids", "labels"]


__all__ = ["OmniASRProcessor"]
