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

from ...audio_utils import AudioInput, make_audio_chat_template_content, make_list_of_audio_chat_template
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

# The language token that selects the model's language-agnostic mode, which is also the default.
LANGUAGE_AGNOSTIC = "auto"


class OmniASRProcessorKwargs(ProcessingKwargs, total=False):  # trf-ignore: TRF019
    _defaults = {
        "audio_kwargs": {"sampling_rate": 16000},
        "text_kwargs": {"padding": True},
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class OmniASRProcessor(ProcessorMixin):
    valid_processor_kwargs = OmniASRProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
        audio_token="<extra_id_1>",
        conv_kernel=None,
        conv_stride=None,
    ):
        r"""
        audio_token (`str`, *optional*, defaults to `"<extra_id_1>"`):
            The placeholder token that stands for one speech encoder frame in the LLM variant's prompt, i.e. the
            token of [`OmniASRConfig.audio_token_id`].
        conv_kernel (`list[int]`, *optional*):
            Kernel size of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASREncoderConfig.conv_kernel`]. Needed by the LLM variant to count the frames an audio input is
            subsampled to, and therefore how many audio placeholders its prompt holds.
        conv_stride (`list[int]`, *optional*):
            Stride of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASREncoderConfig.conv_stride`].
        """
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.conv_kernel = list(conv_kernel) if conv_kernel is not None else None
        self.conv_stride = list(conv_stride) if conv_stride is not None else None
        # Only the LLM variant is prompted, so only its checkpoints ship a chat template.
        if self.chat_template is None:
            # CTC decoding which require tokens to be grouped
            self.group_tokens = True
        else:
            self.group_tokens = False
            if self.conv_kernel is None or self.conv_stride is None:
                raise ValueError(
                    f"{self.__class__.__name__} needs `conv_kernel` and `conv_stride` to count the audio placeholders of "
                    "the LLM variant's prompt."
                )

    @auto_docstring
    def __call__(
        self,
        audio: AudioInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str`, `list[str]`, *optional*):
            For the CTC variant, the transcription(s), one per audio input, from which the CTC `labels` are built.
            For the LLM variant, the prompt(s) rendered by [`~OmniASRProcessor.apply_chat_template`], each holding
            one `audio_token` per audio input; prefer [`~OmniASRProcessor.apply_transcription_request`].
        output_labels (`bool`, *optional*, defaults to `False`):
            LLM variant only: whether to return `labels` for training, which only cover what follows each prompt's
            closing BOS, i.e. the assistant's transcription and its EOS.

        Returns:
            [`BatchFeature`]: `input_values` and its `padding_mask`. For the CTC variant, `labels` is added whenever
            `text` is given. For the LLM variant, the decoder prompt as `input_ids` and its `attention_mask`, and
            `labels` when `output_labels=True`.
        """
        if self.chat_template is None:
            return self._call_ctc(audio, text=text, **kwargs)

        model_inputs = super().__call__(audio=audio, text=text, **kwargs)

        if output_labels:
            input_ids = model_inputs["input_ids"]
            # The prompt closes with the BOS the transcription is decoded from, so everything up to it (the audio
            # placeholders, the language markers and the left padding) is masked out of the loss.
            prompt_mask = (input_ids == self.tokenizer.bos_token_id).flip(-1).cumsum(-1).flip(-1) > 0
            labels = input_ids.masked_fill(prompt_mask | (model_inputs["attention_mask"] == 0), -100)
            model_inputs["labels"] = labels

        return model_inputs

    def _call_ctc(
        self,
        audio: AudioInput,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            OmniASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])
            labels = encodings["input_ids"]
            # Mask padding positions with -100 so the CTC loss ignores them.
            if "attention_mask" in encodings:
                labels[encodings["attention_mask"] == 0] = -100
            inputs["labels"] = labels

        return inputs

    def validate_inputs(
        self,
        audio: AudioInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(audio=audio, text=text, **kwargs)

        if text is not None and audio is not None and len(text) != len(audio):
            raise ValueError(f"Got {len(text)} text but {len(audio)} audios; they must match 1:1.")

    def _process_audio(self, audio: AudioInput, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)
        audio_inputs["num_audio_tokens"] = self._get_num_audio_tokens(audio_inputs["padding_mask"].sum(-1))
        audio_replacements = [self.replace_audio_token(audio_inputs, audio_idx=idx) for idx in range(len(audio))]
        return audio_inputs, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["num_audio_tokens"][audio_idx])

    def _get_num_audio_tokens(self, audio_lengths: "torch.Tensor") -> "torch.Tensor":
        """
        Number of speech encoder frames each audio length is subsampled to, i.e. how many audio placeholders its
        prompt holds. Mirrors `OmniASRPreTrainedModel._get_subsampling_output_length`.
        """
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            audio_lengths = torch.div(audio_lengths - kernel, stride, rounding_mode="floor") + 1
        return audio_lengths

    def apply_transcription_request(
        self,
        audio: str | list[str] | AudioInput,
        language: str | list[str] | None = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for the LLM variant's speech recognition without manually writing the chat template.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            language (`str` or `list[str]`, *optional*):
                Language code(s) (e.g. `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`), either one for the whole batch
                or one per audio. `None` or `"auto"` selects the model's language-agnostic mode; naming the language
                explicitly gives better transcription quality.
            **kwargs:
                Additional keyword arguments forwarded to [`~OmniASRProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`OmniASRForConditionalGeneration.generate`].
        """
        if self.chat_template is None:
            raise ValueError(
                f"{self.__class__.__name__} has no chat template: only the LLM variant is prompted. The CTC variant "
                "takes the audio directly, e.g. `processor(audio)`."
            )
        audio_items = list(make_list_of_audio_chat_template(audio))
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if language is None or isinstance(language, str):
            language = [language] * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")

        conversations = []
        for audio_item, lang in zip(audio_items, language):
            content = [make_audio_chat_template_content(audio_item)]
            if lang is not None:
                content.append({"type": "language", "language": self._resolve_language(lang)})
            conversations.append([{"role": "user", "content": content}])

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            **kwargs,
        )

    def _resolve_language(self, language: str) -> str:
        """Lower-case the language code, and check that the vocabulary holds its token."""
        code = language.lower()
        if self.tokenizer.convert_tokens_to_ids(f"<|lang:{code}|>") == self.tokenizer.unk_token_id:
            languages = sorted(
                token[len("<|lang:") : -len("|>")]
                for token in self.tokenizer.get_vocab()
                if token.startswith("<|lang:") and token != f"<|lang:{LANGUAGE_AGNOSTIC}|>"
            )
            raise ValueError(
                f"Unknown `language={language!r}`. Pass {LANGUAGE_AGNOSTIC!r} for the language-agnostic mode, or one "
                f"of the {len(languages)} supported codes, e.g. {languages[:5]}."
            )
        return code

    def decode(self, *args, **kwargs):
        # CTC decoding collapses runs of identical tokens; the autoregressive LLM variant must keep them.
        kwargs.setdefault("group_tokens", self.group_tokens)
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        if self.chat_template is None:
            # CTC variant: the transcription is only tokenized into the CTC `labels`.
            return self.feature_extractor.model_input_names + ["labels"]
        return super().model_input_names

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]


__all__ = ["OmniASRProcessor"]
