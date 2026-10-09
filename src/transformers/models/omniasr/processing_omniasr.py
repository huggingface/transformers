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

# ISO 639-1 codes (e.g. `"en"`) mapped to the model's language codes, for the languages that have one.
# fmt: off
ISO_639_1_TO_LANGUAGE = {
    "ab": "abk_cyrl", "af": "afr_latn", "ak": "aka_latn", "am": "amh_ethi", "an": "arg_latn", "ar": "arb_arab",
    "as": "asm_beng", "av": "ava_cyrl", "ay": "ayr_latn", "az": "aze_latn", "ba": "bak_cyrl", "be": "bel_cyrl",
    "bg": "bul_cyrl", "bi": "bis_latn", "bm": "bam_latn", "bn": "ben_beng", "bo": "bod_tibt", "br": "bre_latn",
    "bs": "bos_latn", "ca": "cat_latn", "ce": "che_cyrl", "cs": "ces_latn", "cv": "chv_cyrl", "cy": "cym_latn",
    "da": "dan_latn", "de": "deu_latn", "dv": "div_thaa", "dz": "dzo_tibt", "ee": "ewe_latn", "el": "ell_grek",
    "en": "eng_latn", "eo": "epo_latn", "es": "spa_latn", "et": "ekk_latn", "eu": "eus_latn", "fa": "fas_arab",
    "ff": "ful_latn", "fi": "fin_latn", "fj": "fij_latn", "fo": "fao_latn", "fr": "fra_latn", "fy": "fry_latn",
    "ga": "gle_latn", "gl": "glg_latn", "gn": "grn_latn", "gu": "guj_gujr", "gv": "glv_latn", "ha": "hau_latn",
    "he": "heb_hebr", "hi": "hin_deva", "hr": "hrv_latn", "ht": "hat_latn", "hu": "hun_latn", "hy": "hye_armn",
    "hz": "her_latn", "ia": "ina_latn", "id": "ind_latn", "ig": "ibo_latn", "ik": "ipk_latn", "is": "isl_latn",
    "it": "ita_latn", "ja": "jpn_jpan", "jv": "jav_latn", "ka": "kat_geor", "ki": "kik_latn", "kj": "kua_latn",
    "kk": "kaz_cyrl", "km": "khm_khmr", "kn": "kan_knda", "ko": "kor_hang", "kr": "knc_latn", "ks": "kas_arab",
    "ku": "kur_arab", "kv": "kpv_cyrl", "kw": "cor_latn", "ky": "kir_cyrl", "la": "lat_latn", "lb": "ltz_latn",
    "lg": "lug_latn", "ln": "lin_latn", "lo": "lao_laoo", "lt": "lit_latn", "lv": "lav_latn", "mg": "mlg_latn",
    "mh": "mah_latn", "mi": "mri_latn", "mk": "mkd_cyrl", "ml": "mal_mlym", "mn": "mon_cyrl", "mr": "mar_deva",
    "ms": "zsm_latn", "mt": "mlt_latn", "my": "mya_mymr", "nb": "nob_latn", "ne": "nep_deva", "ng": "ndo_latn",
    "nl": "nld_latn", "nn": "nno_latn", "no": "nob_latn", "ny": "nya_latn", "oc": "oci_latn", "om": "orm_latn",
    "or": "ory_orya", "os": "oss_cyrl", "pa": "pan_guru", "pl": "pol_latn", "ps": "pus_arab", "pt": "por_latn",
    "rn": "run_latn", "ro": "ron_latn", "ru": "rus_cyrl", "rw": "kin_latn", "sc": "srd_latn", "sd": "snd_arab",
    "sg": "sag_latn", "si": "sin_sinh", "sk": "slk_latn", "sl": "slv_latn", "sm": "smo_latn", "sn": "sna_latn",
    "so": "som_latn", "sq": "als_latn", "sr": "srp_cyrl", "su": "sun_latn", "sv": "swe_latn", "sw": "swh_latn",
    "ta": "tam_taml", "te": "tel_telu", "tg": "tgk_cyrl", "th": "tha_thai", "ti": "tir_ethi", "tk": "tuk_latn",
    "tl": "tgl_latn", "tn": "tsn_latn", "tr": "tur_latn", "ts": "tso_latn", "tt": "tat_cyrl", "tw": "twi_latn",
    "ug": "uig_arab", "uk": "ukr_cyrl", "ur": "urd_arab", "uz": "uzb_latn", "vi": "vie_latn", "wo": "wol_latn",
    "xh": "xho_latn", "yi": "ydd_hebr", "yo": "yor_latn", "za": "zyb_latn", "zh": "cmn_hans", "zu": "zul_latn",
}
# fmt: on


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
            The placeholder token that stands for one speech encoder frame in the prompt, i.e. the token of
            [`OmniASRConfig.audio_token_id`].
        conv_kernel (`list[int]`, *optional*):
            Kernel size of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASRAudioConfig.conv_kernel`]. Needed to count the frames an audio input is subsampled to, and
            therefore how many audio placeholders its prompt holds. Defaults to `[10, 3, 3, 3, 3, 2, 2]`.
        conv_stride (`list[int]`, *optional*):
            Stride of each convolution of the speech encoder's feature encoder, i.e.
            [`OmniASRAudioConfig.conv_stride`]. Defaults to `[5, 2, 2, 2, 2, 2, 2]`.
        """
        self.conv_kernel = conv_kernel if conv_kernel is not None else [10, 3, 3, 3, 3, 2, 2]
        self.conv_stride = conv_stride if conv_stride is not None else [5, 2, 2, 2, 2, 2, 2]
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

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
            The prompt(s) rendered by [`~OmniASRProcessor.apply_chat_template`], each holding one `audio_token` per
            audio input; prefer [`~OmniASRProcessor.apply_transcription_request`].
        output_labels (`bool`, *optional*, defaults to `False`):
            Whether to return `labels` for training, which only cover what follows each prompt's closing BOS, i.e.
            the assistant's transcription and its EOS.

        Returns:
            [`BatchFeature`]: `input_values` and its `padding_mask`, the decoder prompt as `input_ids` and its
            `attention_mask`, and `labels` when `output_labels=True`.
        """
        model_inputs = super().__call__(audio=audio, text=text, **kwargs)

        if output_labels:
            input_ids = model_inputs["input_ids"]
            # The prompt closes with the BOS the transcription is decoded from, so everything up to it (the audio
            # placeholders, the language markers and the left padding) is masked out of the loss.
            prompt_mask = (input_ids == self.tokenizer.bos_token_id).flip(-1).cumsum(-1).flip(-1) > 0
            labels = input_ids.masked_fill(prompt_mask | (model_inputs["attention_mask"] == 0), -100)
            model_inputs["labels"] = labels

        return model_inputs

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
        transcription: str | list[str] | None = None,
        **kwargs: Unpack[OmniASRProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare inputs for speech recognition without manually writing the chat template, either for inference or,
        when `transcription` is given, for training.

        Args:
            audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by
                the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.
            language (`str` or `list[str]`, *optional*):
                Language code(s) (e.g. `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`), either one for the whole batch
                or one per audio. ISO 639-1 codes (e.g. `"en"`) are also accepted for the languages that have one.
                `None` or `"auto"` selects the model's language-agnostic mode; naming the language explicitly gives
                better transcription quality.
            transcription (`str` or `list[str]`, *optional*):
                Target transcript(s) for training, one per audio. When given, each transcript is written as the
                assistant turn and `labels` are returned (see `output_labels` in [`~OmniASRProcessor.__call__`]).
            **kwargs:
                Additional keyword arguments forwarded to [`~OmniASRProcessor.apply_chat_template`] (for example
                `text_kwargs`, `audio_kwargs`, ...).

        Returns:
            [`BatchFeature`]: Processor outputs ready to be passed to
            [`OmniASRForConditionalGeneration.generate`], or to [`OmniASRForConditionalGeneration.forward`] to compute
            the loss when `transcription` is given.
        """
        audio_items = list(make_list_of_audio_chat_template(audio))
        batch_size = len(audio_items)
        if batch_size == 0:
            raise ValueError("`audio` must contain at least one sample.")

        if language is None or isinstance(language, str):
            language = [language] * batch_size
        if len(language) != batch_size:
            raise ValueError(f"Got {len(language)} language(s) for {batch_size} sample(s); counts must match.")

        is_training = transcription is not None
        if is_training:
            if isinstance(transcription, str):
                transcription = [transcription]
            if len(transcription) != batch_size:
                raise ValueError(
                    f"Got {len(transcription)} transcription(s) for {batch_size} sample(s); counts must match."
                )
            kwargs["processor_kwargs"] = {"output_labels": True, **kwargs.get("processor_kwargs", {})}
        else:
            transcription = [None] * batch_size

        conversations = []
        for audio_item, lang, transcript in zip(audio_items, language, transcription):
            content = [make_audio_chat_template_content(audio_item)]
            if lang is not None:
                content.append({"type": "language", "language": lang})
            conversation = [{"role": "user", "content": content}]
            if transcript is not None:
                conversation.append({"role": "assistant", "content": [{"type": "text", "text": transcript}]})
            conversations.append(conversation)

        return self.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=not is_training,
            return_dict=True,
            **kwargs,
        )

    def apply_chat_template(self, conversation, *args, **kwargs):
        """
        Same as [`~ProcessorMixin.apply_chat_template`], but resolves the language codes of the conversation first.
        The chat template writes the language token as is, so this override is needed to accept ISO 639-1 codes (e.g.
        `"en"` for `"eng_Latn"`), and to raise on unsupported codes instead of silently writing an unknown token.
        """
        is_batched = isinstance(conversation[0], (list, tuple))
        conversations = [
            [
                {
                    **message,
                    "content": [
                        {**item, "language": self._resolve_language(item["language"])}
                        if item.get("type") == "language"
                        else item
                        for item in message["content"]
                    ],
                }
                if isinstance(message.get("content"), list)
                else message
                for message in conv
            ]
            for conv in (conversation if is_batched else [conversation])
        ]
        return super().apply_chat_template(conversations if is_batched else conversations[0], *args, **kwargs)

    def _resolve_language(self, language: str) -> str:
        """Lower-case the language code, map an ISO 639-1 code, and check that the vocabulary holds its token."""
        code = language.lower()
        code = ISO_639_1_TO_LANGUAGE.get(code, code)
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

    @property
    def unused_input_names(self) -> list[str]:
        "Input names returned always by subprocessors but not used in model's `forward`"
        return ["num_audio_tokens"]


__all__ = ["OmniASRProcessor"]
