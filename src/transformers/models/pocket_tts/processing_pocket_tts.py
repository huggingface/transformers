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

import re
from pathlib import Path

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring, is_soundfile_available, is_torch_available, requires_backends
from ...utils.import_utils import requires


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf


class PocketTTSAudioKwargs(AudioKwargs, total=False):
    """
    end_on_pause (`bool`, *optional*):
        Whether to end each voice-prompt waveform on a short pause, as the original does before cloning a voice.
        Defaults to the feature extractor's `end_on_pause`.
    """

    end_on_pause: bool


class PocketTTSProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: PocketTTSAudioKwargs
    # call defaults (`padding`, `padding_side`, `return_tensors`) ship in the checkpoint's `tokenizer_config.json`
    _defaults = {}


@requires(backends=("torch",))
@auto_docstring
class PocketTTSProcessor(ProcessorMixin):
    audio_tokenizer_class = "PocketTTSAudioTokenizerModel"
    valid_processor_kwargs = PocketTTSProcessorKwargs

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer=None,
        chat_template=None,
        short_text_max_words: int = 4,
        frames_after_eos_short_text: int = 4,
        frames_after_eos: int = 2,
        fade_in_duration: float = 0.005,
    ):
        r"""
        audio_tokenizer (`PocketTTSAudioTokenizerModel`, *optional*):
            The codec: it encodes audio into latents, and decodes generated latents into audio.
        short_text_max_words (`int`, *optional*, defaults to 4):
            Texts of at most this many words get `frames_after_eos_short_text` frames after the end-of-speech frame.
        frames_after_eos_short_text (`int`, *optional*, defaults to 4):
            Number of frames generated after the end-of-speech frame for short texts.
        frames_after_eos (`int`, *optional*, defaults to 2):
            Number of frames generated after the end-of-speech frame for longer texts.
        fade_in_duration (`float`, *optional*, defaults to 0.005):
            Duration in seconds of the linear fade-in applied to the start of each decoded waveform: a fresh decoder
            state starts with a small step, heard as a click.
        """
        self.short_text_max_words = short_text_max_words
        self.frames_after_eos_short_text = frames_after_eos_short_text
        self.frames_after_eos = frames_after_eos
        self.fade_in_duration = fade_in_duration
        # prompt layout written by the chat template: `audio_prompt_bos_token` `audio_token` text `audio_bos_token`. The
        # voice BOS is optional: a derived model's tokenizer may have none.
        self.audio_prompt_bos_token = getattr(tokenizer, "audio_prompt_bos_token", None)
        self.audio_token = tokenizer.audio_prompt_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_latent_token = tokenizer.audio_latent_token
        self.audio_bos_token = tokenizer.audio_bos_token
        placeholder_tokens = [
            self.audio_prompt_bos_token,
            self.audio_token,
            self.audio_bos_token,
            self.audio_latent_token,
        ]
        self.placeholder_tokens = [token for token in placeholder_tokens if token is not None]
        super().__init__(feature_extractor, tokenizer, audio_tokenizer=audio_tokenizer, chat_template=chat_template)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "audio_prompt_latents",
            "audio_prompt_latents_mask",
            "frames_after_eos",
        ]

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        audio: AudioInput | None = None,
        audio_prompt_latents: "list[torch.Tensor] | None" = None,
        labels: "list[torch.Tensor] | None" = None,
        **kwargs: Unpack[PocketTTSProcessorKwargs],
    ) -> BatchFeature:
        r"""
        text (`str` or `list[str]`):
            The prompt(s) rendered by [`~PocketTTSProcessor.apply_chat_template`].
        audio (`np.ndarray`, `torch.Tensor`, or a list of them, *optional*):
            The waveforms of the conversations, mono at 24 kHz, in conversation order: each voice, and in training each
            target.
        audio_prompt_latents (`list[torch.Tensor]`, *optional*):
            Precomputed voices, one `(num_frames, latent_dim)` tensor per voice placeholder, instead of waveforms.
        labels (`list[torch.Tensor]`, *optional*):
            Precomputed targets (training), one `(num_frames, latent_dim)` tensor per target placeholder, instead of
            waveforms.

        Returns:
            [`BatchFeature`] with `input_ids`, `attention_mask`, `audio_prompt_latents` and `audio_prompt_latents_mask`;
            for generation `frames_after_eos` (frames generated after the end-of-speech frame, by text length), for
            training `labels` and `labels_mask`.
        """
        output_kwargs = self._merge_kwargs(
            PocketTTSProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        return_tensors = output_kwargs["text_kwargs"].get("return_tensors", None)

        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        precomputed = audio_prompt_latents is not None or labels is not None
        if audio is not None and precomputed:
            raise ValueError(
                "Pass the audio either as waveforms (`audio`) or as latents (`audio_prompt_latents` and `labels`)."
            )

        text = [text] if isinstance(text, str) else text
        self.validate_inputs(text=text, audio_prompt_latents=audio_prompt_latents, labels=labels)

        # in the flat list of audio, identify which items are training targets and which are prompt voices
        is_target = self._audio_is_target(text)

        if any(is_target):
            # training audio is encoded as is, as the original's loader does: no pause appended to the voices
            kwargs["end_on_pause"] = False

        # audio can also be provided via precomputed latents
        # for those, expand the placeholders as `_process_audio` does for waveforms
        latents = None
        if precomputed:
            voices, targets = iter(audio_prompt_latents or []), iter(labels or [])
            latents = [next(targets) if target else next(voices) for target in is_target]
            replacements = [self.audio_token * len(row) for row in latents]
            text, _ = self.get_text_with_replacements(text, audio_replacements=replacements)

        inputs = super().__call__(text=text, audio=audio, **kwargs)

        # split the conversations' audio, in order, into the voices and the training targets
        if audio is not None:
            audio_latents, audio_latents_mask = inputs.pop("audio_latents"), inputs.pop("audio_latents_mask")
            latents = [row[mask.bool()] for row, mask in zip(audio_latents, audio_latents_mask)]
        if latents is not None:
            voices = [row for row, target in zip(latents, is_target) if not target]
            targets = [row for row, target in zip(latents, is_target) if target]
            # right-padded, with the mask of each row's frames
            for name, rows in (("audio_prompt_latents", voices), ("labels", targets)):
                if rows:
                    inputs[name] = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
                    inputs[f"{name}_mask"] = torch.nn.utils.rnn.pad_sequence(
                        [torch.ones(len(row), dtype=torch.long) for row in rows], batch_first=True
                    )

        if not any(is_target):
            # The words of the text as the tokenizer prepares it. The original counts them before appending the final
            # punctuation, which can merge a trailing standalone dash or comma ("one two three four —") into the period:
            # such a text gets one word less here.
            num_words = [
                len(self.tokenizer.decode(ids, skip_special_tokens=True).split()) for ids in inputs["input_ids"]
            ]
            inputs["frames_after_eos"] = torch.tensor(
                [
                    self.frames_after_eos_short_text if words <= self.short_text_max_words else self.frames_after_eos
                    for words in num_words
                ]
            )

        return inputs

    def validate_inputs(
        self,
        text: TextInput | list[TextInput] | None = None,
        audio: AudioInput | None = None,
        audio_prompt_latents: "list[torch.Tensor] | None" = None,
        labels: "list[torch.Tensor] | None" = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(text=text, audio=audio, **kwargs)

        if text is None or any(self.audio_bos_token not in prompt for prompt in text):
            raise ValueError(
                f"Every prompt holds the first audio position `{self.audio_bos_token}`: render the conversations with "
                "`processor.apply_chat_template`."
            )

        num_placeholders = sum(prompt.count(self.audio_token) for prompt in text)
        if audio is not None and len(audio) != num_placeholders:
            raise ValueError(f"Got {len(audio)} waveforms for {num_placeholders} audio placeholders.")

        if audio_prompt_latents is not None or labels is not None:
            is_target = self._audio_is_target(text)
            num_voices, num_targets = len(audio_prompt_latents or []), len(labels or [])
            if num_voices != is_target.count(False) or num_targets != is_target.count(True):
                raise ValueError(
                    f"Got {num_voices} voices and {num_targets} targets for {is_target.count(False)} voice and "
                    f"{is_target.count(True)} target placeholders."
                )

    def _audio_is_target(self, text: list[str]) -> list[bool]:
        """Role of each audio placeholder, in order: a training target follows `audio_bos_token`; a voice is any other."""
        pattern = f"({re.escape(self.audio_bos_token)})?{re.escape(self.audio_token)}"
        return [match.group(1) is not None for prompt in text for match in re.finditer(pattern, prompt)]

    def _process_audio(self, audio: AudioInput, **kwargs):
        """Full audio processing, voices and targets in one batch: feature extraction → audio tokenizer latents."""
        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise ValueError(
                f"{self.__class__.__name__} has no audio tokenizer, so audio cannot be encoded: pass the voice as "
                "`audio_prompt_latents` (the checkpoint's `voices/`)."
            )
        if audio_tokenizer.config.encoder_config is None:
            raise ValueError(
                "This processor's audio tokenizer has no encoder (it was loaded with `encoder_config=None`), so only "
                "decoding is available: load it with its encoder to encode audio, or pass the voice as "
                "`audio_prompt_latents` (the checkpoint's `voices/`)."
            )

        audio_inputs = self.feature_extractor(audio, **kwargs).to(audio_tokenizer.device)
        with torch.no_grad():
            encoded = audio_tokenizer.encode(
                audio_inputs["input_values"].to(audio_tokenizer.dtype), audio_inputs["padding_mask"]
            )

        processed_audio = {
            "audio_latents": encoded.latents.float().cpu(),
            "audio_latents_mask": encoded.latents_mask.cpu(),
        }
        audio_replacements = [self.replace_audio_token(processed_audio, audio_idx=idx) for idx in range(len(audio))]
        return processed_audio, audio_replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["audio_latents_mask"][audio_idx].sum())

    def get_text_with_replacements(
        self,
        text: list[str],
        images_replacements: list[str] = [],
        videos_replacements: list[str] = [],
        audio_replacements: list[str] = [],
    ):
        # the chat template writes every audio item as `audio_token`: a training target expands to latent positions
        if audio_replacements:
            is_target = self._audio_is_target(text)
            audio_replacements = [
                replacement.replace(self.audio_token, self.audio_latent_token) if target else replacement
                for replacement, target in zip(audio_replacements, is_target, strict=True)
            ]
        return super().get_text_with_replacements(text, images_replacements, videos_replacements, audio_replacements)

    def split_text(self, text: str, max_tokens: int = 50) -> list[str]:
        r"""
        Splits `text` into sentence-aligned chunks of at most `max_tokens` tokens, as the original's
        `split_into_best_sentences`: the prepared text is cut after sentence ends (not on decimal points), sentences
        longer than `max_tokens` after clause ends, and the pieces are greedily grouped. Each chunk is meant to be one
        conversation of a batch.

        Args:
            text (`str`): The text to split.
            max_tokens (`int`, *optional*, defaults to 50): Maximum number of tokens per chunk.
        """

        def is_decimal_point(tokens: list[int], idx: int) -> bool:
            prefix, suffix = self.tokenizer.decode(tokens[:idx]), self.tokenizer.decode(tokens[idx:])
            return len(prefix) >= 2 and prefix[-1] == "." and prefix[-2].isdigit() and suffix[:1].isdigit()

        def split_after(
            tokens: list[int], boundary_ids: set[int], skip_decimal_points: bool = False
        ) -> list[list[int]]:
            """Splits `tokens` after each run of `boundary_ids`."""
            starts = [0] + [
                idx
                for idx in range(1, len(tokens))
                if tokens[idx - 1] in boundary_ids
                and tokens[idx] not in boundary_ids
                and not (skip_decimal_points and is_decimal_point(tokens, idx))
            ]
            return [tokens[start:end] for start, end in zip(starts, starts[1:] + [len(tokens)])]

        # the text prepared by the chat template and the tokenizer
        prompt = self.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": text}]}])
        for token in self.placeholder_tokens:
            prompt = prompt.replace(token, "")
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        # the markers the vocabulary holds
        sentence_end_ids = set(self.tokenizer.convert_tokens_to_ids([".", "!", "...", "?"])) - {
            self.tokenizer.unk_token_id
        }
        clause_end_ids = set(self.tokenizer.convert_tokens_to_ids([",", ";", ":"])) - {self.tokenizer.unk_token_id}

        segments = []
        for sentence in split_after(tokens, sentence_end_ids, skip_decimal_points=True):
            # sentences too long are cut again after clause ends
            segments += split_after(sentence, clause_end_ids) if len(sentence) > max_tokens else [sentence]
        segments = [(len(segment), self.tokenizer.decode(segment)) for segment in segments]

        chunks, chunk, chunk_len = [], "", 0
        for num_tokens, segment in segments:
            if chunk and chunk_len + num_tokens > max_tokens:
                chunks.append(chunk.strip())
                chunk, chunk_len = "", 0
            chunk = f"{chunk} {segment}" if chunk else segment
            chunk_len += num_tokens
        if chunk:
            chunks.append(chunk.strip())

        return chunks

    def decode(
        self, audio_latents: "torch.Tensor", audio_latents_mask: "torch.Tensor | None" = None
    ) -> "torch.Tensor | list[torch.Tensor]":
        r"""
        Decodes generated latents into waveforms with the audio tokenizer, and fades the first `fade_in_duration` seconds
        of each in, as the original does.

        Args:
            audio_latents (`torch.Tensor` of shape `(batch_size, num_frames, latent_dim)` or `(num_frames, latent_dim)`):
                The latents `generate` returns (`PocketTTSGenerateOutput.audio_latents`), decoded as one batch, or one
                sample's latents.
            audio_latents_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Mask of each sample's frames (`PocketTTSGenerateOutput.audio_latents_mask`). Without it, every row is
                decoded at full length.

        Returns:
            `list[torch.Tensor]` of 1D waveforms on CPU, one per row, or one waveform for one sample's latents.
        """
        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is None:
            raise ValueError(f"{self.__class__.__name__} has no audio tokenizer, so latents cannot be decoded.")
        if audio_tokenizer.config.decoder_config is None:
            raise ValueError(
                "This processor's audio tokenizer has no decoder (it was loaded with `decoder_config=None`), so only "
                "encoding is available: load it with its decoder to decode audio."
            )
        is_single = audio_latents.ndim == 2
        if is_single:
            audio_latents = audio_latents[None]
        if audio_latents_mask is None:
            audio_latents_mask = torch.ones(audio_latents.shape[:2], dtype=torch.long)

        # the decoder is causal: the padding after a row's frames leaves its samples unchanged
        with torch.no_grad():
            audio_values = audio_tokenizer.decode(
                audio_latents.to(audio_tokenizer.device, audio_tokenizer.dtype)
            ).audio_values

        num_samples = audio_latents_mask.sum(-1) * audio_tokenizer.config.frame_size
        fade_in = torch.linspace(0, 1, int(self.fade_in_duration * self.feature_extractor.sampling_rate))
        waveforms = []
        for waveform, length in zip(audio_values[:, 0], num_samples.tolist()):
            waveform = waveform[:length].float().cpu()
            num_fade_samples = min(len(fade_in), len(waveform))
            waveform[:num_fade_samples] *= fade_in[:num_fade_samples]
            waveforms.append(waveform)

        return waveforms[0] if is_single else waveforms

    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        sampling_rate: int | None = None,
    ):
        r"""
        Writes the generated waveform(s) to WAV file(s).

        audio (`torch.Tensor`, `np.ndarray` or a list of them):
            The waveform(s), as returned by [`~PocketTTSProcessor.decode`].
        saving_path (`str`, `Path` or a list of them):
            One path per waveform.
        sampling_rate (`int`, *optional*):
            Sampling rate of the waveforms, the feature extractor's (the audio tokenizer's) by default.
        """
        sampling_rate = self.feature_extractor.sampling_rate if sampling_rate is None else sampling_rate
        # TODO: @eustlb, this should be in AudioProcessor
        requires_backends(self, ["soundfile"])
        audio = make_list_of_audio(audio)
        if isinstance(saving_path, (str, Path)):
            saving_path = [saving_path]
        elif not (isinstance(saving_path, (list, tuple)) and all(isinstance(p, (str, Path)) for p in saving_path)):
            raise ValueError("Invalid input path. Please provide a string, or a list of strings")
        if len(audio) != len(saving_path):
            raise ValueError("The number of audio and saving paths must be the same")
        for audio_value, path in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(path, audio_value, sampling_rate)


__all__ = ["PocketTTSProcessor"]
