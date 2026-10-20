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

import unicodedata

import torch
from huggingface_hub.dataclasses import strict
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from torch import nn

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import auto_docstring
from ...utils.type_validators import activation_fn_key, interval, positive_int_field
from ..pocket_tts.configuration_pocket_tts import PocketTTSConfig, PocketTTSFlowHeadConfig
from ..pocket_tts.modeling_pocket_tts import (
    PocketTTSForConditionalGeneration,
    PocketTTSModel,
    PocketTTSPreTrainedModel,
    PocketTTSTimestepNorm,
)
from ..pocket_tts.tokenization_pocket_tts import PocketTTSTokenizer
from .generation_gradium_phonon_tts import GradiumPhononTTSGenerationMixin


@auto_docstring(checkpoint="eustlb/gradium-phonon-tts")
@strict
class GradiumPhononTTSFlowHeadConfig(PocketTTSFlowHeadConfig):
    r"""
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the continuous audio latents the head predicts (the latent dimension of the audio tokenizer).
    condition_dim (`int`, *optional*, defaults to 768):
        Dimension of the backbone hidden state the head is conditioned on (the backbone `hidden_size`).
    frequency_embedding_size (`int`, *optional*, defaults to 256):
        Size of the sinusoidal embedding of the flow times.
    layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the layer normalizations inside the residual blocks and the final layer.
    rms_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the normalization closing the time embeddings.
    max_period (`float`, *optional*, defaults to 10000.0):
        Maximum period of the sinusoidal time embedding, which the checkpoint stores as `freqs`.
    flow_type (`str`, *optional*, defaults to `"lsd"`):
        Objective the head was trained with, which sets how it is conditioned and sampled. `"lsd"` (Lagrangian
        self-distillation): conditioned on the start and end times of a flow step, sampled by Euler steps from Gaussian
        noise. `"drifting"`: no time conditioning, the head maps noise to the latent in one call.
    loss_weighting_hidden_size (`int`, *optional*, defaults to 32):
        Width of the MLP that learns the uncertainty weighting of the LSD loss terms from their flow times (training
        only).
    loss_weighting_num_layers (`int`, *optional*, defaults to 3):
        Number of hidden layers of that MLP.
    """

    condition_dim: int = positive_int_field(default=768)


@auto_docstring(checkpoint="eustlb/gradium-phonon-tts")
@strict
class GradiumPhononTTSConfig(PocketTTSConfig):
    r"""
    flow_head_config (`Union[GradiumPhononTTSFlowHeadConfig, dict]`, *optional*):
        The config of the flow head that samples each latent from the backbone hidden state.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the layer normalizations of the backbone.
    latent_dim (`int`, *optional*, defaults to 32):
        Dimension of the audio latents the backbone takes as input (`audio_latents`); equals the flow head's and the audio
        tokenizer's `latent_dim`.
    audio_prompt_token_id (`int`, *optional*, defaults to 5):
        Placeholder token id of the voice-prompt frames in `input_ids` (`<|audio_prompt|>`).
    audio_latent_token_id (`int`, *optional*, defaults to 6):
        Placeholder token id of the audio latent positions in `input_ids` (`<|audio_latent|>`).
    audio_bos_token_id (`int`, *optional*, defaults to 4):
        Token id of the first audio position, closing the prompt: it holds the learnt BOS latent, from which the first
        frame is generated (`<|audio_bos|>`).
    frame_rate (`float`, *optional*, defaults to 12.5):
        Number of audio latents per second of audio: `generate` derives its default frame budget from it.
    eos_loss_weight (`float`, *optional*, defaults to 0.1):
        Weight of the end-of-speech loss in the training loss.
    flow_loss_num_samples (`int`, *optional*, defaults to 4):
        Number of noise draws per target frame in the flow loss.
    lsd_distill_probability (`float`, *optional*, defaults to 0.25):
        Probability, per training step, of computing the self-distillation term of the LSD loss (reweighted to keep the
        expected loss); 1 computes it at every step.
    lsd_diagonal_weight (`float`, *optional*, defaults to 0.75):
        Weight of the diagonal (instantaneous flow) term of the LSD loss; the self-distillation term weighs the rest.
    lsd_time_logit_mean (`float`, *optional*, defaults to 0.4):
        Mean of the logit-normal distribution of the flow times drawn by the LSD loss.
    lsd_time_logit_std (`float`, *optional*, defaults to 1.0):
        Standard deviation of the logit-normal distribution of the flow times drawn by the LSD loss.
    latent_stats_ema_steps (`int`, *optional*, defaults to 1000):
        Number of first training steps during which `latent_mean` and `latent_std` follow an exponential moving average
        of the target latents; 0 keeps them frozen.
    latent_stats_ema_decay (`float`, *optional*, defaults to 0.999):
        Decay of that moving average.
    audio_prompt_dim (`int`, *optional*, defaults to 512):
        Dimension of the voice-prompt latents (`audio_prompt_latents`), the unquantized latents of Gradium's speaker
        codec (a Mimi encoder, not the audio tokenizer's), which the backbone projects to `hidden_size`.
    num_speakers (`int`, *optional*, defaults to 1):
        Value of the `num_speakers` condition: the row of its lookup table added to every audio latent position.
    num_speakers_vocab_size (`int`, *optional*, defaults to 32):
        Number of rows of the `num_speakers` lookup table.
    num_speakers_embedding_dim (`int`, *optional*, defaults to 16):
        Width of the `num_speakers` lookup table, projected to `hidden_size`.
    duration_delta (`float`, *optional*, defaults to 0.0):
        Value of the `duration_delta` condition, embedded sinusoidally and added to every audio latent position.
    duration_delta_embedding_dim (`int`, *optional*, defaults to 128):
        Size of the sinusoidal embedding of `duration_delta`, projected to `hidden_size`.
    duration_delta_scale (`float`, *optional*, defaults to 1000.0):
        Factor applied to `duration_delta` before its sinusoidal embedding.
    duration_delta_max_period (`float`, *optional*, defaults to 10000.0):
        Maximum period of the sinusoidal embedding of `duration_delta`.
    """

    vocab_size: int = positive_int_field(default=4001)
    hidden_size: int = positive_int_field(default=768)
    num_attention_heads: int = positive_int_field(default=12)
    intermediate_size: int = positive_int_field(default=3072)
    hidden_act: str = activation_fn_key(default="gelu")
    audio_prompt_dim: int = positive_int_field(default=512)
    audio_prompt_token_id: int = 5
    audio_latent_token_id: int = 6
    audio_bos_token_id: int = 4
    num_speakers: int = interval(min=0)(default=1)
    num_speakers_vocab_size: int = positive_int_field(default=32)
    num_speakers_embedding_dim: int = positive_int_field(default=16)
    duration_delta: float = 0.0
    duration_delta_embedding_dim: int = positive_int_field(default=128)
    duration_delta_scale: float = 1000.0
    duration_delta_max_period: float = interval(min=0.0)(default=10000.0)

    # Difference to Pocket TTS: no learnt vector before the voice prompt
    audio_prompt_bos_token_id = AttributeError()


class GradiumPhononTTSTimestepNorm(PocketTTSTimestepNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_square = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(mean_square + self.variance_epsilon)


class GradiumPhononTTSPreTrainedModel(PocketTTSPreTrainedModel): ...


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS backbone: a causal transformer over one stream made of the voice prompt (speaker-codec
    latents), the text tokens and the audio latents, returning the normalised hidden states.
    """
)
class GradiumPhononTTSModel(PocketTTSModel):
    def __init__(self, config: GradiumPhononTTSConfig):
        super().__init__(config)
        self.audio_prompt_proj = nn.Linear(config.audio_prompt_dim, config.hidden_size, bias=False)

        self.num_speakers_embed = nn.Embedding(config.num_speakers_vocab_size, config.num_speakers_embedding_dim)
        self.num_speakers_proj = nn.Linear(config.num_speakers_embedding_dim, config.hidden_size, bias=False)
        self.duration_delta_proj = nn.Linear(config.duration_delta_embedding_dim, config.hidden_size, bias=False)

    def get_condition_embeds(self, device: torch.device) -> torch.Tensor:
        """The sum of the conditions, one `hidden_size` vector (built on `device`, the modules called for offloading)."""
        num_speakers = torch.tensor(self.config.num_speakers, device=device)
        condition_embeds = self.num_speakers_proj(self.num_speakers_embed(num_speakers))
        half_dim = self.config.duration_delta_embedding_dim // 2
        exponents = torch.arange(half_dim, device=device, dtype=torch.float32) / (half_dim - 1)
        phases = self.config.duration_delta_scale * self.config.duration_delta
        phases = phases / self.config.duration_delta_max_period**exponents
        duration_embeds = torch.cat([phases.cos(), phases.sin()]).to(condition_embeds.dtype)
        condition_embeds = condition_embeds + self.duration_delta_proj(duration_embeds).to(condition_embeds.device)
        return condition_embeds

    def embed_audio_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """
        Embeds normalised audio latents (the learnt BOS latent or the generated ones) into the backbone stream, with the
        speaker conditioning added to every audio latent position.
        """
        latent_embeds = self.audio_latent_proj(audio_latents)
        return latent_embeds + self.get_condition_embeds(latent_embeds.device).to(latent_embeds.device)


# Text normalisation of Gradium's runtime (`ptts/src/preprocess.rs`, `preprocess/rewrite.rs`): characters the model never
# saw (typographic quotes, bullets, arrows, emoji) are folded or dropped, symbols that are read aloud (`@`, `+`, `=`) and
# large numbers are spelled out in the language, and `;`, parentheses and a spaced `:` become commas.
SUPPORTED_LANGUAGES = ("en", "fr", "de", "es", "pt")
SPOKEN_SYMBOLS = {
    "en": {"@": "at", "+": "plus", "=": "equals"},
    "fr": {"@": "arobaze", "+": "plus", "=": "égal"},
    "de": {"@": "ät", "+": "Plus", "=": "Gleich"},
    "es": {"@": "arroba", "+": "mas", "=": "igual"},
    "pt": {"@": "arroba", "+": "mais", "=": "igual"},
}
DECIMAL_SEPARATORS = {"en": "point", "fr": "virgule", "de": "Komma", "es": "coma", "pt": "vírgula"}
MINUS_WORDS = {"en": "minus", "fr": "moins", "de": "minus", "es": "menos", "pt": "menos"}
# scale words from a billion down: (divisor, singular, plural)
SCALES = {
    "en": [(1_000_000_000, "billion", "billion"), (1_000_000, "million", "million"), (1000, "thousand", "thousand")],
    "fr": [(1_000_000_000, "milliard", "milliards"), (1_000_000, "million", "millions"), (1000, "mille", "mille")],
    "de": [
        (1_000_000_000, "Milliarde", "Milliarden"),
        (1_000_000, "Million", "Millionen"),
        (1000, "Tausend", "Tausend"),
    ],
    "es": [(1_000_000_000, "mil millones", "mil millones"), (1_000_000, "millón", "millones"), (1000, "mil", "mil")],
    "pt": [(1_000_000_000, "bilhão", "bilhões"), (1_000_000, "milhão", "milhões"), (1000, "mil", "mil")],
}
DOUBLE_QUOTES = "»«“”„‟″‶⹂❝❞❠〝〞〟＂"
SINGLE_QUOTES = "‘’‚‛′‵❛❜❟＇ʼʻʹ´`"
DASHES = "‐‑‒―"
DROPPED_TO_SPACE = "–*—[]{}•‣◦·→←↑↓➡➜"
NUMBER_SUFFIX_CHARS = "!?.:,;…"


def push_char(buffer: list[str], char: str):
    """Appends `char`; a `.` or `,` first swallows the whitespace and ASCII punctuation (quotes excepted) before it."""
    if char in ".,":
        while buffer and (
            buffer[-1].isspace() or (buffer[-1].isascii() and not buffer[-1].isalnum() and buffer[-1] not in "\"'")
        ):
            buffer.pop()
    buffer.append(char)


def push_whitespace(buffer: list[str]):
    if buffer and not buffer[-1].isspace():
        push_char(buffer, " ")


def rewrite_number(word: str, language: str) -> str | None:
    """
    Large numbers with their scale words ("1,234.56" -> "1 thousand 234 point 56"); values under a thousand and years stay
    digits, a leading zero marks a code. Returns `None` when the rule does not claim the word.
    """
    if len(word) > 1 and word[0] == "0" and word[1].isdigit() and word[1].isascii():
        return None
    # English groups thousands with commas and marks decimals with a dot, French does neither; the other languages mark
    # decimals with a comma and group with dots only when every group after the first has three digits
    if language == "en":
        group, point = ",", "."
    elif language == "fr":
        group, point = None, ","
    else:
        integer = word.rsplit(",", 1)[0] if "," in word else word
        grouped = "." in integer and all(i == 0 or len(part) == 3 for i, part in enumerate(integer.split(".")))
        group, point = ("." if grouped else None), ","

    negative = word.startswith("-")
    rest = word[1:] if negative else word
    end = next(
        (i for i, char in enumerate(rest) if not (char.isascii() and char.isdigit()) and char != group), len(rest)
    )
    integer = rest[:end]
    if group is not None:
        integer = integer.rstrip(group)
    if not integer:
        return None
    rest = rest[len(integer) :]

    decimal = None
    if rest.startswith(point):
        after = rest[1:]
        decimal_end = next((i for i, char in enumerate(after) if not (char.isascii() and char.isdigit())), len(after))
        if decimal_end > 0:
            decimal, rest = after[:decimal_end], after[decimal_end:]
    # only trailing punctuation may follow the number
    if any(char not in NUMBER_SUFFIX_CHARS for char in rest):
        return None

    digits = "".join(char for char in integer if char.isascii() and char.isdigit())
    if not digits or int(digits) >= 1_000_000_000_000:
        return None
    value = int(digits)
    if value < 1000 or 1900 <= value < 2100:
        words = str(value)
    else:
        parts = []
        for divisor, one, many in SCALES[language]:
            count = (value // divisor) % 1000
            if count == 1:
                if language in ("fr", "es", "pt") and divisor == 1000:
                    # "mille" and "mil" are numeral adjectives: no "1" in front of them
                    parts.append(one)
                elif language == "de":
                    # a bare "1" would be read "eins": German inflects the count like an article
                    parts.append(f"ein {one}" if divisor == 1000 else f"eine {one}")
                else:
                    parts.append(f"1 {one}")
            elif count > 1:
                parts.append(f"{count} {many}")
        if value % 1000 != 0:
            parts.append(str(value % 1000))
        words = " ".join(parts)
    if decimal is not None:
        words = f"{words} {DECIMAL_SEPARATORS[language]} {decimal}"
    if negative:
        words = f"{MINUS_WORDS[language]} {words}"
    return words + rest


class GradiumPhononTTSTokenizer(PocketTTSTokenizer, TokenizersBackend):
    """
    Construct a Gradium Phonon TTS tokenizer (backed by HuggingFace's *tokenizers* library), based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    Before tokenization, the text is prepared the way Gradium's runtime does, in `language`: see
    [`~GradiumPhononTTSTokenizer.normalize_text`]. Only the text between special tokens is prepared, so a prompt rendered
    by the processor's chat template keeps its placeholders.

    This tokenizer inherits from [`TokenizersBackend`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab (`list[tuple[str, float]]`, *optional*):
            The Unigram vocabulary, as `(piece, score)` pairs. If not provided, a minimal vocabulary is created with the
            special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding.
        normalize (`bool`, *optional*, defaults to `True`):
            Whether to prepare the text before tokenization. Can be overridden per call, e.g. `normalize=False` for the
            transcripts of training samples, which are kept as written.
        language (`str`, *optional*, defaults to `"en"`):
            Language the text is normalized in, which sets the spoken forms of symbols and numbers: one of `"en"`,
            `"fr"`, `"de"`, `"es"` and `"pt"`. `None` skips the normalization and only applies the runtime's
            `prepare_text_prompt`.
    """

    def __init__(
        self,
        vocab: list[tuple[str, float]] | None = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        normalize: bool = True,
        language: str | None = "en",
        **kwargs,
    ):
        if language is not None and language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}: expected one of {SUPPORTED_LANGUAGES}, or `None`.")
        self.normalize = normalize
        self.language = language

        if vocab is None:
            vocab = [(str(unk_token), 0.0), (str(bos_token), 0.0), (str(eos_token), 0.0), (str(pad_token), 0.0)]
        self._tokenizer = Tokenizer(Unigram(vocab, unk_id=0, byte_fallback=True))
        self._tokenizer.normalizer = normalizers.Prepend("▁")
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        self._tokenizer.decoder = decoders.Sequence(
            [
                decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )

        TokenizersBackend.__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            normalize=normalize,
            language=language,
            **kwargs,
        )

    def normalize_text(self, text: str) -> str:
        r"""
        Prepares `text` the way Gradium's runtime does. Its `normalize_text`, in `language`: typographic quotes and dashes
        are folded to ASCII, bullets, arrows, emoji and control characters become spaces, `…` a period, `@`, `+` and `=`
        their spoken forms, `;`, `(`, `)` and a `:` not between two characters a comma, punctuation does not pile up
        before a `.` or `,`, and large numbers get their scale words. Then its `prepare_text_prompt`: collapse the
        whitespace, upper-case the first character, end text whose last character is alphanumeric with a period.

        Args:
            text (`str`): The text to prepare.

        Returns:
            `str`: The prepared text.
        """
        if self.language is not None:
            spoken = SPOKEN_SYMBOLS[self.language]
            buffer = []
            for index, char in enumerate(text):
                previous = text[index - 1] if index > 0 else None
                following = text[index + 1] if index + 1 < len(text) else None
                if char in DOUBLE_QUOTES:
                    push_char(buffer, '"')
                elif char in SINGLE_QUOTES:
                    push_char(buffer, "'")
                elif char in DASHES:
                    push_char(buffer, "-")
                elif char in DROPPED_TO_SPACE:
                    push_whitespace(buffer)
                elif char == "…":
                    push_char(buffer, ".")
                elif char in spoken:
                    push_whitespace(buffer)
                    for spoken_char in spoken[char]:
                        push_char(buffer, spoken_char)
                    push_whitespace(buffer)
                elif (
                    char == ":"
                    and previous is not None
                    and not previous.isspace()
                    and following is not None
                    and not following.isspace()
                ):
                    # a `:` between two non-space characters, as in `10:30`, is kept
                    push_char(buffer, ":")
                elif char in ";:()":
                    while buffer and buffer[-1].isspace():
                        buffer.pop()
                    push_char(buffer, ",")
                    push_whitespace(buffer)
                elif (
                    0x1F600 <= ord(char) <= 0x1FAFF
                    or 0x2600 <= ord(char) <= 0x27BF
                    or 0x1F1E6 <= ord(char) <= 0x1F1FF
                    or unicodedata.category(char) == "Cc"
                    or char.isspace()
                ):
                    # emoji, control characters and whitespace
                    push_whitespace(buffer)
                else:
                    push_char(buffer, char)
            words = "".join(buffer).rstrip().split(" ")
            text = " ".join(rewrite_number(word, self.language) or word for word in words)

        text = " ".join(text.replace("\n", " ").replace("\r", " ").split())
        if text:
            text = text[0].upper() + text[1:]
            if text[-1].isalnum():
                text += "."
        return text


@auto_docstring(
    custom_intro="""
    The Gradium Phonon TTS text-to-speech model: the backbone, the flow head that samples one audio latent per step and
    the end-of-speech head. It generates audio tokenizer latents, which the processor decodes into 24 kHz audio.
    """
)
class GradiumPhononTTSForConditionalGeneration(PocketTTSForConditionalGeneration, GradiumPhononTTSGenerationMixin):
    # Gradium's runtime checkpoint also holds the codec (the processor's audio tokenizer) and training-only tensors
    _keys_to_ignore_on_load_unexpected = [r"^mimi\.", r"^flow_lm\.flow\.w_s_t\.", r"\.learnt_padding$"]


__all__ = [
    "GradiumPhononTTSConfig",
    "GradiumPhononTTSFlowHeadConfig",
    "GradiumPhononTTSPreTrainedModel",
    "GradiumPhononTTSModel",
    "GradiumPhononTTSForConditionalGeneration",
    "GradiumPhononTTSTokenizer",
]
