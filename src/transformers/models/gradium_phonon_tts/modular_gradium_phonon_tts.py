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

import torch
from huggingface_hub.dataclasses import strict
from tokenizers import Regex, normalizers
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


# Gradium's runtime prepares the text in a language (`normalize_text` in `ptts/src/preprocess.rs`): the spoken forms of
# `@`, `+` and `=`, and the words of its `numbers` rewrite (`preprocess/rewrite.rs`): the decimal separator, the minus sign
# and the scales from a billion down, as (divisor, a count of one, plural).
LANGUAGES = {
    "en": {
        "spoken": {"@": "at", "+": "plus", "=": "equals"},
        "point": "point",
        "minus": "minus",
        "scales": [(10**9, "1 billion", "billion"), (10**6, "1 million", "million"), (1000, "1 thousand", "thousand")],
    },
    "fr": {
        "spoken": {"@": "arobaze", "+": "plus", "=": "égal"},
        "point": "virgule",
        "minus": "moins",
        "scales": [(10**9, "1 milliard", "milliards"), (10**6, "1 million", "millions"), (1000, "mille", "mille")],
    },
    "de": {
        "spoken": {"@": "ät", "+": "Plus", "=": "Gleich"},
        "point": "Komma",
        "minus": "minus",
        # a bare "1" would be read "eins": German inflects the count like an article
        "scales": [
            (10**9, "eine Milliarde", "Milliarden"),
            (10**6, "eine Million", "Millionen"),
            (1000, "ein Tausend", "Tausend"),
        ],
    },
    "es": {
        "spoken": {"@": "arroba", "+": "mas", "=": "igual"},
        "point": "coma",
        "minus": "menos",
        "scales": [(10**9, "1 mil millones", "mil millones"), (10**6, "1 millón", "millones"), (1000, "mil", "mil")],
    },
    "pt": {
        "spoken": {"@": "arroba", "+": "mais", "=": "igual"},
        "point": "vírgula",
        "minus": "menos",
        "scales": [(10**9, "1 bilhão", "bilhões"), (10**6, "1 milhão", "milhões"), (1000, "mil", "mil")],
    },
}


def rewrite_number(word: str, language: str) -> str | None:
    """
    The runtime's `numbers` rewrite: a large number gets its scale words ("1,234.56" -> "1 thousand 234 point 56"), values
    under a thousand and years stay digits, a leading zero marks a code. `None` when the word is not a number followed by
    punctuation.
    """
    if re.match("0[0-9]", word):
        return None
    if language == "en":
        group, point = ",", "."
    elif language == "fr":
        group, point = "", ","
    else:
        # a dot groups thousands only when every group after the first has three digits: "1.234", not "1.23"
        groups = word.rsplit(",", 1)[0].split(".")
        group, point = ("." if len(groups) > 1 and all(len(g) == 3 for g in groups[1:]) else ""), ","
    match = re.fullmatch(rf"(-?)([0-9{group}]*?[0-9])(?:{re.escape(point)}([0-9]+))?([!?.:,;…]*)", word)
    if match is None:
        return None
    negative, integer, decimal, suffix = match.groups()
    value = int(integer.replace(group, "") if group else integer)
    if value >= 10**12:
        return None
    words = LANGUAGES[language]
    if value < 1000 or 1900 <= value < 2100:
        text = str(value)
    else:
        counts = [(value // divisor % 1000, one, many) for divisor, one, many in words["scales"]]
        parts = [one if count == 1 else f"{count} {many}" for count, one, many in counts if count]
        text = " ".join(parts + ([str(value % 1000)] if value % 1000 else []))
    if decimal is not None:
        text = f"{text} {words['point']} {decimal}"
    return f"{words['minus']} {text}{suffix}" if negative else text + suffix


class GradiumPhononTTSTokenizer(PocketTTSTokenizer, TokenizersBackend):
    """
    Construct a Gradium Phonon TTS tokenizer (backed by HuggingFace's *tokenizers* library), based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    The text is prepared the way Gradium's runtime does, in `language`: see [`~GradiumPhononTTSTokenizer.normalize_text`].
    The number rewriting reads the normalized words, so the normalizer runs before it rather than in the backend
    tokenizer, whose `tokenizer.json` stays the runtime's. Special tokens are left as they are, so a prompt rendered by the
    processor's chat template keeps its placeholders.

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
            transcripts of training samples, which are read as written.
        language (`str`, *optional*, defaults to `"en"`):
            Language the text is normalized in, which sets the spoken forms of symbols and numbers: one of `"en"`,
            `"fr"`, `"de"`, `"es"` and `"pt"`. `None` only collapses the whitespace, upper-cases the first character and
            ends the text with a period (the runtime's `prepare_text_prompt`).
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
        if language is not None and language not in LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}: expected one of {list(LANGUAGES)}, or `None`.")
        self.normalize = normalize
        self.language = language
        self._tokenizer = self._build_tokenizer(
            vocab or [(str(token), 0.0) for token in (unk_token, bos_token, eos_token, pad_token)]
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

    def _normalizer(self, normalize: bool) -> normalizers.Normalizer:
        # the text is prepared in `_preprocess_text`: a second pass would turn a kept `:` (`(:x`) into a comma
        return normalizers.Prepend("▁")

    def _text_normalizers(self) -> list[normalizers.Normalizer]:
        """The steps of the text preparation a normalizer can do, the ones before the number rewriting."""
        steps = []
        if self.language is not None:
            steps = [
                # `;`, `(`, `)` and a `:` not between two characters (as in `10:30`) are pauses: commas
                normalizers.Replace(Regex(r"(?<!\S):|:(?!\S)|[;()]"), ", "),
                normalizers.Replace(Regex("[»«“”„‟″‶⹂❝❞❠〝〞〟＂]"), '"'),
                normalizers.Replace(Regex("[‘’‚‛′‵❛❜❟＇ʼʻʹ´`]"), "'"),
                normalizers.Replace(Regex("[‐‑‒―]"), "-"),
                normalizers.Replace("…", "."),
                *[
                    normalizers.Replace(symbol, f" {word} ")
                    for symbol, word in LANGUAGES[self.language]["spoken"].items()
                ],
                # dashes, brackets, bullets, arrows, emoji and control characters become spaces
                normalizers.Replace(
                    Regex(r"[–*—\[\]{}•‣◦·→←↑↓➡➜\p{Cc}\x{1F600}-\x{1FAFF}\x{2600}-\x{27BF}\x{1F1E6}-\x{1F1FF}]"), " "
                ),
                # a `.` or `,` swallows the whitespace and ASCII punctuation (quotes excepted) before it
                normalizers.Replace(Regex(r"[\s!#-&(-/:-@\[-`{-~]+(?=[.,])"), ""),
            ]
        # then the start of the runtime's `prepare_text_prompt`: collapse the whitespace
        return steps + [normalizers.Replace(Regex(r"\s+"), " "), normalizers.Strip()]

    def _preprocess_text(self, text: str) -> str:
        """The text preparation: the normalizer, the number rewriting, then the capitalization and the final period."""
        text = self._text_normalizer.normalize_str(text)
        if self.language is not None:
            text = " ".join(rewrite_number(word, self.language) or word for word in text.split(" "))
        text = text[:1].upper() + text[1:]
        return text + "." if text[-1:].isalnum() else text

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
        return self._preprocess_text(text)


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
