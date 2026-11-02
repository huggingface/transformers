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

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, is_torch_available
from ...utils.import_utils import requires


if is_torch_available():
    import torch


# The instruction opening the prompt, per `cot` mode: the released checkpoint's.
DEFAULT_INSTRUCTIONS = {
    "off": "Generate music with codec tokens from the given conditions.",
    "melody": (
        "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from "
        "the given conditions."
    ),
    "full": (
        "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given "
        "conditions."
    ),
}


@requires(backends=("torch",))
@auto_docstring
class YuE2Processor(ProcessorMixin):
    audio_tokenizer_class = "YuE2AcousticTokenizerModel"

    def __init__(self, tokenizer, audio_tokenizer=None, instructions: dict[str, str] | None = None):
        r"""
        audio_tokenizer (`YuE2AcousticTokenizerModel`, *optional*):
            The acoustic tokenizer, which decodes the generated latents into audio.
        instructions (`dict[str, str]`, *optional*):
            The instruction opening the prompt for each `cot` mode (`"off"`, `"melody"` and `"full"`). Defaults to the
            released checkpoint's.
        """
        self.instructions = instructions if instructions is not None else dict(DEFAULT_INSTRUCTIONS)
        self.abc_start_token = "<abc>"
        self.abc_end_token = "</abc>"
        self.music_start_token = "<music>"
        super().__init__(tokenizer, audio_tokenizer=audio_tokenizer)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "negative_input_ids", "negative_attention_mask"]

    @auto_docstring
    def __call__(
        self,
        lyrics: str | list[str],
        style: str | list[str],
        cot: str = "full",
        abc: str | list[str] | None = None,
    ) -> BatchFeature:
        r"""
        lyrics (`str` or `list[str]`):
            The lyrics of the song(s), sections marked as in the original (e.g. `[verse]`, `[chorus]`).
        style (`str` or `list[str]`):
            The style prompt of each song: genre, instruments, vocals, mood...
        cot (`str`, *optional*, defaults to `"full"`):
            How the model plans the song before generating it, as the original's `cot`: `"full"` writes a
            chord-annotated symbolic score (ABC notation), `"melody"` a melody-only one, `"off"` none.
        abc (`str` or `list[str]`, *optional*):
            A symbolic score (ABC notation) to follow instead of writing one, e.g. one edited from a previous
            generation, or transcribed from an existing song for a cover. Requires `cot="melody"` or `cot="full"`.

        Returns:
            [`BatchFeature`] with the left-padded `input_ids` and `attention_mask` of the prompts, and the
            `negative_input_ids` and `negative_attention_mask` of the classifier-free guidance.
        """
        if cot not in self.instructions:
            raise ValueError(f"`cot` must be one of {list(self.instructions)}, got {cot!r}.")
        lyrics = [lyrics] if isinstance(lyrics, str) else list(lyrics)
        style = [style] * len(lyrics) if isinstance(style, str) else list(style)
        if abc is None or isinstance(abc, str):
            abc = [abc] * len(lyrics)
        if not len(lyrics) == len(style) == len(abc):
            raise ValueError(
                f"Got {len(lyrics)} lyrics, {len(style)} styles and {len(abc)} scores: pass one per song (or one style "
                "for all)."
            )
        if cot == "off" and any(score is not None for score in abc):
            raise ValueError('A score (`abc`) requires `cot="melody"` or `cot="full"`.')

        tokenizer = self.tokenizer
        bos = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id]
        abc_start, abc_end, music_start = tokenizer.convert_tokens_to_ids(
            [self.abc_start_token, self.abc_end_token, self.music_start_token]
        )
        instruction = self.instructions[cot]
        texts = [f"{instruction}\n[Tags]\n{text_style}\n[Lyrics]\n{text}\n" for text, text_style in zip(lyrics, style)]
        text_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        score_ids = [tokenizer(score, add_special_tokens=False)["input_ids"] if score else None for score in abc]

        prompts = []
        for ids, score in zip(text_ids, score_ids):
            if cot == "off":
                suffix = [abc_start, abc_end, music_start]
            elif score is None:
                # the model writes the score
                suffix = [abc_start]
            else:
                suffix = [abc_start] + score + [abc_end, music_start]
            prompts.append(bos + ids + suffix)
        # the instruction alone: `generate` completes it with each song's score
        negative_ids = bos + tokenizer(instruction, add_special_tokens=False)["input_ids"]

        input_ids, attention_mask = self._left_pad(prompts, tokenizer.pad_token_id)
        negative_input_ids, negative_attention_mask = self._left_pad(
            [negative_ids] * len(prompts), tokenizer.pad_token_id
        )
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "negative_input_ids": negative_input_ids,
                "negative_attention_mask": negative_attention_mask,
            },
            tensor_type="pt",
        )

    @staticmethod
    def _left_pad(rows: list[list[int]], pad_token_id: int) -> tuple[list[list[int]], list[list[int]]]:
        length = max(len(row) for row in rows)
        input_ids = [[pad_token_id] * (length - len(row)) + row for row in rows]
        attention_mask = [[0] * (length - len(row)) + [1] * len(row) for row in rows]
        return input_ids, attention_mask

    def decode(
        self, audio_latents: "torch.Tensor", audio_latents_mask: "torch.Tensor | None" = None
    ) -> list["torch.Tensor"]:
        """
        Decodes generated acoustic latents into waveforms with the acoustic tokenizer, in float32, clipped to [-1, 1] as
        the original.

        Args:
            audio_latents (`torch.Tensor` of shape `(batch_size, num_frames, latent_dim)`):
                The `audio_latents` returned by `generate`.
            audio_latents_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
                Their mask, to decode each song's own frames only.

        Returns:
            `list[torch.Tensor]`: one waveform of shape `(audio_channels, num_samples)` per song, on the CPU, at the
            acoustic tokenizer's `sampling_rate`.
        """
        if self.audio_tokenizer is None:
            raise ValueError("Decoding needs the processor's `audio_tokenizer`.")
        if audio_latents_mask is None:
            audio_latents_mask = torch.ones(audio_latents.shape[:2], dtype=torch.long)
        audio_tokenizer = self.audio_tokenizer
        device = audio_tokenizer.device
        waveforms = []
        for latents, mask in zip(audio_latents, audio_latents_mask):
            latents = latents[mask.bool().to(latents.device)][None].to(device, torch.float32)
            with torch.no_grad():
                audio = audio_tokenizer.decode(latents).audio_values[0]
            waveforms.append(audio.clamp(-1, 1).float().cpu())
        return waveforms


__all__ = ["YuE2Processor"]
