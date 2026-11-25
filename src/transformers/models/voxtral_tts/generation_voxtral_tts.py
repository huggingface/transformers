# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch

from ...cache_utils import DynamicCache
from ...utils import ModelOutput, logging


logger = logging.get_logger(__name__)


@dataclass
class VoxtralTtsGenerateOutput(ModelOutput):
    r"""
    audio (`list[torch.FloatTensor]`):
        Generated waveforms (one 1D tensor per batch element, at `config.codec_config.sampling_rate`).
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_frames, num_codebooks)`):
        Generated audio codes in the model vocabulary, including the final `[END_AUDIO]` frame. Frames generated
        after a sequence has ended are filled with `[EMPTY_AUDIO]` codes.
    audio_lengths (`torch.LongTensor` of shape `(batch_size,)`):
        Number of audio frames of each sequence, excluding the final `[END_AUDIO]` frame.
    """

    audio: list[torch.FloatTensor] | None = None
    audio_codes: torch.LongTensor | None = None
    audio_lengths: torch.LongTensor | None = None


class VoxtralTtsGenerationMixin:
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        voice_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_frames: int = 2000,
        cfg_alpha: float = 1.2,
        generator: torch.Generator | None = None,
        output_audio: bool = True,
        **kwargs,
    ) -> VoxtralTtsGenerateOutput:
        r"""
        Generates speech for the given prompt(s).

        At every step, the hidden state of the last position is turned into one audio frame (37 codes) by the
        acoustic transformer (greedy semantic code, flow-matching acoustic codes); the embedding of that frame is fed
        back to the language model until every sequence emits `[END_AUDIO]`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Prompt token ids, as returned by [`VoxtralTtsProcessor`] (left-padded for batches).
            voice_embeddings (`torch.FloatTensor` of shape `(num_audio_placeholders, hidden_size)`, *optional*):
                Voice preset embeddings replacing the `[AUDIO]` placeholder tokens of the prompt(s).
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention mask of the prompt(s).
            max_new_frames (`int`, *optional*, defaults to 2000):
                Maximum number of audio frames to generate (one frame is 80 ms of audio).
            cfg_alpha (`float`, *optional*, defaults to 1.2):
                Classifier-free guidance scale of the flow-matching velocity.
            generator (`torch.Generator`, *optional*):
                Random generator for the flow-matching noise, for reproducible generations.
            output_audio (`bool`, *optional*, defaults to `True`):
                Whether to decode the generated codes into waveforms with the codec.

        Returns:
            [`VoxtralTtsGenerateOutput`]
        """
        if kwargs:
            raise ValueError(f"Unexpected arguments passed to `generate`: {list(kwargs)}")
        from .modeling_voxtral_tts import EMPTY_AUDIO_TOKEN, END_AUDIO_TOKEN

        batch_size = input_ids.shape[0]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

        cache = DynamicCache(config=self.config.text_config)
        outputs = self.model(
            input_ids=input_ids,
            voice_embeddings=voice_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=True,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1]
        next_position_ids = position_ids[:, -1:] + 1

        frames = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        for _ in range(max_new_frames):
            frame = self.acoustic_transformer.generate_frame(
                last_hidden_state, cfg_alpha=cfg_alpha, generator=generator
            )
            frame[finished] = EMPTY_AUDIO_TOKEN
            frames.append(frame)
            just_finished = (frame[:, 0] == END_AUDIO_TOKEN) & ~finished
            audio_lengths += (~finished & ~just_finished).long()
            finished |= just_finished
            if finished.all():
                break

            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
            outputs = self.model(
                audio_codes=frame[:, None],
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                cache=cache,
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state[:, -1]
            next_position_ids = next_position_ids + 1
        else:
            logger.warning(
                f"Reached `max_new_frames={max_new_frames}` before every sequence emitted `[END_AUDIO]`; the audio may "
                "be truncated."
            )

        audio_codes = torch.stack(frames, dim=1)
        audio = None
        if output_audio:
            num_special = self.config.acoustic_transformer_config.num_special_audio_tokens
            codec_codes = (audio_codes - num_special).clamp(min=0)
            # The codec is causal, so frames past the end of a sequence do not affect its waveform
            audio_values = self.codec_model.decode(codec_codes).audio_values
            hop_length = self.config.codec_config.hop_length
            audio = [audio_values[i, : int(audio_lengths[i]) * hop_length] for i in range(batch_size)]

        return VoxtralTtsGenerateOutput(audio=audio, audio_codes=audio_codes, audio_lengths=audio_lengths)
