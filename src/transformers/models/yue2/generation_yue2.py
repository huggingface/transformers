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

import copy
from dataclasses import dataclass

import torch

from ...cache_utils import DynamicCache
from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin
from ...generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from ...utils import logging


logger = logging.get_logger(__name__)

# Defaults of the YuE2-specific generation parameters, the released checkpoint's values, used when neither the
# `generate` call nor the generation config sets them.
_GENERATION_DEFAULTS = {"repetition_penalty_window": 50, "num_flow_steps": 32, "abc_generation_config": {}}
# Guidance of the semantic tokens when `guidance_scale` is not set: without a symbolic score, the original guides them
# slightly with the instruction alone.
_DEFAULT_GUIDANCE_WITHOUT_SCORE = 1.01


@dataclass
class YuE2GenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of `YuE2ForConditionalGeneration.generate`.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Each prompt followed by its symbolic score (ABC notation, between `abc_start_token_id` and
            `abc_end_token_id`), and its semantic tokens (between `music_start_token_id` and `music_end_token_id`),
            left-padded.
        audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
            The acoustic latents of the songs, one frame per semantic token, right-padded with zeros.
            [`YuE2Processor.decode`] turns them into waveforms.
        audio_latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`):
            Mask of each song's frames (`1`) as opposed to padding (`0`).
    """

    audio_latents: torch.FloatTensor | None = None
    audio_latents_mask: torch.LongTensor | None = None


class YuE2AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    Restricts generation to the token ids in `[start, end)` and to `end_token_id`: the ordinary text tokens for the
    symbolic score, the semantic tokens for the music.

    Args:
        start (`int`):
            First allowed token id.
        end (`int`):
            End (excluded) of the allowed token ids.
        end_token_id (`int`):
            The token id that ends the generation, allowed too.
    """

    def __init__(self, start: int, end: int, end_token_id: int):
        self.start = start
        self.end = end
        self.end_token_id = end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
        allowed[self.start : self.end] = True
        allowed[self.end_token_id] = True
        return scores.masked_fill(~allowed, -float("inf"))


class YuE2RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    YuE2's repetition penalty: each token is penalized by `penalty` to the power of its number of occurrences among the
    last `window` generated tokens (the prompt excluded), dividing positive scores and multiplying negative ones.
    Unlike [`RepetitionPenaltyLogitsProcessor`], the penalty grows with the count and forgets older tokens.

    Args:
        penalty (`float`):
            Penalty per occurrence; 1.0 means no penalty.
        window (`int`):
            Number of last generated tokens considered.
        prompt_length (`int`):
            Length of the prompt, whose tokens are not penalized.
    """

    def __init__(self, penalty: float, window: int, prompt_length: int):
        self.penalty = penalty
        self.window = window
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        recent_ids = input_ids[:, self.prompt_length :][:, -self.window :]
        if self.penalty == 1.0 or recent_ids.shape[1] == 0:
            return scores
        counts = torch.zeros_like(scores).scatter_add_(-1, recent_ids, torch.ones_like(recent_ids, dtype=scores.dtype))
        penalty = self.penalty**counts
        return torch.where(scores < 0, scores * penalty, scores / penalty)


class YuE2ClassifierFreeGuidanceLogitsProcessor(UnbatchedClassifierFreeGuidanceLogitsProcessor):
    """
    Classifier-free guidance as YuE2 applies it: on the raw logits, `unconditional + guidance_scale * (conditional -
    unconditional)`, with the unconditional logits of the negative prompts. Unlike
    [`UnbatchedClassifierFreeGuidanceLogitsProcessor`], there is no log-softmax, which matters for the repetition
    penalty that follows (it depends on the sign of the scores), and the guidance scale is per row.

    Args:
        guidance_scale (`torch.FloatTensor` of shape `(batch_size,)`):
            Guidance scale of each row; rows with a scale of 1 are left unchanged.
        model (`PreTrainedModel`):
            The model, run on the negative prompts.
        unconditional_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The negative prompts, left-padded.
        unconditional_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Their attention mask.
    """

    def __init__(
        self,
        guidance_scale: torch.FloatTensor,
        model,
        unconditional_ids: torch.LongTensor,
        unconditional_attention_mask: torch.LongTensor,
    ):
        super().__init__(1.0, model, unconditional_ids, unconditional_attention_mask, use_cache=True)
        self.guidance_scale = guidance_scale

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        unconditional = self.get_unconditional_logits(input_ids)[:, -1].to(scores.device, scores.dtype)
        guidance_scale = self.guidance_scale[:, None].to(scores.device, scores.dtype)
        guided = unconditional + guidance_scale * (scores - unconditional)
        return torch.where(guidance_scale == 1, scores, guided)


class YuE2GenerationMixin(GenerationMixin):
    def generate(
        self,
        inputs: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        negative_input_ids: torch.LongTensor | None = None,
        negative_attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> YuE2GenerateOutput | list[torch.FloatTensor]:
        r"""
        Generates songs, as the original pipeline: the symbolic score (ABC notation) when the prompt asks for one, then
        the semantic tokens, both with the regular generation loop, then the acoustic latents of each semantic token by
        flow matching. This method overrides [`~generation.utils.GenerationMixin.generate`].

        The two token stages have their own sampling parameters: the semantic tokens use the generation config, and the
        score uses its `abc_generation_config` overrides (`temperature`, `top_p`, `top_k`, `repetition_penalty`,
        `repetition_penalty_window`, `min_new_tokens`, `max_new_tokens`). `repetition_penalty` is YuE2's
        ([`YuE2RepetitionPenaltyLogitsProcessor`], over the last `repetition_penalty_window` generated tokens) and
        `guidance_scale` its classifier-free guidance of the semantic tokens
        ([`YuE2ClassifierFreeGuidanceLogitsProcessor`]); without it, the semantic tokens are guided by 1.01 when the prompt
        has no score and not at all otherwise, as the original. `num_flow_steps` is the number of midpoint steps of the
        flow-matching ODE.

        The flow matching draws its initial noise from the global CPU random generator, the whole song at once, as the
        original does with its seed: call `torch.manual_seed(seed)` before `generate` for reproducible audio.

        Parameters:
            inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The `input_ids` prepared by [`YuE2Processor`], left-padded. Can also be passed as `input_ids=`.
            generation_config (`GenerationConfig`, *optional*):
                The generation config; defaults to the model's.
            negative_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompts of the classifier-free guidance, prepared by [`YuE2Processor`]: the instruction
                alone, to which `generate` appends the song's score.
            negative_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Their attention mask.
            kwargs (`dict[str, Any]`, *optional*):
                `attention_mask`, and generation parameters overriding the generation config.

        Returns:
            [`~models.yue2.generation_yue2.YuE2GenerateOutput`] when `return_dict_in_generate=True` (the checkpoint's
            generation config), with the padded `audio_latents` and their `audio_latents_mask`, which
            [`YuE2Processor.decode`] decodes. Otherwise the list of the songs' latents, one `(num_frames, latent_dim)`
            tensor each.
        """
        input_ids = inputs if inputs is not None else kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("`generate` needs the `input_ids` prepared by `YuE2Processor`.")
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        yue2_kwargs = {key: kwargs.pop(key) for key in _GENERATION_DEFAULTS if key in kwargs}
        # the defaults are filled by each stage's own `generate` call
        generation_config = copy.deepcopy(
            generation_config if generation_config is not None else self.generation_config
        )
        model_kwargs = generation_config.update(**kwargs)
        for key, default in _GENERATION_DEFAULTS.items():
            if key in yue2_kwargs:
                setattr(generation_config, key, yue2_kwargs[key])
            elif getattr(generation_config, key, None) is None:
                setattr(generation_config, key, copy.deepcopy(default))

        config = self.config
        prompts = [row[mask.bool()] for row, mask in zip(input_ids, attention_mask)]

        # 1. The symbolic score, for the prompts that end on an open score
        needs_score = [row[-1].item() == config.abc_start_token_id for row in prompts]
        if any(needs_score):
            score_config = copy.deepcopy(generation_config)
            score_config.update(**generation_config.abc_generation_config, allow_custom_entries=True)
            score_config.guidance_scale = 1.0
            indices = [index for index, needed in enumerate(needs_score) if needed]
            scores, _ = self._generate_tokens(
                [prompts[index] for index in indices],
                score_config,
                allowed_tokens=(0, config.text_vocab_size),
                end_token_id=config.abc_end_token_id,
                **model_kwargs,
            )
            closing = input_ids.new_tensor([config.abc_end_token_id, config.music_start_token_id])
            for index, score in zip(indices, scores):
                prompts[index] = torch.cat([prompts[index], score, closing])

        # 2. The semantic tokens
        scores = [self._get_score(prompt) for prompt in prompts]
        guidance_scale = generation_config.guidance_scale
        guidance_scales = torch.tensor(
            [
                guidance_scale
                if guidance_scale is not None
                else (_DEFAULT_GUIDANCE_WITHOUT_SCORE if score is None else 1.0)
                for score in scores
            ]
        )
        negative_prompts = None
        if (guidance_scales != 1).any():
            if negative_input_ids is None:
                raise ValueError(
                    "Classifier-free guidance needs the negative prompts: pass the `negative_input_ids` and "
                    "`negative_attention_mask` of `YuE2Processor`, or `guidance_scale=1.0`."
                )
            if negative_attention_mask is None:
                negative_attention_mask = torch.ones_like(negative_input_ids)
            negative_prompts = [
                self._get_negative_prompt(row[mask.bool()], score)
                for row, mask, score in zip(negative_input_ids, negative_attention_mask, scores)
            ]
        music_config = copy.deepcopy(generation_config)
        # YuE2's guidance replaces the default one (with a neutral value, see `_generate_tokens`)
        music_config.guidance_scale = 1.0
        codecs, ended = self._generate_tokens(
            prompts,
            music_config,
            allowed_tokens=(config.codec_token_offset, config.codec_token_offset + config.codec_vocab_size),
            end_token_id=config.music_end_token_id,
            guidance_scales=guidance_scales,
            negative_prompts=negative_prompts,
            **model_kwargs,
        )

        # 3. The acoustic latents, by flow matching
        latents = [
            self._sample_audio_latents(prompt, codec, generation_config.num_flow_steps)
            for prompt, codec in zip(prompts, codecs)
        ]

        if not generation_config.return_dict_in_generate:
            return latents

        music_end = input_ids.new_tensor([config.music_end_token_id])
        sequences = [
            torch.cat([prompt, codec, music_end] if has_ended else [prompt, codec])
            for prompt, codec, has_ended in zip(prompts, codecs, ended)
        ]
        sequences, _ = self._left_pad(sequences, generation_config.pad_token_id)
        num_frames = torch.tensor([latent.shape[0] for latent in latents])
        audio_latents = latents[0].new_zeros(len(latents), int(num_frames.max()), config.latent_dim)
        for index, latent in enumerate(latents):
            audio_latents[index, : latent.shape[0]] = latent
        audio_latents_mask = (torch.arange(audio_latents.shape[1]) < num_frames[:, None]).long()
        return YuE2GenerateOutput(
            sequences=sequences, audio_latents=audio_latents, audio_latents_mask=audio_latents_mask
        )

    def _generate_tokens(
        self,
        prompts: list[torch.LongTensor],
        generation_config: GenerationConfig,
        allowed_tokens: tuple[int, int],
        end_token_id: int,
        guidance_scales: torch.FloatTensor | None = None,
        negative_prompts: list[torch.LongTensor] | None = None,
        **model_kwargs,
    ) -> tuple[list[torch.LongTensor], list[bool]]:
        """
        One token stage of the song: generates after each prompt among the `allowed_tokens` ids until `end_token_id`.
        Returns the tokens generated for each prompt, without the end token, and whether it was generated.
        """
        generation_config = copy.deepcopy(generation_config)
        generation_config.eos_token_id = end_token_id
        generation_config.return_dict_in_generate = True
        # YuE2's repetition penalty replaces the default one. Neutral values, not `None`: `generate` would fill `None` back
        # from the model's generation config.
        penalty, generation_config.repetition_penalty = generation_config.repetition_penalty, 1.0
        input_ids, attention_mask = self._left_pad(prompts, generation_config.pad_token_id)

        logits_processor = LogitsProcessorList()
        if negative_prompts is not None:
            negative_ids, negative_mask = self._left_pad(negative_prompts, generation_config.pad_token_id)
            # the guidance applies to the raw logits, before the other processors
            logits_processor.append(
                YuE2ClassifierFreeGuidanceLogitsProcessor(guidance_scales, self, negative_ids, negative_mask)
            )
        logits_processor.append(YuE2AllowedTokensLogitsProcessor(*allowed_tokens, end_token_id))
        if penalty is not None:
            logits_processor.append(
                YuE2RepetitionPenaltyLogitsProcessor(
                    penalty, generation_config.repetition_penalty_window, input_ids.shape[1]
                )
            )

        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            **model_kwargs,
        )
        new_tokens = outputs.sequences[:, input_ids.shape[1] :]
        tokens, ended = [], []
        for row in new_tokens:
            end_positions = (row == end_token_id).nonzero()
            has_ended = len(end_positions) > 0
            tokens.append(row[: end_positions[0, 0]] if has_ended else row)
            ended.append(has_ended)
        return tokens, ended

    def _get_score(self, prompt: torch.LongTensor) -> torch.LongTensor | None:
        """The symbolic score ending the prompt, `None` when the prompt has none (generation without a score)."""
        start = (prompt == self.config.abc_start_token_id).nonzero()[-1, 0]
        end = (prompt == self.config.abc_end_token_id).nonzero()[-1, 0]
        score = prompt[start + 1 : end]
        return score if len(score) > 0 else None

    def _get_negative_prompt(self, instruction: torch.LongTensor, score: torch.LongTensor | None) -> torch.LongTensor:
        """
        The negative prompt of the guidance, as the original: the instruction alone without a score, and the instruction
        followed by the same score otherwise.
        """
        config = self.config
        if score is None:
            return torch.cat([instruction, instruction.new_tensor([config.music_start_token_id])])
        return torch.cat(
            [
                instruction,
                instruction.new_tensor([config.abc_start_token_id]),
                score,
                instruction.new_tensor([config.abc_end_token_id, config.music_start_token_id]),
            ]
        )

    @staticmethod
    def _left_pad(rows: list[torch.LongTensor], pad_token_id: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        length = max(len(row) for row in rows)
        input_ids = rows[0].new_full((len(rows), length), pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, length - len(row) :] = row
            attention_mask[index, length - len(row) :] = 1
        return input_ids, attention_mask

    @torch.no_grad()
    def _sample_audio_latents(
        self, prompt: torch.LongTensor, codec: torch.LongTensor, num_flow_steps: int
    ) -> torch.FloatTensor:
        """
        Generates the acoustic latents of one song, one frame per semantic token, by flow matching. As the original, the
        noise of the whole song is drawn at once (on the CPU, in float32), and songs too long for the context are cut
        into chunks, each conditioned on the prompt and its own semantic tokens.
        """
        config = self.config
        num_frames = len(codec)
        noise = torch.randn(num_frames, config.latent_dim, dtype=torch.float32)
        context = config.max_position_embeddings
        # each chunk holds its prompt, its semantic tokens (and their end), and their latents (and the two frames around)
        chunk_size = min((context - len(prompt) - 3) // 2, context)
        if num_frames == 0 or chunk_size < 1:
            raise ValueError(
                f"No room for the acoustic latents of {num_frames} frames after a prompt of {len(prompt)}."
            )

        music_end = prompt.new_tensor([config.music_end_token_id])
        latents = []
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            tokens = torch.cat([prompt, codec[start:end], music_end])[None].to(self.device)
            past_key_values = DynamicCache(config=config)
            self(input_ids=tokens, past_key_values=past_key_values, use_cache=True, logits_to_keep=1)
            latents.append(self._solve_flow(noise[start:end], past_key_values, num_flow_steps))
        return torch.cat(latents)

    def _solve_flow(self, noise: torch.FloatTensor, past_key_values, num_flow_steps: int) -> torch.FloatTensor:
        """
        Integrates the flow-matching ODE from the noise (time 1) to the latents (time 0) with `num_flow_steps` midpoint
        steps, in the model's dtype, as the original.
        """
        dtype = self.velocity_head.weight.dtype
        state = noise.to(self.device, dtype)[None]
        step_size = 1.0 / num_flow_steps
        for step in range(num_flow_steps):
            time = 1.0 - step * step_size
            velocity = self.get_audio_latent_velocity(state, self._get_flow_timestep(time, dtype), past_key_values)
            midpoint = state - velocity * (step_size / 2)
            velocity = self.get_audio_latent_velocity(
                midpoint, self._get_flow_timestep(time - step_size / 2, dtype), past_key_values
            )
            state = state - velocity * step_size
        return state[0].float().cpu()

    def _get_flow_timestep(self, time: float, dtype: torch.dtype) -> torch.Tensor:
        """
        The model input for flow time `time`, through the original's arithmetic: its logit (in float64, clamped to
        [-20, 20]) back through a sigmoid in the model's dtype, then the shift of the time schedule.
        """
        logit = torch.logit(torch.tensor(time, dtype=torch.float64)).clamp(-20, 20).item()
        timestep = torch.sigmoid(torch.tensor([logit], dtype=dtype, device=self.device))
        shift = self.config.timestep_shift
        return shift * timestep / (1 + (shift - 1) * timestep)
