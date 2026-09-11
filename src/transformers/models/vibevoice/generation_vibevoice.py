# Copyright 2026 The Microsoft Team and The HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ...cache_utils import QuantizedLayer
from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    GenerationState,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING
from ...utils import ModelOutput, add_start_docstrings, is_diffusers_available, logging
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConv1dPaddingCache,
)


logger = logging.get_logger(__name__)


@dataclass
class VibeVoiceGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of VibeVoiceForConditionalGeneration.generate.

    Args:
        audio (`list(torch.FloatTensor)` of length `batch_size`):
            The generated audio.
    """

    audio: list[torch.FloatTensor] | None = None


class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """
    Constrains token generation to only diffusion-related tokens during audio generation, as the role of the
    language model is to emit:
    - another audio-diffusion placeholder (which triggers the diffusion head to synthesize the next acoustic latent)
    - or an EOS token (which signals the end of the audio generation).
    The actual audio comes from the diffusion head, not from sampling the vocabulary.
    """

    def __init__(self, valid_token_ids: list[int], vocab_size: int, device: torch.device = None):
        # Pre-build a fixed 1D logits mask: 0 for allowed tokens, -inf for all others.
        logits_mask = torch.full((vocab_size,), float("-inf"))
        logits_mask[valid_token_ids] = 0.0
        self.logits_mask = logits_mask.to(device)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + self.logits_mask


class VibeVoiceGenerationMixin(GenerationMixin):
    """
    Generation mixin for VibeVoice.

    The language model only decides, at each step, between "another audio latent" (`audio_token_id`) and "stop"
    (`audio_eos_token_id` / `eos_token_id`): the tokens are selected by argmax, and the audio comes from a diffusion
    head conditioned on the last hidden state, sampled with classifier-free guidance (CFG). The decoding step is
    expressed with the per-step hooks of [`~generation.GenerationMixin`]:

    - `_select_next_tokens`: argmax, then one step of CFG diffusion for the rows that asked for an audio latent
      (negative branch forward, denoising loop, acoustic decoding). The audio chunk goes to `state.extras`.
    - `_update_model_kwargs_with_next_tokens`: the next step is fed embeddings instead of the placeholder tokens (the
      embedding of the latent on the rows that just synthesized one), see `prepare_inputs_for_generation`.
    - `_build_generate_output`: assembles the audio, one waveform per batch item.

    The CFG negative (unconditional) branch runs its own forward pass with its own KV cache, set up at the first step
    by `_init_audio_generation`. The CFG logic is adapted from the original VibeVoice implementation, where it all
    lives inline in a single `generate` method:
    https://github.com/vibevoice-community/VibeVoice/blob/07cb79feadd2d3fd7f47530d4c964a12857936a0/vibevoice/modular/modeling_vibevoice_inference.py#L327
    Here it is factored into helpers, and the negative-branch helpers mirror what the base loop does for the positive
    branch (prepare inputs -> forward -> update model kwargs -> append the new token).
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        processors = super()._get_logits_processor(*args, **kwargs)
        valid_tokens = [
            self.config.audio_bos_token_id,
            self.config.audio_eos_token_id,
            self.config.audio_token_id,
            self.config.eos_token_id,
        ]
        processors.append(VibeVoiceTokenConstraintProcessor(valid_tokens, self.config.vocab_size, device=self.device))
        return processors

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs
    ) -> tuple[GenerationConfig, dict]:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].

        It extracts VibeVoice-specific parameters for the generation config.

        VibeVoice-specific parameters include:
        - `noise_scheduler`: A custom noise scheduler instance. Optional: if not provided, a default one is built
            from `noise_scheduler_class`/`noise_scheduler_config` on the generation config (requires `diffusers`).
        - `monitor_progress`: Whether to display a progress bar tracking audio generation. Defaults to `False`.
        - `num_diffusion_steps`: Number of diffusion steps for audio latent sampling. Defaults to `10`.
        """

        # Pop arguments that aren't part of standard GenerationConfig() or for the model's forward pass
        monitor_progress = kwargs.pop("monitor_progress", False)
        num_diffusion_steps = kwargs.pop("num_diffusion_steps", None)
        noise_scheduler_kwarg = kwargs.pop("noise_scheduler", None)

        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

        # Resolve the noise scheduler used to sample audio latents from the diffusion head. Priority:
        #   1. a `noise_scheduler` instance passed to `generate(...)` (custom scheduler),
        #   2. an instance already cached on the generation config (e.g. from a previous call),
        #   3. a default built from `noise_scheduler_class` + `noise_scheduler_config` on the generation config
        #      (this is what the released checkpoints ship with, and requires `diffusers`).
        noise_scheduler = noise_scheduler_kwarg or getattr(generation_config, "noise_scheduler", None)
        if noise_scheduler is None:
            noise_scheduler = self._build_default_noise_scheduler(generation_config)
        if noise_scheduler is None:
            raise ValueError(
                "VibeVoice generation requires a `noise_scheduler`. Either pass one to `generate(...)`, e.g. "
                "`diffusers.DPMSolverMultistepScheduler(beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')`, "
                "or set `noise_scheduler_class` (and optionally `noise_scheduler_config`) on the model's generation config."
            )
        if not (
            hasattr(noise_scheduler, "set_timesteps")
            and hasattr(noise_scheduler, "step")
            and hasattr(noise_scheduler, "timesteps")
        ):
            raise ValueError(
                f"The provided noise_scheduler ({type(noise_scheduler).__name__}) is not compatible with VibeVoice "
                "generation. It must implement `set_timesteps` and `step` methods, and have a `timesteps` attribute."
            )
        generation_config.noise_scheduler = noise_scheduler
        generation_config.monitor_progress = monitor_progress
        generation_config.num_diffusion_steps = (
            num_diffusion_steps or getattr(generation_config, "num_diffusion_steps", None) or 10
        )
        # Fallback to a default guidance scale of 1.0 if not set
        if generation_config.guidance_scale is None:
            generation_config.guidance_scale = 1.0
        if generation_config.do_sample:
            logger.warning_once(
                "VibeVoice generation does not support sampling-based token selection. "
                "Tokens will be selected using argmax regardless of do_sample=True."
            )
        return generation_config, model_kwargs

    @staticmethod
    def _build_default_noise_scheduler(generation_config: GenerationConfig):
        scheduler_class_name = getattr(generation_config, "noise_scheduler_class", "DPMSolverMultistepScheduler")

        if not is_diffusers_available():
            raise ImportError(
                f"The default VibeVoice noise scheduler (`{scheduler_class_name}`) requires `diffusers`. Install it "
                "with `pip install diffusers`, or pass a custom `noise_scheduler` instance to `generate(...)`."
            )

        import diffusers

        try:
            scheduler_class = getattr(diffusers, scheduler_class_name)
        except AttributeError:
            raise ValueError(
                f"Could not find noise scheduler `{scheduler_class_name}` in `diffusers`. Set `noise_scheduler_class` "
                "on the generation config to a valid `diffusers` scheduler, or pass a custom `noise_scheduler`."
            )
        scheduler_config = getattr(
            generation_config,
            "noise_scheduler_config",
            {"beta_schedule": "squaredcos_cap_v2", "prediction_type": "v_prediction"},
        )
        return scheduler_class(**scheduler_config)

    def _prepare_negative_generation(
        self,
        batch_size: int,
        generation_config: GenerationConfig,
        device: torch.device,
    ) -> tuple[torch.LongTensor, dict]:
        """
        Set up the unconditional branch used for classifier-free guidance (CFG).

        This mirrors the positive branch's own setup in `generate` (resolve the generation config, prepare the model
        inputs and KV cache), applied to the negative branch. See the class docstring for the original implementation.

        Returns the initial negative `input_ids` and prepared `model_kwargs` for the negative pass.
        The negative sequence starts with a single `audio_bos_token_id` token and its KV cache is
        sized to match the positive generation's maximum length.
        """
        # Negative branch starts from a single audio_bos token with no prior context
        negative_kwargs = {
            "input_ids": torch.full((batch_size, 1), self.config.audio_bos_token_id, dtype=torch.long, device=device),
            "attention_mask": torch.ones((batch_size, 1), dtype=torch.long, device=device),
        }
        # Reuse the standard config prep to resolve generation defaults for the negative branch
        negative_generation_config, negative_model_kwargs = self._prepare_generation_config(
            generation_config, **negative_kwargs
        )
        # Build the remaining model kwargs (e.g. attention mask) expected by generation
        _, _, negative_model_kwargs = self._prepare_model_inputs(
            None, model_kwargs=negative_model_kwargs, bos_token_id=self.config.audio_bos_token_id
        )
        # Prepare generation config for the negative branch
        self._prepare_special_tokens(negative_generation_config, True, device=device)
        negative_input_ids = negative_kwargs["input_ids"]
        negative_has_default_max_length = (
            negative_kwargs.get("max_length") is None and negative_generation_config.max_length is not None
        )
        negative_has_default_min_length = (
            negative_kwargs.get("min_length") is None and negative_generation_config.min_length is not None
        )
        negative_generation_config = self._prepare_generated_length(
            generation_config=negative_generation_config,
            has_default_max_length=negative_has_default_max_length,
            has_default_min_length=negative_has_default_min_length,
            model_input_name="input_ids",
            inputs_tensor=negative_kwargs["input_ids"],
            input_ids_length=negative_input_ids.shape[1],
        )
        # Allocate the negative branch's own KV cache. A static cache's length is memorized under
        # `_previous_max_negative_cache_length` rather than the positive branch's `_previous_max_cache_length`, so the
        # negative cache is not over-allocated to the positive branch's (much longer, prompt-sized) length.
        self._prepare_cache_for_generation(
            negative_generation_config,
            negative_model_kwargs,
            None,
            batch_size,
            negative_generation_config.max_length - 1,
            max_cache_length_attr="_previous_max_negative_cache_length",
        )
        return negative_input_ids, negative_model_kwargs

    def _get_negative_compiled_call(self, compile_config: GenerationConfig | None):
        """
        Return a `torch.compile`'d version of `self.__call__` dedicated to the CFG negative branch.

        This mirrors [`~PreTrainedModel.get_compiled_call`] but stores the compiled callable under a
        separate attribute (`self._negative_compiled_call`). The negative branch keeps its own
        `StaticCache`, with a different length than the positive branch's.
        """
        compile_config = compile_config or self._default_compile_config()
        if (
            not hasattr(self, "_negative_compiled_call")
            or getattr(self, "_last_negative_compile_config", None) != compile_config
        ):
            self._last_negative_compile_config = compile_config
            self._negative_compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._negative_compiled_call

    def _reset_negative_cache_for_audio_start(
        self,
        diffusion_start_mask: torch.Tensor,
        negative_input_ids: torch.LongTensor,
        negative_model_kwargs: dict,
    ) -> None:
        """
        When `audio_bos_token_id` is generated for a subset of sequences, reset those sequences'
        negative KV cache to a single-token context so the unconditional CFG pass starts fresh.
        """
        attention_mask = negative_model_kwargs["attention_mask"]
        # Build a mask that only attends to the last (new bos) position
        reset_attention_mask = torch.zeros_like(attention_mask)
        reset_attention_mask[:, -1] = 1
        # Only apply the reset mask to sequences that just started a new audio chunk
        negative_model_kwargs["attention_mask"] = torch.where(
            diffusion_start_mask.unsqueeze(-1), reset_attention_mask, attention_mask
        )
        last_input_id = negative_input_ids[:, -1]
        # Overwrite the last input id with audio_bos_token_id for sequences starting a new chunk
        negative_input_ids[:, -1] = torch.where(
            diffusion_start_mask, torch.full_like(last_input_id, self.config.audio_bos_token_id), last_input_id
        )
        if negative_model_kwargs.get("past_key_values") is not None:
            mask_4d = diffusion_start_mask.view(-1, 1, 1, 1)
            # Reset the KV cache
            for layer in negative_model_kwargs["past_key_values"].layers:
                if isinstance(layer, QuantizedLayer):
                    continue
                if layer.keys is not None and layer.values is not None:
                    layer_mask_4d = mask_4d.to(layer.keys.device)
                    layer.keys[:, :, -1:, :] = torch.where(
                        layer_mask_4d, layer.keys[:, :, 0:1, :], layer.keys[:, :, -1:, :]
                    )
                    layer.values[:, :, -1:, :] = torch.where(
                        layer_mask_4d, layer.values[:, :, 0:1, :], layer.values[:, :, -1:, :]
                    )

    def _step_negative_branch(
        self,
        diffusion_mask: torch.Tensor,
        next_tokens: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None,
        negative_input_ids: torch.LongTensor,
        negative_model_kwargs: dict,
        negative_forward: Callable,
    ) -> tuple:
        """
        Advance the negative (unconditional) CFG branch by one step.

        This mirrors the positive branch's per-step logic in the base decoding loop: build the model inputs with
        `prepare_inputs_for_generation`, run the forward pass, advance the cache/kwargs with
        `_update_model_kwargs_for_generation`, and append the new token. It is run here as a separate branch with
        its own `negative_input_ids` and KV cache. See the class docstring for the original implementation.
        """
        use_cache = negative_model_kwargs.get("use_cache", True)
        next_sequence_length = 1 if use_cache else None
        # Prepare inputs for the negative branch's next forward step. Both branches share the same input token, so
        # the positive branch's step embeddings are reused (`prepare_inputs_for_generation` feeds them after the
        # prefill).
        negative_model_inputs = self.prepare_inputs_for_generation(
            negative_input_ids,
            next_sequence_length=next_sequence_length,
            inputs_embeds=inputs_embeds,
            **negative_model_kwargs,
        )
        # Run the unconditional (negative) forward pass
        negative_outputs = negative_forward(**negative_model_inputs, return_dict=True)
        negative_condition = negative_outputs.last_hidden_state[diffusion_mask, -1, :]
        # Advance the negative branch's cache/kwargs for the next generation step
        negative_model_kwargs = self._update_model_kwargs_for_generation(
            negative_outputs, negative_model_kwargs, is_encoder_decoder=False
        )
        # Append the just-generated tokens to the negative branch's sequence
        negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)
        return negative_condition, negative_input_ids, negative_model_kwargs

    def _sample_audio_latent(
        self,
        positive_condition: torch.FloatTensor,
        negative_condition: torch.FloatTensor,
        noise_scheduler: Any,
        num_diffusion_steps: int,
        guidance_scale: float,
    ) -> torch.FloatTensor:
        """
        Run the diffusion denoising loop with classifier-free guidance. Adapted from `sample_speech_tokens` in the
        original implementation: https://github.com/vibevoice-community/VibeVoice/blob/07cb79feadd2d3fd7f47530d4c964a12857936a0/vibevoice/modular/modeling_vibevoice_inference.py#L700
        """
        # Stack positive/negative conditions so the diffusion head can compute both in a single forward pass
        diffusion_head_device = next(self.model.diffusion_head.parameters()).device
        condition = torch.cat([positive_condition, negative_condition], dim=0).to(diffusion_head_device)
        noisy_audio_latent = torch.randn(condition.shape[0], self.config.audio_config.hidden_size).to(condition)
        noise_scheduler.set_timesteps(num_inference_steps=num_diffusion_steps)
        half = len(noisy_audio_latent) // 2
        # Diffusion process
        for timestep in noise_scheduler.timesteps:
            combined = torch.cat([noisy_audio_latent[:half], noisy_audio_latent[:half]], dim=0)
            eps = self.model.diffusion_head(
                combined, timestep.repeat(combined.shape[0]).to(combined), condition=condition
            )
            cond_eps, uncond_eps = torch.split(eps, half, dim=0)
            # Classifier-free guidance (CFG)
            guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([guided_eps, guided_eps], dim=0)
            noisy_audio_latent = noise_scheduler.step(eps, timestep, noisy_audio_latent).prev_sample
        return noisy_audio_latent[:half].unsqueeze(1)

    def _decode_audio_latent(
        self,
        audio_latent: torch.FloatTensor,
        diffusion_mask: torch.Tensor,
        batch_size: int,
        acoustic_cache: VibeVoiceAcousticTokenizerConv1dPaddingCache | None,
    ) -> Any:
        """Decode `audio_latent` to an audio waveform."""
        scaled_latent = audio_latent / self.model.latent_scaling_factor.to(
            audio_latent.device
        ) - self.model.latent_bias_factor.to(audio_latent.device)
        if not diffusion_mask.all():
            padded_latent = torch.zeros(batch_size, *scaled_latent.shape[1:]).to(
                scaled_latent.device, scaled_latent.dtype
            )
            padded_latent[diffusion_mask] = scaled_latent
        else:
            padded_latent = scaled_latent
        return self.model.audio_tower.decode(
            padded_latent.to(self.model.audio_tower.device),
            padding_cache=acoustic_cache,
            use_cache=True,
        )

    def _init_audio_generation(
        self,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        batch_size: int,
        device: torch.device,
    ) -> None:
        """
        First step: set up the CFG negative branch, the audio buffers and the progress bar in `state.extras`. No hook
        runs before the decoding loop with access to `state`, hence the lazy initialization at `state.step == 0`.
        """
        negative_input_ids, negative_model_kwargs = self._prepare_negative_generation(
            batch_size, generation_config, device=device
        )
        negative_forward = (
            self._get_negative_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )
        progress_bar = None
        if generation_config.monitor_progress:
            # `cur_len` is still the prompt length at the first step
            progress_bar = logging.tqdm(
                total=generation_config.max_length - state.cur_len, desc="Generating audio", unit=" tokens"
            )
        state.extras.update(
            negative_input_ids=negative_input_ids,
            negative_model_kwargs=negative_model_kwargs,
            negative_forward=negative_forward,
            inputs_embeds=None,
            acoustic_cache=None,
            semantic_cache=None,
            audio_chunks=[[] for _ in range(batch_size)],
            progress_bar=progress_bar,
        )

    def _select_next_tokens(
        self,
        next_token_scores: torch.FloatTensor,
        generation_config: GenerationConfig,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> torch.LongTensor:
        """
        Argmax token selection (the language model only decides between "another audio latent" and "stop"), then,
        for the rows that asked for an audio latent, one step of the classifier-free-guided diffusion: negative
        branch forward, denoising loop, acoustic decoding. The audio chunk is stored in `state.extras`, and the
        embedding of the latent is kept for `_update_model_kwargs_with_next_tokens`. The diffusion runs here rather
        than in that hook because it needs `generation_config` (noise scheduler, `num_diffusion_steps`,
        `guidance_scale`, `compile_config`), which the update hook does not receive.
        """
        # no pre-loop hook sees `state`: the negative branch and the buffers are created at the first step
        if state.step == 0:
            self._init_audio_generation(
                state,
                generation_config,
                model_kwargs,
                batch_size=next_token_scores.shape[0],
                device=next_token_scores.device,
            )
        extras = state.extras
        if extras["progress_bar"] is not None:
            extras["progress_bar"].update(1)
        next_tokens = torch.argmax(next_token_scores, dim=-1)

        unfinished = state.unfinished_sequences.bool()
        # When audio_bos is predicted, reset the negative branch KV cache so the unconditional CFG pass starts from
        # a clean single-token context for this sequence.
        diffusion_start_mask = unfinished & (next_tokens == self.config.audio_bos_token_id)
        self._reset_negative_cache_for_audio_start(
            diffusion_start_mask, extras["negative_input_ids"], extras["negative_model_kwargs"]
        )

        # When audio_token is predicted, run the diffusion head to synthesize the next audio chunk
        diffusion_mask = unfinished & (next_tokens == self.config.audio_token_id)
        extras["diffusion"] = None
        if diffusion_mask.any():
            # The negative branch is fed the previous step's positive `inputs_embeds` (None at the first step). Its
            # sequence receives the tokens before finished rows are masked; those rows' negative outputs are ignored.
            negative_condition, extras["negative_input_ids"], extras["negative_model_kwargs"] = (
                self._step_negative_branch(
                    diffusion_mask,
                    next_tokens,
                    extras["inputs_embeds"],
                    extras["negative_input_ids"],
                    extras["negative_model_kwargs"],
                    extras["negative_forward"],
                )
            )
            positive_condition = outputs.last_hidden_state[diffusion_mask, -1, :]
            audio_latent = self._sample_audio_latent(
                positive_condition,
                negative_condition,
                generation_config.noise_scheduler,
                generation_config.num_diffusion_steps,
                generation_config.guidance_scale,
            )
            audio_output = self._decode_audio_latent(
                audio_latent, diffusion_mask, next_tokens.shape[0], extras["acoustic_cache"]
            )
            extras["acoustic_cache"] = audio_output.padding_cache
            for i, sample_idx in enumerate(diffusion_mask.nonzero(as_tuple=False).view(-1)):
                extras["audio_chunks"][sample_idx.item()].append(audio_output.audio[i])

            # embedding of the latent: the input of the next LM step for these rows
            semantic_outputs = self.model.semantic_tokenizer_encoder(
                audio_output.audio, padding_cache=extras["semantic_cache"], use_cache=True
            )
            semantic_features = semantic_outputs.latents[diffusion_mask.to(semantic_outputs.latents.device)]
            acoustic_embed = self.model.multi_modal_projector(audio_latent)
            semantic_embed = self.model.semantic_connector(semantic_features)
            extras["diffusion"] = (diffusion_mask, acoustic_embed + semantic_embed.to(acoustic_embed.device))
            extras["semantic_cache"] = semantic_outputs.padding_cache
        return next_tokens

    def _update_model_kwargs_with_next_tokens(
        self,
        next_tokens: torch.LongTensor,
        outputs: ModelOutput | None,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> dict[str, Any]:
        # The next step is fed embeddings, not the placeholder tokens: the token embedding for text-like tokens, the
        # embedding of the audio latent for the rows that just synthesized one.
        next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)
        diffusion = state.extras.get("diffusion")
        if diffusion is not None:
            diffusion_mask, diffusion_embeds = diffusion
            next_inputs_embeds[diffusion_mask] = diffusion_embeds.to(next_inputs_embeds.device)
        # The negative branch reuses them next step. Kept apart from `model_kwargs["inputs_embeds"]`, which at the
        # first step may hold a user-provided prompt while the negative branch must start from `None`.
        state.extras["inputs_embeds"] = next_inputs_embeds
        model_kwargs["inputs_embeds"] = next_inputs_embeds
        # the audio prompt was consumed by the prefill
        model_kwargs.pop("input_values", None)
        model_kwargs.pop("padding_mask", None)
        return model_kwargs

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> VibeVoiceGenerateOutput | list[torch.Tensor | None]:
        output = super()._build_generate_output(sequences, state, generation_config, model_kwargs, **kwargs)
        if state.extras["progress_bar"] is not None:
            state.extras["progress_bar"].close()
        generated_audio = [torch.cat(chunks, dim=-1) if chunks else None for chunks in state.extras["audio_chunks"]]
        if generation_config.return_dict_in_generate:
            return VibeVoiceGenerateOutput(**output, audio=generated_audio)
        # the generated tokens are placeholders; the audio is the result
        return generated_audio
