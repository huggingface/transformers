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
from typing import TYPE_CHECKING, Any, Optional

import torch

from ...cache_utils import DynamicCache, StaticCache
from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.configuration_utils import ALL_STATIC_CACHE_IMPLEMENTATIONS
from ...generation.logits_process import LOGITS_PROCESSOR_INPUTS_DOCSTRING
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.utils import ALL_CACHE_NAMES, GenerateNonBeamOutput
from ...utils import add_start_docstrings, is_diffusers_available, logging
from ..vibevoice_acoustic_tokenizer.modeling_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConv1dPaddingCache,
)


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer


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

    The classifier-free guidance (CFG) logic (the negative/unconditional branch and the diffusion sampling)
    is adapted from the original VibeVoice implementation, where it all lives inline in a single `generate`
    method:
    https://github.com/vibevoice-community/VibeVoice/blob/07cb79feadd2d3fd7f47530d4c964a12857936a0/vibevoice/modular/modeling_vibevoice_inference.py#L327
    Here it is factored into helpers, and the negative-branch helpers mirror what the positive branch does in
    `_sample` (prepare inputs -> forward -> update model kwargs -> append the new token).
    """

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

        This mirrors the positive branch's own setup in `generate`/`_sample` (resolve the generation config,
        prepare the model inputs and KV cache), applied to the negative branch. See the class docstring for the
        original implementation.

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
        # Allocate/prepare the negative branch's own KV cache, sized to the positive max length. We store it
        # under `_negative_cache` (rather than the positive branch's `_cache`) via the `cache_attr_name` argument
        # of our `_prepare_cache_for_generation` override, so the two branches keep independent caches.
        self._prepare_cache_for_generation(
            negative_generation_config,
            negative_model_kwargs,
            None,
            batch_size,
            negative_generation_config.max_length - 1,
            cache_attr_name="_negative_cache",
        )
        return negative_input_ids, negative_model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        generation_mode: Any,
        batch_size: int,
        max_cache_length: int,
        cache_attr_name: str = "_cache",
    ) -> None:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_cache_for_generation].

        The base implementation always persists the generation cache on the fixed `self._cache` attribute.
        VibeVoice runs a second, independent generation pass for the classifier-free guidance negative branch,
        which needs its own persistent cache. The added `cache_attr_name` argument lets the caller pick which
        attribute the cache lives on (`"_cache"` for the positive branch, `"_negative_cache"` for the negative
        one). Only the code paths exercised by VibeVoice are kept: decoder-only dynamic and static caches.
        """
        cache_name = "past_key_values"

        # Quick escape route 1: if the user specifies a cache, we only need to check for conflicting arguments.
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                    "Cache object) is unsupported. Please use only one of the two."
                )
            if isinstance(user_defined_cache, tuple):
                raise ValueError(
                    "Passing a tuple of `past_key_values` is not supported anymore. Please use a `Cache` instance."
                )
            return

        # Quick escape route 2: if the user specifies no cache is to be used.
        if generation_config.use_cache is False:
            return

        if generation_config.cache_implementation in ALL_STATIC_CACHE_IMPLEMENTATIONS:
            # `max_cache_len` sizes the static cache for the worst case across calls, so later calls with a longer
            # prompt or larger `max_new_tokens` (up to that ceiling) reuse the same cache instead of reallocating
            # (and recompiling under `torch.compile`). See #46424.
            if generation_config.max_cache_len is not None:
                max_cache_length = max(max_cache_length, generation_config.max_cache_len)
            cache_batch_size = max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size
            model_kwargs[cache_name] = self._prepare_static_cache(
                cache_implementation=generation_config.cache_implementation,
                batch_size=cache_batch_size,
                max_cache_len=max_cache_length,
                prefill_chunk_size=generation_config.prefill_chunk_size,
                cache_attr_name=cache_attr_name,
            )
        else:
            # i.e. `cache_implementation` in [None, "dynamic", "offloaded"]
            dynamic_cache_kwargs = {"config": self.config.get_text_config(decoder=True)}
            if generation_config.cache_implementation == "offloaded":
                dynamic_cache_kwargs["offloading"] = True
            model_kwargs[cache_name] = DynamicCache(**dynamic_cache_kwargs)

    def _prepare_static_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        prefill_chunk_size: int | None,
        cache_attr_name: str = "_cache",
    ) -> StaticCache:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_static_cache].

        Same behavior as the base method (a persistent `StaticCache` reused across `generate` calls, only
        reallocated when a larger cache or different batch size is required), except the cache is stored under the
        attribute named by `cache_attr_name` instead of always `self._cache`. This lets the CFG negative branch
        keep an independent `self._negative_cache`. Only the decoder-only path used by VibeVoice is kept.

        Returns the resulting cache object.
        """
        offload_cache = "offloaded" in cache_implementation

        existing_cache = getattr(self, cache_attr_name, None)
        cache_to_check = existing_cache if isinstance(existing_cache, StaticCache) else None

        need_new_cache = (
            cache_to_check is None
            or cache_to_check.offloading != offload_cache
            or cache_to_check.batch_size != batch_size
            or cache_to_check.get_max_length() < max_cache_len
        )

        if need_new_cache:
            cache = StaticCache(
                config=self.config.get_text_config(decoder=True),
                max_cache_len=max_cache_len,
                offloading=offload_cache,
            )
            if prefill_chunk_size is not None:
                # Chunked prefill compiles the prefill, so eagerly init the fresh cache to avoid a recompile next
                # call (#46421). Skipped (-> lazy init) when it can't be initialized on a single device.
                init_shape = self._get_static_cache_init_shape()
                if init_shape is not None:
                    num_heads, head_dim = init_shape
                    cache.early_initialization(
                        batch_size=batch_size,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    )
        else:
            cache = existing_cache
            cache.reset()
        setattr(self, cache_attr_name, cache)
        return cache

    def _get_negative_compiled_call(self, compile_config: GenerationConfig | None):
        """
        Return a `torch.compile`'d version of `self.__call__` dedicated to the CFG negative branch.

        This mirrors [`~PreTrainedModel.get_compiled_call`] but stores the compiled callable under a
        separate attribute (`self._negative_compiled_call`). The negative branch keeps its own
        `StaticCache` (`self._negative_cache`) with a different length than the positive branch.
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

        This mirrors the positive branch's per-step logic in `_sample`: build the model inputs with
        `prepare_inputs_for_generation`, run the forward pass, advance the cache/kwargs with
        `_update_model_kwargs_for_generation`, and append the new token. It is run here as a separate branch with
        its own `negative_input_ids` and KV cache. See the class docstring for the original implementation.
        """
        use_cache = negative_model_kwargs.get("use_cache", True)
        next_sequence_length = 1 if use_cache else None
        # Prepare inputs for the negative branch's next forward step
        negative_model_inputs = self.prepare_inputs_for_generation(
            negative_input_ids, next_sequence_length=next_sequence_length, **negative_model_kwargs
        )
        # Reuse the positive branch's step embeddings since both branches share the same input token
        if negative_model_inputs.get("inputs_embeds") is None and inputs_embeds is not None:
            negative_model_inputs["inputs_embeds"] = inputs_embeds
            negative_model_inputs["input_ids"] = None
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

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        This method overrides [~generation.utils.GenerationMixin._sample].
        To ease maintenance, modifications are marked with the comment "VibeVoice specific".
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # `pad_token_id` is created on `inputs_tensor.device` in `_prepare_special_tokens`. For multimodal models
        # (e.g. BLIP-2, LLaVA) sharded across devices via `device_map="auto"`, `inputs_tensor` (e.g. `pixel_values`
        # on the vision encoder) and `input_ids` (on the language model) can live on different devices, so we need to
        # realign `pad_token_id` with `input_ids` to avoid cross-device ops below.
        if pad_token_id is not None:
            pad_token_id = pad_token_id.to(input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        prefill_consumed = False
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )

        # *************** VibeVoice specific ***************
        noise_scheduler = generation_config.noise_scheduler
        monitor_progress = generation_config.monitor_progress
        num_diffusion_steps = generation_config.num_diffusion_steps
        if do_sample:
            logger.warning_once(
                "VibeVoice generation does not support sampling-based token selection. "
                "Tokens will be selected using argmax regardless of do_sample=True."
            )

        # State tracking
        acoustic_cache, semantic_cache, inputs_embeds = None, None, None
        audio_chunks = [[] for _ in range(batch_size)]
        cur_len = input_ids.shape[1]

        # Setup negative generation for classifier-free guidance
        negative_input_ids, negative_model_kwargs = self._prepare_negative_generation(
            batch_size, generation_config, device=input_ids.device
        )
        negative_forward = (
            self._get_negative_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        # Generation limits for progress tracking
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs["attention_mask"].sum(dim=-1)
        max_step_per_sample = torch.min(
            generation_config.max_length - initial_length_per_sample,
            torch.full_like(initial_length_per_sample, generation_config.max_length - initial_length),
        )
        if monitor_progress:
            progress_bar = logging.tqdm(total=int(max_step_per_sample.max()), desc="Generating audio", unit=" tokens")
        else:
            progress_bar = None
        # ============================================

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # *************** VibeVoice specific ***************
            if progress_bar is not None:
                progress_bar.update(1)
            # ============================================

            if prefill_consumed:
                next_sequence_length = 1 if model_kwargs["use_cache"] else None
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                # *************** VibeVoice specific ***************
                # Subsequent steps use embeddings from previous step
                model_inputs.pop("input_values", None)
                # `padding_mask` is used with `input_values` so we don't need it for subsequent steps
                model_inputs.pop("padding_mask", None)
                model_inputs["inputs_embeds"] = inputs_embeds
                # ============================================
                with self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # token selection
            # *************** VibeVoice specific ***************
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            # ============================================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # *************** VibeVoice specific ***************
            next_inputs_embeds = self.get_input_embeddings()(next_tokens).unsqueeze(1)

            # When audio_bos is predicted, reset the negative branch KV cache so the unconditional
            # CFG pass starts from a clean single-token context for this sequence.
            diffusion_start_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_bos_token_id)
            self._reset_negative_cache_for_audio_start(diffusion_start_mask, negative_input_ids, negative_model_kwargs)

            # When audio_token is predicted, run the diffusion head to synthesize the next audio chunk
            # and compute the embedding for the next LM step.
            diffusion_mask = unfinished_sequences.bool() & (next_tokens == self.config.audio_token_id)
            if diffusion_mask.any():
                negative_condition, negative_input_ids, negative_model_kwargs = self._step_negative_branch(
                    diffusion_mask,
                    next_tokens,
                    inputs_embeds,
                    negative_input_ids,
                    negative_model_kwargs,
                    negative_forward,
                )
                positive_condition = outputs.last_hidden_state[diffusion_mask, -1, :]
                audio_latent = self._sample_audio_latent(
                    positive_condition,
                    negative_condition,
                    noise_scheduler,
                    num_diffusion_steps,
                    generation_config.guidance_scale,
                )
                audio_output = self._decode_audio_latent(audio_latent, diffusion_mask, batch_size, acoustic_cache)
                acoustic_cache = audio_output.padding_cache
                for i, sample_idx in enumerate(diffusion_mask.nonzero(as_tuple=False).view(-1)):
                    audio_chunks[sample_idx.item()].append(audio_output.audio[i])

                # prepare inputs for next LM step
                semantic_outputs = self.model.semantic_tokenizer_encoder(
                    audio_output.audio,
                    padding_cache=semantic_cache,
                    use_cache=True,
                )
                semantic_features = semantic_outputs.latents[diffusion_mask.to(semantic_outputs.latents.device)]
                acoustic_embed = self.model.multi_modal_projector(audio_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed.to(acoustic_embed.device)
                next_inputs_embeds[diffusion_mask] = diffusion_embeds.to(next_inputs_embeds.device)
                semantic_cache = semantic_outputs.padding_cache

            inputs_embeds = next_inputs_embeds
            cur_len += 1
            # ============================================

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        # *************** VibeVoice specific ***************
        if progress_bar is not None:
            progress_bar.close()
        # ============================================

        if streamer is not None:
            streamer.end()

        # *************** VibeVoice specific ***************
        generated_audio = [torch.cat(chunks, dim=-1) if chunks else None for chunks in audio_chunks]
        # ============================================

        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]
            # *************** VibeVoice specific ***************
            return VibeVoiceGenerateOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
                audio=generated_audio,
            )
        else:
            # NOTE (ebezzam): new tokens in input_ids are simply audio tokens (mainly `audio_token_id` to
            # trigger generation) so returning `input_ids` is insufficient for generating audio
            return generated_audio
        # ============================================
