# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ...cache_utils import (
    Cache,
    DynamicCache,
    QuantizedCache,
    StaticCache,
)
from ...generation import (
    EosTokenCriteria,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from ...generation.configuration_utils import (
    ALL_CACHE_IMPLEMENTATIONS,
    ALL_STATIC_CACHE_IMPLEMENTATIONS,
    DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS,
    STATIC_CACHE_IMPLEMENTATIONS,
)
from ...generation.streamers import BaseStreamer
from ...modeling_outputs import ModelOutput
from ...utils import logging


logger = logging.get_logger(__name__)


# TODO(joaogante): block audio and video tokens from gemma4 from being sampled? (some logits processor)
class DiffusionGemmaGenerationConfig(GenerationConfig):
    # no-format
    """
    A GenerationConfig class with parameterization customized for [`DiffusionGemmaGenerationMixin.generate`].

    Args:
        > Parameters that control the length of the output

        max_new_tokens (`int`, *optional*):
            The maximum number of tokens to generate, ignoring the number of tokens in the prompt.
        max_length (`int`, *optional*):
            The maximum length of the output sequence. `max_new_tokens` is recommended for controlling how many tokens
            the model generates.

        > Diffusion parameters

        max_denoising_steps (`int`):
            The maximum number of denoising steps to perform.
        sampler_config (`EntropyBoundSamplerConfig`):
            The configuration for the sampler. See [`EntropyBoundSampler`] to learn how a sampler operates in a
            text diffusion model.
        t_min (`float`):
            The final temperature in the schedule, i.e. at the last denoising step. See
            [`LinearTemperatureScheduleLogitsProcessor`] for more details.
        t_max (`float`):
            The initial temperature in the schedule, i.e. at the first denoising step. See
            [`LinearTemperatureScheduleLogitsProcessor`] for more details.
        stability_threshold (`int`):
            The number of steps for which the accepted canvas must be the same to trigger the stopping criteria.
            See [`StableAndConfidentStoppingCriteria`] for more details.
        confidence_threshold (`float`):
            The threshold for the mean of the entropy of temperature-scaled logits to trigger the stopping criteria.
            See [`StableAndConfidentStoppingCriteria`] for more details.

        > Parameters that control the cache

        cache_implementation (`str`, *optional*):
            Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:

            - `"dynamic"`: [`DynamicCache`]
            - `"static"`: [`StaticCache`]
            - `"offloaded"`: [`DynamicCache(offloaded=True)`]
            - `"offloaded_static"`: [`StaticCache(offloaded=True)`]
            - `"quantized"`: [`QuantizedCache`]

            If none is specified, we will use the default cache for the model (which is often [`DynamicCache`]). See
            our [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) for further information.
        cache_config (`dict`, *optional*, default to `None`):
            Arguments used in the key-value cache class can be passed in `cache_config`.

        > Special tokens that can be used at generation time

        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, list[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, **kwargs):
        # TODO(joao): test other common `GenerationConfig` flags like top-k, and whitelist them.

        # We intentionally DON'T call super().__init__(): we don't want most of the attributes of the parent class.

        # Parameters that control the length of the output
        self.max_new_tokens: int | None = kwargs.pop("max_new_tokens", None)
        self.max_length: int | None = kwargs.pop("max_length", None)

        # Diffusion parameters
        # There can be only one sampler at a time, but multiple logits processors and/or stopping criteria.
        self.max_denoising_steps: int = kwargs.pop("max_denoising_steps", None)
        self.sampler_config: EntropyBoundSamplerConfig = kwargs.pop("sampler_config", None)
        self.t_min: float = kwargs.pop("t_min", None)
        self.t_max: float = kwargs.pop("t_max", None)
        self.stability_threshold: int = kwargs.pop("stability_threshold", None)
        self.confidence_threshold: float = kwargs.pop("confidence_threshold", None)

        # Parameters that control the cache
        self.cache_implementation: str | None = kwargs.pop("cache_implementation", None)
        self.cache_config: dict[str, Any] | None = kwargs.pop("cache_config", None)

        # Special tokens that can be used at generation time
        self.bos_token_id: int | None = kwargs.pop("bos_token_id", None)
        self.pad_token_id: int | None = kwargs.pop("pad_token_id", None)
        self.eos_token_id: list[int] | int | None = kwargs.pop("eos_token_id", None)

        # Metadata
        self._commit_hash: str | None = kwargs.pop("_commit_hash", None)
        self._from_model_config: bool | None = kwargs.pop("_from_model_config", None)
        self.transformers_version: str | None = kwargs.pop("transformers_version", None)

        # kwargs must be empty at this point. If it is not, then it received unexpected kwargs.
        if len(kwargs) > 0:
            raise ValueError(f"Unexpected kwargs: {kwargs.keys()}")

        # Validate the values of the attributes
        self._resolve_dataclasses()
        self.validate()

    def validate(self, **unused_kwargs):
        # 1. Diffusion-specific attributes
        if self.max_denoising_steps is not None and (
            not isinstance(self.max_denoising_steps, int) or self.max_denoising_steps <= 0
        ):
            raise ValueError(f"`max_denoising_steps` must be a positive integer, but got {self.max_denoising_steps}")
        if self.sampler_config is not None and not isinstance(self.sampler_config, (EntropyBoundSamplerConfig)):
            raise ValueError(
                f"`sampler_config` must be an instance of `EntropyBoundSamplerConfig`, but got {type(self.sampler_config)}"
            )

        if self.t_min is not None and self.t_min < 0:
            raise ValueError(f"`t_min` must be >= 0.0 (got {self.t_min})")
        if self.t_max is not None and self.t_max < 0:
            raise ValueError(f"`t_max` must be >= 0.0 (got {self.t_max})")
        if self.t_min is not None and self.t_max is not None and self.t_max <= self.t_min:
            raise ValueError(f"`t_max` must be >= t_min` (got {self.t_max} < {self.t_min})")

        if self.stability_threshold is not None and (
            not (isinstance(self.stability_threshold, int)) or self.stability_threshold < 0
        ):
            raise ValueError(f"`stability_threshold` must be an integer >= 0 (got {self.entropy_bound})")
        if self.confidence_threshold is not None and (
            not (isinstance(self.confidence_threshold, float)) or self.confidence_threshold <= 0
        ):
            raise ValueError(f"`confidence_threshold` must be a float > 0 (got {self.entropy_bound})")

        # 2. Other attributes (often used in AR)
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"`max_length` must be a positive integer, but got {self.max_length}")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be a positive integer, but got {self.max_new_tokens}")
        if self.cache_implementation is not None and self.cache_implementation not in ALL_CACHE_IMPLEMENTATIONS:
            raise ValueError(
                f"`cache_implementation` must be one of {ALL_CACHE_IMPLEMENTATIONS}, but got "
                f"{self.cache_implementation}"
            )

    def _resolve_dataclasses(self):
        """
        At serialization time, dataclasses get stored as a dictionary with an extra "_cls_name" field.
        This function converts those dictionaries back into their dataclass format, if they exist.

        NOTE: this dictionary input format is intentionally not documented in __init__, to ensure
        users use the dataclasses -- they have built-in validation.
        """
        # Assumption: all dataclasses that we want to load can be instantiated in this file
        current_module = sys.modules[__name__]

        for attr_name in ("sampler_config",):
            attr = getattr(self, attr_name)
            # Load the right dataclass using the `_cls_name` field
            if isinstance(attr, dict):
                cls_name = attr.pop("_cls_name", None)
                config_dataclass = getattr(current_module, cls_name)
                loaded_attr = config_dataclass(**attr)
                setattr(self, attr_name, loaded_attr)

    @staticmethod
    def _get_default_generation_params() -> dict[str, Any]:
        """
        Defaults to be applied when unset by the model OR by the user, such that `model.generate()` works with minimal
        parameterization.

        Pretrained checkpoints should set these as appropriate in their `generation_config.json`, to establish
        a better default baseline. Be mindful that tests may use use these values.
        """
        return {
            "max_new_tokens": 256,
            "max_denoising_steps": 48,
            "sampler_config": EntropyBoundSamplerConfig(entropy_bound=0.1),
            "t_min": 0.4,
            "t_max": 0.8,
            "stability_threshold": 1,
            "confidence_threshold": 0.005,
        }

    # Overriding GenerationMixin-related functions that are not relevant to DiffusionGemma.
    # (These functions being tightly coupled to the GenerationMixin is a sign they should be moved into GenerationMixin)
    def get_generation_mode(self, *args, **kwargs):
        raise NotImplementedError("DiffusionGemmaGenerationConfig does not support `get_generation_mode`")

    # Legacy support from `GenerationConfig`
    def from_model_config(self, *args, **kwargs):
        raise NotImplementedError("DiffusionGemmaGenerationConfig does not support `from_model_config`")


@dataclass
class DiffusionGemmaGenerationOutput(ModelOutput):
    """
    Output class for DiffusionGemma generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences, including the prompt if `input_ids` was provided to the `generate` method.
        tokens_per_forward (`torch.LongTensor` of shape (`batch_size`)):
            The number of tokens per forward in this `generate` call, for each member in the batch. This is often
            used as a secondary evaluation metric for text diffusion models.
        past_key_values (`Cache`):
            The cache used for generation. It can be passed to subsequent calls to `generate` to speed up generation,
            in multi-turn sessions.
        logits (`None`):
            Unused. Kept in the interface for BC.
        scores (`None`):
            Unused. Kept in the interface for BC.
        hidden_states (`None`):
            Unused. Kept in the interface for BC.
    """

    sequences: torch.LongTensor
    tokens_per_forward: int | None = None
    past_key_values: Cache | None = None
    logits: None = None  # Unused for now, kept in the interface for BC with AR generation
    scores: None = None  # Unused for now, kept in the interface for BC with AR generation
    hidden_states: None = None  # Unused for now, kept in the interface for BC with AR generation


class LinearTemperatureScheduleLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor that applies a linear temperature schedule to the logits. This is similar to
    `TemperatureLogitsWarper`, except that the temperature is a function of the current step.

    At step n out of N, the temperature t is given by t = t_min + ((t_max - t_min) * (n/N)).

    Args:
        t_min (`float`):
            The final temperature in the schedule, i.e. at the last denoising step.
        t_max (`float`):
            The initial temperature in the schedule, i.e. at the first denoising step.
        max_denoising_steps (`int`):
            The maximum number of denoising steps.
    """

    def __init__(self, t_min: float, t_max: float, max_denoising_steps: int):
        self.t_min = t_min
        self.t_max = t_max
        self.max_denoising_steps = max_denoising_steps

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, cur_step: int) -> torch.FloatTensor:
        """
        Applies the linear temperature schedule to the logits.

        NOTE: remember that in text diffusion models, `cur_step` corresponds to the number of steps *remaining* in the
        denoising process.

        Args:
            input_ids (`torch.LongTensor`):
                The input ids.
            scores (`torch.FloatTensor`):
                The logits.
            cur_step (`int`):
                The current step.

        Returns:
            `torch.FloatTensor`: The logits after applying the linear temperature schedule.
        """
        temperature = self.t_min + ((self.t_max - self.t_min) * (cur_step / self.max_denoising_steps))
        return scores / temperature


@dataclass
class EntropyBoundSamplerConfig:
    """
    Configuration class for the entropy bound sampler.

    Args:
        entropy_bound (`float`):
            The entropy bound. The higher this value is, the more tokens will be accepted. See the docstring of
            [`EntropyBoundSampler.accept_canvas`] for more details on how it is applied.
    """

    entropy_bound: float

    def __post_init__(self):
        if not (isinstance(self.entropy_bound, float)) or self.entropy_bound <= 0:
            raise ValueError(f"`entropy_bound` must be a float > 0 (got {self.entropy_bound})")

    def to_dict(self):
        # Stores the class name as well, so we can load it back
        obj_dict = copy.deepcopy(self.__dict__)
        obj_dict["_cls_name"] = self.__class__.__name__
        return obj_dict


class EntropyBoundSampler:
    r"""
    Sampler class that initializes a canvas with random tokens, accepts tokens based on token-level entropy, and
    renoises non-accepted tokens.

    Here is a rough sketch of how the sampler loop works:
    ```
              +-----------------------+
              | Canvas initialization |
              | x_T ∈ U(V)            |
              +-----------+-----------+
                          |
                          v
               +----------+---------+       +---------------------+
    +--------->| Current canvas x_t |------>| Denoiser canvas x_D |
    |          +----------+---------+       +----------+----------+
    |                      \                          /
    |                       \                        /
    |                        \   Acceptance logic   /
    |                         v                    v
    |                       +-------------------------+
    | Stop if max           | Accepted canvas x_{t-1} |
    | denosing steps        +------------+------------+      +-------------------+
    | reached or                          \                  | New canvas ∈ U(V) |
    | adaptive stopping                    \                 +---------+---------+
    | triggers                              \    Renoising logic      /
    |                                        v                       v
    |                                       +-------------------------+
    +---------------------------------------| Next canvas x_{t-1}     |
                                            +-------------------------+
    ```

    Args:
        config (`EntropyBoundSamplerConfig`):
            The configuration of the sampler.
        canvas_length (`int`):
            The length of the canvas.
        vocab_size (`int`):
            The size of the vocabulary.
        max_denoising_steps (`int`):
            The maximum number of denoising steps. (Unused in this sampler)
    """

    def __init__(
        self, config: EntropyBoundSamplerConfig, canvas_length: int, vocab_size: int, max_denoising_steps: int
    ):
        self.entropy_bound = config.entropy_bound
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.accepted_token_mask = None  # keeps track of the positions of the accepted tokens

    def initialize_canvas(self, batch_size: int, device: torch.device) -> torch.LongTensor:
        """
        Initializes and returns a new canvas of `canvas_length` tokens with random values from the vocabulary.
        """
        canvas_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, self.canvas_length),
            device=device,
        )
        return canvas_ids

    def accept_canvas(
        self,
        current_canvas: torch.LongTensor,
        denoiser_canvas: torch.LongTensor,
        logits: torch.FloatTensor,
        cur_step: int,
    ) -> torch.LongTensor:
        """
        Accepts tokens from the denoiser based on an entropy bound. More concretely, sampling proceeds by accepting
        k tokens with lowest entropy, such that

        sum_i^k entropy_i - max(entropy_1, ..., entropy_k) <= entropy_bound,

        where the LHS is the upper bound on the joint mutual information between these tokens, and thus the sampler
        chooses k tokens that they are approximately independent.

        Originally proposed in https://arxiv.org/pdf/2505.24857

        Args:
            current_canvas (`torch.LongTensor`):
                The current canvas.
            denoiser_canvas (`torch.LongTensor`):
                The canvas sampled from the denoiser predictions.
            logits (`torch.FloatTensor`):
                The logits from the denoiser.
            cur_step (`int`):
                The current step.

        Returns:
            torch.LongTensor: The accepted canvas.
        """
        dist = torch.distributions.Categorical(logits=logits)
        token_entropy = dist.entropy()  # (batch_size, canvas_length)
        sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
        cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)

        # Note: sorted_token_entropy = cumulative maximum entropy, because it's sorted in ascending order
        sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= self.entropy_bound
        self.accepted_token_mask = torch.scatter(
            input=torch.zeros_like(sorted_selection_mask), dim=-1, index=sorted_indices, src=sorted_selection_mask
        )
        accepted_canvas = torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)
        return accepted_canvas

    def renoise_canvas(self, accepted_canvas: torch.LongTensor, cur_step: int) -> torch.LongTensor:
        """
        Renoises all non-accepted tokens.

        Args:
            accepted_canvas (`torch.LongTensor`):
                The accepted canvas.
            cur_step (`int`):
                The current step. (Unused in this sampler)

        Returns:
            torch.LongTensor: The renoised canvas.
        """
        device = accepted_canvas.device
        batch_size = accepted_canvas.shape[0]

        renoise_mask = ~self.accepted_token_mask
        random_canvas = self.initialize_canvas(batch_size, device)
        renoised_canvas = torch.where(renoise_mask, random_canvas, accepted_canvas)
        return renoised_canvas


class DiffusionGemmaAdaptiveStopping(ABC):
    """
    Base class for DiffusionGemma adaptive stopping strategies. It may be stateful or stateless.
    """

    @abstractmethod
    def __call__(self, argmax_canvas: torch.LongTensor, logits: torch.FloatTensor, **kwargs) -> torch.BoolTensor: ...

    def reset(self):
        pass  # Default no-op for stateless stoppers


class StableAndConfidentStoppingCriteria(DiffusionGemmaAdaptiveStopping):
    """
    Adaptive stopping strategy that stops when the diffusion process is confident and stable. To be more specific:
    - The diffusion process is stable when the accepted canvas are the same across `stability_threshold` steps.
    - The diffusion process is confident when the mean of the entropy of the processed logits is below
      `confidence_threshold`.

    Args:
        stability_threshold (`int`):
            The number of steps for which the accepted canvas must be the same to trigger the stopping criteria.
        confidence_threshold (`float`):
            The threshold for the mean of the entropy of temperature-scaled logits to trigger the stopping criteria.
    """

    def __init__(self, stability_threshold: int, confidence_threshold: float):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.argmax_canvas_history = None

    def __call__(self, argmax_canvas: torch.LongTensor, logits: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        Applies the stable and confident adaptive stopping strategy, returning a boolean tensor indicating whether to
        stop for each sample in the batch.

        Args:
            argmax_canvas(`torch.LongTensor`):
                The argmax of the latest denoiser prediction.
            logits (`torch.FloatTensor`):
                The predicted logits, after applying logits processors.

        Returns:
            torch.BoolTensor: A boolean tensor indicating whether to stop.
        """
        # 1. Stability criteria
        if self.stability_threshold == 0:
            stable = torch.ones((logits.shape[0]), device=logits.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas

        # 2. Confidence criteria
        dist = torch.distributions.Categorical(logits=logits)
        token_entropy = dist.entropy()
        confident = torch.mean(token_entropy, dim=-1) < self.confidence_threshold

        return stable & confident

    def reset(self):
        self.argmax_canvas_history = None


class DiffusionGemmaGenerationMixin:
    """
    Mixin class for DiffusionGemma generation. Contains all the model-level methods.
    """

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        streamer: BaseStreamer | None = None,
        generation_config: DiffusionGemmaGenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        **kwargs,
    ) -> DiffusionGemmaGenerationOutput:
        """
        Generates text using the diffusion model.

        It contains an outer loop doing autoregressive generation of canvases (blocks of tokens), and an inner
        loop doing diffusion on each canvas. The algorithm works roughly as follows:
        ```
        1. Autoregressive canvas generation loop:
            a. Encode all previous tokens using the encoder, to get the KV cache.
            b. Prepare data for the new denoising loop
            c. For each denoising (diffusion) step:
                i.   Run the decoder, taking the current canvas, the encoder KV cache, and the self-conditioning logits
                     (if available) as inputs.
                ii.  Select new canvas tokens from the output logits.
                iii. Apply the sampler acceptance and renoising logic.
                iv.  Update the diffusion stopping criteria.
                v.   Use the output logits as self-conditioning logits for the next step.
            d. Append the new denoised canvas to the sequence of generated tokens.
            e. Check if any autoregressive stopping criteria are met, and break the outer loop if all sequences have
               met them. Replaces generated tokens in finished sequences by pad.
            f. Prepare tensors for the next block
        ```

        Parameters:
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                The sequence used as a prompt for the generation.
            past_key_values ([`Cache`], *optional*):
                Cache object containing the past key values and past attention masks for the decoder. If it is set,
                `input_ids` and/or `pixel_values` must correspond to uncached data only.
            streamer ([`BaseStreamer`], *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing. If the
                streamer object has a `put_draft` method, tokens from the denoising steps will be sent there.

            > Additional arguments for power users

            generation_config ([`DiffusionGemmaGenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call, overriding
                the model defaults. If the model checkpoint has a `generation_config.json` file, the model default
                will be loaded from there. Otherwise, it will be an empty `DiffusionGemmaGenerationConfig` instance.
                As an additional shortcut, `**kwargs` matching attributes in the `generation_config` will override them.
            logits_processor ([`LogitsProcessorList`], *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config, to be applied on the diffusion logits. If provided, these processors will be first
                to be applied. This feature is intended for advanced users. You can, for instance, pass here the
                logits processors commonly used with AR LLMs.
            stopping_criteria ([`StoppingCriteriaList`], *optional*):
                Custom stopping criteria that complements the default block autoregressive stopping criteria built
                from arguments and a generation config. If provided, these criteria will be first to be applied. This
                feature is intended for advanced users. You can, for instance, pass here the stopping criteria commonly
                used with AR LLMs.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. For instance, you can set the starting canvas with
                `decoder_input_ids`.

        Return:
            [`DiffusionGemmaGenerationOutput`]: a `ModelOutput` instance containing the generated text (`sequences`),
            as well as other optional outputs.

        Examples:

        ```python
        >>> from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

        >>> model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        ...     "google/diffusiongemma-26B-A4B-it", device_map="auto",
        >>> )

        >>> chat = [{"role": "user", "content": "Why is the sky blue?"},]
        >>> processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")
        >>> input_ids = processor.apply_chat_template(chat, tokenize=True, return_tensors="pt")

        >>> streamer = TextDiffusionStreamer(tokenizer=processor.tokenizer)
        >>> model.generate(input_ids.to(model.device), max_new_tokens=512, streamer=streamer)
        ```
        """
        # 0. Input preparation
        # 0.a. Prepare the generation config, respecting the kwarg-based parameterization from the original AR
        # `generate`
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)

        # 0.b. Set generation or output control variables. As in AR generation, `max_new_tokens` takes precedence
        # over `max_length` (we check against the default value, 256).
        batch_size, cur_len = input_ids.shape
        initial_input_ids_len = cur_len
        if past_key_values is not None:
            cur_len += past_key_values.get_seq_length()
        max_length, max_new_tokens = self._prepare_generated_length(generation_config, cur_len)
        max_new_canvases = math.ceil(max_new_tokens / self.config.canvas_length)

        # 0.c. Sanity-checks, before spending time in the generation loop
        if past_key_values is not None and generation_config.cache_implementation is not None:
            raise ValueError("Cannot provide both `past_key_values` and `generation_config.cache_implementation`.")
        if (
            "pixel_values" not in model_kwargs
            and input_ids is not None
            and (input_ids == self.config.image_token_id).any()
        ):
            logger.warning_once(
                "Your input tokens contain image tokens, but you haven't set `pixel_values`.\n\n"
                "If you're using HF's processor classes, make sure you process your chat template with "
                "`return_dict=True`, and pass the resulting dictionary to `generate`."
            )

        # 0.d. Initialize tensor or tensor-based data and variables
        device = input_ids.device
        canvas_length = self.config.canvas_length
        current_canvas = None
        eos_tensor = None
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        decoder_forward_passes = torch.zeros(batch_size, dtype=torch.int, device=device)
        if past_key_values is None:
            past_key_values = self._prepare_cache_for_generation(
                generation_config=generation_config,
                batch_size=batch_size,
                max_length=max_length - canvas_length,  # the last generated canvas won't be cached
            )
        if generation_config.eos_token_id is not None:
            eos_tensor = torch.tensor(generation_config.eos_token_id, device=input_ids.device)

        encoder_position_ids = torch.arange(
            cur_len - input_ids.shape[1], cur_len, dtype=torch.int32, device=input_ids.device
        ).unsqueeze(0)
        decoder_position_ids = torch.arange(
            cur_len, cur_len + canvas_length, dtype=torch.int32, device=input_ids.device
        ).unsqueeze(0)

        if "attention_mask" in kwargs:
            if len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")
            attention_mask = model_kwargs.pop("attention_mask").bool()
        else:
            attention_mask = torch.ones((batch_size, cur_len), dtype=torch.bool, device=input_ids.device)

        # 0.e. Initialize samplers, logits processors, and stopping criteria
        sampler = self._prepare_sampler(generation_config)
        logits_processor = self._prepare_logits_processor(generation_config, logits_processor)
        stopping_criteria = self._prepare_ar_stopping_criteria(generation_config, stopping_criteria)
        diffusion_stopping_criteria = self._prepare_diffusion_stopping_criteria(generation_config)
        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 0.f performance tuning
        is_compiling = past_key_values is not None and past_key_values.is_compileable
        if is_compiling:
            encoder_forward_after_prefill, decoder_forward, sampler, diffusion_stopping_criteria = (
                self._compile_functions(sampler, diffusion_stopping_criteria)
            )

            decoder_attention_mask = torch.zeros(
                (batch_size, past_key_values.max_cache_len + canvas_length),
                dtype=torch.bool,
                device=attention_mask.device,
            )
            decoder_attention_mask[:, : attention_mask.shape[1]] = attention_mask
            decoder_attention_mask[:, -canvas_length:] = 1
        else:
            decoder_forward = self.forward
            encoder_forward_after_prefill = self.model.encoder
            decoder_attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)

        # 1. Autoregressive canvas generation loop
        # NOTE: please keep the docstring in sync with this section's comments.
        is_prefill = True
        for _ in range(max_new_canvases):
            # 1.a. Encode all previous tokens using the encoder, to get the KV cache.
            unprocessed_input_ids, encoder_mask_mapping = self._prepare_encoder_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_position_ids=encoder_position_ids,
                past_key_values=past_key_values,
                is_prefill=is_prefill,
                canvas_length=canvas_length,
                batch_size=batch_size,
                **model_kwargs,
            )

            encoder_forward = self.model.encoder if is_prefill else encoder_forward_after_prefill
            encoder_outputs = encoder_forward(
                input_ids=unprocessed_input_ids,
                attention_mask=encoder_mask_mapping,
                past_key_values=past_key_values,
                position_ids=encoder_position_ids,
                **model_kwargs,
            )
            past_key_values = encoder_outputs.past_key_values
            is_prefill = False

            # 1.b. Prepare data for the new denoising loop
            current_canvas, self_conditioning_logits, mask_mapping, finished_denoising = self._prepare_denoiser_inputs(
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                sampler=sampler,
                diffusion_stopping_criteria=diffusion_stopping_criteria,
                batch_size=batch_size,
                device=device,
                model_kwargs=model_kwargs,  # passed as a dict, because some contents will be popped
            )
            argmax_canvas = current_canvas

            # 1.c For each denoising (diffusion) step:
            # NOTE: we iterate in reverse order, as denoising is the reverse diffusion process (N..1).
            for cur_step in reversed(range(1, generation_config.max_denoising_steps + 1)):
                # Unfinished batch items get their decoder forward pass counter incremented
                # Finished batch items wouldn't have this decoder pass if we were running with bsz == 1
                decoder_forward_passes += ~(finished_denoising | finished_sequences)

                current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = self._denoising_step(
                    decoder_forward=decoder_forward,
                    current_canvas=current_canvas,
                    argmax_canvas=argmax_canvas,
                    input_ids=input_ids,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=self_conditioning_logits,
                    mask_mapping=mask_mapping,
                    past_key_values=past_key_values,
                    finished_denoising=finished_denoising,
                    cur_step=cur_step,
                    sampler=sampler,
                    logits_processor=logits_processor,
                    diffusion_stopping_criteria=diffusion_stopping_criteria,
                    **model_kwargs,
                )

                # If we have a draft-compatible streamer, put out the latest draft. We consider `argmax_canvas`
                # to be the draft, as it is often the closest to the final output.
                if streamer is not None and hasattr(streamer, "put_draft"):
                    streamer_kwargs = {"value": argmax_canvas.cpu()}
                    if getattr(streamer, "_takes_logits", False):
                        streamer_kwargs = {"logits": self_conditioning_logits.cpu()}
                    streamer.put_draft(**streamer_kwargs)

                # Early exit if no more denoising steps are needed
                if torch.all(finished_denoising):
                    break

            # 1.d. Append the new denoised canvas to the sequence of generated tokens.
            input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)

            # 1.e. Check if any autoregressive stopping criteria are met, and break the outer loop if all sequences
            # have met them. Replaces generated tokens in finished sequences by pad.
            input_ids, finished_sequences = self._finalize_canvas(
                input_ids=input_ids,
                finished_sequences=finished_sequences,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                canvas_length=canvas_length,
                eos_tensor=eos_tensor,
            )

            if streamer is not None:
                streamer.put(input_ids[:, -canvas_length:].cpu())

            if torch.all(finished_sequences):
                break

            # 1.f. Prepare tensors for the next block
            cur_len, decoder_attention_mask, attention_mask, encoder_position_ids, decoder_position_ids = (
                self._prepare_kwargs_for_next_canvas(
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_position_ids=decoder_position_ids,
                    past_key_values=past_key_values,
                    canvas_length=canvas_length,
                    cur_len=cur_len,
                    is_compiling=is_compiling,
                )
            )

        # 2. Finalize and return
        if streamer is not None:
            streamer.end()

        tokens_per_forward = self._compute_tokens_per_forward(
            input_ids, decoder_forward_passes, initial_input_ids_len, generation_config.pad_token_id
        )
        return DiffusionGemmaGenerationOutput(
            sequences=input_ids, tokens_per_forward=tokens_per_forward, past_key_values=past_key_values
        )

    @staticmethod
    def _compute_tokens_per_forward(
        input_ids: torch.Tensor,
        decoder_forward_passes: torch.Tensor,
        initial_input_ids_len: int,
        pad_token_id: int | None,
    ) -> torch.Tensor:
        """
        Computes and returns the tokens per forward of the diffusion step.

        It is defined as # generated tokens / # denoising steps, where:
        - # generated tokens EXCLUDES all pad tokens (i.e. tokens after EOS)
        - # denoising steps EXCLUDES the batched denoising steps after which a given row has hit the stopping criteria
        """
        new_tokens = input_ids[:, initial_input_ids_len:]
        if pad_token_id is not None:
            num_valid_tokens = (new_tokens != pad_token_id).sum(dim=-1)
        else:
            num_valid_tokens = new_tokens.shape[1]
        tokens_per_forward = num_valid_tokens / decoder_forward_passes
        return tokens_per_forward

    def _prepare_generation_config(
        self, generation_config: DiffusionGemmaGenerationConfig, **kwargs: Any
    ) -> DiffusionGemmaGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.

        # priority for baseline parameterization: ad hoc kwargs passed to `generate` > provided `generation_config` >
        # `self.generation_config` > global defaults
        generation_config = generation_config or self.generation_config or DiffusionGemmaGenerationConfig()
        # copy: don't modify the original generation config when applying global defaults or kwargs
        generation_config = copy.deepcopy(generation_config)
        # apply global defaults to unset parameters
        global_defaults = generation_config._get_default_generation_params()
        generation_config.update(**global_defaults, defaults_only=True)
        # kwargs rejected from updating the generation config are model_kwargs
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        return generation_config, model_kwargs

    def _prepare_generated_length(
        self,
        generation_config: DiffusionGemmaGenerationConfig,
        cur_len: int,
    ):
        """Prepared max length in generation configs to avoid clashes between similar attributes"""
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.

        if generation_config.max_length and generation_config.max_new_tokens == 256:
            max_length = generation_config.max_length
            max_new_tokens = max_length - cur_len
        else:
            max_new_tokens = generation_config.max_new_tokens
            max_length = max_new_tokens + cur_len
        return max_length, max_new_tokens

    def _prepare_cache_for_generation(
        self, generation_config: DiffusionGemmaGenerationConfig, batch_size: int, max_length: int
    ) -> Cache:
        """
        Prepares and returns the cache for generation, given the parameterization in `generation_config`.

        (NOTE: Originally copied from `GenerationMixin._prepare_cache_for_generation` on 2026-03-27, and stripped down
        for DiffusionGemma.)
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.

        # Static Caches
        if generation_config.cache_implementation in ALL_STATIC_CACHE_IMPLEMENTATIONS:
            if generation_config.cache_implementation in DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS:
                logger.warning_once(
                    f"Using `cache_implementation='{generation_config.cache_implementation}' is deprecated "
                    f"and will be removed in v5.13. Please only use one of {STATIC_CACHE_IMPLEMENTATIONS}, "
                    "and the layer structure will be inferred automatically."
                )
            past_key_values = self._prepare_static_cache(
                cache_implementation=generation_config.cache_implementation,
                batch_size=batch_size,
                max_length=max_length,
            )
        elif generation_config.cache_implementation == "quantized":
            cache_config = generation_config.cache_config if generation_config.cache_config is not None else {}
            cache_config.setdefault("config", self.config.get_text_config(decoder=True))
            backend = cache_config.pop("backend", "quanto")
            past_key_values = QuantizedCache(backend=backend, **cache_config)

        # Dynamic Caches
        else:
            dynamic_cache_kwargs = {}
            if generation_config.cache_implementation != "dynamic_full":
                dynamic_cache_kwargs["config"] = self.config.get_text_config(decoder=True)
            if generation_config.cache_implementation == "offloaded":
                dynamic_cache_kwargs["offloading"] = True
            past_key_values = DynamicCache(**dynamic_cache_kwargs)

        return past_key_values

    def _prepare_encoder_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_position_ids: torch.Tensor,
        past_key_values: Cache,
        is_prefill: bool,
        canvas_length: int,
        batch_size: int,
        **model_kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Prepares the inputs for the encoder"""
        unprocessed_input_ids = input_ids if is_prefill else input_ids[:, -canvas_length:]
        # Clone with `memory_format=torch.contiguous_format` to prevent stride-related graph breaks
        unprocessed_input_ids = unprocessed_input_ids.clone(memory_format=torch.contiguous_format)

        # 2D -> 4D attention mask mapping. Calling it in advance prevents graph breaks
        dummy_input_embeds = torch.empty(
            (batch_size, unprocessed_input_ids.shape[1], 0), dtype=self.dtype, device=input_ids.device
        )
        encoder_mask_mapping = self.model.encoder.create_masks_for_generate(
            config=self.config,
            # we only need batch size, seq_length, dtype and device here - so we pass a 0-sized tensor with only the metadata
            inputs_embeds=dummy_input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=encoder_position_ids,
            mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
        )
        return unprocessed_input_ids, encoder_mask_mapping

    def _prepare_denoiser_inputs(
        self,
        decoder_attention_mask: torch.Tensor,
        past_key_values: Cache,
        sampler: EntropyBoundSampler,
        diffusion_stopping_criteria: DiffusionGemmaAdaptiveStopping | None,
        batch_size: int,
        device: torch.device,
        model_kwargs: dict,
    ) -> tuple:
        """Prepares the inputs for the denoising loop"""
        # These `model_kwargs` keys, when set, are consumed in the first encoder call
        for key in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
            if key in model_kwargs:
                del model_kwargs[key]

        # Randomly initialize a canvas of `canvas_length` tokens and prepare the 4D decoder attention mask
        # (The exception is if a user provides their own starting canvas, which gets consumed in the first
        # decoder call)
        current_canvas = model_kwargs.pop(
            "decoder_input_ids", sampler.initialize_canvas(batch_size=batch_size, device=device)
        )
        # (The same applies to the self-conditioning logits)
        self_conditioning_logits = model_kwargs.pop("self_conditioning_logits", None)

        mask_mapping = self.model.decoder.create_diffusion_decoder_attention_mask(
            config=self.config.text_config,
            inputs_embeds=current_canvas.unsqueeze(-1),  # we only need a dummy tensor with the same shape[:2] here
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        finished_denoising = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if diffusion_stopping_criteria is not None:
            diffusion_stopping_criteria.reset()

        return current_canvas, self_conditioning_logits, mask_mapping, finished_denoising

    def _denoising_step(
        self,
        decoder_forward: Callable,
        current_canvas: torch.Tensor,
        argmax_canvas: torch.Tensor,
        input_ids: torch.LongTensor,
        decoder_position_ids: torch.LongTensor,
        self_conditioning_logits: torch.Tensor,
        mask_mapping: dict[str, torch.Tensor],
        past_key_values: Cache,
        finished_denoising: torch.Tensor,
        cur_step: int,
        sampler: EntropyBoundSampler,
        logits_processor: LogitsProcessorList,
        diffusion_stopping_criteria: DiffusionGemmaAdaptiveStopping | None,
        **model_kwargs,
    ):
        """
        Runs one denoising step. Please refer to the docstring in `generate` for more details.
        """
        # if we're compiling inner functions, `cur_step` as a plain `int` will trigger recompilations
        cur_step = torch.tensor(cur_step, device=current_canvas.device, dtype=torch.int32)
        torch.compiler.cudagraph_mark_step_begin()  # needed for the compiled EB sampler

        # 1.c.i Run the decoder, taking the current canvas, the encoder KV cache, and the self-conditioning
        # logits (if available) as inputs.
        decoder_outputs = decoder_forward(
            decoder_input_ids=current_canvas,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=mask_mapping,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
            **model_kwargs,
        )
        raw_logits = decoder_outputs.logits

        # 1.c.ii Select new canvas tokens from the output logits.
        processed_logits = logits_processor(input_ids, raw_logits, cur_step=cur_step)
        probs = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
        # `torch.multinomial` only works on 2D tensors, so we flatten/unflatten
        vocab_size = self.config.text_config.vocab_size
        batch_size, canvas_length = current_canvas.shape
        denoiser_canvas = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
        denoiser_canvas = denoiser_canvas.squeeze(-1).view(batch_size, canvas_length)
        new_argmax_canvas = torch.argmax(processed_logits, dim=-1)

        # 1.c.iii Apply the sampler acceptance and renoising logic.
        accepted_canvas = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits, cur_step)
        accepted_canvas = accepted_canvas.clone()  # clone needed for compiled sampler
        new_current_canvas = sampler.renoise_canvas(accepted_canvas, cur_step)
        new_current_canvas = new_current_canvas.clone()  # clone needed for compiled sampler

        # 1.c.iv Update the diffusion stopping criteria.
        if diffusion_stopping_criteria is not None:
            # If we have any batch item that has finished before, we don't want to update its results!
            if finished_denoising.any():
                new_argmax_canvas = torch.where(finished_denoising[:, None], argmax_canvas, new_argmax_canvas)
                new_current_canvas = torch.where(finished_denoising[:, None], current_canvas, new_current_canvas)
                processed_logits = torch.where(
                    finished_denoising[:, None, None], self_conditioning_logits, processed_logits
                )

            finished_denoising |= diffusion_stopping_criteria(new_argmax_canvas, processed_logits)

        # 1.c.v Use the output logits as self-conditioning logits for the next step.
        embeddings_dtype = self.model.decoder.embed_tokens.weight.dtype
        self_conditioning_logits = processed_logits.to(embeddings_dtype)

        return (
            new_current_canvas,
            new_argmax_canvas,
            self_conditioning_logits,
            finished_denoising,
        )

    @staticmethod
    def _finalize_canvas(
        input_ids: torch.Tensor,
        finished_sequences: torch.Tensor,
        generation_config: DiffusionGemmaGenerationConfig,
        stopping_criteria: StableAndConfidentStoppingCriteria,
        canvas_length: int,
        eos_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finalizes a newly generated canvas"""
        finished_this_canvas = stopping_criteria(
            input_ids,
            None,
            # `new_token_length` is used in the EosTokenCriteria to look for eos tokens in the whole canvas
            new_token_length=canvas_length,
        )
        previously_finished_sequences = finished_sequences
        finished_sequences = previously_finished_sequences | finished_this_canvas
        pad_mask = None
        if generation_config.pad_token_id is not None and torch.any(finished_sequences):
            # finished sequences from previous canvases: all generated tokens get replaced by pad
            input_ids[previously_finished_sequences, -canvas_length:] = generation_config.pad_token_id
            # finished sequences from this canvas: all tokens after eos get replaced by pad
            if generation_config.eos_token_id is not None and torch.any(finished_this_canvas):
                new_tokens = input_ids[:, -canvas_length:]
                is_eos = torch.isin(new_tokens, eos_tensor)
                eos_cumsum = is_eos.cumsum(dim=-1)
                pad_mask = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)
                new_tokens[pad_mask] = generation_config.pad_token_id  # replaces `input_ids`
        return input_ids, finished_sequences

    @staticmethod
    def _prepare_kwargs_for_next_canvas(
        attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        decoder_position_ids: torch.Tensor,
        past_key_values: Cache,
        canvas_length: int,
        cur_len: int,
        is_compiling: bool,
    ) -> tuple:
        """Prepares model inputs for the next canvas"""
        cur_len += canvas_length
        if is_compiling:
            valid_cache_length = past_key_values.get_seq_length()
            decoder_attention_mask[:, valid_cache_length : valid_cache_length + canvas_length] = 1
        else:
            decoder_attention_mask = torch.nn.functional.pad(decoder_attention_mask, (0, canvas_length), value=True)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, canvas_length), value=True)
        encoder_position_ids = decoder_position_ids
        decoder_position_ids = torch.arange(
            cur_len, cur_len + canvas_length, dtype=torch.int32, device=decoder_position_ids.device
        ).unsqueeze(0)
        return cur_len, decoder_attention_mask, attention_mask, encoder_position_ids, decoder_position_ids

    def _prepare_static_cache(self, cache_implementation: str, batch_size: int, max_length: int) -> Cache:
        """
        Sets a cache for `generate`, **that will persist across calls**. A new cache will only be initialized if a
        new `generate` call requires a larger cache or uses a different batch size.

        Returns the resulting cache object.

        (NOTE: Originally copied from `GenerationMixin._prepare_static_cache` on 2026-03-27, and stripped down
        for DiffusionGemma.)
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.
        offload_cache = "offloaded" in cache_implementation

        cache_to_check: StaticCache | None = None
        if hasattr(self, "_cache") and isinstance(self._cache, StaticCache):
            cache_to_check = self._cache

        need_new_cache = (
            cache_to_check is None
            or cache_to_check.offloading != offload_cache
            or cache_to_check.max_batch_size != batch_size
            or cache_to_check.max_cache_len < max_length
        )

        if need_new_cache:
            cache_kwargs = {
                "config": self.config.get_text_config(decoder=True),
                "max_cache_len": max_length,
                "offloading": offload_cache,
            }
            self._cache = StaticCache(**cache_kwargs)
        else:
            self._cache.reset()
        return self._cache

    def _prepare_logits_processor(
        self, generation_config: DiffusionGemmaGenerationConfig, logits_processor: LogitsProcessorList | None = None
    ) -> LogitsProcessorList:
        """
        Prepares and returns the logits processor for generation, given the parameterization in `generation_config`.
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.

        # Externally defined `logits_processor` will be applied first.
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        if generation_config.t_min is not None and generation_config.t_max is not None:
            logits_processor.append(
                LinearTemperatureScheduleLogitsProcessor(
                    t_min=generation_config.t_min,
                    t_max=generation_config.t_max,
                    max_denoising_steps=generation_config.max_denoising_steps,
                )
            )

        return logits_processor

    def _prepare_ar_stopping_criteria(
        self,
        generation_config: DiffusionGemmaGenerationConfig,
        stopping_criteria: StoppingCriteriaList | None = None,
    ) -> StoppingCriteriaList:
        """
        Prepares and returns the autoregressive stopping criteria for generation, given the parameterization in
        `generation_config`.
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.

        # Externally defined `stopping_criteria` will be applied first.
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()

        if generation_config.max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(generation_config.max_length))
        if generation_config.eos_token_id is not None:
            stopping_criteria.append(EosTokenCriteria(generation_config.eos_token_id))

        return stopping_criteria

    def _prepare_diffusion_stopping_criteria(
        self, generation_config: DiffusionGemmaGenerationConfig
    ) -> StableAndConfidentStoppingCriteria | None:
        """
        Prepares and returns the diffusion stopping criteria for generation, given the parameterization in
        `generation_config`.
        """
        if generation_config.stability_threshold is not None and generation_config.confidence_threshold is not None:
            diffusion_stopping_criteria = StableAndConfidentStoppingCriteria(
                stability_threshold=generation_config.stability_threshold,
                confidence_threshold=generation_config.confidence_threshold,
            )
        else:
            diffusion_stopping_criteria = None
        return diffusion_stopping_criteria

    def _prepare_sampler(self, generation_config: DiffusionGemmaGenerationConfig) -> EntropyBoundSampler:
        """
        Prepares and returns the sampler for generation, given the parameterization in `generation_config`.
        """
        # Assumption: validation of the type in `sampler_config` happens in `generation_config.validate()`
        return EntropyBoundSampler(
            config=generation_config.sampler_config,
            canvas_length=self.config.canvas_length,
            vocab_size=self.config.text_config.vocab_size,
            max_denoising_steps=generation_config.max_denoising_steps,
        )

    def _compile_functions(self, sampler, diffusion_stopping_criteria):
        """
        Compiles some (but not all) pieces of the decoding loop. Some pieces have e.g. dynamic shapes
        Stores compiled code in `self`, to avoid recompiling between calls.
        """
        if not hasattr(self, "_compiled_encoder"):
            self._compiled_encoder = torch.compile(self.model.encoder, mode="reduce-overhead", fullgraph=True)
        encoder_forward_after_prefill = self._compiled_encoder

        if not hasattr(self, "_compiled_decoder_forward"):
            self._compiled_decoder_forward = torch.compile(self.forward, mode="reduce-overhead", fullgraph=True)
        decoder_forward = self._compiled_decoder_forward

        if not hasattr(self, "_compiled_accept_canvas"):
            self._compiled_accept_canvas = torch.compile(sampler.accept_canvas, mode="reduce-overhead", fullgraph=True)
        sampler.accept_canvas = self._compiled_accept_canvas

        if not hasattr(self, "_compiled_renoise_canvas"):
            self._compiled_renoise_canvas = torch.compile(
                sampler.renoise_canvas, mode="reduce-overhead", fullgraph=True
            )
        sampler.renoise_canvas = self._compiled_renoise_canvas

        if diffusion_stopping_criteria is not None:
            if not hasattr(self, "_compiled_diffusion_stopping_criteria"):
                self._compiled_diffusion_stopping_criteria = torch.compile(
                    diffusion_stopping_criteria.__call__, mode="reduce-overhead", fullgraph=True
                )
            diffusion_stopping_criteria.__call__ = self._compiled_diffusion_stopping_criteria

        return encoder_forward_after_prefill, decoder_forward, sampler, diffusion_stopping_criteria

    def adjust_generation_fn(
        self,
        generation_config,
        from_auto_class,
        from_pipeline,
        pretrained_model_name_or_path,
        cache_dir,
        force_download,
        proxies,
        local_files_only,
        token,
        revision,
        subfolder,
        trust_remote_code,
        **kwargs,
    ):
        """
        Logic used at `model_cls.from_pretrained()` time, to set a model-level generation config.

        (NOTE: Originally copied from `GenerationMixin.adjust_generation_fn` on 2026-05-04, and stripped down
        for DiffusionGemma.)
        """
        # TODO(joao, raushan): refactor `GenerationMixin` and this to reuse logic without requiring inheritance.
        del trust_remote_code  # unused

        if self.can_generate() and generation_config is not None:
            self.generation_config = self.generation_config.from_dict(generation_config.to_dict())
        elif self.can_generate() and pretrained_model_name_or_path is not None:
            repo_loading_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "subfolder": subfolder,
                **kwargs,
            }
            # Load generation config
            try:
                self.generation_config = self.generation_config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **repo_loading_kwargs,
                )
            except OSError:
                logger.info("Generation config file not found, using the default generation config.")


__all__ = [
    "DiffusionGemmaGenerationOutput",
    "DiffusionGemmaGenerationMixin",
    "DiffusionGemmaGenerationConfig",
    "EntropyBoundSamplerConfig",
    "EntropyBoundSampler",
    "StableAndConfidentStoppingCriteria",
    "LinearTemperatureScheduleLogitsProcessor",
]
