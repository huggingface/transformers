# Copyright 2025 The HuggingFace Inc. team.
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
from abc import ABC, abstractmethod

import torch

from ..logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .requests import FutureRequestState, logger


# Abstract base class for all continuous batching logits processors
class ContinuousBatchingLogitsProcessor(ABC):
    # Kwargs that this processor uses, mapped to their expected type. Only the type is checked at runtime, value-range
    # validation (e.g. temperature > 0) is not performed to keep a light API. You can open a PR if this is needed.
    supported_kwargs: dict[str, type]
    # Kwargs that this processor recognizes but ignores
    ignored_kwargs: tuple[str, ...]

    @abstractmethod
    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default values for this processor."""
        pass

    @abstractmethod
    def prepare_tensor_args(self, requests_in_batch: list[FutureRequestState]) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies the logits processor in a per-token manner.
        Args:
            - scores (torch.FloatTensor): The scores to process, with shape [num_tokens, vocab_size]
            - tensor_arg (torch.Tensor): The tensor argument to use for the logits processor, with shape
                [max_num_tokens] and dtype torch.int32. The dtype might not be representative of the actual data, for
                instance it's common to have a float32 tensor viewed as int32 (eg. temperature)
        Returns:
            - torch.FloatTensor: The processed scores, with shape [num_tokens, vocab_size]
        """
        pass


# Main class for managing a list of processors (CB version or not) for batched generation
class ContinuousBatchingLogitsProcessorList:
    """A class to hold logits processors for continuous batching (CB).

    Each processor has a base class, which is the one used in regular `generate` and some have a per-request version
    adapted for CB. The list of logits processors present is generated using  the `_get_logits_processor` method from
    the model, which will only include processors if their presence is required by the generation config. For instance,
    if you want to use temperature scaling, you need to specify a temperature that's neither None nor 1.0. Otherwise
    no processors will be created for temperature, and per-request temperature scaling will not be available.

    On support of base processors:
        Some base processors are not supported by CB and will be dropped when this class is instantiated. Some
        processors have not yet been categorized as supported or not and will be kept but with a warning. All processors
        can be kept by setting the flag `drop_unsupported_processors` to False.
    On per-request processors:
        Some base processors have a per-request version adapted for CB and will be converted to their per-request
        version when this class is instantiated. This is the default behavior unless the flag `per_request_processors`
        is set to False.
    """

    def __init__(
        self,
        logits_processor: LogitsProcessorList,
        per_request_processors: bool = False,
        drop_unsupported_processors: bool = True,
    ) -> None:
        self.logits_processor = logits_processor
        self.tensors_required = 0  # number of tensors required to store CB logits processors arguments
        # If needed, convert compatible logits processors to their per-request versions
        if per_request_processors:
            self._convert_to_per_request_processors()
        # Validate and optionally filter processors based on their CB support
        self._validate_processors(drop_unsupported_processors)
        self._retrieve_processors_kwargs()
        # Static boolean to know if there is any logits processing to do. Helps with torch.compile().
        self.do_processing = len(self.logits_processor) > 0

    def __repr__(self) -> str:
        return f"ContinuousBatchingLogitsProcessorList(logits_processor={self.logits_processor}, tensors_required={self.tensors_required})"

    def clear(self) -> None:
        self.logits_processor = LogitsProcessorList()
        self.tensors_required = 0
        self.supported_keys = {}
        self.ignored_keys = set()
        self.do_processing = False

    def _convert_to_per_request_processors(self) -> None:
        """Replaces the compatible logits processors with their per-request versions."""
        for i, processor in enumerate(self.logits_processor):
            for regular_cls, cb_cls in CLASSIC_TO_CB_PROCESSORS_MAP.items():
                if isinstance(processor, regular_cls):
                    self.logits_processor[i] = cb_cls(processor)
                    self.tensors_required += 1  # in the future, this might be more than 1 (will be stored in mapping)
                    break

    def _validate_processors(self, drop_unsupported: bool) -> None:
        """Validates the logits processors and optionally removes unsupported ones. When drop_unsupported is True,
        processors explicitly marked as unsupported are removed. Otherwise, all processors are kept but warnings are
        logged for unsupported or unknown ones.
        """
        filtered_processors = []
        for processor in self.logits_processor:
            class_name = processor.__class__.__name__
            supported = getattr(processor, "supports_continuous_batching", None)

            # Keep all ContinuousBatchingLogitsProcessor or supported processors
            if isinstance(processor, ContinuousBatchingLogitsProcessor) or supported:
                filtered_processors.append(processor)
            # Keep processors with support status unknown
            elif supported is None:
                logger.warning(f"Processor {class_name} might not be supported by CB.")
                filtered_processors.append(processor)
            # Otherwise, processor is not supported, then behavior depends on the flag drop_unsupported
            elif drop_unsupported:
                logger.warning(f"Processor {class_name} isn't supported by CB. Dropping it.")
            else:
                logger.warning(f"Processor {class_name} isn't supported by CB. Kept it because {drop_unsupported = }.")
                filtered_processors.append(processor)

        # Update the list of logits processors (preserve LogitsProcessorList type)
        self.logits_processor = LogitsProcessorList(filtered_processors)

    def __bool__(self) -> bool:
        return bool(self.logits_processor)

    def _retrieve_processors_kwargs(self) -> None:
        """Retrieves the supported (with types) and ignored kwargs from continuous batching processors."""
        self.supported_keys: dict[str, type] = {}
        self.ignored_keys = set()
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                self.supported_keys.update(processor.supported_kwargs)
                self.ignored_keys.update(processor.ignored_kwargs)

    def check_kwargs(self, kwargs: dict) -> None:
        """Checks that the provided kwargs are compatible with the current CB processors. Warn for ignored kwargs."""
        if not kwargs:
            return None
        # Validate types for supported keys, detect unsupported keys
        problematic_keys = set()
        for key, value in kwargs.items():
            if key not in self.supported_keys:
                problematic_keys.add(key)
            else:
                expected_type = self.supported_keys[key]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"logit_processor_kwargs['{key}'] has type {type(value).__name__}, expected {expected_type.__name__}"
                    )
        # Stop if there are only supported keys
        if not problematic_keys:
            return
        # Check if there are unknown keys
        unknown_keys = problematic_keys - self.ignored_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown logit_processor_kwargs: {unknown_keys}. {self.supported_keys = } and {self.ignored_keys = }"
                "If you expect a key to not be ignored, make sure its default value (in the generation config) is not "
                "None. Eg. if temperature is None or 1.0 at creation time, no processor will be created for temperature"
            )
        # If there are none, throw a warning about the ignored keys
        logger.warning(
            f"Ignored logit_processor_kwargs: {problematic_keys}. {self.supported_keys = } and {self.ignored_keys = }"
        )

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default values for this processor."""
        i = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                processor.fill_defaults(int32_tensor[i])
                i += 1

    def prepare_tensor_args(
        self, requests_in_batch: list[FutureRequestState], arg_storage: torch.Tensor
    ) -> torch.Tensor:
        current_arg_id = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                tensorized_arg = processor.prepare_tensor_args(requests_in_batch)
                arg_storage[current_arg_id, : tensorized_arg.size(0)] = tensorized_arg.to(arg_storage.device)
                current_arg_id += 1
        return arg_storage

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, logits_processor_args: torch.Tensor
    ) -> torch.FloatTensor:
        current_arg_id = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                scores = processor(scores, logits_processor_args[current_arg_id])
                current_arg_id += 1
            else:
                scores = processor(input_ids, scores)
        return scores


# Here are all the continuous batching logits processors that are supported
class ContinuousBatchingTemperatureLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"temperature": float}
    ignored_kwargs: tuple[str, ...] = ()

    def __init__(self, temperature_processor: TemperatureLogitsWarper) -> None:
        self.temperature = temperature_processor.temperature

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default temperature."""
        default = torch.empty_like(int32_tensor, dtype=torch.float32)
        default.fill_(self.temperature)
        int32_tensor.copy_(default.view(dtype=torch.int32))

    def prepare_tensor_args(self, requests_in_batch: list[FutureRequestState]) -> torch.Tensor:
        data = []
        for request in requests_in_batch:
            temp = request.state.logit_processor_kwargs.get("temperature", self.temperature)
            data.extend([temp] * request.query_length)
        tensorized = torch.tensor(data, dtype=torch.float32, device="cpu")
        # View the output with the bulk storage dtype (int32) but keeps the underlying data the same
        return tensorized.view(dtype=torch.int32)

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        temperatures = tensor_arg[: scores.size(0)].view(dtype=torch.float32)  # shape [N]
        return scores / temperatures.unsqueeze(-1)  # broadcast [N, 1] over [N, V]


class ContinuousBatchingTopKLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"top_k": int}
    ignored_kwargs: tuple[str, ...] = ("filter_value", "min_tokens_to_keep")

    def __init__(self, top_k_processor: TopKLogitsWarper):
        self.top_k = top_k_processor.top_k
        self.filter_value = top_k_processor.filter_value
        self.min_tokens_to_keep = top_k_processor.min_tokens_to_keep

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default top_k."""
        int32_tensor.fill_(self.top_k)

    def prepare_tensor_args(self, requests_in_batch: list[FutureRequestState]) -> torch.Tensor:
        top_ks = []
        for request in requests_in_batch:
            top_k = request.state.logit_processor_kwargs.get("top_k", self.top_k)
            top_k = max(top_k, self.min_tokens_to_keep)
            top_ks.extend([top_k] * request.query_length)
        # Prepare tensor arg with int32 as the main type
        tensor_args = torch.tensor(top_ks, dtype=torch.int32, device="cpu")
        return tensor_args

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies top-k selection to the scores tensor (shape [N, V])."""
        top_k = tensor_arg[: scores.size(0)]  # shape [N]
        # Sort descending, get threshold at position (top_k - 1) which is the k-th largest
        sorted_scores = torch.sort(scores, dim=-1, descending=True)[0]  # [N, V]
        top_k_indices = (top_k - 1).unsqueeze(-1).to(dtype=torch.int64)  # [N, 1]
        thresholds = sorted_scores.gather(dim=-1, index=top_k_indices)  # [N, 1]
        return scores.masked_fill(scores < thresholds, self.filter_value)


class ContinuousBatchingTopPLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"top_p": float}
    ignored_kwargs: tuple[str, ...] = ("filter_value", "min_tokens_to_keep")

    def __init__(self, top_p_processor: TopPLogitsWarper):
        self.top_p = top_p_processor.top_p
        self.filter_value = top_p_processor.filter_value
        self.min_tokens_to_keep = top_p_processor.min_tokens_to_keep

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default top_p."""
        default = torch.empty_like(int32_tensor, dtype=torch.float32)
        default.fill_(self.top_p)
        int32_tensor.copy_(default.view(dtype=torch.int32))

    def prepare_tensor_args(self, requests_in_batch: list[FutureRequestState]) -> torch.Tensor:
        top_ps = []
        for request in requests_in_batch:
            # Retrieve config for this request
            top_p = request.state.logit_processor_kwargs.get("top_p", self.top_p)
            top_ps.extend([top_p] * request.query_length)
        # Store top_p as float32 viewed as int32 to match the bulk storage dtype
        tensorized = torch.tensor(top_ps, dtype=torch.float32, device="cpu")
        return tensorized.view(dtype=torch.int32)

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies top-p (nucleus) sampling to the scores tensor (shape [N, V])."""
        top_p = tensor_arg[: scores.size(0)].view(dtype=torch.float32)  # shape [N]

        # Sort logits in ascending order
        sorted_logits, sorted_indices = torch.sort(scores, descending=False, dim=-1)  # [N, V]
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # [N, V]

        # Remove tokens with cumulative probability <= (1 - top_p)
        threshold = (1 - top_p).unsqueeze(-1)  # [N, 1]
        sorted_indices_to_remove = cumulative_probs <= threshold  # [N, V]

        # Keep at least min_tokens_to_keep (always keep the last tokens in sorted order = highest prob)
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = False

        # Scatter sorted mask back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        return scores.masked_fill(indices_to_remove, self.filter_value)


CLASSIC_TO_CB_PROCESSORS_MAP = {
    TemperatureLogitsWarper: ContinuousBatchingTemperatureLogitsWarper,
    TopKLogitsWarper: ContinuousBatchingTopKLogitsWarper,
    TopPLogitsWarper: ContinuousBatchingTopPLogitsWarper,
}
