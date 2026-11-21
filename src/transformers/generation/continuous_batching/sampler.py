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
from typing import Any, NamedTuple

import torch
from torch import nn

from ..logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
from .cb_logits_processors import (
    ContinuousBatchingLogitsProcessor,
    ContinuousBatchingLogitsProcessorList,
    ContinuousBatchingTemperatureLogitsWarper,
    ContinuousBatchingTopKLogitsWarper,
    ContinuousBatchingTopPLogitsWarper,
)
from .requests import FutureRequestState


class SamplingMode(NamedTuple):
    """How the next tokens of a batch are picked, chosen on the host as the batch is prepared. Part of the CUDA graph
    key, as each mode records a graph of its own:
    - `"greedy"`: every request is greedy (a top-k of 1), its token is the argmax of its logits
    - `"window"`: every top-k fits in `window`, so the candidates of a row are its `window` largest logits
    - `"full"`: the logits processors on the whole vocabulary, then a softmax and a multinomial (or an argmax)
    """

    kind: str
    window: int = 0


FULL_SAMPLING = SamplingMode("full")
GREEDY_SAMPLING = SamplingMode("greedy")
# The window of the largest logits the candidates are taken from, when every top-k of a batch fits in it
MIN_WINDOW = 32
MAX_WINDOW = 1024
# The processors applied to a window, with a setting per request (continuous batching's) or for the whole batch
FUSED_PROCESSORS = {
    ContinuousBatchingTemperatureLogitsWarper: "temperature",
    TemperatureLogitsWarper: "temperature",
    ContinuousBatchingTopKLogitsWarper: "top_k",
    TopKLogitsWarper: "top_k",
    ContinuousBatchingTopPLogitsWarper: "top_p",
    TopPLogitsWarper: "top_p",
}
FUSED_ORDER = ("temperature", "top_k", "top_p")


class ContinuousBatchingSampler:
    """
    Picks the next token of each request of a batch from its logits, with the request's own sampling settings.

    When the temperature, top-k and top-p (per request or for the whole batch) are the last logits processors, and no
    logprobs are returned, it never sorts the vocabulary: a greedy request takes the argmax, and the others are
    sampled among the largest logits of their row, from the same distribution as the processors followed by a softmax
    over the vocabulary. Otherwise, the processors are applied to the whole vocabulary.

    Args:
        logit_processor (`ContinuousBatchingLogitsProcessorList`): The logits processors of the batches.
        do_sample (`bool`): Whether tokens are sampled, rather than the argmax.
        return_logprobs (`bool`): Whether the log probability of each token is returned with it.
    """

    def __init__(
        self, logit_processor: ContinuousBatchingLogitsProcessorList, do_sample: bool, return_logprobs: bool
    ) -> None:
        self.logit_processor = logit_processor
        self.do_sample = do_sample
        self.return_logprobs = return_logprobs
        self.fused_args = self._find_fused_args() if do_sample and not return_logprobs else None
        # Applied to the whole vocabulary before the others are applied to a window
        self.leading_processors = [
            processor for processor in logit_processor.logits_processor if type(processor) not in FUSED_PROCESSORS
        ]

    def _find_fused_args(self) -> dict[str, tuple[int | None, Any]] | None:
        """The argument row (None for a setting of the whole batch) and processor of the temperature, top-k and top-p,
        when they can be applied to a window of the logits: a top-k, the processors in the order temperature, top-k,
        top-p (temperature and top-p are optional), no processor after them, and no per-request one before them."""
        found = {}
        arg_id = 0
        for processor in self.logit_processor.logits_processor:
            name = FUSED_PROCESSORS.get(type(processor))
            per_request = isinstance(processor, ContinuousBatchingLogitsProcessor)
            if name is None:
                if found or per_request:
                    return None
                continue
            if found and FUSED_ORDER.index(name) <= FUSED_ORDER.index(list(found)[-1]):
                return None
            found[name] = (arg_id if per_request else None, processor)
            if per_request:
                arg_id += 1
        return found if "top_k" in found else None

    def _settings(
        self, name: str, logits_processor_args: torch.Tensor, logits: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """The `name` setting of each row of `logits`, of shape [num_logits, 1]: its request's, or the batch's."""
        arg_id, processor = self.fused_args[name]
        if arg_id is None:
            return torch.full((logits.size(0), 1), getattr(processor, name), dtype=dtype, device=logits.device)
        return logits_processor_args[arg_id, : logits.size(0)].view(dtype=dtype).unsqueeze(-1)

    def prepare_args(self, requests_in_batch: list[FutureRequestState], arg_storage: torch.Tensor) -> SamplingMode:
        """Writes the per-request arguments of the logits processors in `arg_storage`, and returns how the batch's next
        tokens are picked."""
        host_args = self.logit_processor.prepare_tensor_args(requests_in_batch, arg_storage)
        if self.fused_args is None:
            return FULL_SAMPLING
        top_k_id, top_k_processor = self.fused_args["top_k"]
        top_ks = [top_k_processor.top_k] if top_k_id is None else host_args[top_k_id].tolist()
        if all(top_k == 1 for top_k in top_ks):
            return GREEDY_SAMPLING
        # Twice the largest top-k, so that the logits equal to the k-th one are kept, as the processors do
        window = max(MIN_WINDOW, 1 << (2 * max(top_ks) - 1).bit_length())
        return SamplingMode("window", window) if window <= MAX_WINDOW else FULL_SAMPLING

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits: torch.Tensor,
        logits_processor_args: torch.Tensor,
        mode: SamplingMode,
        output_ids: torch.Tensor,
    ) -> None:
        """Writes the next token of each row of `logits` (shape [num_logits, vocab_size]) in `output_ids`, and with
        `return_logprobs`, its log probability in the row after them."""
        if mode.kind == "full":
            self._sample_vocabulary(input_ids, logits, logits_processor_args, output_ids)
            return
        for processor in self.leading_processors:
            logits = processor(input_ids, logits)
        next_tokens = logits.argmax(dim=-1)
        if mode.kind == "window":
            values, indices = self._candidates(logits, logits_processor_args, mode.window)
            choice = torch.multinomial(values.softmax(dim=-1), num_samples=1)
            sampled_tokens = indices.gather(dim=-1, index=choice).squeeze(-1)
            is_greedy = self._settings("top_k", logits_processor_args, logits, torch.int32).squeeze(-1) == 1
            next_tokens = torch.where(is_greedy, next_tokens, sampled_tokens)
        output_ids[0, : next_tokens.size(0)].copy_(next_tokens)

    def _candidates(
        self, logits: torch.Tensor, logits_processor_args: torch.Tensor, window: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the temperature, top-k and top-p to the `window` largest logits of each row. Returns the processed
        logits, those filtered out set to the filter value, and their token ids, both of shape [num_logits, window]."""
        window = min(window, logits.size(-1))
        values, indices = torch.topk(logits, window, dim=-1)  # descending
        if "temperature" in self.fused_args:
            values = values / self._settings("temperature", logits_processor_args, logits, torch.float32)

        _, top_k_processor = self.fused_args["top_k"]
        top_k = self._settings("top_k", logits_processor_args, logits, torch.int32)
        kth_position = (top_k.clamp(min=1, max=window) - 1).to(dtype=torch.int64)
        thresholds = values.gather(dim=-1, index=kth_position)
        values = values.masked_fill(values < thresholds, top_k_processor.filter_value)

        if "top_p" in self.fused_args:
            _, top_p_processor = self.fused_args["top_p"]
            top_p = self._settings("top_p", logits_processor_args, logits, torch.float32)
            cumulative_probs = values.flip(-1).softmax(dim=-1).cumsum(dim=-1)  # ascending, as the processor
            to_remove = cumulative_probs <= (1 - top_p)
            to_remove[..., -min(top_p_processor.min_tokens_to_keep, window) :] = False
            values = values.masked_fill(to_remove.flip(-1), top_p_processor.filter_value)
        return values, indices

    def _sample_vocabulary(
        self,
        input_ids: torch.LongTensor,
        logits: torch.Tensor,
        logits_processor_args: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """The logits processors on the whole vocabulary, then the softmax and the multinomial (or the argmax)."""
        if self.logit_processor.do_processing:
            logits = self.logit_processor(input_ids, logits, logits_processor_args)
        # Apply softmax if we are sampling or if we are generating log probabilities
        if self.do_sample or self.return_logprobs:
            probs = nn.functional.softmax(logits, dim=-1)  # shape [num_logits, vocab_size]
        else:
            probs = logits
        if self.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1)  # shape [num_logits, 1]
        else:
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)  # shape [num_logits, 1]

        tokens = next_tokens.size(0)
        output_ids[0, :tokens].copy_(next_tokens.squeeze(-1))
        if self.return_logprobs:
            logprobs = probs.gather(dim=1, index=next_tokens).squeeze(-1).log()  # shape [num_logits]
            # In order to match the dtype of output_ids, we cast the fp32 logprobs as int32 without changing the
            # underlying data. It's just a trick to use the same storage for both tensors.
            output_ids[1, :tokens].copy_(logprobs.view(dtype=torch.int32))
