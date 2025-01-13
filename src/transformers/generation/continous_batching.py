# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import math
import statistics
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from flash_attn import flash_attn_with_kvcache

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ..utils import (
    is_accelerate_available,
    logging,
)
from .configuration_utils import (
    GenerationConfig,
    PretrainedConfig,
)
from .logits_process import (
    LogitsProcessorList,
)
from .stopping_criteria import (
    StoppingCriteriaList,
)
from .utils import (
    is_torchdynamo_compiling,
    logging,
)


if TYPE_CHECKING:
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    pass


from .. import GenerationMixin


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> torch.Tensor:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = flash_attn_with_kvcache(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None

ALL_ATTENTION_FUNCTIONS["paged_attention_forward"] = paged_attention_forward


class ContinuousBatch:
    ...


def compute_optimal_blocks(
    device: torch.device,
    generation_config: PretrainedConfig,
    inputs: List[List[int]],
    head_dim: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    safety_margin: float = 0.9  # Safety margin for memory usage
):
    # Get device memory properties
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - max(allocated_memory, reserved_memory)
    else:
        raise ValueError("This function currently supports CUDA devices only.")

    tokens_to_generate = generation_config.max_new_tokens
    median_prefill_length = statistics.median(len(elem) for elem in inputs) # we don't need too many blocks

    # Apply safety margin
    available_memory *= safety_margin
    # Memory per tensor element
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    # Estimate memory usage per block
    input_memory = median_prefill_length * head_dim * num_kv_heads * dtype_size
    output_memory = (median_prefill_length+tokens_to_generate) * head_dim * num_kv_heads * dtype_size
    per_block_memory = input_memory + output_memory

    # Compute the optimal number of blocks and block size
    max_blocks = available_memory // per_block_memory
    block_size = available_memory // (per_block_memory * head_dim * num_kv_heads * dtype_size)
    next_block_size = pow(2, math.ceil(math.log(block_size)/math.log(2))) # Round to next power of 2

    return int(max_blocks),  int(next_block_size)

class PagedAttentionCache:
    def __init__(
        self,
        config: PretrainedConfig,
        input_shapes:List[int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        num_blocks = config.num_blocks
        page_block_size = config.page_block_size
        if num_blocks is None or page_block_size is None:
            # We determine the best size,
            num_blocks, page_block_size = compute_optimal_blocks(device, config, input_shapes)

        cache_shape = (num_blocks, page_block_size,  self.num_key_value_heads,  self.head_dim)

        self.dtype = dtype

        self.key_cache: torch.Tensor = []
        self.value_cache: torch.Tensor = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                self.register_buffer(f"value_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

        # a single sequence should take at most  config.max_num_blocks_per_seq
        self.block_table = torch.zeros((config.batch_size, config.max_num_blocks_per_seq))


class ContinuousMixin(GenerationMixin):
    batches = None

    @torch.no_grad()
    def fast_generate(
        self,
        inputs: list = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        r"""
            Fast generate that leverages continuous batching.

            1. You prepare the inputs
            2. You run forward pass
            3. You check for eos:
                a. Evict requests that are done
                b. Add new requests
            4. Pack finished sequences to return them
            5. Go to step 2.

            NO ATTENTION MASKS
            NO ASSISTANT MODEL

        """
        paged_attention_cache = PagedAttentionCache(self, generation_config) # init with throughput desiered, batch, etc
        current_batch = ContinuousBatch(inputs, generation_config)
        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        while current_batch.len() > 0:
            continous_batch = current_batch.prepare_next_inputs()
            generated_tokens = self.forward(continous_batch, paged_attention_cache.cache_positions, paged_attention_cache.block_tables, ...)

            # predict next token
            generated_ids = torch.argmax(generated_tokens, dim=-1)
            current_batch.evict_finished_sequences(generated_ids)
            if len(current_batch.finished_sequences) > paged_attention_cache.batch_size:
                yield current_batch.finished_sequences

        return current_batch.all_generated_sequences
