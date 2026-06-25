# Copyright 2026 The HuggingFace Inc. team
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


from collections import deque
from itertools import repeat
from typing import Any

import torch

from ...configuration_utils import PretrainedConfig
from .requests import RequestState


# TODO: add block-based indexing
# TODO: add hash-based indexing for multimodal inputs
class EncoderCache:
    cache: torch.Tensor
    REQUEST_ID_KEY: str = "_cb_request_id"

    def __init__(
        self,
        config: PretrainedConfig,
        modality: str,
        max_batch_tokens: int,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Create the actual cache tensor
        cache_size = max(16384, max_batch_tokens)
        cache_shape = (cache_size, config.text_config.hidden_size)
        self.cache = torch.empty(cache_shape, dtype=model_dtype, device=device)
        # Create bookkeeping data structures
        self.free_blocks = deque(range(cache_size))
        self.allocated_blocks_masks: dict[str, torch.Tensor] = {}
        self.embeddings_lengths: dict[str, int] = {}
        # Specialize on modality the encoder cache object
        self._specialize_on_modality(config, modality)

    def _specialize_on_modality(self, config: PretrainedConfig, modality: str) -> None:
        """Specialize the encoder cache depending on the model's modality by retrieving the special token ID."""
        # Infer the name of the special token ID
        if modality == "image":
            possible_token_names = ["image_token_id", "image_token_index"]
        elif modality == "audio":
            possible_token_names = ["audio_token_id", "audio_token_index"]
        else:
            raise ValueError(f"Invalid modality: {modality}")
        # Retrieve the actual token ID
        token_id = None
        for token_name in possible_token_names:
            token_id = getattr(config, token_name, None)
            if token_id is not None:
                break
        if not isinstance(token_id, int) or token_id <= 0:
            raise ValueError(f"{token_name} token ID must be a positive integer but got {token_id = }")
        # Save attribute values
        self.special_token_id = token_id
        self.modality = modality

    def extract_mm_embeddings(self, encoding_output: Any) -> torch.Tensor:
        """Extracts the multimodal embeddings from the encoding output."""
        og_encoding_output = encoding_output
        # In most cases, the relevant outputs are located in the pooler_output attribute
        if hasattr(encoding_output, "pooler_output"):
            encoding_output = encoding_output.pooler_output
        # If the pooler output is a tuple or a list, use concatenate to get a tensor
        if isinstance(encoding_output, tuple) or isinstance(encoding_output, list):
            return torch.cat(encoding_output, dim=0)
        # Otherwise, return if the output is a tensor
        if isinstance(encoding_output, torch.Tensor):
            return encoding_output
        raise ValueError(f"Invalid encoding output: {og_encoding_output}")

    def can_store_mm_embeddings(self, state: RequestState) -> bool:
        """Checks if there is enough space in the encoder cache to store the multimodal embeddings."""
        # Retrieve the number of multimodal embeddings from the multimodal data (or compute and cache it)
        num_mm_embeddings = self.embeddings_lengths.get(state.request_id)
        if num_mm_embeddings is None:
            input_ids = torch.tensor(state.initial_tokens, device="cpu", dtype=torch.int32)
            num_mm_embeddings = (input_ids == self.special_token_id).sum().item()
            self.embeddings_lengths[state.request_id] = num_mm_embeddings
        return len(self.free_blocks) >= num_mm_embeddings

    def allocate_blocks(self, state: RequestState) -> None:
        """Allocates blocks for a request. This should only be called once per request."""
        # Get the list of allocated blocks for the request
        num_mm_embeddings = self.embeddings_lengths.pop(state.request_id)  # this value will never be used again
        allocated_blocks = [self.free_blocks.popleft() for _ in range(num_mm_embeddings)]
        # Infer the allocated blocks mask
        input_ids = torch.tensor(state.initial_tokens, device="cpu", dtype=torch.int32)
        img_mask = input_ids == self.special_token_id
        input_ids.fill_(-1)
        input_ids[img_mask] = torch.tensor(allocated_blocks, device="cpu", dtype=torch.int32)
        self.allocated_blocks_masks[state.request_id] = input_ids
        # TODO: this could be optimized by truncating from the first and last img tokens

    def extend_read_indices(
        self, request_id: str, past_length: int, query_length: int, read_indices: list[int]
    ) -> tuple[bool, bool]:
        """
        Extends the list of indices being read from the encoder cache for a given request. Returns a tuple of booleans:
            - cache_read: True if any multimodal embedding is read by this request
            - to_free: True if the request has all its multimodal embeddings read and can be freed from the cache
        For instance, if the initial tokens and allocated blocks are as follows:

            Initial tokens:   [xxx, xxx, xxx, img, img, img, xxx]
            Allocated blocks: [ -1,  -1,  -1,   0,   1,   3,  -1]
        Then for a past length of 3 and a query length of 5, the read indices will be:

            Read indices:     [                 0,   1,   3,  -1,  -1]

        and the function will return (True, True) because there are actual cache reads (block 0, 1 and 3 are read) and
        all its multimodal embeddings have been read: they can be freed from the cache.
        """
        to_free = False
        block_table = self.allocated_blocks_masks.get(request_id)
        # Only compute read indices if the request has allocated blocks
        if block_table is not None:
            intersection = block_table[past_length : past_length + query_length].tolist()
            missing_indices = query_length - len(intersection)
            # Check if any of the multimodal embeddings for this request are read in this batch
            cache_read = (block_table[past_length : past_length + query_length] != -1).any().item()
            # Check if all the multimodal embeddings for this request have been read
            if past_length + query_length >= len(block_table):
                to_free = True
        else:
            intersection = []
            missing_indices = query_length
            cache_read = False
        # Extend the read indices
        read_indices.extend(intersection)
        read_indices.extend(repeat(-1, missing_indices))
        return cache_read, to_free

    def store_mm_embeddings(self, request_id: str, mm_embedding: torch.Tensor) -> None:
        """Stores the multimodal embeddings for a request in the encoder cache."""
        # Retrieve the allocated blocks mask for the request
        allocated_blocks_mask = self.allocated_blocks_masks.get(request_id)
        if allocated_blocks_mask is None:
            raise ValueError(f"Request {request_id} has no allocated blocks mask")
        # Extract the allocated blocks from the mask
        mask = allocated_blocks_mask != -1
        allocated_blocks = allocated_blocks_mask[mask].to(self.cache.device)
        # Store the multimodal embeddings in the cache
        self.cache[allocated_blocks] = mm_embedding

    def release_cache_for_requests(self, requests: set[str]) -> None:
        """Releases the cache for the given requests from the encoder cache. The set of request for which to release
        cache is kept by the InputAndOutputs object, because in the case of asynchronous batching, a request for which
        the multimodal embeddings were fully processed in batch N cannot be freed in batch N+1."""
        # Loop until there are no requests for which to release cache
        while requests:
            request_id = requests.pop()
            # Retrieve the list of blocks to free
            allocated_blocks_mask = self.allocated_blocks_masks.pop(request_id, None)
            if allocated_blocks_mask is not None:
                mask = allocated_blocks_mask != -1
                blocks_to_free = allocated_blocks_mask[mask].tolist()
                # Actually free the blocks
                self.free_blocks.extend(blocks_to_free)
            # Also pop the embedding length for the request
            self.embeddings_lengths.pop(request_id, None)

    def free_all_requests(self) -> None:
        """Frees all requests from the encoder cache, whatever their state."""
        all_requests_stored = set(self.allocated_blocks_masks.keys()) | set(self.embeddings_lengths.keys())
        self.release_cache_for_requests(all_requests_stored)
