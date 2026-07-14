
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


# TODO: add block-based indexing (group several embeddings per allocation to cut bookkeeping)
# TODO: add hash-based indexing for multimodal inputs
class EmbeddingsCache:
    # One embedding is stored per row of the storage tensor. Rows are named this way to avoid confusion with the KV
    # cache "blocks" (each of which spans block_size tokens).
    storage: torch.Tensor
    REQUEST_ID_KEY: str = "_cb_request_id"

    def __init__(
        self,
        config: PretrainedConfig,
        modality: str,
        max_batch_tokens: int,
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        text_config = config.get_text_config(decoder=True)
        # Create the actual storage tensor
        self.cache_size = max(16384, max_batch_tokens)
        storage_shape = (self.cache_size, text_config.hidden_size)
        self.storage = torch.empty(storage_shape, dtype=model_dtype, device=device)
        # Create bookkeeping data structures
        self.free_rows = deque(range(self.cache_size))
        self.allocated_rows_masks: dict[str, torch.Tensor] = {}
        self.embeddings_lengths: dict[str, int] = {}
        # Specialize on modality the embeddings cache object
        self._specialize_on_modality(config, modality)

    def _specialize_on_modality(self, config: PretrainedConfig, modality: str) -> None:
        """Specialize the embeddings cache depending on the model's modality by retrieving the special token ID."""
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

    def can_ever_fit_mm_embeddings(self, state: RequestState) -> bool:
        """Checks if there is enough space in the embeddings cache to fit the multimodal embeddings if it is the only
        request."""
        num_mm_embeddings = state.count_mm_embeddings(self.special_token_id)
        return self.cache_size >= num_mm_embeddings

    def can_store_mm_embeddings(self, state: RequestState) -> bool:
        """Checks if there is enough space in the embeddings cache to store the multimodal embeddings."""
        # Retrieve the number of multimodal embeddings from the multimodal data (or compute and cache it)
        num_mm_embeddings = self.embeddings_lengths.get(state.request_id)
        if num_mm_embeddings is None:
            num_mm_embeddings = state.count_mm_embeddings(self.special_token_id)
            self.embeddings_lengths[state.request_id] = num_mm_embeddings
        return len(self.free_rows) >= num_mm_embeddings

    def allocate_rows(self, state: RequestState) -> None:
        """Allocates storage rows for a request. This should only be called once per request."""
        # Get the list of allocated rows for the request
        num_mm_embeddings = self.embeddings_lengths.pop(state.request_id)  # this value will never be used again
        allocated_rows = [self.free_rows.popleft() for _ in range(num_mm_embeddings)]
        # Infer the allocated rows mask
        input_ids = torch.tensor(state.initial_tokens, device="cpu", dtype=torch.int32)
        img_mask = input_ids == self.special_token_id
        input_ids.fill_(-1)
        input_ids[img_mask] = torch.tensor(allocated_rows, device="cpu", dtype=torch.int32)
        self.allocated_rows_masks[state.request_id] = input_ids
        # TODO: this could be optimized by truncating from the first and last img tokens

    def extend_read_indices(
        self, request_id: str, past_length: int, query_length: int, read_indices: list[int]
    ) -> tuple[bool, bool]:
        """
        Extends the list of indices being read from the embeddings cache for a given request. Returns a tuple of booleans:
            - cache_read: True if any multimodal embedding is read by this request
            - to_free: True if the request has all its multimodal embeddings read and can be freed from the cache
        For instance, if the initial tokens and allocated rows are as follows:

            Initial tokens:   [xxx, xxx, xxx, img, img, img, xxx]
            Allocated rows:   [ -1,  -1,  -1,   0,   1,   3,  -1]
        Then for a past length of 3 and a query length of 5, the read indices will be:

            Read indices:     [                 0,   1,   3,  -1,  -1]

        and the function will return (True, True) because there are actual cache reads (row 0, 1 and 3 are read) and
        all its multimodal embeddings have been read: they can be freed from the cache.
        """
        to_free = False
        rows_mask = self.allocated_rows_masks.get(request_id)
        # Only compute read indices if the request has allocated rows
        if rows_mask is not None:
            submask = rows_mask[past_length : past_length + query_length]
            read_ids = submask.tolist()
            missing_indices = query_length - len(read_ids)
            # Check if any of the multimodal embeddings for this request are read in this batch
            cache_read = (submask != -1).any().item()
            # Check if all the multimodal embeddings for this request have been read
            if past_length + query_length >= len(rows_mask):
                to_free = True
        else:
            read_ids = []
            missing_indices = query_length
            cache_read = False
        # Extend the read indices
        read_indices.extend(read_ids)
        read_indices.extend(repeat(-1, missing_indices))
        return cache_read, to_free

    def store_mm_embeddings(self, request_id: str, mm_embedding: torch.Tensor) -> None:
        """Stores the multimodal embeddings for a request in the embeddings cache."""
        # Retrieve the allocated rows mask for the request
        allocated_rows_mask = self.allocated_rows_masks.get(request_id)
        if allocated_rows_mask is None:
            raise ValueError(f"Request {request_id} has no allocated rows mask")
        # Extract the allocated rows from the mask
        mask = allocated_rows_mask != -1
        allocated_rows = allocated_rows_mask[mask].to(self.storage.device)
        # Flatten along the image dimension; the row count must match the number of special-token positions
        mm_embedding = mm_embedding.reshape(-1, mm_embedding.shape[-1])
        if mm_embedding.shape[0] != allocated_rows.numel():
            raise ValueError(
                f"Request {request_id} has {allocated_rows.numel()} multimodal tokens but the encoder produced "
                f"{mm_embedding.shape[0]} embeddings."
            )
        # Store the multimodal embeddings in the cache
        self.storage[allocated_rows] = mm_embedding

    def release_cache_for_requests(self, requests: set[str]) -> None:
        """Releases the cache for the given requests from the embeddings cache. The set of request for which to release
        cache is kept by the InputAndOutputs object, because in the case of asynchronous batching, a request for which
        the multimodal embeddings were fully processed in batch N cannot be freed in batch N+1."""
        for request_id in requests:
            # Retrieve the list of rows to free
            allocated_rows_mask = self.allocated_rows_masks.pop(request_id, None)
            if allocated_rows_mask is not None:
                mask = allocated_rows_mask != -1
                rows_to_free = allocated_rows_mask[mask].tolist()
                # Actually free the rows
                self.free_rows.extend(rows_to_free)
            # Also pop the embedding length for the request
            self.embeddings_lengths.pop(request_id, None)

    def free_all_requests(self) -> None:
        """Frees all requests from the embeddings cache, whatever their state."""
        all_requests_stored = set(self.allocated_rows_masks.keys()) | set(self.embeddings_lengths.keys())
        self.release_cache_for_requests(all_requests_stored)
