# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team
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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

from ...utils.logging import logging
from ...utils.metrics import traced


# We centralize the logger here to coordinate between logging and progress bar
logger = logging.getLogger("ContinuousBatchingLogger")
logger.setLevel(logging.INFO)


@staticmethod
def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        # MPS memory reporting (PyTorch 2.0+)
        total_memory = torch.mps.driver_allocated_memory()
        allocated_memory = total_memory - torch.mps.recommended_max_memory()
        reserved_memory = 0  # MPS does not track reserved separately
    else:
        device = torch.device("cpu")
        total_memory = None
        reserved_memory = 0
        allocated_memory = 0
    return device, total_memory, reserved_memory, allocated_memory


class RequestStatus(Enum):
    """Status of a generation request through its lifecycle."""

    PENDING = "pending"
    PREFILLING = "prefilling"
    PREFILLING_SPLIT = "prefilling_split"
    SPLIT_PENDING_REMAINDER = "split_pending_remainder"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class GenerationOutput:
    """Tracks the output of a generation request.

    Attributes:
        request_id (str): The ID of the generation request.
        prompt_ids (list[int]): The IDs of the prompt tokens.
        generated_tokens (list[int]): The generated tokens.
        logprobs (list[float]): The log probabilities of the generated tokens.
        error (Optional[str]): Any error message associated with the request. When None, the request was successful.
        status (RequestStatus): The status of the request.
        created_time (float): The time the request was created.
        next_token (Optional[int]): The next token to be generated.
    """

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    error: Optional[str] = None
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.time)
    next_token: Optional[int] = field(default_factory=int)


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        request_id (str): The ID of the generation request.
        full_prompt_ids (list[int] | None): The tokens IDs of the full prompt.
        prompt_ids (list[int] | None): The tokens IDs currently being processed.
        remaining_prompt_ids (list[int]): The tokens IDs remaining to be processed (for split requests).
        static_outputs (list[int]): The generated tokens.
        allocated_blocks (list[int]): The identifiers of the allocated blocks to the request.
        position_offset (int): The current position in the sequence for position_ids.
        status (RequestStatus): The status of the request: can be one of PENDING, PREFILLING, PREFILLING_SPLIT,
                                SPLIT_PENDING_REMAINDER, DECODING, FINISHED, FAILED
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_token_id (int): The ID of the end-of-sequence token.
        created_time (float): The time the request was created.
        error (Optional[str]): Any error message associated with the request. When None, has had no error yet.
        next_token (Optional[str]): The next token to be generated.
    """

    # Required fields
    request_id: str
    full_prompt_ids: Optional[list[int]] = None  # Full initial prompt
    prompt_ids: Optional[list[int]] = None  # Tokens IDs currently being processed (initial + generated)
    remaining_prompt_ids: list[int] = field(default_factory=list)  # For split requests, prefill left to process
    static_outputs: list[int] = field(default_factory=list)  # Generated tokens
    allocated_blocks: list[int] = field(default_factory=list)  # Block IDs allocated to the request
    position_offset: int = 0  # Current position in the sequence for position_ids
    _status: RequestStatus = RequestStatus.PENDING  # Status of the request, hidden behind a property
    max_new_tokens: int = 20  # Maximum number of new tokens to generate
    eos_token_id: int = -1  # ID of the end-of-sequence token
    created_time: float = field(default_factory=time.time)  # Time the request was created
    error: Optional[str] = None  # Error message if the request failed
    next_token: Optional[str] = None  # Next token to be generated
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)

    @property
    def status(self) -> RequestStatus:
        return self._status

    @status.setter
    def status(self, value: RequestStatus):
        if self._status == RequestStatus.PENDING:
            self.lifespan = (time.time(), -1)
        elif value == RequestStatus.FINISHED:
            self.lifespan = (self.lifespan[0], time.time())
            self.log_end_of_request()
        self._status = value

    def log_end_of_request(self):
        prefill_len = len(self.full_prompt_ids)
        decode_len = self.generated_len()
        start_time = self.lifespan[0] - self.created_time
        end_time = self.lifespan[1] - self.created_time
        logger.info(
            f"Request {self.request_id} finished: {prefill_len = } {decode_len = } {start_time = } {end_time = }"
        )

    def current_len(self) -> int:
        """Get the current length of the sequence (prompt + generated tokens)."""
        return self.position_offset

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.static_outputs)

    # TODO: this logic seems one token off, check it out
    @traced
    def update_with_token(self, token_id: int) -> bool:
        """Update the request with a newly generated token and check for completion.

        Args:
            token_id: The token ID to add to the output sequence

        Returns:
            bool: True if the request is now complete, False otherwise
        """
        # Only update if we're in decoding state
        if self.status != RequestStatus.DECODING:
            return False

        is_eos = token_id == self.eos_token_id and self.eos_token_id != -1
        is_max_len = self.generated_len() >= self.max_new_tokens

        # Only add the token if we're not finishing due to max length
        # (EOS tokens should still be added to the output)
        if not (is_max_len and not is_eos):
            self.static_outputs.extend([token_id])

        if is_eos or is_max_len:
            self.status = RequestStatus.FINISHED
            return True
        return False

    def __repr__(self):
        msg = [
            f"request_id={self.request_id}",
            f"status={self._status}",
            f"out_tokens={self.generated_len()}",
            f"query_length={len(self.prompt_ids)}",
            f"remaining_tokens={len(self.remaining_prompt_ids)}",
            f"kv_length={self.position_offset}",
            f"full_prompt_length={len(self.full_prompt_ids)}",
            f"allocated_blocks={self.allocated_blocks}",
            f"generated_tokens={self.static_outputs}",
        ]
        return "RequestState(\n\t" + ",\n\t".join(msg) + "\n)"

    def to_generation_output(self):
        """Convert the request state to a GenerationOutput object."""
        return GenerationOutput(
            request_id=self.request_id,
            prompt_ids=self.full_prompt_ids,
            status=self.status,
            generated_tokens=self.static_outputs,
            logprobs=[],
            error=self.error,
            next_token=self.next_token,
        )
