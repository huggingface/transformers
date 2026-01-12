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

import torch

from ...utils import is_torch_xpu_available
from ...utils.logging import logging
from ...utils.metrics import traced


# We centralize the logger here to coordinate between logging and progress bar
logger = logging.getLogger("ContinuousBatchingLogger")


def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
    elif is_torch_xpu_available():
        device = torch.device("xpu")
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
        total_memory = torch.xpu.get_device_properties(device).total_memory
        reserved_memory = torch.xpu.memory_reserved(device)
        allocated_memory = torch.xpu.memory_allocated(device)
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
    """

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    error: str | None = None
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.perf_counter)
    timestamps: list[float] | None = None  # Timestamps of the generated tokens

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        request_id (str): The ID of the generation request.
        initial_tokens (list[int]): The initial prompt tokens.
        num_children (int): The number of children requests
        full_prompt_ids (list[int] | None): The tokens IDs of the full prompt.
        prompt_ids (list[int] | None): The tokens IDs currently being processed.
        remaining_prompt_ids (list[int]): The tokens IDs remaining to be processed (for split requests).
        static_outputs (list[int]): The generated tokens.
        allocated_blocks (int): The number of blocks allocated to the request.
        position_offset (int): The current position in the sequence for position_ids.
        status (RequestStatus): The status of the request: can be one of PENDING, PREFILLING, PREFILLING_SPLIT,
                                SPLIT_PENDING_REMAINDER, DECODING, FINISHED, FAILED
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_token_id (int): The ID of the end-of-sequence token.
        streaming (bool): Whether to stream tokens as they're generated
        created_time (float): The time the request was created.
        error (Optional[str]): Any error message associated with the request. When None, has had no error yet.
    """

    # Required fields
    request_id: str
    initial_tokens: list[int]  # Initial prompt tokens
    # Optional fields
    record_timestamps: bool = False  # Whether to record timestamps for the generated tokens
    num_children: int = 0  # Number of children requests
    # Internal fields
    tokens_to_process: list[int] | None = None  # Tokens IDs currently being processed
    remaining_prefill_tokens: list[int] = field(default_factory=list)  # For split requests, prefill left to process
    generated_tokens: list[int] = field(default_factory=list)  # Generated tokens
    allocated_blocks: int = 0  # Number of blocks allocated to the request
    position_offset: int = 0  # Current position in the sequence for position_ids
    _status: RequestStatus = RequestStatus.PENDING  # Status of the request, hidden behind a property
    max_new_tokens: int = 20  # Maximum number of new tokens to generate
    eos_token_id: int = -1  # ID of the end-of-sequence token
    streaming: bool = False  # Whether to stream tokens as they're generated
    created_time: float = field(default_factory=time.perf_counter)  # Time the request was created
    error: str | None = None  # Error message if the request failed
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)
    _timestamps: list[float] = field(default_factory=list)  # Timestamps of the generated tokens

    @property
    def status(self) -> RequestStatus:
        return self._status

    @status.setter
    def status(self, value: RequestStatus):
        if self._status == RequestStatus.PENDING:
            self.lifespan = (time.perf_counter(), -1)
        elif value == RequestStatus.FINISHED:
            self.lifespan = (self.lifespan[0], time.perf_counter())
            self.log_end_of_request()
        self._status = value

    @property
    def timestamps(self) -> list[float] | None:
        return self._timestamps if self.record_timestamps else None

    def log_end_of_request(self):
        prefill_len = len(self.initial_tokens)
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
        return len(self.generated_tokens)

    # TODO: this logic seems one token off, check it out
    @traced
    def update_and_check_completion(self, token_id: int) -> bool:
        """Update the request with a newly generated token and check for completion.

        Args:
            token_id: The token ID to add to the output sequence

        Returns:
            bool: True if the request is now complete, False otherwise
        """
        # Only update if we're in decoding state # TODO: seems useless (always true) -- remove this
        if self.status != RequestStatus.DECODING:
            return False

        # If we're recording timestamps, add timestamp to the list
        if self.record_timestamps:
            self._timestamps.append(time.perf_counter())

        is_eos = token_id == self.eos_token_id and self.eos_token_id != -1
        is_max_len = self.generated_len() >= self.max_new_tokens

        # Only add the token if we're not finishing due to max length
        # (EOS tokens should still be added to the output)
        if not (is_max_len and not is_eos):
            self.generated_tokens.extend([token_id])

        if is_eos or is_max_len:
            self.status = RequestStatus.FINISHED
            return True
        return False

    def __repr__(self):
        msg = [
            f"request_id={self.request_id}",
            f"status={self._status}",
            f"out_tokens={self.generated_len()}",
            f"query_length={len(self.tokens_to_process)}",
            f"remaining_tokens={len(self.remaining_prefill_tokens)}",
            f"kv_length={self.position_offset}",
            f"full_prompt_length={len(self.initial_tokens)}",
            f"allocated_blocks={self.allocated_blocks}",
            f"generated_tokens={self.generated_tokens}",
        ]
        return "RequestState(\n\t" + ",\n\t".join(msg) + "\n)"

    def to_generation_output(self):
        """Convert the request state to a GenerationOutput object."""
        return GenerationOutput(
            request_id=self.request_id,
            prompt_ids=self.initial_tokens,
            status=self.status,
            generated_tokens=self.generated_tokens,
            logprobs=[],
            error=self.error,
            timestamps=self.timestamps,
        )

    def fork(self, new_request_id: str) -> "RequestState":
        """Fork the request into a new request with the same state expect for request_id, created_time and lifespan."""
        t = time.perf_counter()
        new_request = RequestState(
            request_id=new_request_id,
            initial_tokens=self.initial_tokens,
            num_children=self.num_children,
            tokens_to_process=self.tokens_to_process[:],
            remaining_prefill_tokens=self.remaining_prefill_tokens[:],
            generated_tokens=self.generated_tokens[:],
            allocated_blocks=self.allocated_blocks,
            position_offset=self.position_offset,
            _status=self.status,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            streaming=self.streaming,
            created_time=t,
            lifespan=(t, -1),
            _timestamps=None if self.timestamps is None else self.timestamps[:],
            error=self.error,
            record_timestamps=self.record_timestamps,
        )
        return new_request
