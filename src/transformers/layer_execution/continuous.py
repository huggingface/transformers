# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Continuous admission and compatible-request batching with request-local execution caches."""

import copy
import queue
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from ..generation.configuration_utils import ContinuousBatchingConfig
from ..generation.continuous_batching.continuous_api import ContinuousBatchingManager, OutputRouter
from ..generation.continuous_batching.distributed import DistributedHelper
from ..generation.continuous_batching.requests import RequestState, RequestStatus
from ..utils import is_torch_distributed_available
from .cache import LayerExecutionCache
from .executor import _get_model_and_decoder


if TYPE_CHECKING:
    from .._typing import GenerativePreTrainedModel

if is_torch_distributed_available():
    from torch.distributed import ReduceOp, all_reduce, broadcast, broadcast_object_list, get_global_rank, get_rank


class LayerExecutionContinuousBatchingManager(ContinuousBatchingManager):
    """Continuously batch compatible requests with ordinary attention kernels and external recurrent state."""

    model: "GenerativePreTrainedModel"

    def __init__(
        self,
        model: "GenerativePreTrainedModel",
        generation_config,
        continuous_batching_config=None,
        workload_hints=None,
    ):
        self.model = model.eval()
        self.generation_config, _ = model._prepare_generation_config(copy.deepcopy(generation_config))
        self.continuous_batching_config = continuous_batching_config or ContinuousBatchingConfig()
        if self.generation_config.num_beams != 1:
            raise ValueError("Continuous batching uses greedy or sampling requests; use generate() for beam search.")
        _, self.decoder = _get_model_and_decoder(model)
        self.plan = model.get_layer_execution_plan()
        self.num_return_sequences = self.generation_config.num_return_sequences or 1
        self.input_queue = queue.Queue()
        self.output_router = OutputRouter()
        self._active, self._caches, self._processors = {}, {}, {}
        self._configs = {}
        self._pending = deque()
        self._max_active = self.continuous_batching_config.max_requests_per_batch or 16
        if type(self._max_active) is not int or self._max_active < 1:
            raise ValueError("max_requests_per_batch must be a positive integer.")
        self._commands = queue.Queue()
        self._ids = set()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self._compute_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._hard_stop = False
        self._destroyed = False
        self._pause_owner = None
        self._generation_thread = None
        self._fatal_error = None
        self.warmed_up = False
        self.distributed_helper = DistributedHelper(
            device_mesh=getattr(model, "_device_mesh", None),
            cpu_group_timeout=self.continuous_batching_config.cpu_group_timeout,
            tp_plan=getattr(model, "tp_plan", {}),
        )
        pipeline = getattr(model, "_pp_stage", None)
        self._group = pipeline.pp_group if pipeline is not None else self.distributed_helper.tp_group
        self.is_tp_driver = self._group is None or get_rank(self._group) == 0

    def switch_to_cb_friendly_attn(self, model, auto_switch_to_flash=True):
        return None

    def warmup(self):
        self.warmed_up = True

    def start(self):
        if self._destroyed:
            raise RuntimeError("A destroyed continuous batching manager cannot be restarted.")
        if not self.is_running():
            self._stop_event.clear()
            self._hard_stop = False
            self._fatal_error = None
            self._generation_thread = threading.Thread(target=self._run, daemon=True, name="layer-execution-batching")
            self._generation_thread.start()

    def is_running(self):
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def stop(self, block=True, timeout=None, keep_for_next_session=False, hard_stop=False):
        if self._pause_owner == threading.get_ident():
            raise RuntimeError("Leave the pause context before stopping continuous batching.")
        self._hard_stop = hard_stop
        self._stop_event.set()
        if block and self._generation_thread is not None:
            self._generation_thread.join(timeout)
            if self._generation_thread.is_alive():
                raise TimeoutError("Continuous batching did not finish stopping within the requested timeout.")
        if keep_for_next_session:
            setattr(self.model, "_cached_continuous_batching_manager", self)

    def destroy(self):
        self.stop(hard_stop=True)
        self._active.clear()
        self._caches.clear()
        self._processors.clear()
        self._configs.clear()
        self._pending.clear()
        self.distributed_helper.destroy_cpu_comm_group()
        self._destroyed = True
        if getattr(self.model, "_cached_continuous_batching_manager", None) is self:
            del self.model._cached_continuous_batching_manager

    @contextmanager
    def pause(self):
        with self._compute_lock:
            self._pause_owner = threading.get_ident()
            try:
                yield
            finally:
                self._pause_owner = None

    def add_request(
        self,
        input_ids,
        request_id=None,
        max_new_tokens=None,
        streaming=False,
        record_timestamps=False,
        eos_token_id=None,
        **logit_processor_kwargs,
    ):
        if not self.is_tp_driver:
            return None
        if self._destroyed or self._stop_event.is_set():
            raise RuntimeError("Continuous batching is stopping or destroyed; start a new session first.")
        if not input_ids or any(type(token) is not int for token in input_ids):
            raise ValueError("A request must contain a non-empty list of integer token IDs.")
        maximum = max_new_tokens if max_new_tokens is not None else self.generation_config.max_new_tokens
        maximum = maximum if maximum is not None else self.generation_config.max_length - len(input_ids)
        if type(maximum) is not int or maximum < 1:
            raise ValueError("A request requires a positive max_new_tokens.")
        with self._request_lock:
            request_id = request_id or f"req_{self._request_counter}"
            self._request_counter += 1
            if request_id in self._ids:
                raise ValueError(f"Duplicate request ID: {request_id!r}.")
            self._ids.add(request_id)
        state = RequestState(
            request_id=request_id,
            initial_tokens=list(input_ids),
            max_new_tokens=maximum,
            eos_token_id=eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id,
            streaming=streaming,
            record_timestamps=record_timestamps,
            logit_processor_kwargs=logit_processor_kwargs,
        )
        self.input_queue.put(state)
        return request_id

    def add_requests(self, inputs, **kwargs):
        result = []
        for tokens in inputs:
            for _ in range(self.num_return_sequences):
                identifier = self.add_request(tokens, **kwargs)
                if identifier is not None:
                    result.append(identifier)
        return result

    def cancel_request(self, request_id):
        if self.is_tp_driver:
            self._commands.put(request_id)

    def get_result(self, request_id=None, timeout=None):
        if not self.is_running() and self.output_router.output_queue.empty():
            return None
        try:
            result = self.output_router.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if request_id is not None and result.request_id != request_id:
            self.output_router.output_queue.put(result)
            return None
        return result

    def _drain(self, incoming):
        values = []
        while True:
            try:
                values.append(incoming.get_nowait())
            except queue.Empty:
                return values

    def _finish(self, state, error=None):
        if error is not None:
            state.error = str(error)
            state.status = RequestStatus.FAILED
        self._active.pop(state.request_id, None)
        self._caches.pop(state.request_id, None)
        self._processors.pop(state.request_id, None)
        self._configs.pop(state.request_id, None)
        if self.is_tp_driver:
            self.output_router.deliver(state.to_generation_output())

    def _admit(self, state):
        config = copy.deepcopy(self.generation_config)
        unknown = config.update(**state.logit_processor_kwargs)
        if unknown:
            raise ValueError(f"Unknown generation options: {tuple(unknown)}.")
        if config.num_beams != 1 or config.cache_implementation != self.generation_config.cache_implementation:
            raise ValueError("Request overrides must keep num_beams=1 and the manager's cache backend.")
        config.eos_token_id = state.eos_token_id
        config.max_length = len(state.initial_tokens) + state.max_new_tokens
        self.model._prepare_special_tokens(config, kwargs_has_attention_mask=False, device=self.model.device)
        self._processors[state.request_id] = self.model._get_logits_processor(
            generation_config=config,
            input_ids_seq_length=len(state.initial_tokens),
            device=self.model.device,
        )
        self._configs[state.request_id] = config
        state.status = RequestStatus.PREFILLING
        state.tokens_to_process = state.initial_tokens[:]
        self._active[state.request_id] = state

    def _step(self):
        groups = defaultdict(list)
        implementation = self.generation_config.cache_implementation or "dynamic"
        for state in self._active.values():
            cache = self._caches.get(state.request_id)
            key = (len(state.tokens_to_process), int(cache.get_seq_length()) if cache is not None else -1)
            if implementation != "dynamic":
                key += (state.request_id,)
            groups[key].append(state)
        limit = self._max_active
        for states in groups.values():
            for start in range(0, len(states), limit):
                batch = states[start : start + limit]
                inputs = torch.tensor([state.tokens_to_process for state in batch], device=self.model.device)
                caches = [self._caches.get(state.request_id) for state in batch]
                cache = LayerExecutionCache.stack(caches) if caches[0] is not None and len(batch) > 1 else caches[0]
                if cache is None:
                    cache = LayerExecutionCache(
                        self.decoder.config,
                        cache_implementation=implementation,
                        max_cache_len=max(len(state.initial_tokens) + state.max_new_tokens for state in batch),
                        cache_config=self.generation_config.cache_config,
                        steps=self.decoder._layer_execution_steps,
                    )
                output = self.model(
                    inputs,
                    **{self.decoder._layer_execution_adapter.cache_name: cache},
                    use_cache=True,
                    return_dict=True,
                )
                split = cache.unstack() if len(batch) > 1 else [cache]
                for index, state in enumerate(batch):
                    self._caches[state.request_id] = split[index]
                    full_ids = torch.tensor([state.initial_tokens + state.generated_tokens], device=inputs.device)
                    scores = self._processors[state.request_id](full_ids, output.logits[index : index + 1, -1].float())
                    probabilities = scores.softmax(dim=-1)
                    token = (
                        torch.multinomial(probabilities, 1)
                        if self._configs[state.request_id].do_sample
                        else scores.argmax(dim=-1, keepdim=True)
                    )
                    if self._group is not None:
                        broadcast(token, src=get_global_rank(self._group, 0), group=self._group)
                    state.position_offset += len(state.tokens_to_process)
                    state.status = RequestStatus.DECODING
                    complete = state.update_and_check_completion(
                        token.item(), probabilities.gather(1, token).log().item()
                    )
                    if complete:
                        self._finish(state)
                    elif state.streaming and self.is_tp_driver:
                        self.output_router.deliver(state.to_generation_output())

    @torch.no_grad()
    def _run(self):
        try:
            if self.model.device.type == "cuda":
                # CUDA's current device is thread-local, including the device used by NCCL object collectives.
                torch.cuda.set_device(self.model.device)
            while True:
                commands = [
                    (self._drain(self.input_queue), self._drain(self._commands)) if self.is_tp_driver else None
                ]
                stop_status = torch.tensor(
                    2 if self._hard_stop else int(self._stop_event.is_set()), device=self.model.device
                )
                if self._group is not None:
                    all_reduce(stop_status, op=ReduceOp.MAX, group=self._group)
                    broadcast_object_list(commands, src=get_global_rank(self._group, 0), group=self._group)
                incoming, cancelled = commands[0]
                self._pending.extend(incoming)
                for identifier in cancelled:
                    if identifier in self._active:
                        self._finish(self._active[identifier], "Request cancelled.")
                    remaining = deque()
                    while self._pending:
                        state = self._pending.popleft()
                        if state.request_id == identifier:
                            self._finish(state, "Request cancelled.")
                        else:
                            remaining.append(state)
                    self._pending = remaining
                if stop_status.item() == 2:
                    for state in [*self._active.values(), *self._pending]:
                        self._finish(state, "Continuous batching stopped.")
                    self._pending.clear()
                    break
                if self.plan != self.model.get_layer_execution_plan():
                    raise ValueError("Execution plan changed while continuous batching was active.")
                while self._pending and len(self._active) < self._max_active:
                    state = self._pending.popleft()
                    try:
                        self._admit(state)
                    except (ValueError, TypeError) as error:
                        self._finish(state, error)
                if self._active:
                    with self._compute_lock:
                        self._step()
                elif stop_status.item():
                    break
                else:
                    self._stop_event.wait(0.01)
        except Exception as error:
            self._fatal_error = error
            for state in list(self._active.values()):
                self._finish(state, error)
            for state in self._drain(self.input_queue):
                self._finish(state, error)
            for state in self._pending:
                self._finish(state, error)
            self._pending.clear()
