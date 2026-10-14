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

"""Complete shared gradients before ZeRO-3 partitions them."""

import weakref
from collections import Counter

import torch
from torch.utils._pytree import tree_flatten

from .executor import _get_model_and_decoder, get_layer_execution_plan


class _SharedGradientCoordinator:
    def __init__(self, engine):
        self.engine = weakref.ref(engine)
        self.pending = {}
        self.queued = False
        self.handles = [
            engine.module.register_forward_pre_hook(self.prepare),
            engine.module.register_forward_hook(self.observe, always_call=True),
        ]

    def restore(self):
        parameters, self.pending = self.pending, {}
        self.queued = False
        for parameter, (exists, value) in parameters.items():
            if exists:
                parameter.ds_grad_is_ready = value
            else:
                del parameter.ds_grad_is_ready
        return parameters

    def remove(self):
        self.restore()
        for handle in self.handles:
            handle.remove()

    def prepare(self, module, inputs):
        if not torch.is_grad_enabled() or self.engine() is None:
            return
        plan = get_layer_execution_plan(module)
        if plan is None:
            return
        _, decoder = _get_model_and_decoder(module)
        layers = decoder._layer_execution_adapter.get_layers(decoder)
        for index, count in Counter(plan.layer_order).items():
            if count < 2:
                continue
            for parameter in layers[index].parameters():
                if (
                    parameter.requires_grad
                    and parameter not in self.pending
                    and getattr(parameter, "ds_grad_is_ready", True)
                ):
                    self.pending[parameter] = (hasattr(parameter, "ds_grad_is_ready"), True)
                    # ZeRO's tiled-gradient protocol also supports multiple contributions to one parameter.
                    parameter.ds_grad_is_ready = False

    def observe(self, module, inputs, output):
        if output is None:
            self.restore()
            return
        if not torch.is_grad_enabled() or not self.pending:
            return

        def queue(gradient):
            if not self.queued:
                # Register before parameter hooks queue ZeRO's post-backward epilogue.
                torch.autograd.Variable._execution_engine.queue_callback(self.complete)
                self.queued = True
            return gradient

        for tensor in tree_flatten(output)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(queue)

    def complete(self):
        parameters = self.restore()
        engine = self.engine()
        if engine is not None:
            for parameter in parameters:
                if parameter.grad is not None:
                    engine.optimizer.reduce_ready_partitions_and_remove_grads(parameter)


def configure_deepspeed_layer_execution(engine):
    """Configure an initialized ZeRO-3 engine before training a decoder execution plan.

    Trainer calls this automatically. Manual DeepSpeed users call it after ``deepspeed.initialize``.
    Repeated source parameters accumulate every contribution before normal ZeRO partition/reduction; the model's
    parameter registrations, optimizer groups, source-layer ownership, and inference caches remain unchanged.
    """
    if engine.zero_optimization_stage() != 3:
        return engine
    prior = getattr(engine.module, "_layer_execution_deepspeed_coordinator", None)
    if not any(hasattr(module, "_layer_execution_adapter") for module in engine.module.modules()):
        if prior is not None:
            prior.remove()
            del engine.module._layer_execution_deepspeed_coordinator
        return engine
    try:
        _, decoder = _get_model_and_decoder(engine)
    except TypeError:
        return engine
    if decoder.config.layer_execution_plan is None:
        return engine
    if not callable(getattr(engine.optimizer, "reduce_ready_partitions_and_remove_grads", None)):
        raise RuntimeError("This DeepSpeed optimizer does not expose the ZeRO-3 shared-gradient integration API.")
    if prior is not None:
        if prior.engine() is engine:
            return engine
        prior.remove()
    engine.module._layer_execution_deepspeed_coordinator = _SharedGradientCoordinator(engine)
    return engine
