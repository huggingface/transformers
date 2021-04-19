# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import collections

from .file_utils import ExplicitEnum, is_torch_available
from .utils import logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class DebugActivationOverflow:
    """
    This debug class helps detect and understand where the model starts getting ``nan`` or ``inf`` in activation
    elements.

    To activate, initialize the object with the model ::

        debug_overflow = DebugActivationOverflow(model)

    then run the training as normal and if any ``nan`` or ``inf`` get detected this module will throw an exception and
    will print several dozens of frames that lead to this event, each line reporting:

    1. the absolute largest element of either input or output variable
    2. the batch number
    3. the fully qualified state_dict key of which element it was run for,
    4. the class name whose ``forward`` was run
    5. and finally whether it was an input or output and its index if it was a tuple.

    Args:
        model (:obj:`nn.Module`):
            The model to debug.
        max_frames_to_save (:obj:`int`, `optional`, defaults to 21):
            How many frames back to record - a few dozens is a good number.
    """

    def __init__(self, model, max_frames_to_save=21):
        self.model = model

        # keep a LIFO buffer of frames to dump as soon as inf/nan is encountered to give context to the problem emergence
        self.frames = collections.deque([], max_frames_to_save)
        self.frame = []
        self.batch_number = 0
        self.detected_overflow = False

        self.analyse_model()

        self.register_forward_hook()

    def save_frame(self, frame=None):
        if frame is not None:
            self.expand_frame(frame)
        self.frames.append("\n".join(self.frame))
        self.frame = []  # start a new frame

    def expand_frame(self, line):
        self.frame.append(line)

    def dump_saved_frames(self):
        print(f"\n\nDetected inf/nan during batch_number={self.batch_number}")
        print(f"last {len(self.frames)} frames:")
        print(f"{'abs min':8} {'abs max':8} metadata")
        print("\n".join(self.frames))
        print("\n\n")

    def analyse_model(self):
        # extract the fully qualified module names, to be able to report at run time. e.g.:
        # encoder.block.2.layer.0.SelfAttention.o
        #
        # for shared weights only the first shared module name will be registered
        self.module_names = {m: name for name, m in self.model.named_modules()}
        self.longest_module_name = max(len(v) for v in self.module_names.values())

    def analyse_variable(self, var, ctx):
        if torch.is_tensor(var):
            self.expand_frame(get_abs_min_max(var, ctx))
            if detect_overflow(var, ctx):
                self.detected_overflow = True

    def register_forward_hook(self):
        self.model.apply(self._register_forward_hook)

    def _register_forward_hook(self, module):
        module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # - input is a tuple of packed inputs (could be non-Tensors)
        # - output could be a Tensor or a tuple of Tensors and non-Tensors

        prefix = "                 "

        # count batch numbers
        if module == self.model:
            self.batch_number += 1
            self.expand_frame(f"{prefix} Start batch_number={self.batch_number}")

        self.expand_frame(f"{prefix} {self.module_names[module]} {module.__class__.__name__}")

        # params
        for name, p in module.named_parameters(recurse=False):
            self.analyse_variable(p, name)

        # inputs
        if len(input) > 1:
            for i, x in enumerate(input):
                self.analyse_variable(x, f"input[{i}]")
        else:
            self.analyse_variable(input[0], "input")

        # outputs
        if isinstance(output, tuple):
            for i, x in enumerate(output):
                # possibly a tuple of tuples
                if isinstance(x, tuple):
                    for j, y in enumerate(x):
                        self.analyse_variable(y, f"output[{i}][{j}]")
                else:
                    self.analyse_variable(x, f"output[{i}]")
        else:
            self.analyse_variable(output, "output")

        self.save_frame()

        if self.detected_overflow:
            self.dump_saved_frames()

            # now we can die, as it's pointless to continue running
            raise ValueError(
                "DebugActivationOverflow: inf/nan detected, aborting as there is no point running further. "
                "Please scroll up above this traceback to see the activation values prior to this event."
            )


def get_abs_min_max(var, ctx):
    abs_var = var.abs()
    return f"{abs_var.min():8.2e} {abs_var.max():8.2e} {ctx}"


def detect_overflow(var, ctx):
    """
    Report the count of ``nan`` and ``inf`` entries in the tensor.

    This is useful for detecting overflows/underflows and best to call right after the function that did some math that
    modified the variable in question.

    Args:
        var: tensor variable to check
        ctx: the message to print as a context

    Return:
        True if inf or nan was detected, False otherwise
    """
    detected = False
    if torch.isnan(var).any().item():
        detected = True
        print(f"{ctx} has nans")
    if torch.isinf(var).any().item():
        detected = True
        print(f"{ctx} has infs")

    # if needed to monitor large elements can enable the following
    if 0:  # and detected:
        n100 = var[torch.ge(var.abs(), 100)]
        if n100.numel() > 0:
            print(f"{ctx}:  n100={n100.numel()}")
        n1000 = var[torch.ge(var.abs(), 1000)]
        if n1000.numel() > 0:
            print(f"{ctx}: n1000={n1000.numel()}")
        n10000 = var[torch.ge(var.abs(), 10000)]
        if n10000.numel() > 0:
            print(f"{ctx}: n10000={n10000.numel()}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e}")

    if 0:
        print(f"min={var.min():9.2e} max={var.max():9.2e} var={var.var():9.2e} mean={var.mean():9.2e} ({ctx})")

    return detected


class DebugOption(ExplicitEnum):
    ACIVATION_OVERFLOW = "activation_overflow"
    TPU_METRICS_DEBUG = "tpu_metrics_debug"
