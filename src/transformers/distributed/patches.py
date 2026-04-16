# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Monkey-patches for DTensor-aware operations.

These patches are applied at model-loading time (during ``from_pretrained``)
so that modeling files stay free of DTensor-specific code.
"""

from __future__ import annotations

import inspect
import sys
from functools import wraps

from torch.distributed.tensor import DTensor, Replicate


def _make_dtensor_rotary_wrapper(original_fn):
    """Return a wrapper that promotes cos/sin to replicated DTensors.

    Models use two ``apply_rotary_pos_emb`` signatures:
      - ``(q, k, cos, sin, ...)``  — most models
      - ``(x, cos, sin, ...)``     — gemma3n, gemma4, glm_moe_dsa

    We detect which one at patch time via parameter count and create
    the matching wrapper.
    """
    params = inspect.signature(original_fn).parameters
    n_required = sum(1 for p in params.values() if p.default is inspect.Parameter.empty)

    if n_required >= 4:

        @wraps(original_fn)
        def _wrapper(q, k, cos, sin, *args, **kwargs):
            if isinstance(q, DTensor) and not isinstance(cos, DTensor):
                replicate = (Replicate(),) * q.device_mesh.ndim
                cos = DTensor.from_local(cos, q.device_mesh, replicate, run_check=False)
                sin = DTensor.from_local(sin, q.device_mesh, replicate, run_check=False)
            return original_fn(q, k, cos, sin, *args, **kwargs)
    else:

        @wraps(original_fn)
        def _wrapper(x, cos, sin, *args, **kwargs):
            if isinstance(x, DTensor) and not isinstance(cos, DTensor):
                replicate = (Replicate(),) * x.device_mesh.ndim
                cos = DTensor.from_local(cos, x.device_mesh, replicate, run_check=False)
                sin = DTensor.from_local(sin, x.device_mesh, replicate, run_check=False)
            return original_fn(x, cos, sin, *args, **kwargs)

    return _wrapper


def patch_dtensor_ops(model):
    """Monkey-patch DTensor-aware wrappers onto the model's modeling module.

    Finds the Python module where the model class is defined and wraps
    ``apply_rotary_pos_emb`` (if present) so that cos/sin tensors are
    automatically promoted to replicated DTensors when the query is a DTensor.

    Called from ``apply_tensor_parallel`` after ``parallelize_module``.
    """
    model_module = sys.modules.get(type(model).__module__)
    if model_module is None:
        return

    original_fn = getattr(model_module, "apply_rotary_pos_emb", None)
    if original_fn is not None:
        model_module.apply_rotary_pos_emb = _make_dtensor_rotary_wrapper(original_fn)
