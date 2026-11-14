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
"""Loading a model an optimum library exported, which writes no export metadata: optimum-intel's OpenVINO IRs
(`intel`), optimum-onnx's ONNX graphs (`onnx`) and optimum-executorch's programs (`executorch`). The metadata is rebuilt
from the graphs and the model's config, with a warning saying so; what a graph takes that `generate` doesn't give it is
supplied by each library's generation mixin, not by the runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

from ...utils import logging
from ...utils.hub import DownloadKwargs
from . import executorch, intel, onnx
from .common import _find, _OptimumGenerationMixin


if TYPE_CHECKING:
    from ..base import ExportedModel


logger = logging.get_logger(__name__)


def _optimum_model_class(model_class: type, mixin: type | None) -> type:
    """`model_class` as an optimum export runs: for one `generate` drives, with its library's generation mixin (if any)
    and `_OptimumGenerationMixin`."""
    from ..generation import ExportedModelForCausalLM

    if not issubclass(model_class, ExportedModelForCausalLM):
        return model_class
    mixins = (mixin, _OptimumGenerationMixin) if mixin is not None else (_OptimumGenerationMixin,)
    return type(model_class.__name__, (*mixins, model_class), {"__module__": __name__})


# Per library: the files that tell its exports apart, its format, its loader, its generation mixin and the exporter to
# re-export with.
_LIBRARIES = {
    "optimum-intel": (
        (intel.OPENVINO_MODEL_FILE, intel.OPENVINO_SEQ2SEQ_FILES["decoder_model"], intel.OPENVINO_LANGUAGE_MODEL_FILE),
        "openvino",
        intel._load_openvino,
        intel._OpenVINOGenerationMixin,
        "OpenVINOExporter",
    ),
    "optimum-onnx": (
        onnx.ONNX_FILES + onnx.ONNX_ENCODER_FILES + onnx.ONNX_SPLIT_FILES,
        "onnx",
        onnx._load_onnx,
        onnx._OnnxGenerationMixin,
        "OnnxExporter",
    ),
    "optimum-executorch": (
        (executorch.EXECUTORCH_MODEL_FILE,),
        "executorch",
        executorch._load_executorch,
        None,
        "ExecutorchExporter",
    ),
}


def optimum_library(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> str | None:
    """Which optimum library exported `pretrained_model_name_or_path`, if one did."""
    for library, (files, *_) in _LIBRARIES.items():
        if _find(pretrained_model_name_or_path, files, **kwargs) is not None:
            return library
    return None


def load_optimum_export(
    pretrained_model_name_or_path,
    cls: type,
    device=None,
    runner_kwargs=None,
    trust_remote_code: bool = False,
    **kwargs: Unpack[DownloadKwargs],
) -> ExportedModel:
    """An optimum export as an [`ExportedModel`] of `cls`, on `device`: its metadata rebuilt from its graphs and the
    model's config, with a warning that it is.

    Raises:
        `OSError`: if no optimum library wrote it either.
        `ValueError`: for an export whose graphs are laid out differently from this runtime's (see each library's
            module).
    """
    from ...generation import GenerationConfig
    from ...models.auto import AutoConfig
    from ..base import EXPORT_MANIFEST_FILE, _exported_model_class

    library = optimum_library(pretrained_model_name_or_path, **kwargs)
    if library is None:
        raise OSError(
            f"{pretrained_model_name_or_path} has no {EXPORT_MANIFEST_FILE}: it isn't an export saved with "
            "`ExportedModel.save_pretrained`, nor one an optimum library wrote."
        )
    _, export_format, load, mixin, exporter = _LIBRARIES[library]
    logger.warning(
        f"{pretrained_model_name_or_path} is an {library} export, which writes no export metadata: it is rebuilt from "
        "its graphs, so calls take the graphs' inputs, an encoder's outputs come back as a dict rather than the model's "
        f"output class, and it can't be saved again. Re-export the model with `{exporter}` (`export`, or "
        "`export_for_generation` for a generative model) and `save_pretrained` it for the full runtime."
    )
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
    try:
        generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except OSError:
        generation_config = None
    components, generation = load(pretrained_model_name_or_path, config, **kwargs)
    model_class = _optimum_model_class(_exported_model_class(components, generation=generation), mixin)
    if not issubclass(model_class, cls):
        raise TypeError(
            f"{pretrained_model_name_or_path} runs as a {model_class.__name__}, not a {cls.__name__}: load it with "
            f"`{model_class.__name__}.from_pretrained` or `ExportedModel.from_pretrained`."
        )
    model = model_class(
        components, export_format, config=config, generation_config=generation_config, generation=generation
    )
    return model.to(device, **(runner_kwargs or {}))
