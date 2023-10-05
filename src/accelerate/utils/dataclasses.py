# Copyright 2022 The HuggingFace Team. All rights reserved.
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
General namespace and dataclass related classes
"""

import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .versions import compare_versions


class KwargsHandler:
    """
    Internal mixin that implements a `to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        # import clear_environment here to avoid circular import problem
        from .other import clear_environment

        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class AutocastKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how `torch.autocast` behaves. Please refer to the
    documentation of this [context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) for more
    information on each argument.

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import AutocastKwargs

    kwargs = AutocastKwargs(cache_enabled=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    enabled: bool = True
    cache_enabled: bool = None


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how your model is wrapped in a
    `torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
    [wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
    information on each argument.

    <Tip warning={true}>

    `gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.

    `static_graph` is only available in PyTorch 1.11.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `GradScaler` is only available in PyTorch 1.5.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs

    kwargs = GradScalerKwargs(backoff_filter=0.25)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True


@dataclass
class InitProcessGroupKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the distributed processes. Please refer
    to the documentation of this
    [method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
    information on each argument.

    ```python
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    backend: Optional[str] = "nccl"
    init_method: Optional[str] = None
    timeout: timedelta = timedelta(seconds=1800)


@dataclass
class FP8RecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision
    training. Please refer to the documentation of this
    [class](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html#transformer_engine.common.recipe.DelayedScaling)
    for more information on each argument.

    ```python
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs

    kwargs = FP8RecipeKwargs(fp8_format="HYBRID")
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
    ```
    """

    margin: int = 0
    interval: int = 1
    fp8_format: str = "E4M3"
    amax_history_len: int = 1
    amax_compute_algo: str = "most_recent"
    override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)

    def __post_init__(self):
        self.fp8_format = self.fp8_format.upper()
        if self.fp8_format not in ["E4M3", "HYBRID"]:
            raise ValueError("`fp8_format` must be 'E4M3' or 'HYBRID'.")
        if self.amax_compute_algo not in ["max", "most_recent"]:
            raise ValueError("`amax_compute_algo` must be 'max' or 'most_recent'")


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **MULTI_NPU** -- Distributed on multiple NPUs.
        - **MULTI_XPU** -- Distributed on multiple XPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    MULTI_NPU = "MULTI_NPU"
    MULTI_XPU = "MULTI_XPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    TPU = "TPU"
    MEGATRON_LM = "MEGATRON_LM"


class SageMakerDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **DATA_PARALLEL** -- using sagemaker distributed data parallelism.
        - **MODEL_PARALLEL** -- using sagemaker distributed model parallelism.
    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    DATA_PARALLEL = "DATA_PARALLEL"
    MODEL_PARALLEL = "MODEL_PARALLEL"


class ComputeEnvironment(str, enum.Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """

    # Subclassing str as well as Enum allows the `ComputeEnvironment` to be JSON-serializable out of the box.
    LOCAL_MACHINE = "LOCAL_MACHINE"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"


class DynamoBackend(str, enum.Enum):
    """
    Represents a dynamo backend (see https://github.com/pytorch/torchdynamo).

    Values:

        - **NO** -- Do not use torch dynamo.
        - **EAGER** -- Uses PyTorch to run the extracted GraphModule. This is quite useful in debugging TorchDynamo
          issues.
        - **AOT_EAGER** -- Uses AotAutograd with no compiler, i.e, just using PyTorch eager for the AotAutograd's
          extracted forward and backward graphs. This is useful for debugging, and unlikely to give speedups.
        - **INDUCTOR** -- Uses TorchInductor backend with AotAutograd and cudagraphs by leveraging codegened Triton
          kernels. [Read
          more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
        - **NVFUSER** -- nvFuser with TorchScript. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_NVFUSER** -- nvFuser with AotAutograd. [Read
          more](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
        - **AOT_CUDAGRAPHS** -- cudagraphs with AotAutograd. [Read
          more](https://github.com/pytorch/torchdynamo/pull/757)
        - **OFI** -- Uses Torchscript optimize_for_inference. Inference only. [Read
          more](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
        - **FX2TRT** -- Uses Nvidia TensorRT for inference optimizations. Inference only. [Read
          more](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
        - **ONNXRT** -- Uses ONNXRT for inference on CPU/GPU. Inference only. [Read more](https://onnxruntime.ai/)
        - **IPEX** -- Uses IPEX for inference on CPU. Inference only. [Read
          more](https://github.com/intel/intel-extension-for-pytorch).

    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    EAGER = "EAGER"
    AOT_EAGER = "AOT_EAGER"
    INDUCTOR = "INDUCTOR"
    NVFUSER = "NVFUSER"
    AOT_NVFUSER = "AOT_NVFUSER"
    AOT_CUDAGRAPHS = "AOT_CUDAGRAPHS"
    OFI = "OFI"
    FX2TRT = "FX2TRT"
    ONNXRT = "ONNXRT"
    IPEX = "IPEX"


class EnumWithContains(enum.EnumMeta):
    "A metaclass that adds the ability to check if `self` contains an item with the `in` operator"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    "An enum class that can get the value of an item with `str(Enum.key)`"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        "Method to list all the possible items in `cls`"
        return list(map(str, cls))


class LoggerType(BaseEnum):
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
    """

    ALL = "all"
    AIM = "aim"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"
    MLFLOW = "mlflow"
    CLEARML = "clearml"


class PrecisionType(BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """

    NO = "no"
    FP8 = "fp8"
    FP16 = "fp16"
    BF16 = "bf16"


class RNGType(BaseEnum):
    TORCH = "torch"
    CUDA = "cuda"
    NPU = "npu"
    XLA = "xla"
    XPU = "xpu"
    GENERATOR = "generator"


class CustomDtype(enum.Enum):
    r"""
    An enum that contains multiple custom dtypes that can be used for `infer_auto_device_map`.
    """
    FP8 = "fp8"
    INT4 = "int4"


# data classes


@dataclass
class TensorInformation:
    shape: torch.Size
    dtype: torch.dtype


@dataclass
class ProjectConfiguration:
    """
    Configuration for the Accelerator object based on inner-project needs.
    """

    project_dir: str = field(default=None, metadata={"help": "A path to a directory for storing data."})
    logging_dir: str = field(
        default=None,
        metadata={
            "help": "A path to a directory for storing logs of locally-compatible loggers. If None, defaults to `project_dir`."
        },
    )
    automatic_checkpoint_naming: bool = field(
        default=False,
        metadata={"help": "Whether saved states should be automatically iteratively named."},
    )

    total_limit: int = field(
        default=None,
        metadata={"help": "The maximum number of total saved states to keep."},
    )

    iteration: int = field(
        default=0,
        metadata={"help": "The current save iteration."},
    )

    def set_directories(self, project_dir: str = None):
        "Sets `self.project_dir` and `self.logging_dir` to the appropriate values."
        self.project_dir = project_dir
        if self.logging_dir is None:
            self.logging_dir = project_dir

    def __post_init__(self):
        self.set_directories(self.project_dir)


@dataclass
class GradientAccumulationPlugin(KwargsHandler):
    """
    A plugin to configure gradient accumulation behavior.
    """

    num_steps: int = field(default=None, metadata={"help": "The number of steps to accumulate gradients for."})
    adjust_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be `True` if the used scheduler was not adjusted for gradient accumulation."
        },
    )
    sync_with_dataloader: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize setting the gradients when at the end of the dataloader. Should only be set to `False` if you know what you're doing."
        },
    )


@dataclass
class TorchDynamoPlugin(KwargsHandler):
    """
    This plugin is used to compile a model with PyTorch 2.0
    """

    backend: DynamoBackend = field(
        default=None,
        metadata={"help": f"Possible options are {[b.value.lower() for b in DynamoBackend]}"},
    )
    mode: str = field(
        default=None, metadata={"help": "Possible options are 'default', 'reduce-overhead' or 'max-autotune'"}
    )
    fullgraph: bool = field(default=None, metadata={"help": "Whether it is ok to break model into several subgraphs"})
    dynamic: bool = field(default=None, metadata={"help": "Whether to use dynamic shape for tracing"})
    options: Any = field(default=None, metadata={"help": "A dictionary of options to pass to the backend."})
    disable: bool = field(default=False, metadata={"help": "Turn torch.compile() into a no-op for testing"})

    def __post_init__(self):
        prefix = "ACCELERATE_DYNAMO_"
        if self.backend is None:
            self.backend = os.environ.get(prefix + "BACKEND", "no")
        self.backend = DynamoBackend(self.backend.upper())
        if self.mode is None:
            self.mode = os.environ.get(prefix + "MODE", "default")
        if self.fullgraph is None:
            self.fullgraph = str_to_bool(os.environ.get(prefix + "USE_FULLGRAPH", "False")) == 1
        if self.dynamic is None:
            self.dynamic = str_to_bool(os.environ.get(prefix + "USE_DYNAMIC", "False")) == 1

    def to_dict(self):
        dynamo_config = copy.deepcopy(self.__dict__)
        dynamo_config["backend"] = dynamo_config["backend"].value.lower()
        return dynamo_config


@dataclass
class DeepSpeedPlugin:
    """
    This plugin is used to integrate DeepSpeed.
    """

    hf_ds_config: Any = field(
        default=None,
        metadata={
            "help": "path to DeepSpeed config file or dict or an object of class `accelerate.utils.deepspeed.HfDeepSpeedConfig`."
        },
    )
    gradient_accumulation_steps: int = field(
        default=None,
        metadata={
            "help": "Number of steps to accumulate gradients before updating optimizer states. If not set, will use the value from the `Accelerator` directly."
        },
    )
    gradient_clipping: float = field(default=None, metadata={"help": "Enable gradient clipping with value"})
    zero_stage: int = field(
        default=None,
        metadata={"help": "Possible options are 0,1,2,3; Default will be taken from environment variable"},
    )
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the train_batch_size"},
    )
    offload_optimizer_device: bool = field(
        default=None,
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3."},
    )
    offload_param_device: bool = field(
        default=None,
        metadata={"help": "Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3."},
    )
    offload_optimizer_nvme_path: str = field(
        default=None,
        metadata={"help": "Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3."},
    )
    offload_param_nvme_path: str = field(
        default=None,
        metadata={"help": "Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3."},
    )
    zero3_init_flag: bool = field(
        default=None,
        metadata={
            "help": "Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
            "Only applicable with ZeRO Stage-3."
        },
    )
    zero3_save_16bit_model: bool = field(
        default=None,
        metadata={"help": "Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3."},
    )

    def __post_init__(self):
        from .deepspeed import HfDeepSpeedConfig

        if self.gradient_accumulation_steps is None:
            gas = os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "auto")
            self.gradient_accumulation_steps = int(gas) if gas.isdigit() else gas

        if self.gradient_clipping is None:
            gradient_clipping = os.environ.get("ACCELERATE_GRADIENT_CLIPPING", "none")
            if gradient_clipping != "none":
                self.gradient_clipping = float(gradient_clipping)

        if self.zero_stage is None:
            self.zero_stage = int(os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", 2))

        if self.offload_optimizer_device is None:
            self.offload_optimizer_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE", "none")

        if self.offload_param_device is None:
            self.offload_param_device = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE", "none")

        if self.offload_optimizer_nvme_path is None:
            self.offload_optimizer_nvme_path = os.environ.get(
                "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH", "none"
            )

        if self.offload_param_nvme_path is None:
            self.offload_param_nvme_path = os.environ.get("ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH", "none")

        if self.zero3_save_16bit_model is None:
            self.zero3_save_16bit_model = (
                os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL", "false") == "true"
            )

        if self.hf_ds_config is None:
            self.hf_ds_config = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "none")
        if (
            isinstance(self.hf_ds_config, dict)
            or (isinstance(self.hf_ds_config, str) and self.hf_ds_config != "none")
            or isinstance(self.hf_ds_config, HfDeepSpeedConfig)
        ):
            if not isinstance(self.hf_ds_config, HfDeepSpeedConfig):
                self.hf_ds_config = HfDeepSpeedConfig(self.hf_ds_config)
            if "gradient_accumulation_steps" not in self.hf_ds_config.config:
                self.hf_ds_config.config["gradient_accumulation_steps"] = 1
            if "zero_optimization" not in self.hf_ds_config.config:
                raise ValueError("Please specify the ZeRO optimization config in the DeepSpeed config.")

            self._deepspeed_config_checks()
            plugin_to_config_mapping = {
                "gradient_accumulation_steps": "gradient_accumulation_steps",
                "gradient_clipping": "gradient_clipping",
                "zero_stage": "zero_optimization.stage",
                "offload_optimizer_device": "zero_optimization.offload_optimizer.device",
                "offload_param_device": "zero_optimization.offload_param.device",
                "offload_param_nvme_path": "zero_optimization.offload_param.nvme_path",
                "offload_optimizer_nvme_path": "zero_optimization.offload_optimizer.nvme_path",
                "zero3_save_16bit_model": "zero_optimization.stage3_gather_16bit_weights_on_model_save",
            }
            kwargs = {v: getattr(self, k) for k, v in plugin_to_config_mapping.items() if getattr(self, k) is not None}
            for key in kwargs.keys():
                self.fill_match(key, **kwargs, must_match=False)
            self.hf_ds_config.set_stage_and_offload()

            # filling the missing values in the class attributes from the DeepSpeed config
            # when using the DeepSpeed config file.
            for key, value in plugin_to_config_mapping.items():
                config_value = self.hf_ds_config.get_value(value)
                if config_value is not None and config_value != "auto":
                    setattr(self, key, config_value)
        else:
            config = {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "zero_optimization": {
                    "stage": self.zero_stage,
                    "offload_optimizer": {
                        "device": self.offload_optimizer_device,
                        "nvme_path": self.offload_optimizer_nvme_path
                        if self.offload_optimizer_device == "nvme"
                        else None,
                    },
                    "offload_param": {
                        "device": self.offload_param_device,
                        "nvme_path": self.offload_param_nvme_path if self.offload_param_device == "nvme" else None,
                    },
                    "stage3_gather_16bit_weights_on_model_save": self.zero3_save_16bit_model,
                },
            }
            if self.gradient_clipping:
                config["gradient_clipping"] = self.gradient_clipping
            self.hf_ds_config = HfDeepSpeedConfig(config)

        self.deepspeed_config = self.hf_ds_config.config
        self.deepspeed_config["steps_per_print"] = float("inf")  # this will stop deepspeed from logging @ stdout
        if self.zero3_init_flag is None:
            self.zero3_init_flag = (
                str_to_bool(os.environ.get("ACCELERATE_DEEPSPEED_ZERO3_INIT", str(self.hf_ds_config.is_zero3()))) == 1
            )
        if self.zero3_init_flag and not self.hf_ds_config.is_zero3():
            warnings.warn("DeepSpeed Zero3 Init flag is only applicable for ZeRO Stage 3. Setting it to False.")
            self.zero3_init_flag = False

    def fill_match(self, ds_key_long, mismatches=None, must_match=True, **kwargs):
        mismatches = [] if mismatches is None else mismatches
        config, ds_key = self.hf_ds_config.find_config_node(ds_key_long)
        if config is None:
            return

        if config.get(ds_key) == "auto":
            if ds_key_long in kwargs:
                config[ds_key] = kwargs[ds_key_long]
                return
            else:
                raise ValueError(
                    f"`{ds_key_long}` not found in kwargs. "
                    f"Please specify `{ds_key_long}` without `auto`(set to correct value) in the DeepSpeed config file or "
                    "pass it in kwargs."
                )

        if not must_match:
            return

        ds_val = config.get(ds_key)
        if ds_val is not None and ds_key_long in kwargs:
            if ds_val != kwargs[ds_key_long]:
                mismatches.append(f"- ds {ds_key_long}={ds_val} vs arg {ds_key_long}={kwargs[ds_key_long]}")

    def deepspeed_config_process(self, prefix="", mismatches=None, config=None, must_match=True, **kwargs):
        """Process the DeepSpeed config with the values from the kwargs."""
        mismatches = [] if mismatches is None else mismatches
        if config is None:
            config = self.deepspeed_config
        for key, value in config.items():
            if isinstance(value, dict):
                self.deepspeed_config_process(
                    prefix=prefix + key + ".", mismatches=mismatches, config=value, must_match=must_match, **kwargs
                )
            else:
                self.fill_match(prefix + key, mismatches, must_match=must_match, **kwargs)
        if len(mismatches) > 0 and prefix == "":
            mismatches_msg = "\n".join(mismatches)
            raise ValueError(
                "Please correct the following DeepSpeed config values that mismatch kwargs "
                f" values:\n{mismatches_msg}\nThe easiest method is to set these DeepSpeed config values to 'auto'."
            )

    def set_mixed_precision(self, mixed_precision):
        ds_config = self.deepspeed_config
        kwargs = {
            "fp16.enabled": mixed_precision == "fp16",
            "bf16.enabled": mixed_precision == "bf16",
        }
        if mixed_precision == "fp16":
            if "fp16" not in ds_config:
                ds_config["fp16"] = {"enabled": True, "auto_cast": True}
        elif mixed_precision == "bf16":
            if "bf16" not in ds_config:
                ds_config["bf16"] = {"enabled": True}

        if mixed_precision != "no":
            diff_dtype = "bf16" if mixed_precision == "fp16" else "fp16"
            if str(ds_config.get(diff_dtype, {}).get("enabled", "False")).lower() == "true":
                raise ValueError(
                    f"`--mixed_precision` arg cannot be set to `{mixed_precision}` when `{diff_dtype}` is set in the DeepSpeed config file."
                )
        for dtype in ["fp16", "bf16"]:
            if dtype not in ds_config:
                ds_config[dtype] = {"enabled": False}
        self.fill_match("fp16.enabled", must_match=False, **kwargs)
        self.fill_match("bf16.enabled", must_match=False, **kwargs)

    def set_deepspeed_weakref(self):
        from .imports import is_transformers_available

        if self.zero3_init_flag:
            if not is_transformers_available():
                raise Exception(
                    "When `zero3_init_flag` is set, it requires Transformers to be installed. "
                    "Please run `pip install transformers`."
                )
            ds_config = copy.deepcopy(self.deepspeed_config)
            if "gradient_accumulation_steps" not in ds_config or ds_config["gradient_accumulation_steps"] == "auto":
                ds_config["gradient_accumulation_steps"] = 1
            if (
                "train_micro_batch_size_per_gpu" not in ds_config
                or ds_config["train_micro_batch_size_per_gpu"] == "auto"
            ):
                ds_config["train_micro_batch_size_per_gpu"] = 1
            if ds_config["train_batch_size"] == "auto":
                del ds_config["train_batch_size"]

            if compare_versions("transformers", "<", "4.33"):
                from transformers.deepspeed import HfDeepSpeedConfig
            else:
                from transformers.integrations import HfDeepSpeedConfig

            self.dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive # noqa

    def is_zero3_init_enabled(self):
        return self.zero3_init_flag

    @contextmanager
    def zero3_init_context_manager(self, enable=False):
        old = self.zero3_init_flag
        if old == enable:
            yield
        else:
            self.zero3_init_flag = enable
            self.dschf = None
            self.set_deepspeed_weakref()
            yield
            self.zero3_init_flag = old
            self.dschf = None
            self.set_deepspeed_weakref()

    def _deepspeed_config_checks(self):
        env_variable_names_to_ignore = [
            "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
            "ACCELERATE_GRADIENT_CLIPPING",
            "ACCELERATE_DEEPSPEED_ZERO_STAGE",
            "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE",
            "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE",
            "ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_NVME_PATH",
            "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_NVME_PATH",
            "ACCELERATE_DEEPSPEED_ZERO3_SAVE_16BIT_MODEL",
            "ACCELERATE_MIXED_PRECISION",
        ]
        env_variable_names_to_ignore = [
            name.replace("ACCELERATE_", "").replace("DEEPSPEED_", "").lower() for name in env_variable_names_to_ignore
        ]

        deepspeed_fields_from_accelerate_config = os.environ.get("ACCELERATE_CONFIG_DS_FIELDS", "").split(",")

        if any(name in env_variable_names_to_ignore for name in deepspeed_fields_from_accelerate_config):
            raise ValueError(
                f"When using `deepspeed_config_file`, the following accelerate config variables will be ignored: {env_variable_names_to_ignore}.\n"
                "Please specify them appropriately in the DeepSpeed config file.\n"
                "If you are using an accelerate config file, remove others config variables mentioned in the above specified list.\n"
                "The easiest method is to create a new config following the questionnaire via `accelerate config`.\n"
                "It will only ask for the necessary config variables when using `deepspeed_config_file`."
            )


@dataclass
class FullyShardedDataParallelPlugin:
    """
    This plugin is used to enable fully sharded data parallelism.
    """

    sharding_strategy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Sharding Strategy of type `torch.distributed.fsdp.fully_sharded_data_parallel.ShardingStrategy`"
        },
    )
    backward_prefetch: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Backward Prefetch of type `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`"
        },
    )
    mixed_precision_policy: "typing.Any" = field(
        default=None,
        metadata={
            "help": "A config to enable mixed precision training with FullyShardedDataParallel. "
            "The 3 flags that are set are `param_dtype`, `reduce_dtype`, `buffer_dtype`. "
            "Each flag expects `torch.dtype` as the value. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.MixedPrecision`."
        },
    )
    auto_wrap_policy: Optional[Callable] = field(
        default=None,
        metadata={"help": "A callable specifying a policy to recursively wrap layers with FSDP"},
    )
    cpu_offload: "typing.Any" = field(
        default=None,
        metadata={
            "help": "Decides Whether to offload parameters and gradients to CPU. "
            "It is of type `torch.distributed.fsdp.fully_sharded_data_parallel.CPUOffload`."
        },
    )
    ignored_modules: Optional[Iterable[torch.nn.Module]] = field(
        default=None,
        metadata={"help": "A list of modules to ignore for FSDP."},
    )
    state_dict_type: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Type of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictType`"
        },
    )
    state_dict_config: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP State Dict Config of type `torch.distributed.fsdp.fully_sharded_data_parallel.StateDictConfig`"
        },
    )
    optim_state_dict_config: "typing.Any" = field(
        default=None,
        metadata={
            "help": "FSDP Optimizer State Dict Config of type `torch.distributed.fsdp.fully_sharded_data_parallel.OptimStateDictConfig`"
        },
    )
    limit_all_gathers: bool = field(
        default=False,
        metadata={
            "help": "If False, then FSDP allows the CPU thread to schedule all-gathers "
            "without any extra synchronization. If True, then FSDP explicitly synchronizes the CPU thread to prevent "
            "too many in-flight all-gathers. This bool only affects the sharded strategies that schedule all-gathers. "
            "Enabling this can help lower the number of CUDA malloc retries."
        },
    )
    use_orig_params: bool = field(
        default=False,
        metadata={
            "help": "If True, allows non-uniform `requires_grad` during init, which means support for interspersed frozen and trainable paramteres. "
            "Useful in cases such as parameter-efficient fine-tuning. "
            "Please refer this [blog](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)"
        },
    )
    param_init_fn: Optional[Callable[[torch.nn.Module], None]] = field(
        default=None,
        metadata={
            "help": "A Callable[torch.nn.Module] -> None that specifies how modules "
            "that are currently on the meta device should be initialized onto an actual device."
        },
    )
    sync_module_states: bool = field(
        default=True,
        metadata={
            "help": "If True, each individually wrapped FSDP unit will broadcast module parameters from rank 0 "
            "to ensure they are the same across all ranks after initialization"
        },
    )
    forward_prefetch: bool = field(
        default=False,
        metadata={
            "help": "If True, then FSDP explicitly prefetches the next upcoming "
            "all-gather while executing in the forward pass. only use with Static graphs."
        },
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, activation checkpointing is a technique to reduce memory usage by clearing activations of "
            "certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time "
            "for reduced memory usage."
        },
    )

    def __post_init__(self):
        from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, CPUOffload, ShardingStrategy

        prefix = "FSDP_"
        if self.sharding_strategy is None:
            self.sharding_strategy = ShardingStrategy(int(os.environ.get(prefix + "SHARDING_STRATEGY", 1)))

        if self.cpu_offload is None:
            if str_to_bool(os.environ.get(prefix + "OFFLOAD_PARAMS", "False")) == 1:
                self.cpu_offload = CPUOffload(offload_params=True)
            else:
                self.cpu_offload = CPUOffload(offload_params=False)

        if self.backward_prefetch is None:
            prefetch_policy = os.environ.get(prefix + "BACKWARD_PREFETCH", "NO_PREFETCH")
            if prefetch_policy != FSDP_BACKWARD_PREFETCH[-1]:
                self.backward_prefetch = BackwardPrefetch(FSDP_BACKWARD_PREFETCH.index(prefetch_policy) + 1)

        if self.state_dict_type is None:
            state_dict_type_policy = os.environ.get(prefix + "STATE_DICT_TYPE", "FULL_STATE_DICT")
            self.set_state_dict_type(state_dict_type_policy)
        self.use_orig_params = str_to_bool(os.environ.get(prefix + "USE_ORIG_PARAMS", "False")) == 1
        self.sync_module_states = str_to_bool(os.environ.get(prefix + "SYNC_MODULE_STATES", "True")) == 1
        self.forward_prefetch = str_to_bool(os.environ.get(prefix + "FORWARD_PREFETCH", "False")) == 1
        self.activation_checkpointing = str_to_bool(os.environ.get(prefix + "ACTIVATION_CHECKPOINTING", "False")) == 1

        if self.sync_module_states:
            self.param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

    @staticmethod
    def get_module_class_from_name(module, name):
        """
        Gets a class from a module by its name.

        Args:
            module (`torch.nn.Module`): The module to get the class from.
            name (`str`): The name of the class.
        """
        modules_children = list(module.children())
        if module.__class__.__name__ == name:
            return module.__class__
        elif len(modules_children) == 0:
            return
        else:
            for child_module in modules_children:
                module_class = FullyShardedDataParallelPlugin.get_module_class_from_name(child_module, name)
                if module_class is not None:
                    return module_class

    def set_auto_wrap_policy(self, model):
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

        default_transformer_cls_names_to_wrap = (
            ",".join(model._no_split_modules) if getattr(model, "_no_split_modules", None) is not None else ""
        )
        if self.auto_wrap_policy is None:
            auto_wrap_policy = os.environ.get("FSDP_AUTO_WRAP_POLICY", "NO_WRAP")
            if auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[0]:
                transformer_cls_names_to_wrap = os.environ.get(
                    "FSDP_TRANSFORMER_CLS_TO_WRAP", default_transformer_cls_names_to_wrap
                ).split(",")
                transformer_cls_to_wrap = set()
                for layer_class in transformer_cls_names_to_wrap:
                    transformer_cls = FullyShardedDataParallelPlugin.get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)

                self.auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            elif auto_wrap_policy == FSDP_AUTO_WRAP_POLICY[1]:
                min_num_params = int(os.environ.get("FSDP_MIN_NUM_PARAMS", 0))
                if min_num_params > 0:
                    self.auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=min_num_params
                    )

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == "fp16":
            dtype = torch.float16
        elif mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown mixed precision value: {mixed_precision}")
        from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision

        if self.mixed_precision_policy is None:
            self.mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def set_state_dict_type(self, state_dict_type_policy):
        from torch.distributed.fsdp.fully_sharded_data_parallel import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        self.state_dict_type = StateDictType(FSDP_STATE_DICT_TYPE.index(state_dict_type_policy) + 1)

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            if self.state_dict_config is None:
                self.state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            if self.optim_state_dict_config is None:
                self.optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


@dataclass
class MegatronLMPlugin:
    """
    Plugin for Megatron-LM to enable tensor, pipeline, sequence and data parallelism. Also to enable selective
    activation recomputation and optimized fused kernels.
    """

    tp_degree: int = field(default=None, metadata={"help": "tensor parallelism degree."})
    pp_degree: int = field(default=None, metadata={"help": "pipeline parallelism degree."})
    num_micro_batches: int = field(default=None, metadata={"help": "number of micro-batches."})
    gradient_clipping: float = field(
        default=None, metadata={"help": "gradient clipping value based on global L2 Norm (0 to disable)"}
    )
    sequence_parallelism: bool = field(
        default=None,
        metadata={"help": "enable sequence parallelism"},
    )
    recompute_activation: bool = field(
        default=None,
        metadata={"help": "enable selective activation recomputation"},
    )
    use_distributed_optimizer: bool = field(
        default=None,
        metadata={"help": "enable distributed optimizer"},
    )
    pipeline_model_parallel_split_rank: int = field(
        default=None, metadata={"help": "Rank where encoder and decoder should be split."}
    )
    num_layers_per_virtual_pipeline_stage: int = field(
        default=None, metadata={"help": "Number of layers per virtual pipeline stage."}
    )
    is_train_batch_min: str = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the micro_batch_size"},
    )
    train_iters: int = field(
        default=None,
        metadata={
            "help": "Total number of iterations to train over all training runs. "
            "Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`"
        },
    )
    train_samples: int = field(
        default=None,
        metadata={
            "help": "Total number of samples to train over all training runs. "
            "Note that either train-iters or train-samples should be provided when using `MegatronLMDummyScheduler`"
        },
    )
    weight_decay_incr_style: str = field(
        default="constant",
        metadata={"help": 'Weight decay increment function. choices=["constant", "linear", "cosine"]. '},
    )
    start_weight_decay: float = field(
        default=None,
        metadata={"help": "Initial weight decay coefficient for L2 regularization."},
    )
    end_weight_decay: float = field(
        default=None,
        metadata={"help": "End of run weight decay coefficient for L2 regularization."},
    )
    lr_decay_style: str = field(
        default="linear",
        metadata={"help": "Learning rate decay function. choices=['constant', 'linear', 'cosine']."},
    )
    lr_decay_iters: int = field(
        default=None,
        metadata={"help": "Number of iterations for learning rate decay. If None defaults to `train_iters`."},
    )
    lr_decay_samples: int = field(
        default=None,
        metadata={"help": "Number of samples for learning rate decay. If None defaults to `train_samples`."},
    )
    lr_warmup_iters: int = field(
        default=None,
        metadata={"help": "number of iterations to linearly warmup learning rate over."},
    )
    lr_warmup_samples: int = field(
        default=None,
        metadata={"help": "number of samples to linearly warmup learning rate over."},
    )
    lr_warmup_fraction: float = field(
        default=None,
        metadata={"help": "fraction of lr-warmup-(iters/samples) to linearly warmup learning rate over."},
    )
    min_lr: float = field(
        default=0,
        metadata={"help": "Minumum value for learning rate. The scheduler clip values below this threshold."},
    )
    consumed_samples: List[int] = field(
        default=None,
        metadata={
            "help": "Number of samples consumed in the same order as the dataloaders to `accelerator.prepare` call."
        },
    )
    no_wd_decay_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to disable weight decay."})
    scale_lr_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to scale learning rate."})
    lr_mult: float = field(default=1.0, metadata={"help": "Learning rate multiplier."})
    megatron_dataset_flag: bool = field(
        default=False,
        metadata={"help": "Whether the format of dataset follows Megatron-LM Indexed/Cached/MemoryMapped format."},
    )
    seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process."},
    )
    encoder_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process for the encoder."},
    )
    decoder_seq_length: int = field(
        default=None,
        metadata={"help": "Maximum sequence length to process for the decoder."},
    )
    tensorboard_dir: str = field(
        default=None,
        metadata={"help": "Path to save tensorboard logs."},
    )
    set_all_logging_options: bool = field(
        default=False,
        metadata={"help": "Whether to set all logging options."},
    )
    eval_iters: int = field(
        default=100, metadata={"help": "Number of iterations to run for evaluation validation/test for."}
    )
    eval_interval: int = field(
        default=1000, metadata={"help": "Interval between running evaluation on validation set."}
    )
    return_logits: bool = field(
        default=False,
        metadata={"help": "Whether to return logits from the model."},
    )

    # custom train step args
    custom_train_step_class: Optional[Any] = field(
        default=None,
        metadata={"help": "Custom train step class."},
    )
    custom_train_step_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Custom train step kwargs."},
    )

    # custom model args
    custom_model_provider_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom model provider function."},
    )
    custom_prepare_model_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom prepare model function."},
    )

    # remaining args such as enabling Alibi/ROPE positional embeddings,
    # wandb logging, Multi-Query Attention, etc.
    other_megatron_args: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Other Megatron-LM arguments. Please refer Megatron-LM"},
    )

    def __post_init__(self):
        prefix = "MEGATRON_LM_"
        if self.tp_degree is None:
            self.tp_degree = int(os.environ.get(prefix + "TP_DEGREE", 1))
        if self.pp_degree is None:
            self.pp_degree = int(os.environ.get(prefix + "PP_DEGREE", 1))
        if self.num_micro_batches is None:
            self.num_micro_batches = int(os.environ.get(prefix + "NUM_MICRO_BATCHES", 1))
        if self.gradient_clipping is None:
            self.gradient_clipping = float(os.environ.get(prefix + "GRADIENT_CLIPPING", 1.0))
        if self.recompute_activation is None:
            self.recompute_activation = str_to_bool(os.environ.get(prefix + "RECOMPUTE_ACTIVATION", "False")) == 1
        if self.use_distributed_optimizer is None:
            self.use_distributed_optimizer = (
                str_to_bool(os.environ.get(prefix + "USE_DISTRIBUTED_OPTIMIZER", "False")) == 1
            )
        if self.sequence_parallelism is None:
            self.sequence_parallelism = str_to_bool(os.environ.get(prefix + "SEQUENCE_PARALLELISM", "False")) == 1

        if self.pp_degree > 1 or self.use_distributed_optimizer:
            self.DDP_impl = "local"
        else:
            self.DDP_impl = "torch"

        if self.consumed_samples is not None:
            if len(self.consumed_samples) == 1:
                self.consumed_samples.extend([0, 0])
            elif len(self.consumed_samples) == 2:
                self.consumed_samples.append(0)

        self.megatron_lm_default_args = {
            "tensor_model_parallel_size": self.tp_degree,
            "pipeline_model_parallel_size": self.pp_degree,
            "pipeline_model_parallel_split_rank": self.pipeline_model_parallel_split_rank,
            "num_layers_per_virtual_pipeline_stage": self.num_layers_per_virtual_pipeline_stage,
            "DDP_impl": self.DDP_impl,
            "use_distributed_optimizer": self.use_distributed_optimizer,
            "sequence_parallel": self.sequence_parallelism,
            "clip_grad": self.gradient_clipping,
            "num_micro_batches": self.num_micro_batches,
            "consumed_samples": self.consumed_samples,
            "no_wd_decay_cond": self.no_wd_decay_cond,
            "scale_lr_cond": self.scale_lr_cond,
            "lr_mult": self.lr_mult,
            "megatron_dataset_flag": self.megatron_dataset_flag,
            "eval_iters": self.eval_iters,
            "eval_interval": self.eval_interval,
        }
        if self.recompute_activation:
            self.megatron_lm_default_args["recompute_granularity"] = "selective"
        if self.tensorboard_dir is not None:
            self.megatron_lm_default_args["tensorboard_dir"] = self.tensorboard_dir
            if self.set_all_logging_options:
                self.set_tensorboard_logging_options()
        if self.other_megatron_args is not None:
            self.megatron_lm_default_args.update(self.other_megatron_args)

    def set_network_size_args(self, model, batch_data=None):
        # Check if the model is either BERT, GPT or T5 else raise error
        # set 'num_layers', 'hidden_size', 'num_attention_heads', 'max_position_embeddings'
        if "megatron-bert" in model.config.model_type.lower():
            model_type_name = "bert"
            num_layers = model.config.num_hidden_layers
            hidden_size = model.config.hidden_size
            num_attention_heads = model.config.num_attention_heads
            max_position_embeddings = model.config.max_position_embeddings
            num_labels = model.config.num_labels
            orig_vocab_size = model.config.vocab_size
            if "maskedlm" in model.__class__.__name__.lower():
                pretraining_flag = True
            if self.seq_length is not None:
                if self.encoder_seq_length is not None:
                    warnings.warn("Both `seq_length` and `encoder_seq_length` are set. Using `encoder_seq_length`.")
                self.seq_length = self.encoder_seq_length
            elif self.encoder_seq_length is not None:
                self.seq_length = self.encoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data["input_ids"].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args["seq_length"] = self.seq_length
        elif "gpt2" in model.config.model_type.lower():
            model_type_name = "gpt"
            num_layers = model.config.n_layer
            hidden_size = model.config.n_embd
            num_attention_heads = model.config.n_head
            max_position_embeddings = model.config.n_positions
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.seq_length is not None:
                if self.decoder_seq_length is not None:
                    warnings.warn("Both `seq_length` and `decoder_seq_length` are set. Using `decoder_seq_length`.")
                self.seq_length = self.decoder_seq_length
            elif self.decoder_seq_length is not None:
                self.seq_length = self.decoder_seq_length
            elif batch_data is not None:
                self.seq_length = batch_data["input_ids"].shape[1]
            else:
                self.seq_length = max_position_embeddings
            self.megatron_lm_default_args["seq_length"] = self.seq_length
            self.megatron_lm_default_args["return_logits"] = self.return_logits
            self.megatron_lm_default_args["tokenizer_type"] = "GPT2BPETokenizer"
        elif "t5" in model.config.model_type.lower():
            model_type_name = "t5"
            num_layers = model.config.num_layers
            hidden_size = model.config.d_model
            num_attention_heads = model.config.num_heads
            max_position_embeddings = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
            orig_vocab_size = model.config.vocab_size
            pretraining_flag = True
            if self.encoder_seq_length is None:
                if batch_data is not None:
                    self.encoder_seq_length = batch_data["input_ids"].shape[1]
                else:
                    self.encoder_seq_length = max_position_embeddings
            if self.decoder_seq_length is None:
                if batch_data is not None:
                    self.decoder_seq_length = batch_data["labels"].shape[1]
                else:
                    self.decoder_seq_length = max_position_embeddings

            self.megatron_lm_default_args["encoder_seq_length"] = self.encoder_seq_length
            self.megatron_lm_default_args["decoder_seq_length"] = self.decoder_seq_length
        else:
            raise ValueError(
                "🤗 Accelerate Megatron-LM integration supports only BERT, GPT and T5 model. "
                "Please check the model you are using is one of those."
            )

        self.megatron_lm_default_args["model_type_name"] = model_type_name
        self.megatron_lm_default_args["num_layers"] = num_layers
        self.megatron_lm_default_args["hidden_size"] = hidden_size
        self.megatron_lm_default_args["num_attention_heads"] = num_attention_heads
        self.megatron_lm_default_args["max_position_embeddings"] = max_position_embeddings
        self.megatron_lm_default_args["pretraining_flag"] = pretraining_flag
        self.megatron_lm_default_args["orig_vocab_size"] = orig_vocab_size
        self.megatron_lm_default_args["model_return_dict"] = model.config.return_dict
        if model_type_name == "bert":
            self.megatron_lm_default_args["num_labels"] = num_labels

    def set_mixed_precision(self, mixed_precision):
        if mixed_precision == "fp16":
            self.megatron_lm_default_args["fp16"] = True
        elif mixed_precision == "bf16":
            self.megatron_lm_default_args["bf16"] = True
            self.DDP_impl = "local"
            self.megatron_lm_default_args["DDP_impl"] = self.DDP_impl

    def set_training_args(self, micro_batch_size, dp_degree):
        self.data_parallel_size = dp_degree
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = dp_degree * micro_batch_size * self.num_micro_batches
        self.megatron_lm_default_args["data_parallel_size"] = self.data_parallel_size
        self.megatron_lm_default_args["micro_batch_size"] = self.micro_batch_size
        self.megatron_lm_default_args["global_batch_size"] = self.global_batch_size

    def set_optimizer_type(self, optimizer):
        optimizer_name = optimizer.__class__.__name__.lower()
        if "adam" in optimizer_name:
            self.megatron_lm_default_args["optimizer"] = "adam"
            self.megatron_lm_default_args["adam_beta1"] = optimizer.defaults["betas"][0]
            self.megatron_lm_default_args["adam_beta2"] = optimizer.defaults["betas"][1]
            self.megatron_lm_default_args["adam_eps"] = optimizer.defaults["eps"]
        elif "sgd" in optimizer_name:
            self.megatron_lm_default_args["optimizer"] = "sgd"
            self.megatron_lm_default_args["sgd_momentum"] = optimizer.defaults["momentum"]
        else:
            raise ValueError(f"Optimizer {optimizer_name} is not supported by Megatron-LM")

        self.megatron_lm_default_args["lr"] = optimizer.defaults["lr"]
        self.megatron_lm_default_args["weight_decay"] = optimizer.defaults["weight_decay"]

    def set_scheduler_args(self, scheduler):
        if self.train_iters is None:
            self.train_iters = scheduler.total_num_steps // self.megatron_lm_default_args["data_parallel_size"]
            if self.train_samples is not None:
                self.train_samples = None
                warnings.warn(
                    "Ignoring `train_samples` as `train_iters` based on scheduler is being used for training."
                )
        if self.lr_warmup_iters is None:
            self.lr_warmup_iters = scheduler.warmup_num_steps // self.megatron_lm_default_args["data_parallel_size"]
            if self.lr_warmup_samples is not None:
                warnings.warn(
                    "Ignoring `lr_warmup_samples` as `lr_warmup_iters` based on scheduler is being used for training."
                )
            self.lr_warmup_samples = 0

        self.megatron_lm_default_args["train_iters"] = self.train_iters
        self.megatron_lm_default_args["lr_warmup_iters"] = self.lr_warmup_iters
        self.megatron_lm_default_args["train_samples"] = self.train_samples
        self.megatron_lm_default_args["lr_warmup_samples"] = self.lr_warmup_samples
        self.megatron_lm_default_args["lr_decay_iters"] = self.lr_decay_iters
        self.megatron_lm_default_args["lr_decay_samples"] = self.lr_decay_samples
        self.megatron_lm_default_args["lr_warmup_fraction"] = self.lr_warmup_fraction
        self.megatron_lm_default_args["lr_decay_style"] = self.lr_decay_style
        self.megatron_lm_default_args["weight_decay_incr_style"] = self.weight_decay_incr_style
        self.megatron_lm_default_args["start_weight_decay"] = self.start_weight_decay
        self.megatron_lm_default_args["end_weight_decay"] = self.end_weight_decay
        self.megatron_lm_default_args["min_lr"] = self.min_lr

    def set_tensorboard_logging_options(self):
        from megatron.arguments import _add_logging_args

        parser = argparse.ArgumentParser()
        parser = _add_logging_args(parser)
        logging_args = parser.parse_known_args()
        self.dataset_args = vars(logging_args[0])
        for key, value in self.dataset_args.items():
            if key.startswith("log_"):
                self.megatron_lm_default_args[key] = True
            elif key.startswith("no_log_"):
                self.megatron_lm_default_args[key.replace("no_", "")] = True


@dataclass
class BnbQuantizationConfig:
    """
    A plugin to enable BitsAndBytes 4bit and 8bit quantization
    """

    load_in_8bit: bool = field(default=False, metadata={"help": "enable 8bit quantization."})

    llm_int8_threshold: float = field(
        default=6.0, metadata={"help": "value of the outliner threshold. only relevant when load_in_8bit=True"}
    )

    load_in_4bit: bool = field(default=False, metadata={"help": "enable 4bit quantization."})

    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "set the quantization data type in the `bnb.nn.Linear4Bit` layers. Options are {'fp4','np4'}."
        },
    )

    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={
            "help": "enable nested quantization where the quantization constants from the first quantization are quantized again."
        },
    )

    bnb_4bit_compute_dtype: bool = field(
        default="fp16",
        metadata={
            "help": "This sets the computational type which might be different than the input time. For example, inputs might be "
            "fp32, but computation can be set to bf16 for speedups. Options are {'fp32','fp16','bf16'}."
        },
    )

    torch_dtype: torch.dtype = field(
        default=None,
        metadata={
            "help": "this sets the dtype of the remaining non quantized layers. `bitsandbytes` library suggests to set the value"
            "to `torch.float16` for 8 bit model and use the same dtype as the compute dtype for 4 bit model "
        },
    )

    skip_modules: List[str] = field(
        default=None,
        metadata={
            "help": "an explicit list of the modules that we don't quantize. The dtype of these modules will be `torch_dtype`."
        },
    )

    keep_in_fp32_modules: List[str] = field(
        default=None,
        metadata={"help": "an explicit list of the modules that we don't quantize. We keep them in `torch.float32`."},
    )

    def __post_init__(self):
        """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_8bit, bool):
            raise ValueError("load_in_8bit must be a boolean")

        if not isinstance(self.load_in_4bit, bool):
            raise ValueError("load_in_4bit must be a boolean")

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8 can't be both True")

        if not self.load_in_4bit and not self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8 can't be both False")

        if not isinstance(self.llm_int8_threshold, (int, float)):
            raise ValueError("llm_int8_threshold must be a float or an int")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError("bnb_4bit_quant_type must be a string")
        elif self.bnb_4bit_quant_type not in ["fp4", "nf4"]:
            raise ValueError(f"bnb_4bit_quant_type must be in ['fp4','nf4'] but found {self.bnb_4bit_quant_type}")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError("bnb_4bit_use_double_quant must be a boolean")

        if isinstance(self.bnb_4bit_compute_dtype, str):
            if self.bnb_4bit_compute_dtype == "fp32":
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.bnb_4bit_compute_dtype == "fp16":
                self.bnb_4bit_compute_dtype = torch.float16
            elif self.bnb_4bit_compute_dtype == "bf16":
                self.bnb_4bit_compute_dtype = torch.bfloat16
            else:
                raise ValueError(
                    f"bnb_4bit_compute_dtype must be in ['fp32','fp16','bf16'] but found {self.bnb_4bit_compute_dtype}"
                )
        elif not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if self.skip_modules is not None and not isinstance(self.skip_modules, list):
            raise ValueError("skip_modules must be a list of strings")

        if self.keep_in_fp32_modules is not None and not isinstance(self.keep_in_fp32_modules, list):
            raise ValueError("keep_in_fp_32_modules must be a list of strings")

        if self.load_in_4bit:
            self.target_dtype = CustomDtype.INT4

        if self.load_in_8bit:
            self.target_dtype = torch.int8

        if self.load_in_4bit and self.llm_int8_threshold != 6.0:
            warnings.warn("llm_int8_threshold can only be used for model loaded in 8bit")

        if isinstance(self.torch_dtype, str):
            if self.torch_dtype == "fp32":
                self.torch_dtype = torch.float32
            elif self.torch_dtype == "fp16":
                self.torch_dtype = torch.float16
            elif self.torch_dtype == "bf16":
                self.torch_dtype = torch.bfloat16
            else:
                raise ValueError(f"torch_dtype must be in ['fp32','fp16','bf16'] but found {self.torch_dtype}")
        if self.load_in_8bit and self.torch_dtype is None:
            self.torch_dtype = torch.float16

        if self.load_in_4bit and self.torch_dtype is None:
            self.torch_dtype = self.bnb_4bit_compute_dtype

        if not isinstance(self.torch_dtype, torch.dtype):
            raise ValueError("torch_dtype must be a torch.dtype")
