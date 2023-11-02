# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ..utils import _LazyModule


_import_structure = {
    "bitsandbytes": [
        "get_keys_to_not_convert",
        "replace_8bit_linear",
        "replace_with_bnb_linear",
        "set_module_8bit_tensor_to_device",
        "set_module_quantized_tensor_to_device",
    ],
    "deepspeed": [
        "HfDeepSpeedConfig",
        "HfTrainerDeepSpeedConfig",
        "deepspeed_config",
        "deepspeed_init",
        "deepspeed_load_checkpoint",
        "deepspeed_optim_sched",
        "is_deepspeed_available",
        "is_deepspeed_zero3_enabled",
        "set_hf_deepspeed_config",
        "unset_hf_deepspeed_config",
    ],
    "integration_utils": [
        "INTEGRATION_TO_CALLBACK",
        "AzureMLCallback",
        "ClearMLCallback",
        "CodeCarbonCallback",
        "CometCallback",
        "DagsHubCallback",
        "FlyteCallback",
        "MLflowCallback",
        "NeptuneCallback",
        "NeptuneMissingConfiguration",
        "TensorBoardCallback",
        "WandbCallback",
        "get_available_reporting_integrations",
        "get_reporting_integration_callbacks",
        "hp_params",
        "is_azureml_available",
        "is_clearml_available",
        "is_codecarbon_available",
        "is_comet_available",
        "is_dagshub_available",
        "is_flyte_deck_standard_available",
        "is_flytekit_available",
        "is_mlflow_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
        "rewrite_logs",
        "run_hp_search_optuna",
        "run_hp_search_ray",
        "run_hp_search_sigopt",
        "run_hp_search_wandb",
    ],
    "peft": ["PeftAdapterMixin"],
}

if TYPE_CHECKING:
    from .bitsandbytes import (
        get_keys_to_not_convert,
        replace_8bit_linear,
        replace_with_bnb_linear,
        set_module_8bit_tensor_to_device,
        set_module_quantized_tensor_to_device,
    )
    from .deepspeed import (
        HfDeepSpeedConfig,
        HfTrainerDeepSpeedConfig,
        deepspeed_config,
        deepspeed_init,
        deepspeed_load_checkpoint,
        deepspeed_optim_sched,
        is_deepspeed_available,
        is_deepspeed_zero3_enabled,
        set_hf_deepspeed_config,
        unset_hf_deepspeed_config,
    )
    from .integration_utils import (
        INTEGRATION_TO_CALLBACK,
        AzureMLCallback,
        ClearMLCallback,
        CodeCarbonCallback,
        CometCallback,
        DagsHubCallback,
        FlyteCallback,
        MLflowCallback,
        NeptuneCallback,
        NeptuneMissingConfiguration,
        TensorBoardCallback,
        WandbCallback,
        get_available_reporting_integrations,
        get_reporting_integration_callbacks,
        hp_params,
        is_azureml_available,
        is_clearml_available,
        is_codecarbon_available,
        is_comet_available,
        is_dagshub_available,
        is_flyte_deck_standard_available,
        is_flytekit_available,
        is_mlflow_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
        rewrite_logs,
        run_hp_search_optuna,
        run_hp_search_ray,
        run_hp_search_sigopt,
        run_hp_search_wandb,
    )
    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
