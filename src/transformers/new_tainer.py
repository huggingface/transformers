import functools
import importlib.metadata
import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch


# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
)

# isort: on

from packaging import version
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .modeling_utils import PreTrainedModel
from .models.auto.modeling_auto import (
    MODEL_MAPPING_NAMES,
)
from .pytorch_utils import (
    is_torch_greater_or_equal_than_2_3,
)
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_pt_utils import (
    LabelSmoother,
)
from .trainer_utils import (
    EvalPrediction,
    enable_full_determinism,
    has_length,
    number_of_arguments,
    set_seed,
)
from .training_args import ParallelMode, TrainingArguments
from .utils import (
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_in_notebook,
    is_peft_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    logging,
)
from .utils.quantization_config import QuantizationMethod


# isort: off

from .modeling_utils import verify_quantization_training_support

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from .utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback


logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftModel

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        GradientAccumulationPlugin,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        pass

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if not is_peft_available():
        return False

    classes_to_check = (PeftModel,)
    if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
        from peft import PeftMixedModel

        classes_to_check += (PeftMixedModel,)

    return isinstance(model, classes_to_check)


class Trainer:
    from .trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.args = args
        # Seed must be set before instantiating the model
        enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

        self.create_accelerator_and_postprocess()

        # Force device and distributed setup init explicitly
        args._setup_devices

        # Init and verify data related items
        if tokenizer is not None and isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor)):
            self.data_collator = DataCollatorWithPadding(tokenizer)
        else:
            self.data_collator = data_collator if data_collator is not None else default_data_collator
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")
        self._signature_columns = None

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.optimizer, self.lr_scheduler = optimizers

        if model is None:
            if model_init is None:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
            if self.optimizer is None or self.lr_scheduler is None:
                raise RuntimeError(
                    "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                    "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
                )
            self.model_init = model_init
            model = self.call_model_init()

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or self.using_model_orchestration
        ):
            self.place_model_on_device = False
        # Bnb quantized models don't support `to` operation
        if (
            self.place_model_on_device
            and not getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        ):
            self.model = self._move_model_to_device(model, args.device)

        self.perform_model_sanity_checks(model)
        self.perform_optimizer_sanity_checks()

        # We keep two references so that later we can check if `self.model is self.model_wrapped`
        self.model_wrapped = self.model = model

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # Mixed precision (when not using `Accelerator`)
        if args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16 and not is_torch_greater_or_equal_than_2_3:
                    raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                elif args.bf16:
                    args.half_precision_backend = "cpu_amp"
                logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or args.bf16) and not (self.is_deepspeed_enabled or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

        # Label smoothing
        self.label_smoother = None
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)

        # Callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_hanlder = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0

        self.hp_search_backend = None
        self.label_names = (
            self.args.label_names if self.args.label_names is not None else find_labels(self.model.__class__)
        )
        self.can_return_loss = can_return_loss(self.model.__class__)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

    @property
    def using_model_orchestration(self):
        "Returns whether the underlying model is using training orchestration such as FSDP or DeepSpeed"
        return self.is_fsdp_enabled or self.is_fsdp_xla_enabled or self.is_deepspeed_enabled

    @property
    def is_fsdp_xla_enabled(self):
        return self.args.fsdp_config.get("xla", False)

    def call_model_init(self, trial=None):
        "Instantiates a model using the model init function."
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("`model_init` should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("`model_init` should not return None.")

        return model

    def perform_model_sanity_checks(self, model):
        """
        Performs a variety of sanity checks on the model:

        1. Checks if the model was initialized properly when using `Zero-3`
        2. Verifies that the model class is suitable for `Trainer`
        3. Checks if quantization training is supported if detected
        """
        if is_deepspeed_zero3_enabled() and not getattr(model, "_transformers_zero3_init_used", True):
            raise ValueError(
                "Model was not initialized with `Zero-3` despite being configured for DeepSpeed Zero-3. Please re-initialize your model via `Model.from_pretrained(...)` or `Model.from_config(...)` after creating your `TrainingArguments`!"
            )

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        verify_quantization_training_support(model)

    def perform_optimizer_sanity_checks(self):
        if is_torch_xla_available() and self.optimizer is not None:
            model_device = next(self.model.parameters()).device
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if self.using_model_orchestration and self.optimizer is not None or self.lr_scheduler is not None:
            raise RuntimeError(
                "Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

    def perform_distributed_sanity_checks(self):
        # Model orchestration verifications
        if self.using_model_orchestration:
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
            if self.args.save_only_model and self.args.load_best_model_at_end:
                raise ValueError(
                    f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`."
                )
            # `auto_find_batch_size` isn't yet supported with DeepSpeed/FSDP
            if self.args.auto_find_batch_size:
                raise NotImplementedError(f"`{wrapper}` doesn't support `auto_find_batch_size`.")
            if len(self.args.fsdp) > 0:
                if self.is_deepspeed_enabled:
                    raise ValueError(
                        "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                    )
                if not self.is_fsdp_xla_enabled and self.args.parallel_mode != ParallelMode.DISTRIBUTED:
                    raise ValueError("Using fsdp only works in distributed training.")

    def setup_model_parallelism(self, model):
        # First check based on model attributes
        self.is_model_parallel = getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False)
        # Then base it off the `device_map`
        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            self.is_model_parallel = len(devices) > 1 or (
                len(devices) == 1 and self.args.device != torch.device(devices[0])
            )
            # warn users
            if self.is_model_parallel:
                logger.warn(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def add_callback(self, callback):
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        self.callback_handler.remove_callback(callback)

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = self.args.accelerator_config.pop("gradient_accumulation_kwargs", {})
        grad_acc_kwargs["sync_with_dataloader"] = False
        if "num_steps" not in grad_acc_kwargs:
            # take the gradient_accumulation_steps setting from TrainingArguments.
            grad_acc_kwargs["num_steps"] = self.args.gradient_accumulation_steps
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            dataloader_config = DataLoaderConfiguration(
                split_batches=accelerator_config.pop("split_batches"),
                dispatch_batches=accelerator_config.pop("dispatch_batches"),
                even_batches=accelerator_config.pop("even_batches"),
                use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
            )
        non_blocking = accelerator_config.pop("non_blocking")
        if not is_accelerate_available("0.30.0") and non_blocking:
            raise ImportError(
                "`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature."
            )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning(
                    "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
                )
            dataloader_config.non_blocking = non_blocking

        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
        }
        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)

        # create accelerator object
        self.accelerator = Accelerator(**args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            if is_accelerate_available("0.23.0"):
                fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                    "activation_checkpointing", fsdp_plugin.activation_checkpointing
                )
                if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                    raise ValueError(
                        "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                        "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                        "when using FSDP."
                    )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        self.perform_distributed_sanity_checks()
