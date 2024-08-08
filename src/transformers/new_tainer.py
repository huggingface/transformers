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
    propagate_args_to_deepspeed,
)

# isort: on

from packaging import version
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler

from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .modeling_utils import PreTrainedModel, is_peft_model
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
    model_sanity_checks,
    optimizer_sanity_checks,
)
from .trainer_utils import (
    EvalPrediction,
    enable_full_determinism,
    has_length,
    number_of_arguments,
    set_seed,
    create_accelerator,
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

        model_sanity_checks(model)
        optimizer_sanity_checks(model, self.optimizer, self.lr_scheduler, self.using_model_orchestration)

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
        # create accelerator object
        self.accelerator = create_accelerator(self.args)
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
        # NOTE: This should be simplified to just build a FSDP plugin manually w/ overrides
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
            propagate_args_to_deepspeed(self.args)

        self.perform_distributed_sanity_checks()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is not None:
            return

        # Determine which model to inspect
        model_to_inspect = self.model
        if is_peft_model(self.model):
            if hasattr(self.model, "get_base_model"):
                model_to_inspect = self.model.get_base_model()
            else:
                # PeftMixedModel does not provide a `get_base_model` method
                model_to_inspect = self.model.base_model.model

        signature = inspect.signature(model_to_inspect.forward)
        self._signature_columns = list(signature.parameters.keys())

        # Labels may be named label or label_ids, the default data collator handles that.
        label_columns = set(["label", "label_ids"] + self.label_names)
        self._signature_columns.extend(label_columns)

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.warning(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature. "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_sampler(self, dataset=None, evaluation:bool = False) -> Optional[torch.utils.data.Sampler]:
        if (
            evaluation and self.args.world_size < 1
             or dataset is None 
             or not has_length(dataset)
        ):
            return None
        
        if evaluation: 
            return SequentialSampler(dataset)

        # Build the sampler.
        if self.args.group_by_length:
            lengths = None
            if (
                is_datasets_available() and 
                isinstance(dataset, datasets.Dataset) and 
                self.args.length_column_name in dataset.column_names
            ):
                lengths = dataset[self.args.length_column_name]
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else:
            return RandomSampler(dataset)

    def _prepare_dataset_for_dataloader(self, dataset, collator, dataset_type:str="train"):
        """
        Prepare a dataset for a dataloader.
        
        Returns: 
            A tuple of (dataset, dataloader_params)
        """
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description=description)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_size": self.args._train_batch_size if dataset_type == "train" else self.args.eval_batch_size,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_sampler(dataset, evaluation=dataset_type != "train")
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if dataset_type == "train":
                dataloader_params["worker_init_fn"] = seed_worker
        return dataset, dataloader_params


    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset, dataloader_params = self._prepare_dataset_for_dataloader(self.train_dataset, self.data_collator)

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])
    
        # Grab the right `eval_dataset` if needed
        if isinstance(eval_dataset, str):
            eval_dataset = self.eval_dataset[eval_dataset]
        elif eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        eval_dataset, dataloader_params = self._prepare_dataset_for_dataloader(eval_dataset, self.data_collator, dataset_type="evaluation")
        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}
        
        return self.accelerator.prepare(eval_dataloader)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_dataset, dataloader_params = self._prepare_dataset_for_dataloader(test_dataset, self.data_collator, dataset_type="test")
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))
