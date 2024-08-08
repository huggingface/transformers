import functools
import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch


# Integrations must be imported before ML frameworks:
# isort: off
from .integrations import (
    get_reporting_integration_callbacks,
    is_deepspeed_available,
    propagate_args_to_deepspeed,
)

# isort: on
import importlib

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
    get_decay_parameter_names,
    model_sanity_checks,
    optimizer_sanity_checks,
    LayerWiseDummyOptimizer,
)
from .trainer_utils import (
    EvalPrediction,
    create_accelerator,
    enable_full_determinism,
    has_length,
    number_of_arguments,
    set_seed,
)
from .trainer_distributed_pt_utils import apply_ipex_optimization, apply_fsdp_xla_optimization
from .training_args import OptimizerGroups, OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_in_notebook,
    is_lomo_available,
    is_peft_available,
    is_sagemaker_mp_enabled,
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
    pass

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        pass

if is_accelerate_available("0.28.0"):
    pass

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


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

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        


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
    
    def _wrap_model(self, model, training=True, dataloader=None):
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model) is not model:
            return model
        
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = apply_ipex_optimization(model, self.optimizer, training, dtype=dtype, is_in_train=self.args.is_in_train)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        if self.args.jit_mode_eval:
            start_time = time.time()
            model = self.torch_jit_model_eval(model, dataloader, training)
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        # Distributed training using PyTorch FSDP on XLA (not supported in accelerate yet)
        if self.is_fsdp_xla_enabled:
            model = apply_fsdp_xla_optimization(model, self.optimizer, self.args.fsdp, self.args.fsdp_config, self.is_fsdp_xla_v2_enabled)

            # Patch `xm.optimizer_step` should not reduce gradients in this case,
            # as FSDP does not need gradient reduction over sharded parameters.
            def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
                loss = optimizer.step(**optimizer_args)
                if barrier:
                    xm.mark_step()
                return loss

            xm.optimizer_step = patched_optimizer_step
        # Sagemaker DP
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        # Otherwise make sure that `Accelerate` has the proper args
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED and not is_torch_neuroncore_available():
            kwargs = {"find_unused_parameters": True}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing

            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            if self.args.ddp_broadcast_buffers is not None:
                kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

            self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)
        return model

    def torch_jit_model_eval(self, model, dataloader, training=False):
        if not training:
            if dataloader is None:
                logger.warning("failed to use PyTorch jit mode due to current dataloader is none.")
                return model
            example_batch = next(iter(dataloader))
            example_batch = self._prepare_inputs(example_batch)
            try:
                jit_model = copy.copy(model)
                jit_model.eval()
                original_forward = jit_model.__dict__.pop("_original_forward", None)
                # remove mixed precision hooks from the model
                if original_forward:
                    jit_model.forward = original_forward
                with self.accelerator.autocast(cache_enabled=False), torch.no_grad():
                    if version.parse(version.parse(torch.__version__).base_version) >= version.parse("2.0.0"):
                        if isinstance(example_batch, dict):
                            jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                        else:
                            jit_model = torch.jit.trace(
                                jit_model,
                                example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                                strict=False,
                            )
                    else:
                        jit_inputs = []
                        for key in example_batch:
                            example_tensor = torch.ones_like(example_batch[key])
                            jit_inputs.append(example_tensor)
                        jit_inputs = tuple(jit_inputs)
                        jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
                jit_model = torch.jit.freeze(jit_model)
                with torch.no_grad():
                    jit_model(**example_batch)
                    jit_model(**example_batch)
                model = jit_model
                self.use_cpu_amp = False
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                logger.warning(f"failed to use PyTorch jit mode due to: {e}.")

        return model
    

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


    def num_examples(self, dataloader: DataLoader) -> int:
        "Attempts to return the total number of examples in a dataloader"
        try:
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        train_tokens = 0
        try:
            for step, batch in enumerate(train_dl):
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps
                train_tokens += tokens
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens
    

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is not None:
            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)
            return self.optimizer
        decay_parameters = get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_cls = None
        optimizer_kwargs = {"lr": args.learning_rate}

        class_name = None
        optimizer_groups = OptimizerGroups.find_groups(args.optim)

        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif "ADAMW" in optimizer_groups:
            optimizer_kwargs.update({
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            })
            class_name = "AdamW"

            # Based on the optimizer, import/grab the right `__init__`
            if args.optim == OptimizerNames.ADAMW_HF:
                from .optimization import AdamW
            elif "ADAMW_TORCH" in optimizer_groups:
                from torch.optim import AdamW
                if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                    optimizer_kwargs.update({"fused": True})
            elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
                import_to_try = "torch_xla.amp.syncfree"
            elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
                import_to_try = "torch_npu.optim"
                class_name = "NpuFusedAdamW"
            elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
                import_to_try = "apex.optimizers"
                class_name = "FusedAdam"
            elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
                import_to_try = "torchdistx.optimizers"
                class_name = "AnyPrecisionAdamW"
                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )

            elif "BNB_COMPATIBLE" in optimizer_groups:
                import_to_try = "bitsandbytes.optim"

            # If we're not needing to import anything, use the earlier imported AdamW
            if import_to_try is None:
                optimizer_cls = AdamW
            else:
                try:
                    module = importlib.import_module(import_to_try)
                    optimizer_cls = getattr(module, class_name)
                except (ModuleNotFoundError, AttributeError):
                    raise ImportError(f"Trainer failed to import {class_name} from {import_to_try}. Please ensure the library you are using is installed.")

        elif "RMSPROP" in optimizer_groups:
            try:
                from bitsandbytes.optim import RMSprop
            except ImportError:
                raise ImportError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            optimizer_cls = RMSprop
            # Above we pass all `adam_kwargs` to the optimizer, here
            # we only pass `optim_args` which can be passed by the user.
            optimizer_kwargs.update(optim_args)
        elif "LION" in optimizer_groups:
            try:
                from bitsandbytes.optim import Lion
            except ImportError:
                raise ImportError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            optimizer_cls = Lion
            optimizer_kwargs.update({"betas": (args.adam_beta1, args.adam_beta2)})

        # Now add in the bnb specifics
        if "BNB_COMPATIBLE" in optimizer_groups:
            optimizer_kwargs.update({
                "optim_bits": 32 if "8bit" not in args.optim else 8,
                "is_paged": "paged" in args.optim and "rmsprop" not in args.optim,
            })

        # And potentially return if ready
        if optimizer_cls is not None:
            return optimizer_cls, optimizer_kwargs

        if "TORCH_NATIVE" in optimizer_groups:
            optimizer_cls = getattr(torch.optim, args.optim)
            return optimizer_cls, optimizer_kwargs

        if "GALORE" in optimizer_groups:
            if not is_galore_torch_available():
                raise ImportError(
                    "You need to install `galore_torch` in order to use GaLore optimizers"
                    " install it with `pip install git+https://github.com/jiaweizzhao/GaLore`"
                )
            from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit

            is_layerwise = args.optim.lower().endswith("layerwise")
            if is_layerwise and args.parallel_mode == ParallelMode.DISTRIBUTED:
                raise NotImplementedError("Layer-wise GaLore does not support DDP at this time")

            if args.optim_target_modules is None:
                raise ValueError(
                    "You need to define a `optim_target_modules` in order to properly use GaLore optimizers"
                )

            if model is None:
                raise ValueError("You need to pass a model in order to correctly initialize a GaLore optimizer.")

            logger.warning(
                "Activated GaLoRE fine-tuning, depending on your model size and hardware, the training might take a while before starting. Please be patient!"
            )

            if args.optim in [OptimizerNames.GALORE_ADAMW, OptimizerNames.GALORE_ADAMW_LAYERWISE]:
                optimizer_cls = GaLoreAdamW
            elif args.optim in [OptimizerNames.GALORE_ADAMW_8BIT, OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE]:
                optimizer_cls = GaLoreAdamW8bit
            elif args.optim in [OptimizerNames.GALORE_ADAFACTOR, OptimizerNames.GALORE_ADAFACTOR_LAYERWISE]:
                optimizer_cls = GaLoreAdafactor

            all_linear = (
                isinstance(args.optim_target_modules, str)
                and args.optim_target_modules.replace("_", "-") == "all-linear"
            )

            galore_params = []
            galore_params_names = []
            for module_name, module in model.named_modules():
                target_module_exists, is_regex = check_target_module_exists(
                    args.optim_target_modules, module_name, return_is_regex=True
                )

                if not isinstance(module, nn.Linear):
                    # Warn in case we match but it's not a linear layer
                    if target_module_exists and not is_regex:
                        logger.warning(
                            f"{module_name} has been matched but ignored as GaLore only supports linear layers. Please double check your `optim_target_modules`!"
                        )

                    continue

                if not target_module_exists and not all_linear:
                    continue

                galore_params.append(module.weight)
                galore_params_names.append(module_name + ".weight")

            if len(galore_params) == 0:
                raise ValueError(
                    f"None of the target modules were found! ({args.optim_target_modules}). Please make sure to pass a valid `target_modules`."
                )

            non_galore_params = [p for n, p in model.named_parameters() if n not in galore_params_names]

            galore_optim_kwargs = {
                "rank": int(optim_args.pop("rank", 128)),
                "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
                "scale": float(optim_args.pop("scale", 0.25)),
                "proj_type": optim_args.pop("proj_type", "std"),
            }

            # The default args are from the official repository: https://github.com/jiaweizzhao/GaLore
            param_groups = [
                {"params": non_galore_params},
                {"params": galore_params, **galore_optim_kwargs},
            ]

            if is_layerwise:
                # For layer-wise optimizers, the optimization step is done through post accumulation
                # gradient hooks. The trick is to first attach these hooks to the model parameters then
                # create a dummy optimizer that will perform no-ops in the Trainer.
                # See the original implementation or the nice implementation from @hiyouga
                # here: https://github.com/hiyouga/LLaMA-Factory/commit/8664262cde3919e10eaecbd66e8c5d356856362e#diff-ebe08ab14496dfb9e06075f0fdd36799ef6d1535cc4dd4715b74c4e3e06fe3ba
                if args.gradient_accumulation_steps != 1:
                    raise ValueError("Layerwise GaLoRE optimizer do not support gradient accumulation !")

                optimizer_dict = {}
                for param in non_galore_params:
                    param_groups = [{"params": [param]}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)
                for param in galore_params:
                    param_groups = [{"params": [param], **galore_optim_kwargs}]
                    optimizer_dict[param] = optimizer_cls(param_groups, **optimizer_kwargs)

                def optimizer_hook(param):
                    if param.grad is not None:
                        optimizer_dict[param].step()
                        optimizer_dict[param].zero_grad()

                for param in model.parameters():
                    if param.requires_grad:
                        param.register_post_accumulate_grad_hook(optimizer_hook)

                optimizer_cls = LayerWiseDummyOptimizer
                optimizer_kwargs.update({"optimizer_dict": optimizer_dict})

            optimizer_kwargs.update({"params": param_groups})

            if args.optim == OptimizerNames.GALORE_ADAFACTOR:
                optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
            return optimizer_cls, optimizer_kwargs

        if "LOMO" in optimizer_groups:
            if not is_lomo_available():
                raise ImportError(
                    "You need to install `lomo_optim` in order to use LOMO optimizers"
                    " install it with `pip install lomo-optim`"
                )
            if not is_accelerate_available("0.30.0"):
                raise ImportError("You need to have `accelerate>=0.30.0` to be able to use LOMO optimizers")

            if model is None:
                raise ValueError("You need to pass a `model` in order to correctly initialize a LOMO optimizer.")

            from lomo_optim import AdaLomo, Lomo

            if "ada" in args.optim:
                optimizer_cls = AdaLomo
            else:
                optimizer_cls = Lomo

            optimizer_kwargs.update({"model": model})
            return optimizer_cls, optimizer_kwargs

        # Finally raise an error if we're here
        raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler