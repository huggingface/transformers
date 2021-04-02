# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..file_utils import WEIGHTS_NAME, is_torch_tpu_available
from ..modeling_utils import PreTrainedModel, unwrap_model
from ..trainer import Trainer
from ..trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    SequentialDistributedSampler,
    nested_detach,
    nested_numpify,
    reissue_pt_warnings,
)
from ..trainer_utils import PREFIX_CHECKPOINT_DIR
from ..utils import logging
from .training_args_sm import is_sagemaker_model_parallel_available


logger = logging.get_logger(__name__)


if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp

    @smp.step()
    def forward_backward(model, inputs, gradient_accumulation_steps=1):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss /= gradient_accumulation_steps
        model.backward(loss)
        return loss

    @smp.step()
    def forward_only(model, inputs):
        return model(**inputs)

    def smp_gather(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_gather(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_gather(v) for k, v in tensor.items()})
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )
        all_tensors = smp.allgather(tensor, smp.CommGroup.DP_GROUP)
        return torch.cat([t.cpu() for t in all_tensors], dim=0)

    def nested_smp_concat(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(nested_smp_concat(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: nested_smp_concat(v) for k, v in tensor.items()})
        # It doesn't seem possible to check here if `tensor` is a StepOutput because StepOutput lives in `smp.step`
        # which is also the name of the decorator so Python is confused.
        return tensor.concat().detach().cpu()


class SageMakerTrainer(Trainer):
    def __init__(self, args=None, **kwargs):
        warnings.warn(
            "`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` "
            "instead.",
            FutureWarning,
        )
        self.is_model_parallel_enabled = is_sagemaker_model_parallel_available()
        super().__init__(args=args, **kwargs)

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        if self.is_model_parallel_enabled:
            return smp.rank() == 0 and smp.local_rank() == 0 and smp.mp_rank() == 0 and smp.dp_rank() == 0
        else:
            return super().is_world_process_zero()

    def _get_train_sampler(self):
        if self.is_model_parallel_enabled:
            if self.args.group_by_length:
                return DistributedLengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, num_replicas=smp.dp_size(), rank=smp.dp_rank()
                )
            elif not self.args.dataloader_drop_last:
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    self.args.per_device_train_batch_size,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                )
            else:
                return DistributedSampler(self.train_dataset, num_replicas=smp.dp_size(), rank=smp.dp_rank())
        else:
            return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if self.is_model_parallel_enabled:
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=smp.dp_size(),
                rank=smp.dp_rank(),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            return super()._get_eval_sampler(eval_dataset)

    def _wrap_model(self, model, training=True):
        if self.is_model_parallel_enabled:
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)
        else:
            return super()._wrap_model(model)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.is_model_parallel_enabled:
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.is_model_parallel_enabled:
            model.train()
            inputs = self._prepare_inputs(inputs)
            loss_mb = forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        else:
            return super().training_step(model, inputs)

    def _gather_and_numpify(self, tensors, name):
        if tensors is None:
            return
        if self.is_model_parallel_enabled:
            tensors = smp_gather(tensors)
            return nested_numpify(tensors)
        else:
            return super()._gather_and_numpify(tensors, name)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """
        if self.is_model_parallel_enabled:
            self._save_smp(output_dir)
        elif is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_process_zero():
            self._save(output_dir)

        # If on sagemaker and we are saving the main model (not a checkpoint so output_dir=None), save a copy to
        # SM_MODEL_DIR for easy deployment.
        if output_dir is None and os.getenv("SM_MODEL_DIR") is not None:
            self.save_model(output_dir=os.getenv("SM_MODEL_DIR"))

    def _save_smp(self, output_dir: Optional[str] = None):
        if smp.dp_rank() != 0:
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Calling the state_dict needs to be done on the wrapped model
        state_dict = self.model_wrapped.state_dict()

        # Rest of the save is done for the main process only
        if self.is_world_process_zero():
            model = self.model
            if not isinstance(model, PreTrainedModel):
                model = unwrap_model(model)
            if isinstance(model, PreTrainedModel):
                model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.is_model_parallel_enabled:
            if smp.dp_rank() != 0:
                return

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self.args.output_dir
            self.store_flos()

            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir)
            # Consolidate the state dict on all processed of dp_rank 0
            opt_state_dict = self.optimizer.state_dict()
            # Save it and the scheduler on the main process
            if self.is_world_process_zero():
                torch.save(opt_state_dict, os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # Maybe delete some older checkpoints.
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
        else:
            super()._save_checkpoint(self, model, trial, metrics=metrics)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if self.is_model_parallel_enabled:
            if checkpoint is None:
                return

            if os.path.isfile(os.path.join(checkpoint, "optimizer.pt")) and os.path.isfile(
                os.path.join(checkpoint, "scheduler.pt")
            ):
                self.optimizer.load_state_dict(
                    torch.load(os.path.join(checkpoint, "optimizer.pt"), map_location="cpu")
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, "scheduler.pt")))
                reissue_pt_warnings(caught_warnings)
        else:
            super()._load_optimizer_and_scheduler(checkpoint)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_model_parallel_enabled:
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)

            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            with torch.no_grad():
                raw_outputs = forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = nested_smp_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = nested_smp_concat(logits_mb)

            if prediction_loss_only:
                return (loss, None, None)

            if len(logits) == 1:
                logits = logits[0]

            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            return (loss, logits, labels)
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
