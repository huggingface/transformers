import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.trainer import get_tpu_sampler


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from .utils import SortishSampler, label_smoothed_nll_loss


logger = logging.getLogger(__name__)


class Seq2SeqTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if self.args.sortish_sampler and self.args.n_gpu <= 1 and self.args.local_rank == -1:
                return SortishSampler

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    # override to support label smoothing
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> float:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`float`: The training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        if self.args.fp16 and _use_native_amp:
            with autocast():
                outputs = model(**inputs)
                logits = outputs[0]
                loss = self._compute_loss(logits, labels, ignore_index=model.config.pad_token_id)
        else:
            outputs = model(**inputs)
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            logits = outputs[0]
            loss = self._compute_loss(logits, labels, ignore_index=model.config.pad_token_id)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _compute_loss(self, logits, labels, ignore_index):
        # assuming label_smoothing is in args
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            assert logits.shape[-1] == self.model.config.vocab_size
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=ignore_index
            )
        return loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.args.predict_from_generate:
                max_length = model.config.max_length
                logits_out = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                # in case the batch is shorter than max length, the output should be padded
                logits = model.config.pad_token_id * torch.ones(
                    (logits_out.shape[0], max_length), dtype=logits_out.dtype, device=logits_out.device
                )
                logits[:, : logits_out.shape[-1]] = logits_out

                if has_labels:
                    outputs = model(**inputs)
                    loss = outputs[0]
                else:
                    loss = None
            else:
                outputs = model(**inputs)
                if has_labels:
                    loss, logits = outputs[:2]
                    loss = loss.mean().item()
                else:
                    loss = None
                    logits = outputs[0]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels)
