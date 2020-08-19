from typing import Any, Dict, Union
from packaging import version
import warnings

import torch
from torch import nn

from transformers import Trainer
from transformers.file_utils import is_apex_available

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

from .utils import label_smoothed_nll_loss

class Seq2SeqTrainer(Trainer):
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
        if hasattr(self, "_training_step"):
            warnings.warn(
                "The `_training_step` method is deprecated and won't be called in a future version, define `training_step` in your subclass.",
                FutureWarning,
            )
            return self._training_step(model, inputs, self.optimizer)

        model.train()
        inputs = self._prepare_inputs(inputs, model)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                outputs = model(**inputs)
                loss = outputs[0]
        else:
            outputs = model(**inputs)
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs[0]

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # assuming label_smoothing is in args
        if self.args.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=model.config.pad_token_id
            )

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()
