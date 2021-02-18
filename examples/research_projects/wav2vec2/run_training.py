#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
from packaging import version

import soundfile as sf
from jiwer import wer
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments, Wav2Vec2ForCTC, Wav2Vec2Tokenizer

from .file_utils import is_apex_available


if is_apex_available():
    from apex import amp


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base")

train_dataset = datasets.load_dataset("librispeech_asr", "clean", split="train.100")
val_dataset = datasets.load_dataset("librispeech_asr", "clean", split="validation")


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


train_dataset = train_dataset.map(map_to_array, remove_columns=["file"])
val_dataset = val_dataset.map(map_to_array, remove_columns=["file"])


def encode_labels(tokenizer, text):
    batch_tokens = []
    for text_str in text:
        batch_tokens.append(
            list(text_str.strip().replace(" ", tokenizer.word_delimiter_token) + tokenizer.word_delimiter_token)
        )

    batch_ids = []
    for tokens in batch_tokens:
        batch_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    return batch_ids


def prepare_dataset(batch):
    batch["input_values"] = tokenizer(batch["speech"]).input_values
    batch["labels"] = encode_labels(tokenizer, batch["text"])
    return batch


train_dataset = train_dataset.map(prepare_dataset, batch_size=16, batched=True, num_proc=32)
val_dataset = val_dataset.map(prepare_dataset, batch_size=16, batched=True, num_proc=32)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: bool = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_values"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(tokenizer=tokenizer, return_attention_mask=False, padding=True)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = 0

    pred_str = tokenizer.batch_decode(pred_ids)
    label_str = tokenizer.batch_decode(pred.label_ids)

    wer_score = wer(label_str, pred_str)

    return {"wer": wer_score}


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
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
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            if model.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["labels"] >= 0).sum()
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=40,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    gradient_accumulation_steps=4,
    save_total_limit=3,
    save_steps=500,
    eval_steps=100,
    logging_steps=50,
    learning_rate=1e-4,
    warmup_steps=3000,
    group_by_length=True,
    logging_dir="./logs",
    fp16=True,
)

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
