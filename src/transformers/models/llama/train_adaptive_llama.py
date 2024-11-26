import pytest
from dataclasses import dataclass, field

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveFanIn, AdaptiveFanOut, AdaptiveFanInOutput, AdaptiveFanOutOutput, AdaptiveLlamaModel


from transformers import GenerationConfig

VOCAB_SIZE = 20
MAX_SEQ_LEN = 10

import random
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_adaptive_llama import AdaptiveLlamaForCausalLM

from transformers import Trainer
from transformers import TrainingArguments

from accelerate.tracking import LOGGER_TYPE_TO_CLASS, GeneralTracker, filter_trackers

import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional, Union, Any, Union, Dict, Tuple

import numpy as np

from transformers.modeling_outputs import BaseModelOutputWithPast

from dataclasses import dataclass, field
import transformers
from transformers import GenerationConfig
from transformers.trainer import nested_detach
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, IterableDatasetShard
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction

from torch.utils.data import DataLoader

import time

from enum import Enum

import torch
from typing import Any, Dict
from transformers import EvalPrediction
import evaluate
from evaluate import Metric
import re
from typing import List, Optional

from transformers import logging

from functools import cached_property

class ComputeMetrics():

    def __call__(self, predictions=None, label_ids=None, losses=None, inputs=None, prefix_ids=None, generated_ids=None, **kwargs) -> Dict:
        accuracy = (generated_ids == kwargs['input_ids'][:, :generated_ids.shape[1]]).sum() / generated_ids.size

        return {
            "accuracy": accuracy
        }


class SequentialNumbersDataset():
    def __init__(self, length=10000, num_numbers=100, max_sequence_length=20):
        self.length = length
        self.num_numbers = num_numbers
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, i):

        current_length = self.max_sequence_length
        start_from = random.randint(3, self.num_numbers - current_length - 3) # pad + bos + eos tokens

        max_padding = self.max_sequence_length - current_length

        inputs_ids = [1] + list(range(start_from, start_from + current_length)) + [2] + ([0] * max_padding)
        labels = [1] + list(range(start_from, start_from + current_length)) + [2] + ([-100] * max_padding)
        attention_mask_length = current_length + 2
        attention_mask = ([1] * attention_mask_length) + ([0] * max_padding)
        attention_mask = torch.tensor(attention_mask)
        special_embeddings_mask = torch.zeros_like(attention_mask)
        special_embeddings_mask[0] = 1
        special_embeddings_mask[attention_mask_length - 1] = 1

        assert len(attention_mask) == len(inputs_ids)

        return {
            "input_ids": inputs_ids,
            "labels": labels,
            "special_embeddings_mask": special_embeddings_mask,
            "attention_mask": attention_mask,
        }


class AdaptiveLlamaTrainer(Trainer):
    def compute_loss(self, model: AdaptiveLlamaForCausalLM, inputs, return_outputs=False, log_metrics=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model.forward(
            input_ids=inputs['input_ids'],
            labels=inputs['labels'],
            special_embeddings_mask=inputs['special_embeddings_mask'],
            attention_mask=inputs['attention_mask'],
        )

        outputs_inverted = model.forward(
            input_ids=inputs['input_ids'],
            labels=inputs['labels'],
            special_embeddings_mask=inputs['special_embeddings_mask'],
            attention_mask=inputs['attention_mask'],
            inverted_merging_map=[ True ]
        )

        loss = outputs.loss + outputs_inverted.loss # + 0.01 * sum([x.fan_in_mlp.weight.norm(2) for x in model.model.adaptive_down])
        # loss = outputs.loss # + (outputs_inversed.loss / 10)

        assert ~ loss.isnan().any(), 'loss cant be none'

        if log_metrics:
            log_info = {
                "debug/straight_loss": outputs.loss.detach().item(),
                "debug/inverted_loss": outputs_inverted.loss.detach().item(),
                "debug/mean_merged_tokens": outputs.mean_merged_tokens,
                "debug/inverted_mean_merged_tokens": outputs_inverted.mean_merged_tokens,
                "debug/total_tokens": inputs['attention_mask'].sum().item(),
            }
            self.log(log_info)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: AdaptiveLlamaForCausalLM, *args, **kwargs):
        result = super().training_step(model, *args, **kwargs)

        # if merger_mpl_grad > 5:
        #     breakpoint()

        extra_log = dict()
        for i, adown in enumerate(model.model.adaptive_down):
            merger_mpl_grad = adown.fan_in_mlp.weight.grad.norm(2).item()
            assert merger_mpl_grad is not None, "merger_mpl_grad is expected to be not none"
            extra_log[f"merger_mpl_grad_norm_{i}"] = merger_mpl_grad

        self.log(extra_log)

        return result


    def update_eval_set_kwargs_containers(self, model, inputs):

        gen_params = {
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "repetition_penalty": 2.5,
            "remove_invalid_values": True,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "forced_eos_token_id": 2,
            "use_cache": False,
            "no_repeat_ngram_size": 4,
            "num_return_sequences": 1,
        }
        genconfig = GenerationConfig()

        caption_legth = inputs['input_ids'].shape[1] - 2
        genconfig.max_length = caption_legth

        batch_size, seq_len = inputs['input_ids'].shape[0], 2
        special_embeddings_mask = torch.ones([batch_size, seq_len], device=inputs['input_ids'].device)
        special_embeddings_mask[:, 1] = 0
        attention_mask = torch.ones([batch_size, seq_len], device=inputs['input_ids'].device)

        prefix_ids = inputs['input_ids'][:, :2]
        all_generation_params = {
            'generation_config': genconfig,
            'max_new_tokens': caption_legth,
            'inputs': prefix_ids,
            'special_embeddings_mask': special_embeddings_mask,
            'attention_mask': attention_mask,
            **gen_params,
        }

        model_generation = model.generate(**all_generation_params)

        return {
            "generated_ids": model_generation,
            "prefix_ids": prefix_ids,
            "input_ids": inputs['input_ids'],
        }

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        
        # print("inputs", inputs.keys())
        # breakpoint()
        
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []


        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True, log_metrics=False)
            loss = loss.mean().detach()

            logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        labels = None

        return (loss, logits, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            # Update containers
            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            extra_eval_set_kwargs = self.update_eval_set_kwargs_containers(model, inputs)
            for key, value in extra_eval_set_kwargs.items():
                if key not in eval_set_kwargs:
                    eval_set_kwargs[key] = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=0)

                eval_set_kwargs[key].add(value)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        predictions=all_preds,
                        label_ids=all_labels,
                        **eval_set_kwargs
                    )

                del losses, logits, labels, inputs, eval_set_kwargs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                for key in eval_set_kwargs.keys():
                    eval_set_kwargs[key].to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        eval_set_kwargs_arrays = dict()
        for key, value in eval_set_kwargs.items():
            eval_set_kwargs_arrays[key] = eval_set_kwargs[key].get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            # and all_preds is not None
            # and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs_arrays["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs_arrays["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                predictions=all_preds,
                label_ids=all_labels,
                **eval_set_kwargs_arrays
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)



@dataclass
class AdaptiveTrainingArguments(TrainingArguments):
    output_dir: str = field(default="llama_for_sequential_numbers",)
    learning_rate: float = field(default=1e-4)
    warmup_steps: int = field(default=100)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=64)
    num_train_epochs: int = field(default=50)
    weight_decay: float = field(default=0.01)
    eval_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    push_to_hub: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    report_to: str = field(default="wandb")
    logging_steps: int = field(default=5)
    dataloader_drop_last: bool = field(default=True)


if __name__ == "__main__":
    snd = SequentialNumbersDataset(length=5000, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)
    snd_eval = SequentialNumbersDataset(length=100, num_numbers=VOCAB_SIZE, max_sequence_length=MAX_SEQ_LEN)

    llama_config = LlamaConfig(
        hidden_size=128,
        vocab_size=VOCAB_SIZE,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        max_position_embeddings=MAX_SEQ_LEN,
        use_cache=False,
        attn_implementation = 'eager'
    )

    model = AdaptiveLlamaForCausalLM(llama_config)
    model.train()

    print("num model parameters:", sum(p.numel() for p in model.parameters()))

    hf_parser = transformers.HfArgumentParser(AdaptiveTrainingArguments)
    (training_args,) = hf_parser.parse_args_into_dataclasses()


    trainer = AdaptiveLlamaTrainer(
        model,
        args=training_args,
        train_dataset=snd,
        eval_dataset=snd_eval,
        compute_metrics=ComputeMetrics(),
    )

    # trainer.accelerator.log_with = filter_trackers("wandb", training_args.output_dir)
    # trackers = filter_trackers(log_with, self.logging_dir)
    # if len(trackers) < 1 and log_with is not None:
    #     warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
    # self.log_with = trackers

    trainer.accelerator.init_trackers(
        project_name="llama_for_sequential_numbers",
    )

    # with torch.autograd.set_detect_anomaly(True):
    trainer.train(
        resume_from_checkpoint=None,
        # resume_from_checkpoint="llama_for_sequential_numbers/checkpoint-1170"
    )

