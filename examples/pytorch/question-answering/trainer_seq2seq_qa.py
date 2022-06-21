# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
A subclass of `Trainer` specific to Question-Answering tasks
"""
from typing import Dict, List, Optional, Iterable, Union, Callable, Tuple

import torch
from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer, is_torch_tpu_available, LogitsProcessorList, StoppingCriteriaList
from transformers.trainer_utils import PredictionOutput


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    # def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List['Constraint']] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
    ) -> Dict[str, float]:
        self._gen_kwargs = {
            "max_length": max_length if max_length is not None else self.args.generation_max_length,
            "num_beams": num_beams if num_beams is not None else self.args.generation_num_beams,
            "min_length": min_length,
            "do_sample": do_sample,
            "early_stopping": early_stopping,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "repetition_penalty": repetition_penalty,
            "bad_words_ids": bad_words_ids,
            "force_words_ids": force_words_ids,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "encoder_no_repeat_ngram_size": encoder_no_repeat_ngram_size,
            "num_return_sequences": num_return_sequences,
            "max_time": max_time,
            "max_new_tokens": max_new_tokens,
            "decoder_start_token_id": decoder_start_token_id,
            "use_cache": use_cache,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": diversity_penalty,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "logits_processor": logits_processor,
            "renormalize_logits": renormalize_logits,
            "stopping_criteria": stopping_criteria,
            "constraints": constraints,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "output_scores": output_scores,
            "return_dict_in_generate": return_dict_in_generate,
            "forced_bos_token_id": forced_bos_token_id,
            "forced_eos_token_id": forced_eos_token_id,
            "remove_invalid_values": remove_invalid_values,
            "synced_gpus": synced_gpus,
            "exponential_decay_length_penalty": exponential_decay_length_penalty,
        }

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test",
                max_length: Optional[int] = None,
                min_length: Optional[int] = None,
                num_beams: Optional[int] = None,
                do_sample: Optional[bool] = None,
                early_stopping: Optional[bool] = None,
                temperature: Optional[float] = None,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                typical_p: Optional[float] = None,
                repetition_penalty: Optional[float] = None,
                bad_words_ids: Optional[Iterable[int]] = None,
                force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
                bos_token_id: Optional[int] = None,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                length_penalty: Optional[float] = None,
                no_repeat_ngram_size: Optional[int] = None,
                encoder_no_repeat_ngram_size: Optional[int] = None,
                num_return_sequences: Optional[int] = None,
                max_time: Optional[float] = None,
                max_new_tokens: Optional[int] = None,
                decoder_start_token_id: Optional[int] = None,
                use_cache: Optional[bool] = None,
                num_beam_groups: Optional[int] = None,
                diversity_penalty: Optional[float] = None,
                prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
                renormalize_logits: Optional[bool] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
                constraints: Optional[List['Constraint']] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                output_scores: Optional[bool] = None,
                return_dict_in_generate: Optional[bool] = None,
                forced_bos_token_id: Optional[int] = None,
                forced_eos_token_id: Optional[int] = None,
                remove_invalid_values: Optional[bool] = None,
                synced_gpus: Optional[bool] = False,
                exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
        ):
        self._gen_kwargs = {
            "max_length": max_length if max_length is not None else self.args.generation_max_length,
            "num_beams": num_beams if num_beams is not None else self.args.generation_num_beams,
            "min_length": min_length,
            "do_sample": do_sample,
            "early_stopping": early_stopping,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "repetition_penalty": repetition_penalty,
            "bad_words_ids": bad_words_ids,
            "force_words_ids": force_words_ids,
            "bos_token_id": bos_token_id,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "encoder_no_repeat_ngram_size": encoder_no_repeat_ngram_size,
            "num_return_sequences": num_return_sequences,
            "max_time": max_time,
            "max_new_tokens": max_new_tokens,
            "decoder_start_token_id": decoder_start_token_id,
            "use_cache": use_cache,
            "num_beam_groups": num_beam_groups,
            "diversity_penalty": diversity_penalty,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "logits_processor": logits_processor,
            "renormalize_logits": renormalize_logits,
            "stopping_criteria": stopping_criteria,
            "constraints": constraints,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "output_scores": output_scores,
            "return_dict_in_generate": return_dict_in_generate,
            "forced_bos_token_id": forced_bos_token_id,
            "forced_eos_token_id": forced_eos_token_id,
            "remove_invalid_values": remove_invalid_values,
            "synced_gpus": synced_gpus,
            "exponential_decay_length_penalty": exponential_decay_length_penalty,
        }

        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)
