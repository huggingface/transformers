# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Check context-parallel prediction restrictions and loss-only evaluation."""

import argparse
import json
import math
import unittest
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, Qwen3Config, Trainer, TrainingArguments, default_data_collator


def count_prediction_rows(prediction):
    return {"rows": len(prediction.predictions)}


class SequenceLengthObserver:
    def __init__(self):
        self.lengths = []

    def __call__(self, module, args, kwargs):
        self.lengths.append(kwargs["input_ids"].shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_cp_size", type=int, required=True)
    parser.add_argument("--use_cpu", action="store_true")
    options = parser.parse_args()
    rows, sequence_length, vocab_size = 8, 128, 1024
    input_ids = torch.randint(vocab_size, (rows, sequence_length), generator=torch.Generator().manual_seed(0))
    datasets = {
        labeled: [
            {"input_ids": tokens, "labels": tokens} if labeled else {"input_ids": tokens} for tokens in input_ids
        ]
        for labeled in (True, False)
    }
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(
        Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=sequence_length,
            attn_implementation="sdpa",
        ),
        dtype=torch.float32 if options.use_cpu else torch.bfloat16,
    )
    observer = SequenceLengthObserver()
    handle = model.register_forward_pre_hook(observer, with_kwargs=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(options.output_dir),
            max_steps=1,
            learning_rate=0.0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            use_cpu=options.use_cpu,
            bf16=not options.use_cpu,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
        ),
        train_dataset=datasets[True],
        data_collator=default_data_collator,
    )
    test_case = unittest.TestCase()
    test_case.assertEqual(trainer.get_cp_size(), options.expected_cp_size)
    # FSDP2 prepares the model together with the optimizer before evaluation.
    trainer.train()
    test_case.assertEqual(trainer.state.global_step, 1)
    metrics = trainer.evaluate(datasets[True])
    test_case.assertTrue(math.isfinite(metrics["eval_loss"]))

    for labeled, dataset in datasets.items():
        if options.expected_cp_size > 1:
            observer.lengths.clear()
            with test_case.assertRaisesRegex(ValueError, "Returning predictions with context parallelism"):
                trainer.predict(dataset)
            test_case.assertEqual(observer.lengths, [])
        else:
            output = trainer.predict(dataset)
            test_case.assertEqual(output.predictions.shape, (rows, sequence_length, vocab_size))
            if labeled:
                test_case.assertEqual(output.label_ids.shape, (rows, sequence_length))
            else:
                test_case.assertIsNone(output.label_ids)

    trainer.compute_metrics = count_prediction_rows
    if options.expected_cp_size > 1:
        with test_case.assertRaisesRegex(ValueError, "Returning predictions with context parallelism"):
            trainer.evaluate(datasets[True])
    else:
        test_case.assertEqual(trainer.evaluate(datasets[True])["eval_rows"], rows)

    trainer.args.prediction_loss_only = True
    for labeled, dataset in datasets.items():
        observer.lengths.clear()
        output = trainer.predict(dataset)
        test_case.assertIsNone(output.predictions)
        test_case.assertIsNone(output.label_ids)
        test_case.assertEqual(set(observer.lengths), {sequence_length // options.expected_cp_size})
        if labeled:
            test_case.assertTrue(math.isfinite(output.metrics["test_loss"]))

    handle.remove()
    if trainer.accelerator.is_main_process:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        with open(options.output_dir / "prediction_checks.json", "w", encoding="utf-8") as result_file:
            json.dump({"cp_size": trainer.get_cp_size(), "eval_loss": metrics["eval_loss"]}, result_file)
    trainer.end()


if __name__ == "__main__":
    main()
