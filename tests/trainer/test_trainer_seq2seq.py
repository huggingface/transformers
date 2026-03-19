# Copyright 2020 the HuggingFace Inc. team.
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
import sys
from pathlib import Path
from unittest.mock import patch

from transformers import (
    AutoModelForSeq2SeqLM,
    BertConfig,
    BertTokenizer,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
)
from transformers.testing_utils import (
    ExtendSysPath,
    TestCasePlus,
    backend_device_count,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_bitsandbytes,
    require_sentencepiece,
    require_torch,
    require_torch_multi_accelerator,
    require_torch_non_multi_accelerator,
    slow,
    torch_device,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed
from transformers.utils import is_datasets_available, is_torch_available


if is_datasets_available():
    import datasets

if is_torch_available():
    import torch


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"
MBART_TINY = "sshleifer/tiny-mbart"


@require_sentencepiece
class Seq2seqTrainerTester(TestCasePlus):
    @slow
    @require_torch
    def test_finetune_bert2bert(self):
        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
            "prajjwal1/bert-tiny",
            "prajjwal1/bert-tiny",
            encoder_config=BertConfig.from_pretrained("prajjwal1/bert-tiny"),
            decoder_config=BertConfig.from_pretrained("prajjwal1/bert-tiny"),
            dtype=torch.float32,
        )
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

        bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
        tokenizer.eos_token_id = tokenizer.sep_token_id
        bert2bert.generation_config.decoder_start_token_id = tokenizer.cls_token_id
        bert2bert.generation_config.max_length = 128

        train_dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:1%]")
        val_dataset = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation[:1%]")

        train_dataset = train_dataset.select(range(32))
        val_dataset = val_dataset.select(range(16))

        batch_size = 4

        def _map_to_encoder_decoder_inputs(batch):
            # Tokenizer will automatically set [BOS] <text> [EOS]
            inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
            outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask

            batch["decoder_input_ids"] = outputs.input_ids
            batch["labels"] = outputs.input_ids.copy()
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
            ]
            batch["decoder_attention_mask"] = outputs.attention_mask

            assert all(len(x) == 512 for x in inputs.input_ids)
            assert all(len(x) == 128 for x in outputs.input_ids)

            return batch

        def _compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            # Replace -100 (ignore index) with pad_token_id before decoding
            import numpy as np

            labels_ids = np.where(labels_ids == -100, tokenizer.pad_token_id, labels_ids)

            # all unnecessary tokens are removed
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            accuracy = sum(int(pred_str[i] == label_str[i]) for i in range(len(pred_str))) / len(pred_str)

            return {"accuracy": accuracy}

        # map train dataset
        train_dataset = train_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )

        # same for validation dataset
        val_dataset = val_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )

        output_dir = self.get_auto_remove_tmp_dir()

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            eval_strategy="steps",
            do_train=True,
            do_eval=True,
            warmup_steps=0,
            eval_steps=2,
            logging_steps=2,
        )

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=bert2bert,
            args=training_args,
            compute_metrics=_compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

        # start training
        trainer.train()

    @slow
    @require_torch
    def test_return_sequences(self):
        # Tests that the number of generated sequences is correct when num_return_sequences > 1
        # and essentially ensuring that `accelerator.gather()` is used instead of `gather_for_metrics`
        INPUT_COLUMN = "question"
        TARGET_COLUMN = "answer"
        MAX_INPUT_LENGTH = 256
        MAX_TARGET_LENGTH = 256

        dataset = datasets.load_dataset("openai/gsm8k", "main", split="train[:38]")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt", padding="longest")
        gen_config = GenerationConfig.from_pretrained(
            "google-t5/t5-small", max_length=None, min_length=None, max_new_tokens=256, min_new_tokens=1, num_beams=5
        )

        training_args = Seq2SeqTrainingArguments(".", predict_with_generate=True)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x: {"samples": x[0].shape[0]},
        )

        def prepare_data(examples):
            # Remove pairs where at least one record is none
            inputs = examples[INPUT_COLUMN]
            targets = examples[TARGET_COLUMN]

            model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
            labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        prepared_dataset = dataset.map(prepare_data, batched=True, remove_columns=[INPUT_COLUMN, TARGET_COLUMN])
        dataset_len = len(prepared_dataset)  # 38

        for num_return_sequences in range(3, 0, -1):
            gen_config.num_return_sequences = num_return_sequences
            metrics = trainer.evaluate(eval_dataset=prepared_dataset, generation_config=gen_config)
            assert metrics["eval_samples"] == dataset_len * num_return_sequences, (
                f"Got {metrics['eval_samples']}, expected: {dataset_len * num_return_sequences}"
            )

    @require_torch
    def test_bad_generation_config_fail_early(self):
        # Tests that a bad generation config causes the trainer to fail early
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt", padding="longest")
        gen_config = GenerationConfig(do_sample=False, top_p=0.9)  # bad: top_p is not compatible with do_sample=False

        training_args = Seq2SeqTrainingArguments(".", predict_with_generate=True, generation_config=gen_config)
        with self.assertRaises(ValueError) as exc:
            _ = Seq2SeqTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                data_collator=data_collator,
                compute_metrics=lambda x: {"samples": x[0].shape[0]},
            )
        self.assertIn("Fix these issues to train your model", str(exc.exception))


@require_torch
class TestTranslationExample(TestCasePlus):
    """Tests for the run_translation.py example script (seq2seq training via CLI)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        examples_dir = Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "translation"
        with ExtendSysPath(str(examples_dir)):
            from run_translation import main as _main

            cls._run_translation_main = staticmethod(_main)

    def _run_translation(
        self,
        distributed=False,
        extra_args_str=None,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        n_gpus_to_use=None,
    ):
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {MBART_TINY}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --test_file {data_dir}/test.json
            --output_dir {output_dir}
            --max_train_samples 8
            --max_source_length 12
            --max_target_length 12
            --do_train
            --num_train_epochs 1
            --per_device_train_batch_size 4
            --learning_rate 3e-3
            --warmup_steps 8
            --logging_steps 0
            --logging_strategy no
            --save_steps 1
            --train_sampling_strategy group_by_length
            --label_smoothing_factor 0.1
            --target_lang ro_RO
            --source_lang en_XX
            --report_to none
        """.split()

        if do_eval:
            args += """
                --do_eval
                --per_device_eval_batch_size 4
                --max_eval_samples 8
                --val_max_target_length 12
                --eval_strategy steps
                --eval_steps 1
            """.split()

        if do_predict:
            args += ["--do_predict"]

        if predict_with_generate:
            args += ["--predict_with_generate"]

        if do_train:
            args += ["--optim", "adafactor"]

        if extra_args_str is not None:
            args += extra_args_str.split()

        if distributed:
            if n_gpus_to_use is None:
                n_gpus_to_use = backend_device_count(torch_device)
            master_port = get_torch_dist_unique_port()
            distributed_args = f"""
                -m torch.distributed.run
                --nproc_per_node={n_gpus_to_use}
                --master_port={master_port}
                {self.examples_dir_str}/pytorch/translation/run_translation.py
            """.split()
            cmd = [sys.executable] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ["run_translation.py"] + args
            with patch.object(sys, "argv", testargs):
                self._run_translation_main()

        return output_dir

    @require_torch_non_multi_accelerator
    def test_run_seq2seq_no_dist(self):
        output_dir = self._run_translation()
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log]
        first_step_stats = eval_metrics[0]
        assert "eval_bleu" in first_step_stats

    @require_torch_multi_accelerator
    def test_run_seq2seq_dp(self):
        output_dir = self._run_translation(distributed=False)
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log]
        first_step_stats = eval_metrics[0]
        assert "eval_bleu" in first_step_stats

    @require_torch_multi_accelerator
    def test_run_seq2seq_ddp(self):
        output_dir = self._run_translation(distributed=True)
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log]
        first_step_stats = eval_metrics[0]
        assert "eval_bleu" in first_step_stats

    @slow
    def test_run_seq2seq_slow(self):
        output_dir = self._run_translation(
            extra_args_str=f"--model_name_or_path {MARIAN_MODEL} --learning_rate 3e-4 --num_train_epochs 10 --max_source_length 128 --max_target_length 128 --eval_steps 2 --save_steps 2",
        )
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]
        assert first_step_stats["eval_loss"] > last_step_stats["eval_loss"], "model learned nothing"
        assert isinstance(last_step_stats["eval_bleu"], float)
        contents = {os.path.basename(p) for p in os.listdir(output_dir)}
        assert "generated_predictions.txt" in contents
        assert "predict_results.json" in contents

    @slow
    @require_bitsandbytes
    def test_run_seq2seq_bnb(self):
        from transformers.training_args import OptimizerNames

        def train_and_return_metrics(optim: str) -> tuple[int, float]:
            output_dir = self._run_translation(
                distributed=True,
                extra_args_str=f"--skip_memory_metrics 0 --model_name_or_path {MARIAN_MODEL} --learning_rate 3e-4 --num_train_epochs 1 --optim {optim} --max_source_length 128 --max_target_length 128",
                do_eval=False,
                do_predict=False,
                n_gpus_to_use=1,
            )
            logs = TrainerState.load_from_json(Path(output_dir, "trainer_state.json")).log_history
            gpu_peak_mem_mb = int(logs[0]["train_mem_gpu_peaked_delta"] / 2**20)
            gpu_alloc_mem_mb = int(logs[0]["train_mem_gpu_alloc_delta"] / 2**20)
            loss = logs[0]["train_loss"]
            return gpu_peak_mem_mb, gpu_alloc_mem_mb, loss

        gpu_peak_mem_orig, gpu_alloc_mem_orig, loss_orig = train_and_return_metrics(OptimizerNames.ADAMW_TORCH.value)
        gpu_peak_mem_bnb, gpu_alloc_mem_bnb, loss_bnb = train_and_return_metrics(OptimizerNames.ADAMW_BNB.value)

        gpu_alloc_mem_diff = gpu_alloc_mem_orig - gpu_alloc_mem_bnb
        gpu_total_mem_orig = gpu_peak_mem_orig + gpu_alloc_mem_orig
        gpu_total_mem_bnb = gpu_peak_mem_bnb + gpu_alloc_mem_bnb
        gpu_total_mem_diff = gpu_total_mem_orig - gpu_total_mem_bnb

        expected_savings = 120
        self.assertGreater(
            gpu_alloc_mem_diff,
            expected_savings,
            f"should use ~150MB less alloc gpu memory with BNB, but got diff={gpu_alloc_mem_diff}MB",
        )
        self.assertGreater(
            gpu_total_mem_diff,
            expected_savings,
            f"should use ~150MB less total gpu memory with BNB, but got diff={gpu_total_mem_diff}MB",
        )
        self.assertAlmostEqual(loss_orig, loss_bnb, 5, f"loss should be the same: {loss_orig} vs {loss_bnb}")
