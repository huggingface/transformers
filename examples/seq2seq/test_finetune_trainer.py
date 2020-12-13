# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from unittest.mock import patch

from transformers import BertTokenizer, EncoderDecoderModel
from transformers.file_utils import is_datasets_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_torch_multi_gpu,
    require_torch_non_multi_gpu,
    slow,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed

from .finetune_trainer import Seq2SeqTrainingArguments, main
from .seq2seq_trainer import Seq2SeqTrainer


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"
MBART_TINY = "sshleifer/tiny-mbart"


class TestFinetuneTrainer(TestCasePlus):
    def finetune_trainer_quick(self, distributed=None):
        output_dir = self.run_trainer(1, "12", MBART_TINY, 1, distributed)
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        assert "eval_bleu" in first_step_stats

    @require_torch_non_multi_gpu
    def test_finetune_trainer_no_dist(self):
        self.finetune_trainer_quick()

    # the following 2 tests verify that the trainer can handle distributed and non-distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_finetune_trainer_dp(self):
        self.finetune_trainer_quick(distributed=False)

    @require_torch_multi_gpu
    def test_finetune_trainer_ddp(self):
        self.finetune_trainer_quick(distributed=True)

    @slow
    def test_finetune_trainer_slow(self):
        # There is a missing call to __init__process_group somewhere
        output_dir = self.run_trainer(
            eval_steps=2, max_len="128", model_name=MARIAN_MODEL, num_train_epochs=10, distributed=False
        )

        # Check metrics
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]

        assert first_step_stats["eval_bleu"] < last_step_stats["eval_bleu"]  # model learned nothing
        assert isinstance(last_step_stats["eval_bleu"], float)

        # test if do_predict saves generations and metrics
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        assert "test_generations.txt" in contents
        assert "test_results.json" in contents

    @slow
    def test_finetune_bert2bert(self):
        if not is_datasets_available():
            return

        import datasets

        bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("prajjwal1/bert-tiny", "prajjwal1/bert-tiny")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size
        bert2bert.config.eos_token_id = tokenizer.sep_token_id
        bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
        bert2bert.config.max_length = 128

        train_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
        val_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")

        train_dataset = train_dataset.select(range(32))
        val_dataset = val_dataset.select(range(16))

        rouge = datasets.load_metric("rouge")

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

            assert all([len(x) == 512 for x in inputs.input_ids])
            assert all([len(x) == 128 for x in outputs.input_ids])

            return batch

        def _compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            # all unnecessary tokens are removed
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])[
                "rouge2"
            ].mid

            return {
                "rouge2_precision": round(rouge_output.precision, 4),
                "rouge2_recall": round(rouge_output.recall, 4),
                "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
            }

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
            evaluation_strategy="steps",
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
        )

        # start training
        trainer.train()

    def run_trainer(
        self, eval_steps: int, max_len: str, model_name: str, num_train_epochs: int, distributed: bool = False
    ):
        data_dir = self.examples_dir / "seq2seq/test_data/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name_or_path {model_name}
            --data_dir {data_dir}
            --output_dir {output_dir}
            --overwrite_output_dir
            --n_train 8
            --n_val 8
            --max_source_length {max_len}
            --max_target_length {max_len}
            --val_max_target_length {max_len}
            --do_train
            --do_eval
            --do_predict
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 4
            --per_device_eval_batch_size 4
            --learning_rate 3e-3
            --warmup_steps 8
            --evaluation_strategy steps
            --predict_with_generate
            --logging_steps 0
            --save_steps {str(eval_steps)}
            --eval_steps {str(eval_steps)}
            --sortish_sampler
            --label_smoothing 0.1
            --adafactor
            --task translation
            --tgt_lang ro_RO
            --src_lang en_XX
        """.split()
        # --eval_beams  2

        if distributed:
            n_gpu = get_gpu_count()
            distributed_args = f"""
                -m torch.distributed.launch
                --nproc_per_node={n_gpu}
                {self.test_file_dir}/finetune_trainer.py
            """.split()
            cmd = [sys.executable] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ["finetune_trainer.py"] + args
            with patch.object(sys, "argv", testargs):
                main()

        return output_dir
