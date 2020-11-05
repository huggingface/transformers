# as due to their complexity multi-gpu tests could impact other tests, and to aid debug we have those in a separate module.

import os
import sys

from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    require_torch_gpu,
    require_torch_multigpu,
    slow,
)

from .test_seq2seq_examples import CHEAP_ARGS, make_test_data_dir
from .utils import load_json


class TestSummarizationDistillerMultiGPU(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        return cls

    @require_torch_multigpu
    def test_multigpu(self):

        updates = dict(
            no_teacher=True,
            freeze_encoder=True,
            gpus=2,
            overwrite_output_dir=True,
            sortish_sampler=True,
        )
        self._test_distiller_cli_fork(updates, check_contents=False)

    def _test_distiller_cli_fork(self, updates, check_contents=True):
        default_updates = dict(
            label_smoothing=0.0,
            early_stopping_patience=-1,
            train_batch_size=1,
            eval_batch_size=2,
            max_epochs=2,
            alpha_mlm=0.2,
            alpha_ce=0.8,
            do_predict=True,
            model_name_or_path="sshleifer/tinier_bart",
            teacher=CHEAP_ARGS["model_name_or_path"],
            val_check_interval=0.5,
        )
        default_updates.update(updates)
        args_d: dict = CHEAP_ARGS.copy()
        tmp_dir = make_test_data_dir(tmp_dir=self.get_auto_remove_tmp_dir())
        output_dir = self.get_auto_remove_tmp_dir()
        args_d.update(data_dir=tmp_dir, output_dir=output_dir, **default_updates)

        def convert(k, v):
            if k in ["tgt_suffix", "server_ip", "server_port", "out", "n_tpu_cores"]:
                return ""
            if v is False or v is None:
                return ""
            if v is True:  # or len(str(v))==0:
                return f"--{k}"
            return f"--{k}={v}"

        cli_args = [x for x in (convert(k, v) for k, v in args_d.items()) if len(x)]
        cmd = [sys.executable, f"{self.test_file_dir}/distillation.py"] + cli_args
        execute_subprocess_async(cmd, env=self.get_env())

        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        ckpt_files = [p for p in contents if p.endswith("ckpt")]
        assert len(ckpt_files) > 0

        self.assertIn("test_generations.txt", contents)
        self.assertIn("test_results.txt", contents)

        # get the following from the module, (we don't have access to `model` here)
        metrics_save_path = os.path.join(output_dir, "metrics.json")
        val_metric = "rouge2"

        metrics = load_json(metrics_save_path)
        # {'test': [{'test_avg_loss': 10.63731575012207, 'test_avg_rouge1': 0.0, 'test_avg_rouge2': 0.0, 'test_avg_rougeL': 0.0, 'test_avg_gen_time': 0.1822289228439331, 'test_avg_gen_len': 142.0, 'step_count': 1}]}
        print(metrics)
        last_step_stats = metrics["val"][-1]
        self.assertGreaterEqual(last_step_stats["val_avg_gen_time"], 0.01)
        self.assertIsInstance(last_step_stats[f"val_avg_{val_metric}"], float)
        self.assertEqual(len(metrics["test"]), 1)
        desired_n_evals = int(args_d["max_epochs"] * (1 / args_d["val_check_interval"]) / 2 + 1)
        self.assertEqual(len(metrics["val"]), desired_n_evals)

    @slow
    @require_torch_gpu
    def test_distributed_eval(self):
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name Helsinki-NLP/opus-mt-en-ro
            --save_dir {output_dir}
            --data_dir test_data/wmt_en_ro
            --num_beams 2
            --task translation
        """.split()

        # we want this test to run even if there is only one GPU, but if there are more we use them all
        n_gpu = get_gpu_count()
        distributed_args = f"""
            -m torch.distributed.launch
            --nproc_per_node={n_gpu}
            {self.test_file_dir}/run_distributed_eval.py
        """.split()
        cmd = [sys.executable] + distributed_args + args
        execute_subprocess_async(cmd, env=self.get_env())

        metrics_save_path = os.path.join(output_dir, "test_bleu.json")
        metrics = load_json(metrics_save_path)
        # print(metrics)
        self.assertGreaterEqual(metrics["bleu"], 25)
