# as due to their complexity multi-gpu tests could impact other tests, and to aid debug we have those in a separate module.

import os
import sys

from transformers.testing_utils import TestCasePlus, execute_subprocess_async, get_gpu_count, require_torch_gpu, slow

from .utils import load_json


class TestSummarizationDistillerMultiGPU(TestCasePlus):
    @classmethod
    def setUpClass(cls):
        return cls

    @slow
    @require_torch_gpu
    def test_distributed_eval(self):
        output_dir = self.get_auto_remove_tmp_dir()
        args = f"""
            --model_name Helsinki-NLP/opus-mt-en-ro
            --save_dir {output_dir}
            --data_dir {self.test_file_dir_str}/test_data/wmt_en_ro
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
