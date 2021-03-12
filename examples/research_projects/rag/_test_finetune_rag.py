import json
import logging
import os
import sys
from pathlib import Path

import finetune_rag
from transformers.file_utils import is_apex_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    require_ray,
    require_torch_gpu,
    require_torch_multi_gpu,
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class RagFinetuneExampleTests(TestCasePlus):
    def _create_dummy_data(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        contents = {"source": "What is love ?", "target": "life"}
        n_lines = {"train": 12, "val": 2, "test": 2}
        for split in ["train", "test", "val"]:
            for field in ["source", "target"]:
                content = "\n".join([contents[field]] * n_lines[split])
                with open(os.path.join(data_dir, f"{split}.{field}"), "w") as f:
                    f.write(content)

    def _run_finetune(self, gpus: int, distributed_retriever: str = "pytorch"):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        output_dir = os.path.join(tmp_dir, "output")
        data_dir = os.path.join(tmp_dir, "data")
        self._create_dummy_data(data_dir=data_dir)

        testargs = f"""
                --data_dir {data_dir} \
                --output_dir {output_dir} \
                --model_name_or_path facebook/rag-sequence-base \
                --model_type rag_sequence \
                --do_train \
                --do_predict \
                --n_val -1 \
                --val_check_interval 1.0 \
                --train_batch_size 2 \
                --eval_batch_size 1 \
                --max_source_length 25 \
                --max_target_length 25 \
                --val_max_target_length 25 \
                --test_max_target_length 25 \
                --label_smoothing 0.1 \
                --dropout 0.1 \
                --attention_dropout 0.1 \
                --weight_decay 0.001 \
                --adam_epsilon 1e-08 \
                --max_grad_norm 0.1 \
                --lr_scheduler polynomial \
                --learning_rate 3e-04 \
                --num_train_epochs 1 \
                --warmup_steps 4 \
                --gradient_accumulation_steps 1 \
                --distributed-port 8787 \
                --use_dummy_dataset 1 \
                --distributed_retriever {distributed_retriever} \
            """.split()

        if gpus > 0:
            testargs.append(f"--gpus={gpus}")
            if is_apex_available():
                testargs.append("--fp16")
        else:
            testargs.append("--gpus=0")
            testargs.append("--distributed_backend=ddp_cpu")
            testargs.append("--num_processes=2")

        cmd = [sys.executable, str(Path(finetune_rag.__file__).resolve())] + testargs
        execute_subprocess_async(cmd, env=self.get_env())

        metrics_save_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_save_path) as f:
            result = json.load(f)
        return result

    @require_torch_gpu
    def test_finetune_gpu(self):
        result = self._run_finetune(gpus=1)
        self.assertGreaterEqual(result["test"][0]["test_avg_em"], 0.2)

    @require_torch_multi_gpu
    def test_finetune_multigpu(self):
        result = self._run_finetune(gpus=2)
        self.assertGreaterEqual(result["test"][0]["test_avg_em"], 0.2)

    @require_torch_gpu
    @require_ray
    def test_finetune_gpu_ray_retrieval(self):
        result = self._run_finetune(gpus=1, distributed_retriever="ray")
        self.assertGreaterEqual(result["test"][0]["test_avg_em"], 0.2)

    @require_torch_multi_gpu
    @require_ray
    def test_finetune_multigpu_ray_retrieval(self):
        result = self._run_finetune(gpus=1, distributed_retriever="ray")
        self.assertGreaterEqual(result["test"][0]["test_avg_em"], 0.2)
