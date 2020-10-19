import os
import tempfile
import unittest
from pathlib import Path

from transformers import AutoConfig, is_tf_available
from transformers.testing_utils import require_tf


if is_tf_available():
    import tensorflow as tf

    from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments


@require_tf
class TFBenchmarkTest(unittest.TestCase):
    def check_results_dict_not_empty(self, results):
        for model_result in results.values():
            for batch_size, sequence_length in zip(model_result["bs"], model_result["ss"]):
                result = model_result["result"][batch_size][sequence_length]
                self.assertIsNotNone(result)

    def test_inference_no_configs_eager(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            eager_mode=True,
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_inference_no_configs_only_pretrain(self):
        MODEL_ID = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
            only_pretrain_model=True,
        )
        benchmark = TensorFlowBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_inference_no_configs_graph(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_inference_with_configs_eager(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        config = AutoConfig.from_pretrained(MODEL_ID)
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            eager_mode=True,
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args, [config])
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_inference_with_configs_graph(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        config = AutoConfig.from_pretrained(MODEL_ID)
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args, [config])
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_train_no_configs(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=True,
            inference=False,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_train_result)
        self.check_results_dict_not_empty(results.memory_train_result)

    def test_train_with_configs(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        config = AutoConfig.from_pretrained(MODEL_ID)
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=True,
            inference=False,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args, [config])
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_train_result)
        self.check_results_dict_not_empty(results.memory_train_result)

    def test_inference_encoder_decoder_with_configs(self):
        MODEL_ID = "patrickvonplaten/t5-tiny-random"
        config = AutoConfig.from_pretrained(MODEL_ID)
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args, configs=[config])
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    @unittest.skipIf(is_tf_available() and len(tf.config.list_physical_devices("GPU")) == 0, "Cannot do xla on CPU.")
    def test_inference_no_configs_xla(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        benchmark_args = TensorFlowBenchmarkArguments(
            models=[MODEL_ID],
            training=False,
            inference=True,
            sequence_lengths=[8],
            batch_sizes=[1],
            use_xla=True,
            multi_process=False,
        )
        benchmark = TensorFlowBenchmark(benchmark_args)
        results = benchmark.run()
        self.check_results_dict_not_empty(results.time_inference_result)
        self.check_results_dict_not_empty(results.memory_inference_result)

    def test_save_csv_files(self):
        MODEL_ID = "sshleifer/tiny-gpt2"
        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_args = TensorFlowBenchmarkArguments(
                models=[MODEL_ID],
                inference=True,
                save_to_csv=True,
                sequence_lengths=[8],
                batch_sizes=[1],
                inference_time_csv_file=os.path.join(tmp_dir, "inf_time.csv"),
                inference_memory_csv_file=os.path.join(tmp_dir, "inf_mem.csv"),
                env_info_csv_file=os.path.join(tmp_dir, "env.csv"),
                multi_process=False,
            )
            benchmark = TensorFlowBenchmark(benchmark_args)
            benchmark.run()
            self.assertTrue(Path(os.path.join(tmp_dir, "inf_time.csv")).exists())
            self.assertTrue(Path(os.path.join(tmp_dir, "inf_mem.csv")).exists())
            self.assertTrue(Path(os.path.join(tmp_dir, "env.csv")).exists())

    def test_trace_memory(self):
        MODEL_ID = "sshleifer/tiny-gpt2"

        def _check_summary_is_not_empty(summary):
            self.assertTrue(hasattr(summary, "sequential"))
            self.assertTrue(hasattr(summary, "cumulative"))
            self.assertTrue(hasattr(summary, "current"))
            self.assertTrue(hasattr(summary, "total"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            benchmark_args = TensorFlowBenchmarkArguments(
                models=[MODEL_ID],
                inference=True,
                sequence_lengths=[8],
                batch_sizes=[1],
                log_filename=os.path.join(tmp_dir, "log.txt"),
                log_print=True,
                trace_memory_line_by_line=True,
                eager_mode=True,
                multi_process=False,
            )
            benchmark = TensorFlowBenchmark(benchmark_args)
            result = benchmark.run()
            _check_summary_is_not_empty(result.inference_summary)
            self.assertTrue(Path(os.path.join(tmp_dir, "log.txt")).exists())
