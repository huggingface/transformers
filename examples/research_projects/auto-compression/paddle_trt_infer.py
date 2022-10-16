import argparse
import os
import time

import numpy as np
from datasets import load_dataset

import evaluate
import paddle
from accelerate import Accelerator
from paddle import inference
from paddle.io import DataLoader
from transformers import AutoTokenizer, default_data_collator


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The directory or name of model.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu", "xpu"],
        help="Device selected for inference.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.",
    )
    parser.add_argument(
        "--use_trt",
        action="store_true",
        help="Whether to use inference engin TensorRT.",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Whether to test performance.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Whether to use int8 inference.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use float16 inference.",
    )
    args = parser.parse_args()
    return args


def reader_wrapper(reader, input_name=["x0", "x1", "x2"]):
    def gen():
        feed_data = []
        data_names = list(reader.dataset[0].keys())
        for data in reader:
            for idx in range(len(input_name)):
                feed_data.append(np.array(data[data_names[idx]]).astype("int64"))
            if "labels" in feed_data:
                feed_data.append(np.array(feed_data["labels"]).reshape(-1, 1))
            yield feed_data

    return gen


class Predictor(object):
    def __init__(self, predictor, input_handles, output_handles):
        self.predictor = predictor
        self.input_handles = input_handles
        self.output_handles = output_handles

    @classmethod
    def create_predictor(cls, args):
        config = paddle.inference.Config(
            args.model_name_or_path + "model.pdmodel", args.model_name_or_path + "model.pdiparams"
        )
        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
            cls.device = paddle.set_device("gpu")
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            cls.device = paddle.set_device("cpu")
        elif args.device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)
        if args.use_trt:
            if args.int8:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Int8,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False,
                )
            elif args.fp16:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Half,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False,
                )
            else:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=inference.PrecisionType.Float32,
                    max_batch_size=args.batch_size,
                    min_subgraph_size=5,
                    use_static=False,
                    use_calib_mode=False,
                )
            print("Enable TensorRT is: {}".format(config.tensorrt_engine_enabled()))

            dynamic_shape_file = os.path.join(args.model_name_or_path, "dynamic_shape.txt")
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
                print("trt set dynamic shape done!")
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape... Please eval again to get real result in TensorRT")
                # sys.exit()

        predictor = paddle.inference.create_predictor(config)

        input_handles = [predictor.get_input_handle(name) for name in predictor.get_input_names()]
        output_handles = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

        return cls(predictor, input_handles, output_handles)

    def predict(self, dataset, collate_fn, args):
        accelerator = Accelerator()
        data_loader = DataLoader(
            dataset, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=True
        )
        data_loader = reader_wrapper(data_loader, ["x0", "x2", "x1", "labels"])
        if args.perf:
            for i, data in enumerate(data_loader):
                for input_field, input_handle in zip(data, self.input_handles):
                    input_handle.copy_from_cpu(
                        input_field.numpy() if isinstance(input_field, paddle.Tensor) else input_field
                    )

                self.predictor.run()

                output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]

                if i > args.perf_warmup_steps:
                    break

            time1 = time.time()
            for i, data in enumerate(data_loader()):
                for input_field, input_handle in zip(data, self.input_handles):
                    input_handle.copy_from_cpu(
                        input_field.numpy() if isinstance(input_field, paddle.Tensor) else input_field
                    )
                self.predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]

            sequences_num = i * args.batch_size
            print("task name: %s, time: %s qps/s, " % (args.task_name, sequences_num / (time.time() - time1)))

        else:
            if args.task_name is not None:
                metric = evaluate.load("glue", args.task_name)
            else:
                metric = evaluate.load("accuracy")

            for i, data in enumerate(data_loader()):
                print(data)
                label = data[-1]
                data = data[:-1]
                print("self.input_handles: ", self.input_handles)
                for input_field, input_handle in zip(data, self.input_handles):
                    input_handle.copy_from_cpu(
                        input_field.numpy() if isinstance(input_field, paddle.Tensor) else input_field
                    )
                self.predictor.run()
                output = [output_handle.copy_to_cpu() for output_handle in self.output_handles]

                predictions = (
                    np.argmax(np.array(output[0]), axis=-1) if not is_regression else np.squeeze(np.array(output[0]))
                )
                predictions, references = accelerator.gather(predictions, np.squeeze(label))
                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )
            eval_metric = metric.compute()
            print("task name: {}, metric: {}, \n".format(args.task_name, eval_metric))


def paddle_data_collator(features):
    batch = default_data_collator(features, return_tensors="np")
    return batch


def main():
    paddle.seed(42)
    args = parse_args()

    predictor = Predictor.create_predictor(args)

    args.task_name = args.task_name.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    raw_datasets = load_dataset("glue", args.task_name)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]
    global is_regression
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    label_to_id = None
    if args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    data_collator = paddle_data_collator

    predictor.predict(eval_dataset, data_collator, args)


if __name__ == "__main__":
    main()
