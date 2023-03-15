# coding=utf-8
# Copyright 2021 NVIDIA Corporation. All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
import argparse
import logging
import os
import time
import timeit

import datasets
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
import torch
from absl import logging as absl_logging
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from utils_qa import postprocess_qa_predictions

import transformers
from transformers import AutoTokenizer, EvalPrediction, default_data_collator, set_seed
from transformers.trainer_pt_utils import nested_concat, nested_truncate


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
absl_logger = absl_logging.get_absl_logger()
absl_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    required=True,
    help="Path to ONNX model: ",
)

parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model checkpoints and predictions will be written.",
)

# Other parameters

parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    required=True,
    help="Pretrained tokenizer name or path if not the same as model_name",
)

parser.add_argument(
    "--version_2_with_negative",
    action="store_true",
    help="If true, the SQuAD examples contain some that do not have an answer.",
)
parser.add_argument(
    "--null_score_diff_threshold",
    type=float,
    default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.",
)

parser.add_argument(
    "--max_seq_length",
    default=384,
    type=int,
    help=(
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    ),
)
parser.add_argument(
    "--doc_stride",
    default=128,
    type=int,
    help="When splitting up a long document into chunks, how much stride to take between chunks.",
)

parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")

parser.add_argument(
    "--n_best_size",
    default=20,
    type=int,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
)
parser.add_argument(
    "--max_answer_length",
    default=30,
    type=int,
    help=(
        "The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another."
    ),
)

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    required=True,
    help="The name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
)
parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision instead of 32-bit",
)
parser.add_argument(
    "--int8",
    action="store_true",
    help="Whether to use INT8",
)

args = parser.parse_args()

if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

logger.info("Training/evaluation parameters %s", args)

args.eval_batch_size = args.per_device_eval_batch_size

INPUT_SHAPE = (args.eval_batch_size, args.max_seq_length)

# TRT Engine properties
STRICT_TYPES = True

engine_name = "temp_engine/bert-fp32.engine"
if args.fp16:
    engine_name = "temp_engine/bert-fp16.engine"
if args.int8:
    engine_name = "temp_engine/bert-int8.engine"

# import ONNX file
if not os.path.exists("temp_engine"):
    os.makedirs("temp_engine")

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
    network, TRT_LOGGER
) as parser:
    with open(args.onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

    # Query input names and shapes from parsed TensorRT network
    network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
    input_names = [_input.name for _input in network_inputs]  # ex: ["actual_input1"]

    with builder.create_builder_config() as config:
        config.max_workspace_size = 1 << 50
        if STRICT_TYPES:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if args.int8:
            config.set_flag(trt.BuilderFlag.INT8)
        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)
        for i in range(len(input_names)):
            profile.set_shape(input_names[i], INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
        engine = builder.build_engine(network, config)

        # serialize_engine and store in file (can be directly loaded and deserialized):
        with open(engine_name, "wb") as f:
            f.write(engine.serialize())


# run inference with TRT
def model_infer(inputs, context, d_inputs, h_output0, h_output1, d_output0, d_output1, stream):
    input_ids = np.asarray(inputs["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(inputs["attention_mask"], dtype=np.int32)
    token_type_ids = np.asarray(inputs["token_type_ids"], dtype=np.int32)

    # Copy inputs
    cuda.memcpy_htod_async(d_inputs[0], input_ids.ravel(), stream)
    cuda.memcpy_htod_async(d_inputs[1], attention_mask.ravel(), stream)
    cuda.memcpy_htod_async(d_inputs[2], token_type_ids.ravel(), stream)
    # start time
    start_time = time.time()
    # Run inference
    context.execute_async(
        bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output0), int(d_output1)], stream_handle=stream.handle
    )
    # Transfer predictions back from GPU
    cuda.memcpy_dtoh_async(h_output0, d_output0, stream)
    cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
    # Synchronize the stream and take time
    stream.synchronize()
    # end time
    end_time = time.time()
    infer_time = end_time - start_time
    outputs = (h_output0, h_output1)
    # print(outputs)
    return outputs, infer_time


# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
accelerator = Accelerator()
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
# 'text' is found. You can easily tweak this behavior (see below).
if args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
else:
    raise ValueError("Evaluation requires a dataset name")
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Preprocessing the datasets.
# Preprocessing is slighlty different for training and evaluation.

column_names = raw_datasets["validation"].column_names

question_column_name = "question" if "question" in column_names else column_names[0]
context_column_name = "context" if "context" in column_names else column_names[1]
answer_column_name = "answers" if "answers" in column_names else column_names[2]

# Padding side determines if we do (question|context) or (context|question).
pad_on_right = tokenizer.padding_side == "right"

if args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )

max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)


# Validation preprocessing
def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


eval_examples = raw_datasets["validation"]
# Validation Feature Creation
eval_dataset = eval_examples.map(
    prepare_validation_features,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not args.overwrite_cache,
    desc="Running tokenizer on validation dataset",
)

data_collator = default_data_collator

eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
eval_dataloader = DataLoader(
    eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
)


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")

# Evaluation!
logger.info("Loading ONNX model %s for evaluation", args.onnx_model_path)
with open(engine_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(
    f.read()
) as engine, engine.create_execution_context() as context:
    # setup for TRT inferrence
    for i in range(len(input_names)):
        context.set_binding_shape(i, INPUT_SHAPE)
    assert context.all_binding_shapes_specified

    def binding_nbytes(binding):
        return trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize

    # Allocate device memory for inputs and outputs.
    d_inputs = [cuda.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)]

    # Allocate output buffer
    h_output0 = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
    h_output1 = cuda.pagelocked_empty(tuple(context.get_binding_shape(4)), dtype=np.float32)
    d_output0 = cuda.mem_alloc(h_output0.nbytes)
    d_output1 = cuda.mem_alloc(h_output1.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    total_time = 0.0
    niter = 0
    start_time = timeit.default_timer()

    all_preds = None
    for step, batch in enumerate(eval_dataloader):
        outputs, infer_time = model_infer(batch, context, d_inputs, h_output0, h_output1, d_output0, d_output1, stream)
        total_time += infer_time
        niter += 1

        start_logits, end_logits = outputs
        start_logits = torch.tensor(start_logits)
        end_logits = torch.tensor(end_logits)

        # necessary to pad predictions and labels for being gathered
        start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
        end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        logits = (accelerator.gather(start_logits).cpu().numpy(), accelerator.gather(end_logits).cpu().numpy())
        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)

    if all_preds is not None:
        all_preds = nested_truncate(all_preds, len(eval_dataset))

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(eval_dataset))
    # Inference time from TRT
    logger.info("Average Inference Time = {:.3f} ms".format(total_time * 1000 / niter))
    logger.info("Total Inference Time =  {:.3f} ms".format(total_time * 1000))
    logger.info("Total Number of Inference =  %d", niter)

prediction = post_processing_function(eval_examples, eval_dataset, all_preds)
eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
logger.info(f"Evaluation metrics: {eval_metric}")
