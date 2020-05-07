# coding=utf-8
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, DistilBert, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    # Only tensorboardX supports writing directly to gs://
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_sampler(dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


def train(args, train_dataset, model, tokenizer, disable_logging=False):
    """ Train the model """
    if xm.is_master_ordinal():
        # Only master writes to Tensorboard
        tb_writer = SummaryWriter(args.tensorboard_logdir)

    train_sampler = get_sampler(train_dataset)
    dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataloader) * args.train_batch_size)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per TPU core = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        (args.train_batch_size * args.gradient_accumulation_steps * xm.xrt_world_size()),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    loss = None
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=disable_logging)
    set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in train_iterator:
        # tpu-comment: Get TPU parallel loader which sends data to TPU in background.
        train_dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", total=len(dataloader), disable=disable_logging)
        for step, batch in enumerate(epoch_iterator):

            # Save model checkpoint.
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                logger.info("Saving model checkpoint to %s", output_dir)

                if xm.is_master_ordinal():
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))

                # Barrier to wait for saving checkpoint.
                xm.rendezvous("mid_training_checkpoint")
                # model.save_pretrained needs to be called by all ordinals
                model.save_pretrained(output_dir)

            model.train()
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                xm.optimizer_step(optimizer)
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics.
                    results = {}
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, disable_logging=disable_logging)
                    loss_scalar = loss.item()
                    logger.info(
                        "global_step: {global_step}, lr: {lr:.6f}, loss: {loss:.3f}".format(
                            global_step=global_step, lr=scheduler.get_lr()[0], loss=loss_scalar
                        )
                    )
                    if xm.is_master_ordinal():
                        # tpu-comment: All values must be in CPU and not on TPU device
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", loss_scalar, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if xm.is_master_ordinal():
        tb_writer.close()
    return global_step, loss.item()


def evaluate(args, model, tokenizer, prefix="", disable_logging=False):
    """Evaluate the model"""
    if xm.is_master_ordinal():
        # Only master writes to Tensorboard
        tb_writer = SummaryWriter(args.tensorboard_logdir)

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        eval_sampler = get_sampler(eval_dataset)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, shuffle=False)
        eval_dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataloader) * args.eval_batch_size)
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=disable_logging):
            model.eval()

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
                outputs = model(**inputs)
                batch_eval_loss, logits = outputs[:2]

                eval_loss += batch_eval_loss
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
        preds = xm.mesh_reduce("eval_preds", preds, np.concatenate)
        out_label_ids = xm.mesh_reduce("eval_out_label_ids", out_label_ids, np.concatenate)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        results["eval_loss"] = eval_loss.item()

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        if xm.is_master_ordinal():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))
                    writer.write("%s = %s\n" % (key, str(results[key])))
                    tb_writer.add_scalar(f"{eval_task}/{key}", results[key])

    if args.metrics_debug:
        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        xm.master_print(met.metrics_report())

    if xm.is_master_ordinal():
        tb_writer.close()

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if not xm.is_master_ordinal():
        xm.rendezvous("load_and_cache_examples")

    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    # Load data features from cache or dataset file
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    if xm.is_master_ordinal():
        xm.rendezvous("load_and_cache_examples")

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            (
                "Output directory ({}) already exists and is not empty." " Use --overwrite_output_dir to overcome."
            ).format(args.output_dir)
        )

    # tpu-comment: Get TPU/XLA Device
    args.device = xm.xla_device()

    # Setup logging
    logging.basicConfig(
        format="[xla:{}] %(asctime)s - %(levelname)s - %(name)s -   %(message)s".format(xm.get_ordinal()),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    disable_logging = False
    if not xm.is_master_ordinal() and args.only_log_master:
        # Disable all non-master loggers below CRITICAL.
        logging.disable(logging.CRITICAL)
        disable_logging = True
    logger.warning("Process rank: %s, device: %s, num_cores: %s", xm.get_ordinal(), args.device, args.num_cores)

    # Set seed to have same initialization
    set_seed(args.seed)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if not xm.is_master_ordinal():
        xm.rendezvous(
            "download_only_once"
        )  # Make sure only the first process in distributed training will download model & vocab

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        xla_device=True,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if xm.is_master_ordinal():
        xm.rendezvous("download_only_once")

    # Send model to TPU/XLA device.
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        # Train the model.
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, disable_logging=disable_logging)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if xm.is_master_ordinal():
            # Save trained model.
            # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

            # Create output directory if needed
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            tokenizer.save_pretrained(args.output_dir)
            # Good practice: save your training arguments together with the trained.
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        xm.rendezvous("post_training_checkpoint")
        # model.save_pretrained needs to be called by all ordinals
        model.save_pretrained(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix, disable_logging=disable_logging)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # TPU Parameters
    parser.add_argument("--num_cores", default=8, type=int, help="Number of TPU cores to use (1 or 8).")
    parser.add_argument("--metrics_debug", action="store_true", help="Whether to print debug metrics.")

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded and features file generated",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--train_batch_size", default=8, type=int, help="Per core batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Per core batch size for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--tensorboard_logdir", default="./runs", type=str, help="Where to write tensorboard metrics.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X update steps.")
    parser.add_argument("--only_log_master", action="store_true", help="Whether to log only from each hosts master.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X update steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()


def _mp_fn(rank, args):
    main(args)


def main_cli():
    args = get_args()
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)


if __name__ == "__main__":
    main_cli()
