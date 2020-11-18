# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# All the modifications on top of 
# https://github.com/W4ngatang/transformers/blob/superglue/examples/run_superglue.py 
# are under the MIT license by Microsoft.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on SuperGLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
""" This is based on Alex Wang's transformers repository, superglue branch. https://github.com/W4ngatang/transformers 
    https://github.com/W4ngatang/transformers/blob/superglue/examples/run_superglue.py """


import sys
import argparse
import glob
import json
import logging
import os
import random
import time
from queue import PriorityQueue
from heapq import heappush, heappop

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch._utils import ExceptionWrapper
from multiprocessing import Process, Queue
from tqdm import tqdm
from torch.nn import MSELoss, CosineSimilarity

from data.superglue import superglue_compute_metrics as compute_metrics
from data.superglue import superglue_convert_examples_to_features as convert_examples_to_features
from data.superglue import superglue_output_modes as output_modes
from data.superglue import superglue_processors as processors
from data.superglue import superglue_tasks_metrics as task_metrics
from data.superglue import superglue_tasks_num_spans as task_spans

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertForSpanClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaForSpanClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
)


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(config_map.keys())
        for config_map in (
            BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
            ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        )
    ),
    (),
)

# only BERT and RoBERTa mdoels are supported
MODEL_CLASSES = {
    "bert": (
        BertConfig,
        BertTokenizer,
        {"classification": BertForSequenceClassification, "span_classification": BertForSpanClassification},
    ),
    "roberta": (
        RobertaConfig,
        RobertaTokenizer,
        {"classification": RobertaForSequenceClassification, "span_classification": RobertaForSpanClassification},
    ),
}

TASK2FILENAME = {
    "boolq": "BoolQ.jsonl",
    "cb": "CB.jsonl",
    "copa": "COPA.jsonl",
    "multirc": "MultiRC.jsonl",
    "record": "ReCoRD.jsonl",
    "rte": "RTE.jsonl",
    "wic": "WiC.jsonl",
    "wsc": "WSC.jsonl",
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:  # number of training steps = number of epochs * number of batches
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * t_total)

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
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        logger.info("Training with fp16.")

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    # )
    train_iterator = range(epochs_trained, int(args.num_train_epochs))

    set_seed(args)  # Added here for reproductibility
    best_val_metric = None
    for epoch_n in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                results = None
                logs = {}
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if (
                        args.local_rank == -1 and args.log_evaluate_during_training and results is None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = evaluate(args, args.task_name, model, tokenizer, use_tqdm=False)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["avg_loss_since_last_log"] = loss_scalar
                    logging_loss = tr_loss

                    logging.info(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.eval_and_save_steps > 0
                    and global_step % args.eval_and_save_steps == 0
                ):
                    # evaluate
                    results, _, _ = evaluate(args, args.task_name, model, tokenizer, use_tqdm=False)
                    for key, value in results.items():
                        logs[f"eval_{key}"] = value
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # save
                    if args.save_only_best:
                        output_dirs = []
                    else:
                        output_dirs = [os.path.join(args.output_dir, f"checkpoint-{global_step}")]
                    curr_val_metric = results[task_metrics[args.task_name]]
                    if best_val_metric is None or curr_val_metric > best_val_metric:
                        # check if best model so far
                        logger.info("Congratulations, best model so far!")
                        output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))
                        best_val_metric = curr_val_metric

                    for output_dir in output_dirs:
                        # in each dir, save model, tokenizer, args, optimizer, scheduler
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", output_dir)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("\tSaved model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step


def distill(args, train_dataset, teacher_model, student_model, tokenizer):
    """ Train the model with distillation
        Assumes that teacher and student models share same token embedding layer.
        So, the same data is loaded and fed to both teacher and student models.

        This function code is base on TinyBERT implementation 
        (https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).
    """

    ############################################################################################
    # no multi-node distributed trainig, continued training and fp16 support for KD
    ############################################################################################

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) # no multi-node
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:  # number of training steps = number of epochs * number of batches
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    num_warmup_steps = int(args.warmup_ratio * t_total)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # layer numbers of teacher and student
    teacher_layer_num = teacher_model.config.num_hidden_layers
    student_layer_num = student_model.config.num_hidden_layers

    # multi-gpu training
    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)

    # Prepare loss functions
    loss_mse = MSELoss()
    loss_cs = CosineSimilarity(dim=2)
    loss_cs_att = CosineSimilarity(dim=3)

    def soft_cross_entropy(predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).sum(dim=-1).mean()

    # Distill!
    logger.info("***** Running distillation training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    student_model.zero_grad()
    train_iterator = range(epochs_trained, int(args.num_train_epochs))

    set_seed(args)  # Added here for reproductibility
    best_val_metric = None
    for epoch_n in train_iterator:
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_cls_loss = 0.
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            student_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            # student model output
            outputs_student = student_model(output_attentions=True, output_hidden_states=True, **inputs)

            # teacher model output
            teacher_model.eval() # set teacher as eval mode
            with torch.no_grad():
                outputs_teacher = teacher_model(output_attentions=True, output_hidden_states=True, **inputs)
            
            # Knowledge Distillation loss
            # 1) logits distillation
            kd_loss = soft_cross_entropy(outputs_student[1], outputs_teacher[1])
            loss = kd_loss
            tr_cls_loss += loss.item()

            # 2) embedding and last hidden state distillation
            if args.state_loss_ratio > 0.0:
                teacher_reps = outputs_teacher[2]
                student_reps = outputs_student[2]

                new_teacher_reps = [teacher_reps[0], teacher_reps[teacher_layer_num]]
                new_student_reps = [student_reps[0], student_reps[student_layer_num]]
                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    # cosine similarity loss
                    if args.state_distill_cs:
                        tmp_loss = 1.0 - loss_cs(student_rep, teacher_rep).mean()
                    # MSE loss
                    else:
                        tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                loss += args.state_loss_ratio * rep_loss
                tr_rep_loss += rep_loss.item()

            # 3) Attentions distillation
            if args.att_loss_ratio > 0.0:
                teacher_atts = outputs_teacher[3]
                student_atts = outputs_student[3]

                assert teacher_layer_num == len(teacher_atts)
                assert student_layer_num == len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                              teacher_att)
                    tmp_loss = 1.0 - loss_cs_att(student_att, teacher_att).mean()
                    att_loss += tmp_loss

                loss += args.att_loss_ratio * att_loss
                tr_att_loss += att_loss.item()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # back propagate
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                student_model.zero_grad()
                global_step += 1

                # change to evaluation mode
                student_model.eval()
                results = None
                logs = {}
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if (
                        args.local_rank == -1 and args.log_evaluate_during_training and results is None
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = evaluate(args, args.task_name, student_model, tokenizer, use_tqdm=False)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["avg_loss_since_last_log"] = loss_scalar
                    logs['cls_loss'] = cls_loss
                    logs['att_loss'] = att_loss
                    logs['rep_loss'] = rep_loss
                    logging_loss = tr_loss

                    logging.info(json.dumps({**logs, **{"step": global_step}}))

                if (
                    args.local_rank in [-1, 0]
                    and args.eval_and_save_steps > 0
                    and global_step % args.eval_and_save_steps == 0
                ):
                    # evaluate
                    results, _, _ = evaluate(args, args.task_name, student_model, tokenizer, use_tqdm=False)
                    for key, value in results.items():
                        logs[f"eval_{key}"] = value
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

                    # save
                    if args.save_only_best:
                        output_dirs = []
                    else:
                        output_dirs = [os.path.join(args.output_dir, f"checkpoint-{global_step}")]
                    curr_val_metric = results[task_metrics[args.task_name]]
                    if best_val_metric is None or curr_val_metric > best_val_metric or args.save_latest:
                        # check if best model so far
                        logger.info("Congratulations, best model so far!")
                        output_dirs.append(os.path.join(args.output_dir, "checkpoint-best"))
                        best_val_metric = curr_val_metric

                    for output_dir in output_dirs:
                        # in each dir, save model, tokenizer, args, optimizer, scheduler
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            student_model.module if hasattr(student_model, "module") else student_model
                        )  # Take care of distributed/parallel training
                        logger.info("Saving model checkpoint to %s", output_dir)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("\tSaved model checkpoint to %s", output_dir)
                
                # change student model back to train mode
                student_model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, task_name, model, tokenizer, split="dev", prefix="", use_tqdm=True):

    results = {}
    if task_name == "record":
        eval_dataset, eval_answers = load_and_cache_examples(args, task_name, tokenizer, split=split)
    else:
        eval_dataset = load_and_cache_examples(args, task_name, tokenizer, split=split)
        eval_answers = None

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.fp16:
        model.half()

    args.eval_batch_size = args.per_instance_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        guids = batch[-1]

        max_seq_length = batch[0].size(1)
        if args.use_fixed_seq_length: # no dynamic sequence length
            batch_seq_length = max_seq_length
        else:
            batch_seq_length = torch.max(batch[-2], 0)[0].item()

        if batch_seq_length < max_seq_length:
            inputs = {"input_ids": batch[0][:,:batch_seq_length].contiguous(),
                      "attention_mask": batch[1][:,:batch_seq_length].contiguous(),
                      "labels": batch[3]}
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2][:,:batch_seq_length].contiguous() if args.model_type 
                        in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        else:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.output_mode == "span_classification":
                inputs["spans"] = batch[4]
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    ex_ids = np.concatenate(ex_ids, axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode in ["classification", "span_classification"] and args.task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        # don't have access to test labels, so skip evaluating on them
        # NB(AW): forcing evaluation on ReCoRD on test (no labels) will error
        result = compute_metrics(task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** {split} results: {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids

def sort_by_importance(weight, bias, importance, num_instances, stride):
    importance_ordered = []
    i = 0
    for heads in importance:
        heappush(importance_ordered, (-heads, i))
        i += 1
    sorted_weight_to_concat = None
    sorted_bias_to_concat = None
    i = 0
    while importance_ordered and i < num_instances:
        head_to_add = heappop(importance_ordered)[1]
        if sorted_weight_to_concat is None:
            sorted_weight_to_concat = (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        else:
            sorted_weight_to_concat += (weight.narrow(0, int(head_to_add * stride), int(stride)), )
        if bias is not None:
            if sorted_bias_to_concat is None:
                sorted_bias_to_concat = (bias.narrow(0, int(head_to_add * stride), int(stride)), )
            else:
                sorted_bias_to_concat += (bias.narrow(0, int(head_to_add * stride), int(stride)), )
        i += 1
    return torch.cat(sorted_weight_to_concat), torch.cat(sorted_bias_to_concat) if sorted_bias_to_concat is not None else None

def prune_rewire(args, task_name, model, tokenizer, prefix="", use_tqdm=True):
    split="dev"
    results = {}
    if args.n_gpu > 1:
        args.n_gpu = 1 # only 1 GPU is supported for pruning
        args.device = 0
    if task_name == "record":
        eval_dataset, eval_answers = load_and_cache_examples(args, task_name, tokenizer, split=split)
    else:
        eval_dataset = load_and_cache_examples(args, task_name, tokenizer, split=split)
        eval_answers = None

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_instance_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # get the model ffn weights and biases
    inter_weights = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size).to(args.device)
    inter_biases = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(args.device)
    output_weights = torch.zeros(model.config.num_hidden_layers, model.config.hidden_size, model.config.intermediate_size).to(args.device)

    layers = model.base_model.encoder.layer
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(args.device)
    ffn_importance = torch.zeros(model.config.num_hidden_layers, model.config.intermediate_size).to(args.device)
    for layer_num in range(model.config.num_hidden_layers):
        inter_weights[layer_num] = layers._modules[str(layer_num)].intermediate.dense.weight.detach().to(args.device)
        inter_biases[layer_num] = layers._modules[str(layer_num)].intermediate.dense.bias.detach().to(args.device)
        output_weights[layer_num] = layers._modules[str(layer_num)].output.dense.weight.detach().to(args.device)

    head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # Eval!
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ex_ids = None
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
    tot_tokens = 0.0
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        guids = batch[-1]

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if args.output_mode == "span_classification":
            inputs["spans"] = batch[4]
        if args.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        outputs = model(output_attentions=True, **inputs, head_mask=head_mask)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()

        # TODO accumulate? absolute value sum?
        tmp_eval_loss.backward()

        # collect attention confidence scores
        head_importance += head_mask.grad.abs().detach()

        # collect gradients of linear layers
        for layer_num in range(model.config.num_hidden_layers):
            ffn_importance[layer_num] += torch.abs(
                torch.sum(layers._modules[str(layer_num)].intermediate.dense.weight.grad.detach()*inter_weights[layer_num], 1) 
                + layers._modules[str(layer_num)].intermediate.dense.bias.grad.detach()*inter_biases[layer_num])

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ex_ids = [guids.detach().cpu().numpy()]
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ex_ids.append(guids.detach().cpu().numpy())

    head_importance /= tot_tokens

    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    # rewire the network
    head_importance = head_importance.cpu()
    ffn_importance = ffn_importance.cpu()
    num_heads = model.config.num_attention_heads
    head_size = model.config.hidden_size / num_heads
    for layer_num in range(model.config.num_hidden_layers):
        # load query, key, value weights
        query_weight = layers._modules[str(layer_num)].attention.self.query.weight
        query_bias = layers._modules[str(layer_num)].attention.self.query.bias
        key_weight = layers._modules[str(layer_num)].attention.self.key.weight
        key_bias = layers._modules[str(layer_num)].attention.self.key.bias
        value_weight = layers._modules[str(layer_num)].attention.self.value.weight
        value_bias = layers._modules[str(layer_num)].attention.self.value.bias

        # sort query, key, value based on the confidence scores
        query_weight, query_bias = sort_by_importance(query_weight,
            query_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.query.weight = torch.nn.Parameter(query_weight)
        layers._modules[str(layer_num)].attention.self.query.bias = torch.nn.Parameter(query_bias)
        key_weight, key_bias = sort_by_importance(key_weight,
            key_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.key.weight = torch.nn.Parameter(key_weight)
        layers._modules[str(layer_num)].attention.self.key.bias = torch.nn.Parameter(key_bias)
        value_weight, value_bias = sort_by_importance(value_weight,
            value_bias,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        layers._modules[str(layer_num)].attention.self.value.weight = torch.nn.Parameter(value_weight)
        layers._modules[str(layer_num)].attention.self.value.bias = torch.nn.Parameter(value_bias)

        # output matrix
        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].attention.output.dense.weight.transpose(0, 1),
            None,
            head_importance[layer_num],
            args.target_num_heads,
            head_size)
        weight_sorted = weight_sorted.transpose(0, 1)
        layers._modules[str(layer_num)].attention.output.dense.weight = torch.nn.Parameter(weight_sorted)

        weight_sorted, bias_sorted = sort_by_importance(
            layers._modules[str(layer_num)].intermediate.dense.weight,
            layers._modules[str(layer_num)].intermediate.dense.bias, 
            ffn_importance[layer_num],
            args.target_ffn_dim,
            1)
        layers._modules[str(layer_num)].intermediate.dense.weight = torch.nn.Parameter(weight_sorted)
        layers._modules[str(layer_num)].intermediate.dense.bias = torch.nn.Parameter(bias_sorted)

        # ffn output matrix input side
        weight_sorted, _ = sort_by_importance(
            layers._modules[str(layer_num)].output.dense.weight.transpose(0, 1),
            None, 
            ffn_importance[layer_num],
            args.target_ffn_dim,
            1)
        weight_sorted = weight_sorted.transpose(0, 1)
        layers._modules[str(layer_num)].output.dense.weight = torch.nn.Parameter(weight_sorted)

    # save pruned model
    from pathlib import Path
    Path(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim))).mkdir(exist_ok=True)

    model.config.hidden_act = 'relu'    # use ReLU activation for the pruned models.
    model.config.num_attention_heads = min([num_heads, args.target_num_heads])
    model.config.intermediate_size = layers._modules['0'].intermediate.dense.weight.size(0)
    model.config.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))

    model.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))
    tokenizer.save_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)))

    ex_ids = np.concatenate(ex_ids, axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode in ["classification", "span_classification"] and args.task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        # don't have access to test labels, so skip evaluating on them
        # NB(AW): forcing evaluation on ReCoRD on test (no labels) will error
        result = compute_metrics(task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** {split} results: {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids

def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    """Copied from psutil code, and modified to fix an error."""
    return sys.modules['psutil'].PROCFS_PATH

def cpu_count_physical():
    """Return the number of physical cores in the system."""
    """Copied from psutil code, and modified to fix an error."""
    # Method #1 doesn't work for some dual socket topologies.
    # # Method #1
    # core_ids = set()
    # for path in glob.glob(
    #         "/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
    #     with open_binary(path) as f:
    #         core_ids.add(int(f.read()))
    # result = len(core_ids)
    # if result != 0:
    #     return result

    # Method #2
    physical_logical_mapping = {}
    mapping = {}
    current_info = {}
    with open('%s/cpuinfo' % get_procfs_path(), "rb") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                # print(current_info)
                # new section
                if (b'physical id' in current_info and
                        b'cpu cores' in current_info):
                    mapping[current_info[b'physical id']] = \
                        current_info[b'cpu cores']
                if (b'physical id' in current_info and
                        b'core id' in current_info and
                        b'processor' in current_info):
                    # print(current_info[b'physical id'] * 1000 + current_info[b'core id'])
                    if current_info[b'physical id'] * 1000 + current_info[b'core id'] not in physical_logical_mapping:
                        physical_logical_mapping[current_info[b'physical id'] * 1000 + current_info[b'core id']] = current_info[b'processor']
                current_info = {}
            else:
                # ongoing section
                if (line.startswith(b'physical id') or
                        line.startswith(b'cpu cores') or
                        line.startswith(b'core id') or
                        line.startswith(b'processor')):
                    key, value = line.split(b'\t:', 1)
                    current_info[key.rstrip()] = int(value.rstrip())

    physical_processor_ids = []
    for key in sorted(physical_logical_mapping.keys()):
        physical_processor_ids.append(physical_logical_mapping[key])

    result = sum(mapping.values())
    # return result or None  # mimic os.cpu_count()
    return result, physical_processor_ids

input_queue = Queue()
result_queue = Queue()

def evaluate_ort_parallel(args, task_name, onnx_session_options, tokenizer, split="dev", prefix="", use_tqdm=True):
    results = {}
    if task_name == "record":
        eval_dataset, eval_answers = load_and_cache_examples(args, task_name, tokenizer, split=split)
    else:
        eval_dataset = load_and_cache_examples(args, task_name, tokenizer, split=split)
        eval_answers = None

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_instance_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-core eval
    # import psutil
    # num_cores = psutil.cpu_count(logical=False)
    num_cores, processor_list = cpu_count_physical()
    # print(processor_list)
    threads_per_instance = args.threads_per_instance
    if args.threads_per_instance < 0:
        threads_per_instance = num_cores
    num_instances = (int) (num_cores / threads_per_instance)

    assert num_instances <= num_cores

    def _worker_proc(input_queue, results_q):
        from onnxruntime import ExecutionMode, InferenceSession, SessionOptions

        onnx_session = InferenceSession(args.model_name_or_path + '/model.onnx', onnx_session_options)
        while True:
            try:
                input = input_queue.get()
                if input is None:   # exit
                    break
                t0 = time.time()
                output = onnx_session.run(None, input[1])
                results_q.put((input[0], output, input[2], input[3], time.time() - t0))
            except Exception:
                output = ExceptionWrapper(
                    where="in guid {}".format(-1))
                results_q.put((-1, output))
                assert False

    # create processes
    for i in range(num_instances):
        p = Process(target=_worker_proc, args=(input_queue, result_queue))
        p.start()
        # pin processes to cores
        lpids = ''
        for j in range(threads_per_instance):
            lpids += str(processor_list[i*threads_per_instance + j])
            if j < threads_per_instance - 1:
                lpids += ','
        os.system("taskset -p -c " + lpids + " " + str(p.pid))

    # Eval!
    wallclock_start = time.time()
    logger.info(f"***** Running evaluation: {prefix} on {task_name} {split} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    preds = None
    out_label_ids = None
    ex_ids = None
    batch_id = 0
    for batch in eval_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        guids = batch[-1]

        labels_input = batch[3]

        batch_seq_length = torch.max(batch[-2], 0)[0].item()

        inputs = {'input_ids': batch[0][:,:batch_seq_length].contiguous().cpu().detach().numpy(),
            'attention_mask': batch[1][:,:batch_seq_length].contiguous().cpu().detach().numpy()}
        if args.model_type in ["bert", "xlnet", "albert"]:
            inputs["token_type_ids"] = batch[2][:,:batch_seq_length].contiguous().cpu().detach().numpy()

        # Queue queries into the Q which ONNX runtime session can consume
        input_queue.put((batch_id, inputs, guids.detach().cpu().numpy(), labels_input.detach().cpu().numpy()))
        batch_id += 1

    # exit signal at the end of the Q
    for _ in range(num_instances):
        input_queue.put(None)

    # It's a little bit slower with heappush, heappop. So, let's just use PQ.
    result_tmp_q = PriorityQueue(batch_id)
    while not result_tmp_q.full():
        if not result_queue.empty():
            output_with_id = result_queue.get()
            result_tmp_q.put((output_with_id[0], output_with_id[1:]))
        else:
            time.sleep(.1)

    total_time = 0.
    while not result_tmp_q.empty():
        output_with_id = result_tmp_q.get()
        logits = output_with_id[1][0]
        guids = output_with_id[1][1]
        input_lables = output_with_id[1][2]
        total_time += output_with_id[1][3]

        if preds is None:
            preds = logits[0]
            out_label_ids = input_lables
            ex_ids = [guids]
        else:
            preds = np.append(preds, logits[0], axis=0)
            out_label_ids = np.append(out_label_ids, input_lables, axis=0)
            ex_ids.append(guids)

    assert len(ex_ids) == batch_id
    print("############## Average latency: ", str(total_time / batch_id))
    print("############## Total time spent (wallclock time): ", str(time.time() - wallclock_start))

    ex_ids = np.concatenate(ex_ids, axis=0)
    if args.output_mode in ["classification", "span_classification"] and args.task_name not in ["record"]:
        preds = np.argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    if split != "test":
        # don't have access to test labels, so skip evaluating on them
        # NB(AW): forcing evaluation on ReCoRD on test (no labels) will error
        result = compute_metrics(task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
        results.update(result)
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info(f"***** {split} results: {prefix} *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, preds, ex_ids

def load_and_cache_examples(args, task, tokenizer, split="train"):
    if args.local_rank not in [-1, 0] and split not in ["dev", "test"]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_tensors_file = os.path.join(
        args.data_dir,
        "tensors_{}_{}_{}_{}_{}".format(
            split, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length), str(task), str(args.do_lower_case)
        ),
    )
    if os.path.exists(cached_tensors_file) and not args.overwrite_cache:
        logger.info("Loading tensors from cached file %s", cached_tensors_file)
        start_time = time.time()
        dataset = torch.load(cached_tensors_file)
        logger.info("\tFinished loading tensors")
        logger.info(f"\tin {time.time() - start_time}s")

    else:
        # no cached tensors, process data from scratch
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if split == "train":
            get_examples = processor.get_train_examples
        elif split == "dev":
            get_examples = processor.get_dev_examples
        elif split == "test":
            get_examples = processor.get_test_examples
        examples = get_examples(args.data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        logger.info("\tFinished creating features")
        if args.local_rank == 0 and split not in ["dev", "train"]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        logger.info("Converting features into tensors")
        all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
        if output_mode in ["classification", "span_classification"]:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        if output_mode in ["span_classification"]:
            # all_starts = torch.tensor([[s[0] for s in f.span_locs] for f in features], dtype=torch.long)
            # all_ends = torch.tensor([[s[1] for s in f.span_locs] for f in features], dtype=torch.long)
            all_spans = torch.tensor([f.span_locs for f in features])
            dataset = TensorDataset(
                all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_spans, all_seq_lengths, all_guids
            )
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_seq_lengths, all_guids)
        logger.info("\tFinished converting features into tensors")
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_tensors_file)
            torch.save(dataset, cached_tensors_file)
            logger.info("\tFinished saving tensors")

    if args.task_name == "record" and split in ["dev", "test"]:
        answers = processor.get_answers(args.data_dir, split)
        return dataset, answers
    else:
        return dataset


def convert_model_to_onnx(args):
    """Converts a pytorch model checkpoint to an ONNX model."""
    from torch.onnx import export

    # Prepare task
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_classes = MODEL_CLASSES[args.model_type]
    model_class = model_classes[args.output_mode]

    # config
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]

    # tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokens = tokenizer.encode_plus("This is a sample input.")
    print(">>>>>>> Sample input: This is a sample input.")
    print(tokens)
    # model
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Only CPU is supported
    model.to(torch.device("cpu"))
    model.eval()

    print(">>>>>>> Model loaded.")

    # onnx convert
    input_names = ['input_ids', 'attention_mask']
    output_names = ['output_0']
    dynamic_axes = {
        'attention_mask': {
            0: 'batch',
            1: 'sequence'
        },
        'input_ids': {
            0: 'batch',
            1: 'sequence'
        },
        'output_0': {
            0: 'batch',
            1: 'sequence'
        }
    }
    if args.model_type in ["bert", "xlnet", "albert"]:
        input_names.append('token_type_ids')
        dynamic_axes["token_type_ids"] = {0: 'batch', 1: 'sequence'}

    model_args = (torch.tensor(tokens['input_ids']).unsqueeze(0),
                  torch.tensor(tokens['attention_mask']).unsqueeze(0))

    if args.model_type in ["bert", "xlnet", "albert"]:
        model_args = model_args + (torch.tensor(tokens['token_type_ids']).unsqueeze(0),)

    print(">>>>>>> ONNX conversion started!")
    torch.onnx.export(
        model,
        model_args,
        f=(args.model_name_or_path + "/model.onnx"),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=11,
    )
    print(">>>>>>> Model converted into ONNX format and saved as: ",
          (args.model_name_or_path + "/model.onnx"))

    # Optimize ONNX graph
    if not args.skip_graph_optimization:
        optimize_onnx_graph(args, config)

    # Run ONNX model after conversion
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
    print("Checking ONNX model loading from: {}".format(args.model_name_or_path + "/model.onnx"))
    try:
        onnx_options = SessionOptions()
        sess = InferenceSession(args.model_name_or_path + "/model.onnx", onnx_options)
        print("Model loaded successfully.")
        if args.model_type in ["bert", "xlnet", "albert"]:
            output_onnx = sess.run(None, {'input_ids': [tokens['input_ids']],
                                'attention_mask': [tokens['attention_mask']],
                                'token_type_ids': [tokens['token_type_ids']]})
        else:
            output_onnx = sess.run(None, {'input_ids': [tokens['input_ids']],
                                'attention_mask': [tokens['attention_mask']]})
        print(output_onnx)
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))

def optimize_onnx_graph(args, config):
    """ Optimize ONNX model with graph optimizations and quantizations """
    import inspect

    from onnx_graph_optimizer.optimizer import optimize_model
    from onnx_graph_optimizer.onnx_model_bert import BertOptimizationOptions

    # various graph optimization options.
    # ZCode uses all the optimizations by default.
    # Whether to use quantization or not can be selected optionally.
    optimization_options = BertOptimizationOptions('bert')
    optimization_options.enable_gelu = True
    optimization_options.enable_layer_norm = True
    optimization_options.enable_attention = True
    optimization_options.enable_attention_fbgemm = False if args.skip_quantization else True
    optimization_options.enable_skip_layer_norm = True
    optimization_options.enable_embed_layer_norm = True
    optimization_options.enable_bias_skip_layer_norm = True
    optimization_options.enable_bias_gelu = True
    optimization_options.enable_gelu_approximation = False
    optimization_options.enable_quantize_matmul = False if args.skip_quantization else True

    logger.warning(">>>>>>> Start optimizing ONNX graph")
    optimizer = optimize_model(args.model_name_or_path + "/model.onnx",
                   model_type='bert',
                   num_heads=config.num_attention_heads,
                   head_size=config.attention_head_size,
                   hidden_size=config.hidden_size,
                   optimization_options=optimization_options,
                   opt_level=0,
                   use_gpu=False,
                   only_onnxruntime=False)

    optimizer.save_model_to_file(args.model_name_or_path + "/model.onnx")
    logger.warning(">>>>>>> Finished optimizing ONNX graph")

def convert_model_to_fp16(args):
    """Converts a fp32 pytorch model checkpoint to a fp16 checkpoint."""
    # Prepare task
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_classes = MODEL_CLASSES[args.model_type]
    model_class = model_classes[args.output_mode]

    # config
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]

    # tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # model.to(args.device)
    model.eval()

    print(">>>>>>> Model loaded.")

    # convert to fp16 and save into fp16 directory
    model.half()

    from pathlib import Path
    Path(args.model_name_or_path + "/fp16").mkdir(exist_ok=True)
    model.save_pretrained(args.model_name_or_path + "/fp16")
    tokenizer.save_pretrained(args.model_name_or_path + "/fp16")
    config.save_pretrained(args.model_name_or_path + "/fp16")

    print(">>>>>>> Model converted into fp16 and saved into: ",
          (args.model_name_or_path + "/fp16"))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--teacher_model_type",
        default=None,
        type=str,
        required=False,
        help="Model type selected in the list for teacher model: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list for student model: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained teacher model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained student model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
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
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
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
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--use_onnxrt",
        action="store_true",
        help="Whether to ONNX runtime for inference evaluation. ONNX converted file needs to be in the same directory."
    )
    parser.add_argument(
        "--do_prune",
        action="store_true",
        help="Whether to prune the model on the dev set. This prunes the model to the target number of heads and the number of FFN states."
    )
    parser.add_argument(
        "--target_num_heads",
        default=12,
        type=int,
        help="The number of attention heads after pruning/rewiring.",
    )
    parser.add_argument(
        "--target_ffn_dim",
        default=3072,
        type=int,
        help="The dimension of FFN intermediate layer after pruning/rewiring.",
    )
    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument(
        "--log_evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_instance_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps as a float.")

    parser.add_argument("--log_energy_consumption", action="store_true", help="Whether to track energy consumption")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_and_save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best", action="store_true", help="Save only when hit best validation score.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--evaluate_test", action="store_true", help="Evaluate on the test splits.")
    parser.add_argument("--skip_evaluate_dev", action="store_true", help="Skip final evaluation on the dev splits.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--use_gpuid", type=int, default=-1, help="Use a specific GPU only")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--convert_onnx", action="store_true", help="Convert a pytorch model to onnx format")
    parser.add_argument("--convert_fp16", action="store_true", help="Convert a pytorch model to a half precision model")
    parser.add_argument("--threads_per_instance", type=int, default=-1, help="Number of threads for one inference instance.")
    parser.add_argument("--state_distill_cs", action="store_true", help="If this is using Cosine similarity for the hidden and embedding state distillation. vs. MSE")
    parser.add_argument('--state_loss_ratio', type=float, default=0.1)
    parser.add_argument('--att_loss_ratio', type=float, default=0.0)
    parser.add_argument("--save_latest", action="store_true", help="Save the last checkpoint regardless of the score.")
    parser.add_argument("--skip_graph_optimization", action="store_true", help="Whether to skip ONNX graph optimization.")
    parser.add_argument("--skip_quantization", action="store_true", help="Whether to skip 8-bit quantization.")
    parser.add_argument("--use_fixed_seq_length", action="store_true", help="Whether to use fixed sequence length.")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.convert_onnx:
        convert_model_to_onnx(args)
        return

    if args.convert_fp16:
        convert_model_to_fp16(args)
        return

    # Launch impact tracker
    if args.log_energy_consumption:
        from experiment_impact_tracker.compute_tracker import ImpactTracker

        logger.info("Launching impact tracker...")
        tracker = ImpactTracker(args.output_dir)
        tracker.launch_impact_monitor()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.use_gpuid > -1:
        device = args.use_gpuid
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.use_onnxrt:
        assert args.do_eval and not args.do_train, f"ONNX runtime can only be used for evaluation mode!"

    # Do all the stuff you want only first process to do
    # e.g. make sure only the first process will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_classes = MODEL_CLASSES[args.model_type]
    model_class = model_classes[args.output_mode]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # For onnx model, it's loaded in the evaluation routine
    if not args.use_onnxrt:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # normal training (fine-tuning)
        if args.teacher_model_type is None or args.teacher_model_name_or_path is None:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer) #, evaluate=False)
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        # distillation
        else:
            # Load pretrained teacher model (use the same tokenizer as student)
            args.teacher_model_type = args.teacher_model_type.lower()
            teacher_config_class, _, teacher_model_classes = MODEL_CLASSES[args.teacher_model_type]
            teacher_model_class = teacher_model_classes[args.output_mode]
            teacher_config = teacher_config_class.from_pretrained(
                args.teacher_model_name_or_path,
                num_labels=num_labels,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            if args.output_mode == "span_classification":
                teacher_config.num_spans = task_spans[args.task_name]
            teacher_model = teacher_model_class.from_pretrained(
                args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in args.teacher_model_name_or_path),
                config=teacher_config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
    
            teacher_model.to(args.device)
    
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, split='train') # evaluate=False)
            global_step, tr_loss = distill(
                args,
                train_dataset,
                teacher_model,
                model,
                tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

        # Evaluation with the best checkpoint
        if args.do_eval and args.local_rank in [-1, 0]:
            tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            checkpoints = [os.path.join(args.output_dir, "checkpoint-best")]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)

            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1]  # if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

                if not args.skip_evaluate_dev:
                    result, preds, ex_ids = evaluate(args, args.task_name, model, tokenizer, prefix=prefix)
                    result = dict((f"{k}_{global_step}", v) for k, v in result.items())

                if args.evaluate_test:
                    # Hack to handle diagnostic datasets
                    eval_task_names = ("rte", "ax-b", "ax-g") if args.task_name == "rte" else (args.task_name,)
                    for eval_task_name in eval_task_names:
                        result, preds, ex_ids = evaluate(
                            args, eval_task_name, model, tokenizer, split="test", prefix=prefix
                        )
                        processor = processors[eval_task_name]()
                        if args.task_name == "record":
                            answers = processor.get_answers(args.data_dir, "test")
                            processor.write_preds(preds, ex_ids, args.output_dir, answers=answers)
                        else:
                            processor.write_preds(preds, ex_ids, args.output_dir)

    # Evaluation only
    if args.do_eval and not args.do_train and args.local_rank in [-1, 0]:
        # onnx based evaluation
        if args.use_onnxrt:
            from onnxruntime import ExecutionMode, InferenceSession, SessionOptions
            onnx_options = SessionOptions()
            onnx_options.intra_op_num_threads = args.threads_per_instance
            onnx_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
            if not args.skip_evaluate_dev:
                result, preds, ex_ids = evaluate_ort_parallel(args, args.task_name, onnx_options, tokenizer, prefix="")
                result = dict((f"{k}", v) for k, v in result.items())

            if args.evaluate_test:
                # Hack to handle diagnostic datasets
                eval_task_names = (args.task_name,) # ("rte", "ax-b", "ax-g") if args.task_name == "rte" else (args.task_name,)
                for eval_task_name in eval_task_names:
                    result, preds, ex_ids = evaluate_ort_parallel(
                        args, eval_task_name, onnx_options, tokenizer, split="test", prefix=""
                    )
                    processor = processors[eval_task_name]()
                    if args.task_name == "record":
                        answers = processor.get_answers(args.data_dir, "test")
                        processor.write_preds(preds, ex_ids, args.output_dir, answers=answers)
                    else:
                        processor.write_preds(preds, ex_ids, args.output_dir)

        # network pruning
        elif args.do_prune:
            result, preds, ex_ids = prune_rewire(args, args.task_name, model, tokenizer, prefix="")
            result = dict((f"{k}", v) for k, v in result.items())
            print("before pruning" + str(result))
            # evaluate after pruning
            config = config_class.from_pretrained(
                args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)) + "/",
                num_labels=num_labels,
                finetuning_task=args.task_name,
            )
            model = model_class.from_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)) + "/")
            model.to(args.device)
            result, preds, ex_ids = evaluate(args, args.task_name, model, tokenizer, prefix="")
            result = dict((f"{k}", v) for k, v in result.items())
            print("after pruning" + str(result))

        # normal evaluation (pytorch)
        else:
            if not args.skip_evaluate_dev:
                result, preds, ex_ids = evaluate(args, args.task_name, model, tokenizer, prefix="", use_tqdm=False)
                result = dict((f"{k}", v) for k, v in result.items())

            if args.evaluate_test:
                # Hack to handle diagnostic datasets
                eval_task_names = (args.task_name,) # ("rte", "ax-b", "ax-g") if args.task_name == "rte" else (args.task_name,)
                for eval_task_name in eval_task_names:
                    result, preds, ex_ids = evaluate(
                        args, eval_task_name, model, tokenizer, split="test", prefix="", use_tqdm=False
                    )
                    processor = processors[eval_task_name]()
                    if args.task_name == "record":
                        answers = processor.get_answers(args.data_dir, "test")
                        processor.write_preds(preds, ex_ids, args.output_dir, answers=answers)
                    else:
                        processor.write_preds(preds, ex_ids, args.output_dir)


if __name__ == "__main__":
    main()
