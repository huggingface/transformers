# coding=utf-8
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  BertForESNLI, DecoderRNN, BertModel,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer, PretrainedConfig,)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors, 
                        Lang, normalize_sentence,
                        tensorFromSentence, pad_seq, 
                        indexesFromSentence, get_text, 
                        get_index, expl_to_expl2label_input)

import sys
import time
import pickle 

file_handler = logging.FileHandler(filename='glue_'+time.strftime("%d:%m") + "_" + time.strftime("%H:%M:%S")+'.log')
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'bert_expl': (BertConfig, BertForESNLI, BertTokenizer),
    'bert_expl_encoder': (BertConfig, BertModel, BertTokenizer)
}

CLS_token = 0 # in decoder language
SEP_token = 1 # in decoder language
PAD_token = 2 # in decoder language
hidden_size = 100 # the hidden_size for decoder. change to 768 and get rid of resize once working on the whole dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train_enc_dec(args, train_dataset, encoder, tokenizer, all_expl):
    '''
    train the encoder and decoder separately
    initialize: 
    - optimizer, scheduler for encoder DONE
    - optimizer, scheduler for decoder DONE
    for each epoch:
        - mark training mode for encoder and decoder DONE
        - call encoder and decoder and get loss TODO
        - loss.backward DONE
        - clip_grad_norm_ DONE
        - if hit accumulation step: encoder_optimizer.zero_grad(), decoder_optimizer.zero_grad(), and step() DONE
    '''
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    # batched data
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare decoder word2index
    decoder_lang = Lang()
    for sentence in all_expl:
        sentence = normalize_sentence(sentence)
        decoder_lang.addSentence(sentence)
    
    MAX_LENGTH = args.max_seq_length
    target_length = args.max_seq_length
    decoder_vocab_size = decoder_lang.n_words

    # initialize decoder
    decoder = DecoderRNN(hidden_size=hidden_size, output_size=decoder_lang.n_words).to('cuda') #initialize decoder
    
    # prepare optimizers
    no_decay = ['bias', 'LayerNorm.weight']
    encoder_grouped_params = [
        {'params': [p for n, p in encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    encoder_optimizer = AdamW(encoder_grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
    decoder_optimizer = AdamW(decoder.parameters(), lr=0.001, eps=args.adam_epsilon)
    
    encoder_scheduler = WarmupLinearSchedule(encoder_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    decoder_scheduler = WarmupLinearSchedule(decoder_optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        encoder, encoder_optimizer = amp.initialize(encoder, encoder_optimizer, opt_level=args.fp16_opt_level)
        decoder, decoder_optimizer = amp.initialize(decoder, decoder_optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        encoder = torch.nn.DataParallel(encoder)
        decoder = torch.nn.DataParallel(decoder)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[args.local_rank],
                                                            output_device=args.local_rank,
                                                            find_unused_parameters=True)
        
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    encoder.zero_grad()
    decoder.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility 
    epoch_loss = 0 #init as a huge number lol
    num_epoch = 0
    for _ in train_iterator:
        num_epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        print('average loss last epoch: ', epoch_loss/len(epoch_iterator))
        epoch_loss = 0 #init as a huge number lol
        if global_step!=0: print('average loss: ', tr_loss/global_step)
        for step, batch in enumerate(epoch_iterator):
            if num_epoch <= 3 and args.freeze:
                encoder.train()
                
            decoder.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None}
            expl_idx = batch[4]
            all_expl = all_expl
            decoder_lang = decoder_lang
            mode = 'teacher'
            batch_size = len(expl_idx)
            
            encoder_outputs = encoder(**inputs)
            bert_output, bert_output_pooled = encoder_outputs[0], encoder_outputs[1] 
            
            generated_expl = torch.zeros(target_length, batch_size, decoder_vocab_size).to('cuda')
        
            target_expl = [all_expl[i] for i in expl_idx]
            target_expl_index = []
            for line in target_expl:
                sentence = normalize_sentence(line)
                indexes = indexesFromSentence(decoder_lang, sentence)
                indexes_padded = pad_seq(indexes, target_length)
                target_expl_index.append(indexes_padded)

            target_expl_index = torch.LongTensor(target_expl_index).transpose(0, 1).to('cuda') 
            #(output_seq_len, bs), where each value is an int between 0 and decode_lang_vocab_size

            decoder_hidden = bert_output_pooled.unsqueeze(0) # (bs, hidden_size) -> (1, bs, hidden_size)
            decoder_input = torch.LongTensor([CLS_token] * batch_size) # (bs)

            for i in range(target_length):
                #print('decoder_input: ', decoder_input.size())
                #print('decoder_hidden: ', decoder_hidden.size())
                
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) 
                #take only premise's encoding results and produce a next sentence for now
                #decoder_output: (bs, n_vocab)
                #decoder_hidden[0]: (1, bs, hidden_size)

                generated_expl[i] = decoder_output   
                topv, topi = decoder_output.topk(1) 
                #decoder_input = topi.squeeze(1)
                #teacher forcing
                decoder_input = (target_expl_index[i] if args.teacher_force else topi.squeeze(1)) # fix dimension
                if args.teacher_force == False and decoder_input.item() == EOS_token: 
                    break

            loss_fct = torch.nn.CrossEntropyLoss()
            generated_expl = generated_expl.transpose(1, 2) # (output_seq_len, bs, n_vocab) -> (output_seq_len, n_vocab, bs)
            loss = loss_fct(generated_expl, target_expl_index)
            
            # sanity check on generated explanations
            #generated_expl_index = get_index(generated_expl) 
            #print(get_text(decoder_lang, generated_expl_index)) 
            #print(get_text(decoder_lang, target_expl_index)) 

            epoch_loss += loss.item()
            
            #if global_step!=0: print('average loss: ', tr_loss/global_step)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                # seems args.fp16 is False anyways.
                print('loss: ', loss)
                with amp.scale_loss(loss, decoder_optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(decoder_optimizer), args.max_grad_norm)
                # check if loss get modified already
                print('loss: ', loss, ' not supposed to change!!!')
                if num_epoch <= 3 and args.freeze:
                    with amp.scale_loss(loss, encoder_optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(encoder_optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if num_epoch <= 3 and args.freeze:
                    encoder_scheduler.step()  # Update learning rate schedule
                    encoder_optimizer.step()
                    encoder.zero_grad()
                    
                decoder_scheduler.step()
                decoder_optimizer.step()
                decoder.zero_grad()
                global_step += 1
                
                '''
                # TODO: write eval first ...
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss
                    '''
                
                
                # save model at checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # Take care of distributed/parallel training
                    encoder_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'encoder_training_args.bin'))
                    logger.info("Saving encoder checkpoint to %s", output_dir)
                    
                    decoder_to_save = decoder.module if hasattr(decoder, 'module') else decoder # Take care of distributed/parallel training
                    torch.save(decoder_to_save.state_dict(), args.output_dir+'decoder_state_dict.pt')
                    decoder = decoder_to_save
                    decoder.to(args.device)
                    logger.info("Saving decoder checkpoint to %s", output_dir)
                    # save decoder_lang using pickle
                    filehandler = open(args.output_dir+'decoder_lang.obj', 'wb') 
                    pickle.dump(decoder_lang, filehandler)
                    logger.info("Saving decoder_lang checkpoint to %s", output_dir)
                    

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, decoder, decoder_lang # global steps, average training loss, decoder


def train(args, train_dataset, model, tokenizer, all_expl=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.expl:
        # prepare decoder word2index
        decoder_lang = Lang()
        for sentence in all_expl:
            sentence = normalize_sentence(sentence)
            decoder_lang.addSentence(sentence)
        model.setOutputVocab(decoder_lang)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            print('train loss: %s' % tr_loss)
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            if args.expl:
                inputs['expl_idx'] = batch[4]
                inputs['all_expl'] = all_expl
                inputs['decoder_lang'] = decoder_lang
                inputs['mode'] = 'teacher'
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            print('loss: ', loss)
            if global_step!=0: print('average loss: ', tr_loss/global_step)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate_enc_dec(args, encoder, decoder, decoder_lang, expl2label_model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, all_expl = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            encoder.eval()
            decoder.eval()
            expl2label_model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0], #(args.per_gpu_eval_bs, args.max_seq_len) LongTensor
                          'attention_mask': batch[1], #(args.per_gpu_eval_bs, args.max_seq_len) LongTensor
                          'token_type_ids': batch[2]  #(args.per_gpu_eval_bs, args.max_seq_len) LongTensor
                         } 
                labels = batch[3]
                expl_idx = batch[4]
                all_expl = all_expl
                decoder_lang = decoder_lang
                batch_size = len(expl_idx)
                
                encoder_outputs = encoder(**inputs)
                bert_output, bert_output_pooled = encoder_outputs[0], encoder_outputs[1] 

                target_length = args.max_seq_length
                decoder_vocab_size = decoder_lang.n_words
                generated_expl = torch.zeros(target_length, batch_size, decoder_vocab_size).to('cuda')
                
                target_expl = [all_expl[i] for i in expl_idx]
                target_expl_index = []
                for line in target_expl:
                    sentence = normalize_sentence(line)
                    indexes = indexesFromSentence(decoder_lang, sentence)
                    indexes_padded = pad_seq(indexes, target_length)
                    target_expl_index.append(indexes_padded)

                target_expl_index = torch.LongTensor(target_expl_index).transpose(0, 1).to('cuda') 
                decoder_hidden = bert_output_pooled.unsqueeze(0) 
                decoder_input = torch.LongTensor([CLS_token] * batch_size) 
                
                for i in range(target_length):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden) 
                    generated_expl[i] = decoder_output   
                    topv, topi = decoder_output.topk(1) 
                    decoder_input = topi.squeeze(1)
                    #teacher forcing
                    #decoder_input = (target_expl_index[i] if args.teacher_force else topi.squeeze(1)) # fix dimension
                    #if args.teacher_force == False and decoder_input.item() == EOS_token: 
                        #break
                        

                loss_fct = torch.nn.CrossEntropyLoss()
                generated_expl = generated_expl.transpose(1, 2) # (output_seq_len, bs, n_vocab) -> (output_seq_len, n_vocab, bs)
                tmp_eval_loss = loss_fct(generated_expl, target_expl_index)
                
                eval_loss += tmp_eval_loss.mean().item()
                
            generated_expl_index = get_index(generated_expl) #(seq_len, bs) 
            generated_expl_text = get_text(decoder_lang, generated_expl_index) # a list (size = bs) of explanation texts
            #target_expl_text = get_text(decoder_lang, target_expl_index)
            
            nb_eval_steps += 1
            
            expl2label_inputs_data = expl_to_expl2label_input(generated_expl_text, args.max_seq_length, tokenizer,
                                                              cls_token_at_end=False,           
                                                              cls_token=tokenizer.cls_token,
                                                              sep_token=tokenizer.sep_token,
                                                              cls_token_segment_id=0,
                                                              pad_on_left=False,                 
                                                              pad_token_segment_id=0)
            
            input_ids = expl2label_inputs_data[0].cuda()
            attention_mask = expl2label_inputs_data[1].cuda()
            token_type_ids = expl2label_inputs_data[2].cuda()
            
            expl2label_inputs = {'input_ids':      input_ids, # (bs, seq_len) based on generated_expl_index
                                 'attention_mask': attention_mask, # (bs, seq_len) based on generated_expl_index
                                 'token_type_ids': token_type_ids, # (bs, seq_len) based on generated_expl_index
                                 'labels': labels}
            #expl2label_inputs['expl_idx'] = None
            #expl2label_inputs['all_expl'] = None
            #expl2label_inputs['decoder_lang'] = None
            
            expl2label_outputs = expl2label_model(**expl2label_inputs)
            expl2label_loss, logits = expl2label_outputs[:2]
            print('expl2label_loss: ', expl2label_loss.item())
            
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = expl2label_inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, expl2label_inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if args.expl:
            eval_dataset, all_expl = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        else:
            eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
            expl=args.expl)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        
    if args.expl:
        all_expl = [f.expl for f in features]
        all_expl_idx = torch.tensor([idx for idx in range(len(features))], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_expl_idx)
        return dataset, all_expl
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    
    parser.add_argument('--expl', type=bool, default=False, const = True,nargs = '?', \
                        help = 'whether to generate expl with esnli decoder, and separately encode premises and hypothesis')
    parser.add_argument('--freeze', type=bool, default=False, const = True,nargs = '?', \
                        help = 'whether freeze encoder training after 3 epoches')
    parser.add_argument('--teacher_force', type=bool, default=False, const = True,nargs = '?', \
                        help = 'whether to use teacher forcing in train and eval')
    
    args = parser.parse_args()
    
    if args.expl:
        args.model_type = 'bert_expl_encoder'
        #args.do_train = True 
        #args.do_eval = True

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() -3 #TODO: get rid of -3 once the other gpu are available 
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        all_expl = None
        if args.expl:
            train_dataset, all_expl = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        else:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, decoder, decoder_lang = train_enc_dec(args, train_dataset, model, tokenizer, all_expl=all_expl)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
        
        # do the same for decoder
        if args.expl:
            decoder_to_save = decoder.module if hasattr(decoder, 'module') else decoder # Take care of distributed/parallel training
            torch.save(decoder_to_save.state_dict(), args.output_dir+'decoder_state_dict.pt')
            decoder = decoder_to_save
            decoder.to(args.device)
            # save decoder_lang using pickle
            filehandler = open(args.output_dir+'decoder_lang.obj', 'wb') 
            pickle.dump(decoder_lang, filehandler)

    # Evaluation
    results = {}
    if (not args.expl) and args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            
    if args.expl and args.do_eval:
        dir1 = args.output_dir
        dir2 = '/tmp/ESNLI_expl_to_labels/'
        
        encoder = BertModel.from_pretrained(dir1)
        # use pickle to load decoder_lang
        filehandler = open(dir1+'decoder_lang.obj', 'rb') 
        decoder_lang = pickle.load(filehandler)
        decoder = DecoderRNN(hidden_size=hidden_size, output_size=decoder_lang.n_words)
        decoder.load_state_dict(torch.load(dir1+'decoder_state_dict.pt'))
        
        expl2label_model = BertForSequenceClassification.from_pretrained(dir2) #store in a different dir from encoder dir
        
        encoder.to(args.device)
        decoder.to(args.device)
        expl2label_model.to(args.device)
        result = evaluate_enc_dec(args, encoder, decoder, decoder_lang, expl2label_model, tokenizer, prefix="")
        result = dict((k + '_{}'.format(""), v) for k, v in result.items())
        results.update(result)

    return results


if __name__ == "__main__":
    main()
