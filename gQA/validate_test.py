import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model_code.data_opinosis import *
from model_code.data_squad import local_num2text_squad as gqa_local_num2text
from model_code.gSUM import gSUM
from torch.optim import RMSprop
import numpy as np
import random
import json
import time
import math
import sys
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from torch.nn.utils import clip_grad_norm_
from pythonrouge.pythonrouge import Pythonrouge
from gQA.utils.beam_decoder import *
from gQA.utils.adamW import AdamW

def fetch_generated_summary(summary, oov, gqa=False):
    eos = 0
    try:
        eos = summaries.index(EOS)
    except:
        eos = len(summary)
    if gqa:
        return gqa_local_num2text(summary[0:eos], None, oov, test=True)
    return local_num2text(summary[0:eos], None, oov, test=True)

def summary_map(summaries, batch, gqa=False):
    return list(map(lambda i: " ".join(fetch_generated_summary(summaries[i], batch.oov[i], gqa)), range(len(summaries))))

def gold_map(batch):
    return list(map(lambda gold: " ".join(gold), batch.gold))

def validate(dataset, run_model, model=None, cpu_embedding=None, \
    loss_function=None, pool=None, args=None, train_gqa=False):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 5

    num_batches = (args.file_limit * 0.05)/dataset.args.batch 
    cur_cache = dataset.get_eval_item(0, cpu_embedding, args.cache, args.PG)
    itt_loss = 0

    for index in tqdm(range(1, int(num_batches), args.cache)):
        #print(index)
        async_result = pool.apply_async(dataset.get_eval_item, (index, cpu_embedding, args.cache, args.PG))
        #print(cur_cache)
        for batch in cur_cache:
            #Some batches have low data quality
            max_sample_size = int(max(batch.sent_sim.sum(dim=1)).item())
            model.change_sample_size(max_sample_size)
            loss, _, _ = run_model(loss_function, model, batch, False, 0., args.PG, train_gqa)

            #Accumulate loss
            itt_loss += loss[0].data.sum()

        #Try to fetch the next cache, if it fails we stored a backup
        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except:
            #If there was an issue with this batch, just load the next batch and continue
            cur_cache = backup
            continue

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old

    return itt_loss

def test_rouge(dataset, run_model, model=None, cpu_embedding=None, 
loss_function=None, pool=None, args=None, distributed = False):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    is_coverage = (args.lambda_coeff > 0)

    num_batches = (args.file_limit * 0.05)/dataset.args.batch
    cur_cache = dataset.get_test_item(0, cpu_embedding, args.cache, args.PG)

    summary_text = []
    golden_text = []
    b = BeamSearch()
    if not distributed:
        for index in tqdm(range(1, int(num_batches), args.cache)):
            #print(index)
            async_result = pool.apply_async(dataset.get_test_item, (index, cpu_embedding, args.cache, args.PG))
            #print(cur_cache)
            for batch in cur_cache:
                #Some batches have low data quality
                returned_beam = b.beam_search([batch], model, args, is_coverage, run_model, False, beam_size=4)

                summary_text.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch))
                golden_text.append(gold_map(batch))

            #Try to fetch the next cache, if it fails we stored a backup
            backup = cur_cache
            try:
                cur_cache = async_result.get()
            except:
                #If there was an issue with this batch, just load the next batch and continue
                cur_cache = backup
                continue
    else:
        for index in range(1, int(num_batches), args.cache):
            #print(index)
            async_result = pool.apply_async(dataset.get_test_item, (index, cpu_embedding, args.cache, args.PG))
            #print(cur_cache)
            for batch in cur_cache:
                #Some batches have low data quality
                returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=4)

                summary_text.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch))
                golden_text.append(gold_map(batch))

            #Try to fetch the next cache, if it fails we stored a backup
            backup = cur_cache
            try:
                cur_cache = async_result.get()
            except:
                #If there was an issue with this batch, just load the next batch and continue
                cur_cache = backup
                continue        

    model.change_sample_size(args.sample_size)
    #print(golden_text)

    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary_text, reference=golden_text,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=False, length=args.gold_len,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)

    score = rouge.calc_score()
    if not distributed:
        print(score)

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old

    return score

def test_qa(dataset, run_model, model=None, cpu_embedding=None, pool=None, args=None):
    model.eval()

    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    args.cache = 10

    num_batches = (args.file_limit * 0.05)/dataset.args.batch
    if args.cheat:
        cur_cache = dataset.get_training_item(0, cpu_embedding, args.cache, args.PG)
    else:    
        cur_cache = dataset.get_test_item(0, cpu_embedding, args.cache, args.PG)

    #Changing default params for testing
    pred_ans = list()
    gold_ans = list()
    b = BeamSearch()
    for index in tqdm(range(1, int(num_batches), args.cache)):
        if args.cheat:
            async_result = pool.apply_async(dataset.get_training_item, (index, cpu_embedding, args.cache, args.PG))
        else:
            async_result = pool.apply_async(dataset.get_test_item, (index, cpu_embedding, args.cache, args.PG))
        for batch in cur_cache:
            num_supporting_sents = int(batch.sent_sim.sum(1).item())
            if args.hard_ss:
                num_supporting_sents = args.sample_size
            model.change_sample_size(num_supporting_sents)
            returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=4)
            pred_ans.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch, True))
            gold_ans.append(gold_map(batch))
            batch.sent_sim[0].sum()
        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except:
            print("something hit me")
            cur_cache = backup
            continue

    num_exact_match = 0
    for i in range(len(gold_ans)):
        gold = gold_ans[i][0]
        pred = pred_ans[i][0]
        if gold.lower() == pred.lower():
            num_exact_match += 1
    accuracy = num_exact_match / len(gold_ans)
    print(f"final_accuracy: {accuracy} on {len(gold_ans)} samples")
    return accuracy


def validate_qa(dataset, run_model, model=None, cpu_embedding=None, pool=None, args=None, samples = 100):
    model.eval()
    random.shuffle(dataset.train)
    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    args.cache = 10

    num_batches = samples * args.cache
    if args.cheat:
        cur_cache = dataset.get_training_item(0, cpu_embedding, args.cache, args.PG)
    else:    
        cur_cache = dataset.get_eval_item(0, cpu_embedding, args.cache, args.PG)

    #Changing default params for testing
    pred_ans = list()
    gold_ans = list()
    b = BeamSearch()

    for index in tqdm(range(1, int(num_batches), args.cache)):
        if args.cheat:
            async_result = pool.apply_async(dataset.get_training_item, (index, cpu_embedding, args.cache, args.PG))
        else:
            async_result = pool.apply_async(dataset.get_eval_item, (index, cpu_embedding, args.cache, args.PG))
        for batch in cur_cache:
            num_supporting_sents = int(batch.sent_sim.sum(1).item())
            #if args.hard_ss:
            #    max_sample_size = args.sample_size
            model.change_sample_size(num_supporting_sents)
            returned_beam = b.beam_search([batch], model, args, False, run_model, False, beam_size=4)
            pred_ans.append(summary_map([returned_beam.tokens[1:len(returned_beam.tokens)]], batch, True))
            gold_ans.append(gold_map(batch))
            batch.sent_sim[0].sum()
        backup = cur_cache
        try:
            cur_cache = async_result.get()
        except:
            print("Exception while validating! Oh no!")
            cur_cache = backup
            continue

    num_exact_match = 0
    for i in range(len(gold_ans)):
        gold = gold_ans[i][0]
        pred = pred_ans[i][0]
        if gold.lower() == pred.lower():
            num_exact_match += 1
    accuracy = num_exact_match / len(gold_ans)

    model.change_sample_size(args.sample_size)
    dataset.args.batch = batch_size_old


    return accuracy

def test(dataset, run_model, model, embedding=None, args=None):
    
    #For when we run out of VRAM
    batch_size_old = dataset.args.batch
    dataset.args.batch = 1 
    is_coverage = (args.lambda_coeff > 0)

    model.eval()
    b = BeamSearch()

    if args.cheat:
        batch = dataset.get_training_item(index=args.test_indx, embedding=embedding, delta=1, pgen=args.PG)
    else:
        batch = dataset.get_eval_item(index=args.test_indx, embedding=embedding, delta=1, pgen=args.PG)
    if args.train_qa:
        num_supporting_sents = int(batch[0].sent_sim.sum(1).item())
    
    #if args.hard_ss:
    #    max_sample_size = args.sample_size
    if args.train_qa:
        model.change_sample_size(num_supporting_sents)

    returned_beam = b.beam_search(batch, model, args, is_coverage, run_model, False, beam_size=4)
    #_, summary, _ = run_model(None, model, batch[0], True, 0, args.PG, train_gqa)

    print("Original Summary")
    print(batch[0].gold[0])
    print("Generated Summary")
    indx = 0

    #Clip at the first EOS
    try:
        indx = returned_beam.tokens.index(EOS)
    except:
        indx = len(returned_beam.tokens)
    try:
        if args.train_qa:
            print(gqa_local_num2text(returned_beam.tokens[1:indx], None, batch[0].oov[0], test=True))
        else:
            print(local_num2text(returned_beam.tokens[1:indx], None, batch[0].oov[0], test=True))
    except:
        print("Error! Word outside of OOV.")
        print(returned_beam.tokens[1:indx])
        print(batch[0].oov[0])

    if not args.train_qa:
        model.change_sample_size(args.sample_size)

    dataset.args.batch = batch_size_old
