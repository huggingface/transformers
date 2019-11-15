import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
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



from gQA.utils.config import *
from gQA.utils.tfr_scheduler import tfr_scheduler
from gQA.utils.adamW import AdamW
from gQA.validate_test import *
from gQA.model_code.data_opinosis import *
from gQA.model_code.gSUM import gSUM
from gQA.model_code.gQA import gQA
xw
def to_var(tensors, cuda=True):
    if cuda:
        return [Variable(t, requires_grad=False).cuda() for t in tensors]
    else:
        return [Variable(t, requires_grad=False) for t in tensors]

def run_model(loss_function, model, data, is_test = False,
        epoch = 0, pointer_generation = None, train_qa = False,
        decode = True, tfr_min = None, tfr_override = None):
    tfr_prop = args.epochs * args.tfr_reset

    teacher_forcing_ratio = 0
    lambda_coeff = 0
    if tfr_min is None:
        tfr_min = args.tfr_min

    lambda_coeff = args.lambda_coeff
    
    #If we are adding lambda to the mix, set TFR = 1
    if lambda_coeff > 0 and not is_test:
        teacher_forcing_ratio = 1.0
    if tfr_override is not None and not is_test:
        teacher_forcing_ratio = tfr_override

    if train_qa:
        question_word, question_ex, question_char, question_word_mask, question_char_mask, length_q = to_var(
        [data.question_word, data.question_word_ex, 
        data.question_char, data.question_word_mask, data.question_char_mask, data.length_q], cuda=args.cuda)

    data_word, data_ex, data_char, data_mask, length_s, extra_zeros = to_var(
        [data.data_word, data.data_word_ex, data.data_char, data.data_mask,
         data.length_s, data.extra_zeros], cuda=args.cuda)

    adj, gold_word, gold_mask_unk, gold_mask, gold_ex = to_var(
        [data.adjm, data.gold_word, data.gold_mask_unk, data.gold_mask, 
         data.gold_word_extended], cuda=args.cuda)

    all_to_all, sent_sim, sent_attn_mask = to_var(
        [data.all_to_all, data.sent_sim, data.sent_attn_mask], cuda=args.cuda)

    if pointer_generation is None:
        pointer_generation = args.PG
    if len(extra_zeros.size()) == 0:
        extra_zeros = None
        pointer_generation = False

    if train_qa:
        return model(loss_function, data_word, data_char,
            data_mask, length_s, adj, all_to_all, sent_sim, sent_attn_mask,
            question_word, question_word_mask, question_ex, question_char, question_char_mask, length_q,
            gold_word, gold_mask_unk, gold_mask, args.cuda, dev=cuda_device, test=is_test, 
            teacher_forcing_ratio = teacher_forcing_ratio,
            gold_len = data.max_gold_len, lambda_coeff=lambda_coeff, 
            PG=pointer_generation, oov=data.oov, data_ex=data_ex, gold_ex=gold_ex,
            extra_zeros=extra_zeros, decode=decode, stage_two = args.stage_two)
    else:
        return model(loss_function, data_word, data_char,
            data_mask, length_s, adj, all_to_all, sent_sim, sent_attn_mask,
            gold_word, gold_mask_unk, gold_mask, args.cuda, dev=cuda_device, test=is_test, 
            teacher_forcing_ratio = teacher_forcing_ratio,
            gold_len = data.max_gold_len, lambda_coeff=lambda_coeff, 
            PG=pointer_generation, oov=data.oov, data_ex=data_ex, gold_ex=gold_ex,
            extra_zeros=extra_zeros, decode=decode)

def save(model, optimizer, epoch, loss = None, scheduler=None):
    if scheduler is not None:
        obj = {'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), \
            'start': epoch+1, 'index' : 1, 'tfr_itt' : scheduler.getItt()}
    else:
        obj = {'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), \
            'start': epoch+1, 'index' : 1}        
    if loss is not None:
        loss_saved = np.loadtxt(open(args.save_path + "_loss_graph.csv", "rb"))
        if loss_saved is not None:
            loss_saved = np.append(np.asarray(loss_saved), loss)
        else:
            loss_saved = np.array(list(loss))
        #max because we use validation accuracy instead of loss
        if args.train_qa:
            best_loss = max(loss_saved)
        else:
            best_loss = min(loss_saved)
        np.savetxt(args.save_path + "_loss_graph.csv", np.asarray(loss_saved))
        if loss == best_loss:
            print("Saving new best model.")
            torch.save(obj, args.save_path+'_best.model')
    torch.save(obj, args.save_path+'.model')


