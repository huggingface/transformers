import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from gQA.model_code.data_opinosis import *
from gQA.model_code.gSUM import gSUM
from torch.optim.lr_scheduler import StepLR
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

class Beam(object):
  def __init__(self, tokens, log_probs, h_state, c_state, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.h_state = h_state
    self.c_state = c_state
    self.coverage = coverage

  def extend(self, token, log_prob, h_states, c_states, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      h_state = h_states,
                      c_state = c_states,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)
    #Remember to set batch_size = 1
    def beam_search(self, batch, model, args, is_coverage, run_model, gqa = False, beam_size = 6):
        gsum_outputs = run_model(None, model, batch[0], True, 0, args.PG, args.train_qa, decode = False)

        decoder_input, decoded_words, decoder_hidden, \
        bridge_output, use_teacher_forcing, batch_size, coverage = gsum_outputs

        h_decoder = decoder_hidden[0]
        c_decoder = decoder_hidden[1]

        #One copy of encoder outputs per beam
        _, t_k, n = bridge_output.outputs[0].size()
        encoder_output = bridge_output.outputs[0]
        encoder_output = encoder_output.expand(beam_size, t_k, n).contiguous()

        encoder_output_ext = bridge_output.outputs[1]
        encoder_output_ext = encoder_output_ext.expand(beam_size, t_k).contiguous()
        bridge_output.outputs = (encoder_output, encoder_output_ext)

        coverage = coverage.expand(beam_size, -1).contiguous()
        bridge_output.mask = bridge_output.mask.expand(beam_size, bridge_output.mask.size(1)).contiguous()

        bridge_output.other = bridge_output.other.expand(beam_size, bridge_output.other.size(1)).contiguous()


        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[SOS],
                      log_probs=[0.0],
                      h_state=h_decoder[0],
                      c_state=c_decoder[0],
                      coverage=(coverage[0] if is_coverage else None))
                 for _ in range(beam_size)]

        
        results = []
        steps = 0
        extra_zeros = None
        if batch[0].extra_zeros is not None:
            extra_zeros = batch[0].extra_zeros.cuda().expand(beam_size, batch[0].extra_zeros.size(1)).contiguous()
        while steps < args.gold_len:
            latest_tokens = [h.latest_token for h in beams]
            #Clip OOV
            latest_tokens = list(map(lambda indx: UNK_WORD if indx >= WORD_VOCAB_SIZE else indx, latest_tokens))

            y_t_1 = Variable(torch.LongTensor(latest_tokens)).cuda()
            all_state_h =[]
            all_state_c = []
            for h in beams:
                h_state = h.h_state
                c_state = h.c_state
                all_state_h.append(h_state)
                all_state_c.append(c_state)

            h_decoder = torch.stack(all_state_h, 0).squeeze(1).unsqueeze(0)
            c_decoder = torch.stack(all_state_c, 0).squeeze(1).unsqueeze(0)

            decoder_hidden = (h_decoder, c_decoder)

            coverage_t_1 = None
            if is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            decoder_output, decoder_hidden, attn, coverage = \
                model.decode_one_step(y_t_1, None, decoder_hidden, bridge_output,
                coverage, extra_zeros, False, beam_size, PG=True, is_coverage=is_coverage)


            log_probs = torch.log(decoder_output)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2)

            h_decoder = decoder_hidden[0].squeeze()
            c_decoder = decoder_hidden[1].squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                h_state_i = h_decoder[i]
                c_state_i = c_decoder[i]
                coverage_i = (coverage[i] if is_coverage else None)

                for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   h_states=h_state_i,
                                   c_states=c_state_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == EOS:
                    results.append(h)
                else:
                    beams.append(h)
                if len(beams) == beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]
