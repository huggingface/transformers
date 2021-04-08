# coding=utf-8
# Copyright 2021 The Ontocord team, the G2P, Melgan, and Fastspeech2 Authors, and the HuggingFace Inc. team. All rights reserved.
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

# This software is based on other open source code. A huge thanks to
# the Huggingface team, and the authors of the Fastspeech2 and Melgan
# papers and the following authors who originally implemented the
# various modules, from which this code is based:

# Chung-Ming Chien's Fastspeech2 implementation is under the MIT license: https://github.com/ming024/FastSpeech2
# Seung-won Park 박승원's Meglan implementation is under BSD-3 license: https://github.com/seungwonpark/melgan
# Kyubyong Park's G2P implementation is under the Apache 2 license: https://github.com/Kyubyong/g2p, and also here for pytorch specifics https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
import math
from .configuration_fastspeech2 import FastSpeech2Config
from transformers import PreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings

### G2P
# A simple Seq2Seq GRU->GRU network to fix and guess phoenemes from
# graphenes. Tie the embeddings to the fastspeech2 embeddings to
# increase generalizations.

class G2PEncoder(nn.Module):
    def __init__(self, config, emb):
        super().__init__()
        self.emb = emb
        self.rnn = nn.GRU(self.emb.embedding_dim,  self.emb.embedding_dim, batch_first=True)
        
    def forward(self, input_ids, seqlens):
        packed_input = self.emb(input_ids)
            
        # packing -> rnn -> unpacking -> position recovery: note that enforce_sorted is set to False.
        packed_input = pack_padded_sequence(packed_input, seqlens, batch_first=True, enforce_sorted=False)   
        outputs, last_hidden = self.rnn(packed_input)

        # last hidden
        last_hidden = last_hidden.permute(1, 2, 0)
        last_hidden = last_hidden.view(last_hidden.size()[0], -1)
        
        return last_hidden

class G2PDecoder(nn.Module):
    global g2idx, idx2g, p2idx, idx2p
    def __init__(self, config, emb):
        super().__init__()
        self.emb = emb
        self.rnn = nn.GRU(self.emb.embedding_dim,  self.emb.embedding_dim, batch_first=True)
        self.fc = nn.Linear(self.emb.embedding_dim, self.emb.num_embeddings, bias=False)
        
    def forward(self, decoder_inputs, h0):
        decoder_inputs = self.emb(decoder_inputs)
           
        outputs, last_hidden = self.rnn(decoder_inputs, h0)
        logits = self.fc(outputs) # (N, T, V)
        y_hat = logits.argmax(-1)
        
        return logits, y_hat, last_hidden

class G2PNet(nn.Module):
    
    def __init__(self, config, emb): 
        super().__init__()
        self.encoder = G2PEncoder(config, emb)
        self.decoder = G2PDecoder(config, emb)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)    
        self.bos_token_id = config.bos_token_id

    def forward(self, input_ids, seqlens, decoder_inputs=None, dec_maxlen=None, target=None):  
        '''
        At training, decoder inputs (ground truth) and teacher forcing is applied. 
        At evaluation, decoder inputs are ignored, and the decoding keeps for `dec_maxlen` steps.
        '''
        device = next(self.parameters()).device  
        if dec_maxlen is None:
          dec_maxlen = int(max(seqlens)*3)      
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if decoder_inputs is not None and decoder_inputs.device != device:
            decoder_inputs = decoder_inputs.to(device)
        last_hidden = self.encoder(input_ids, seqlens)
        h0 = last_hidden.unsqueeze(0)
        if self.training and decoder_inputs is not None: 
            logits, y_hat, h0 = self.decoder(decoder_inputs, h0)
            #loss = self.loss_fn(logits.view(-1, logits.shape[-1]), decoder_inputs.view(-1))
            loss = None
        else: # evaluation
            if decoder_inputs is not None:
                decoder_inputs = decoder_inputs[:, :1] # "<s>"
            else:
                decoder_inputs = torch.LongTensor([self.bos_token_id]).expand(input_ids.size()[0]).unsqueeze(-1).to(device)
            logits, y_hat = [], []
            for t in range(dec_maxlen):
                _logits, _y_hat, h0 =self.decoder(decoder_inputs, h0) # _logits: (N, 1, V), _y_hat: (N, 1), h0: (1, N, N)
                logits.append(_logits)
                y_hat.append(_y_hat)
                decoder_inputs = _y_hat
        
            logits = torch.cat(logits, 1)
            y_hat = torch.cat(y_hat, 1)
            loss = None
        return logits, y_hat, loss

### MELGAN 
MAX_WAV_VALUE = 32768.0

class Discriminator(nn.Module):
    def __init__(self):


        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.Conv1d(1, 16, kernel_size=15, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.mel_channel = config.n_mel_channels
        self.hop_length = config.hop_length
        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(self.mel_channel, 512, kernel_size=7, stride=1),

            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),

            ResStack(256),

            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),

            ResStack(128),

            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),

            ResStack(64),

            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),

            ResStack(32),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.Conv1d(32, 1, kernel_size=7, stride=1),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        device = next(self.parameters()).device
        if mel.device != device:
            mel = mel.to(device)
        zero = torch.full((mel.size()[0], self.mel_channel, 10), -11.5129).to(device)
        mel = torch.cat((mel, zero), dim=2).type(mel.dtype)
        audio = self.forward(mel)
        audio = audio.squeeze() # collapse all dimension except time axis
        if len(audio.size()) > 1:
          audio = audio[:, :-(self.hop_length*10)]
        else:
          audio = audio[:-(self.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        return audio



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )
        
        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

    def forward(self, x):
        ret = list()

        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret # [(feat, score), (feat, score), (feat, score)]


class ResStack(nn.Module):
    def __init__(self, channel):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channel, channel, kernel_size=1),
            )
            for i in range(3)
        ])

        self.shortcuts = nn.ModuleList([
            nn.Conv1d(channel, channel, kernel_size=1)
            for i in range(3)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


### FASTSPEECH2 

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, config):
        super().__init__()
        self.duration_predictor = VariancePredictor(config)
        self.length_regulator = LengthRegulator(config)
        self.pitch_predictor = VariancePredictor(config)
        self.energy_predictor = VariancePredictor(config)
        self.log_offset = config.log_offset        
        self.pitch_bins = nn.Parameter(torch.exp(torch.linspace(np.log(config.f0_min), np.log(config.f0_max), config.n_bins-1)))
        self.energy_bins = nn.Parameter(torch.linspace(config.energy_min, config.energy_max, config.n_bins-1))
        self.pitch_embedding = nn.Embedding(config.n_bins, config.encoder_hidden)
        self.energy_embedding = nn.Embedding(config.n_bins, config.encoder_hidden)
    
    def get_mask_from_lengths(self, lengths, max_len=None):
        device = next(self.parameters()).device
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

        return mask

    def forward(self, x, src_mask, mel_mask=None, duration_target=None, pitch_target=None, energy_target=None, max_len=None):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-self.log_offset), min=0)
            #if max_len is None:
            #  max_len = x.shape[1]*256
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = self.get_mask_from_lengths(mel_len)
        if mel_mask is not None and mel_mask.size()[-1]==0:
          pitch_prediction = self.pitch_predictor(x, None)
        else:
          pitch_prediction = self.pitch_predictor(x, mel_mask)
        dtype = pitch_prediction.dtype
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target.float(), self.pitch_bins))
        else:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_prediction.float(), self.pitch_bins))
        
        energy_prediction = self.energy_predictor(x, mel_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target.float(), self.energy_bins))
        else:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_prediction.float(), self.energy_bins))
        
        x = x + pitch_embedding + energy_embedding
        
        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, config):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        device = x.device
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, config):
        super(VariancePredictor, self).__init__()

        self.input_size = config.encoder_hidden
        self.filter_size = config.variance_predictor_filter_size
        self.kernel = config.variance_predictor_kernel_size
        self.conv_output_size = config.variance_predictor_filter_size
        self.dropout =config.variance_predictor_dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, config):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, config):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=config.fft_conv1d_kernel_size[0], padding=(config.fft_conv1d_kernel_size[0]-1)//2)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=config.fft_conv1d_kernel_size[1], padding=(config.fft_conv1d_kernel_size[1]-1)//2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        device = q.device
        if mask is not None and mask.device != device:
          mask = mask.to(device)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    
    return torch.FloatTensor(sinusoid_table)

from itertools import groupby

class UpsampleEncoder(nn.Module):
    ''' UpsampleEncoder to map LM output to Phonemes'''

    def __init__(self, config, src_word_emb):
        super(UpsampleEncoder, self).__init__()
        vocab_size=config.vocab_size # len(vocab)+1
        len_max_seq=config.max_seq_len
        embedding_size=config.encoder_hidden
        n_layers=config.encoder_layer
        n_head=config.encoder_head
        d_k=config.encoder_hidden // config.encoder_head
        d_v=config.encoder_hidden // config.encoder_head
        d_model=config.encoder_hidden
        d_inner=config.fft_conv1d_filter_size
        dropout=config.encoder_dropout
        self.src_word_emb = src_word_emb
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len
        self.encoder_hidden= config.encoder_hidden

        self.upsample_factor = config.upsample_factor
        n_position = len_max_seq + 1
        self.softmax = nn.Softamx(dim=-1)
        self.conv1d =  nn.Conv1d(in_channels=config.upsample_in, out_channels=self.upsample_factor*embedding_size, kernel_size=3, padding=1)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, embedding_size).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, config=config) for _ in range(n_layers)])
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_ctc_output(self, inputs, logits):
      predicted_ids = torch.argmax(logits, dim=-1)
      batch =[]
      for tokens in predicted_ids.tolist():
            # group same tokens into non-repeating tokens in CTC style decoding
            grouped_tokens2 =  [list(token_group[1]) for token_group in groupby(tokens)]
            grouped_tokens3 = []
            i = 0
            word = []
            for group in grouped_tokens2:
              if group[0] == self.eos_token_id:
                break
              elif group[0] == self.pad_token_id and grouped_tokens3 == [] and word == []:
                pass
              elif word == []:
                if group[0] == tokenizer.pad_token_id:
                  word = [[]]
                else:
                  word = [list(zip(group, range(i, i+len(group))))]
              else:
                  if group[0] != tokenizer.pad_token_id:
                    word[0].extend(list(zip(group, range(i, i+len(group)))))
              i+= len(group)
            batch.append([inputs[[a[1] for a in word for word in grouped_tokens3]].mean(dim=-1)])
      # TODO, we need to pad and bactch and return ctc_len (or a mask)
      return torch.Tensor(batch)

    def forward(self, src_seq, src_len, max_src_len, return_attns=False, labels=None):
        device = next(self.parameters()).device
        if src_seq.device != device:
            src_seq = src_seq.to(device)
        src_seq = self.conv1d(src_seq)
        input_lengths = src_len*self.upsample_factor
        src_mask = self.get_mask_from_lengths(input_lengths, max_src_len)
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        if src_seq.shape[1] > self.max_seq_len:
            position = get_sinusoid_encoding_table(src_seq.shape[1], self.encoder_hidden)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.dtype).to(device)
            position.requires_grad=False
            enc_output = src_seq + position
        else:
            enc_output = src_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        logits = self.lm_head(enc_output)
        ctc_output = self.get_ctc_output(enc_output, logits)
        # get ctc_output, taking into account labels len if any
        ctc_output = self.fc(ctc_output)

        total_loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            #attention_mask = (
            #    attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            #)
            #input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                total_loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.pad_token_id,
                    reduction=self.ctc_loss_reduction,
                    zero_infinity=self.ctc_zero_infinity,
                )

            # do mse_loss of actual embeddings values and ctc_output, taking into account the mask.
            target =  self.src_word_emb(labels)
            target_mask = self.get_mask_from_lengths(ctc_len, max_input_len)
            loss = self.mse_loss(ctc_output, target)
            loss_dtype = loss.dtype
            loss = (loss * target_mask.type(loss_dtype)).sum() # gives \sigma_euclidean over unmasked elements
            non_zero_elements = target_mask.sum()
            total_loss = total_loss + (loss / non_zero_elements)


        return ctc_output, ctc_len, total_loss 



class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self, config):
        super(Encoder, self).__init__()
        vocab_size=config.vocab_size # len(vocab)+1
        len_max_seq=config.max_seq_len
        embedding_size=config.encoder_hidden
        n_layers=config.encoder_layer
        n_head=config.encoder_head
        d_k=config.encoder_hidden // config.encoder_head
        d_v=config.encoder_hidden // config.encoder_head
        d_model=config.encoder_hidden
        d_inner=config.fft_conv1d_filter_size
        dropout=config.encoder_dropout
        self.pad_token_id = config.pad_token_id
        self.encoder_hidden = config.encoder_hidden
        self.max_seq_len =config.max_seq_len
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(vocab_size, embedding_size, padding_idx=self.pad_token_id)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, embedding_size).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, config=config) for _ in range(n_layers)])

    def forward(self, src_seq, mask, return_attns=False):
        device = next(self.parameters()).device
        if src_seq.device != device:
            src_seq = src_seq.to(device)

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        # -- Forward
        if src_seq.shape[1] > self.max_seq_len:
            position = get_sinusoid_encoding_table(src_seq.shape[1], self.encoder_hidden)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.dtype).to(device)
            position.requires_grad = False
            enc_output = self.src_word_emb(src_seq) + position
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()
        len_max_seq=config.max_seq_len
        embedding_size=config.encoder_hidden
        n_layers=config.decoder_layer
        n_head=config.decoder_head
        d_k=config.decoder_hidden // config.decoder_head
        d_v=config.decoder_hidden // config.decoder_head
        d_model=config.decoder_hidden
        d_inner=config.fft_conv1d_filter_size
        dropout=config.decoder_dropout
        self.max_seq_len= config.max_seq_len
        self.decoder_hidden = config.decoder_hidden
        n_position = len_max_seq + 1

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, embedding_size).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, config=config) for _ in range(n_layers)])

    def forward(self, enc_seq, mask, return_attns=False):
        device = next(self.parameters()).device
        if enc_seq.device != device:
            enc_seq = enc_seq.to(device)
        if mask.device != device:
            mask = mask.to(device)

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if enc_seq.shape[1] > self.max_seq_len:
            position =  get_sinusoid_encoding_table(enc_seq.shape[1], self.decoder_hidden)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.dtype).to(device)
            position.requires_grad=False
            dec_output = enc_seq + position
        else:
            dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        dec_output=dec_output.type(enc_seq.dtype)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output



class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v, config):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, config)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, config)
    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self,
                 n_mel_channels=80,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'),

                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size,
                             stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'),

                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim,
                         n_mel_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'),

                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x

class FastSpeech2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2Config
    base_model_prefix = "fastspeech2"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight.data)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()


class FastSpeech2(FastSpeech2PreTrainedModel):
    """ FastSpeech2 """

    def __init__(self, config):
        super().__init__(config)

        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(config.decoder_hidden, config.n_mel_channels)
        self.g2p = G2PNet(config, self.encoder.src_word_emb)
        self.use_postnet = config.use_postnet
        if self.use_postnet:
            self.postnet = PostNet()
        self.generator = Generator(config)
        self.hop_length = config.hop_length
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.init_weights()

    def get_input_embeddings(self):
        return self.encoder.src_word_emb

    def set_input_embeddings(self, value):
        self.encoder.src_word_emb = value
        self.g2p.encoder.emb = value
        self.g2p.decoder.emb = value

    def get_mask_from_lengths(self, lengths, max_len=None):
        device = next(self.parameters()).device
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

        return mask

    def to_wav_list(self, wav_tensor, wav_len):
      wav = [w[:l] for w, l in zip(wav_tensor, wav_len)]
      return wav

    def forward(self, src_seq, attention_mask, return_wav_list=True, mel_len=None, d_target=None, p_target=None, e_target=None, max_src_len=None, max_mel_len=None, log_d_target=None, use_postnet=None):
        #src_mask = self.get_mask_from_lengths(src_len, max_src_len)

        device = next(self.parameters()).device
        if self.training and return_wav_list:
          warnings.warn("In train mode, a tensor for all wavs should be returned. automatically setting return_wav_list=False")
          return_wav_list=False
        if src_seq.device != device:
            src_seq = src_seq.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        src_mask = attention_mask == 0.
        mel_mask = self.get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        encoder_output = self.encoder(src_seq, src_mask)
        if d_target is not None:
            variance_adaptor_output, log_d_predicted, p_predicted, e_predicted, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, log_d_predicted, p_predicted, e_predicted, mel_len, mel_mask = self.variance_adaptor(
                    encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        decoder_output = self.decoder(variance_adaptor_output, mel_mask)
        mel_output = self.mel_linear(decoder_output)
        
        if use_postnet is not False and self.use_postnet:
            mel_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_postnet = mel_output
        wav = self.generator.inference(mel_postnet.transpose(1, 2))
       
        
        if len(wav.size())>1: 
          wav_mask = torch.zeros(wav.size()).to(device)
          if return_wav_list:
            wav = [w[:l*self.hop_length] for w, l in zip(wav, mel_len)]
          #there's probably a more efficient way to do this using arange
          for i, l in enumerate(mel_len):
            wav_mask[i, :l*self.hop_length] = 1.
          wav_len =  [l*self.hop_length for l in  mel_len]
        else:
          wav_len =  [len(wav)*self.hop_length]
          wav_mask = torch.ones((1, len(wav))).to(device)
          if return_wav_list:
            wav = [wav]
          else:
           wav = wav.unsqueeze(0)
          
        if log_d_target is not None:
            log_d_target_prev_req_grad = log_d_target.requires_grad
            p_target_prev_req_grad = p_target.requires_grad
            e_target_prev_req_grad = e_target.requires_grad
            mel_target_prev_req_grad = mel_target.requires_grad

            log_d_target.requires_grad = False
            p_target.requires_grad = False
            e_target.requires_grad = False
            mel_target.requires_grad = False

            log_d_predicted = log_d_predicted.masked_select(src_mask)
            log_d_target = log_d_target.masked_select(src_mask)
            p_predicted = p_predicted.masked_select(mel_mask)
            p_target = p_target.masked_select(mel_mask)
            e_predicted = e_predicted.masked_select(mel_mask)
            e_target = e_target.masked_select(mel_mask)

            mel = mel.masked_select(mel_mask.unsqueeze(-1))
            mel_postnet = mel_postnet.masked_select(mel_mask.unsqueeze(-1))
            mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))

            mel_loss = self.mse_loss(mel, mel_target)
            mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

            d_loss = self.mae_loss(log_d_predicted, log_d_target)
            p_loss = self.mae_loss(p_predicted, p_target)
            e_loss = self.mae_loss(e_predicted, e_target)

            log_d_target.requires_grad = log_d_target_prev_req_grad
            p_target.requires_grad = p_target_prev_req_grad
            e_target.requires_grad = e_target_prev_req_grad
            mel_target.requires_grad = mel_target_prev_req_grad
            total_loss = mel_loss + mel_postnet_loss + d_loss + p_loss + e_loss
        else:
            total_loss = None

        return mel_output, mel_postnet, log_d_predicted, p_predicted, e_predicted, src_mask, mel_mask, mel_len, wav_len, wav_mask, wav, total_loss
    


class FastSpeech2ForPretraining(FastSpeech2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fastspeech2 = FastSpeech2(config)
        self.init_weights()
        self.tie_weights()

    def get_output_embeddings(self):
        return self.fastspeech2.g2p.decoder.fc

    def set_output_embeddings(self, new_embeddings):
        self.fastspeech2.g2p.decoder.fc = new_embeddings
    
    def to_wav_list(self, wav_tensor, wav_len):
      return self.fastspeech2.to_wav_list(wav_tensor, wav_len)
    
    def forward(self, input_ids, attention_mask, **vargs):
      return self.fastspeech2(input_ids, attention_mask, **vargs)


