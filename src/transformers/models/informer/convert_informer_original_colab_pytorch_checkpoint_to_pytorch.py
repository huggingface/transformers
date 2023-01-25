# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Informer checkpoint."""

"""
Assumption: 

Informer2020 repository is git-cloned from
https://github.com/elisim/Informer2020/tree/hf

"hf" branch. There, I created a Informer's checkpoint from the official colab notebook.

See also: https://github.com/elisim/Informer2020/blob/hf/create_checkpoint_from_offical_colab.ipynb
"""
import argparse
import os
from pathlib import Path

import torch
from torch import nn

from transformers import InformerConfig, InformerModel
from transformers.utils import logging


import sys
if not 'Informer2020' in sys.path:
    sys.path += ['Informer2020']


from Informer2020.exp.exp_informer import Exp_Informer

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def _create_informer_args():
    """
    Arguments are taken from the offical colab example:
    https://colab.research.google.com/drive/1_X7O2BkFLvqyCdZzDZvV2MB0aAvYALLC

    I only comment arguments that are not needed for the model creation (e.g. data_path, use_gpu)
    """
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict()

    ### BoilerCode
    args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
    # args.data = 'ETTh1'  # data
    # args.root_path = './ETDataset/ETT-small/'  # root path of data file
    # args.data_path = 'ETTh1.csv'  # data file
    # args.checkpoints = './informer_checkpoints'  # location of model checkpoints

    ### TS
    args.features = 'M'  # forecasting task, options:[M, S, MS]
    # M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'OT'  # target feature in S or MS task
    args.freq = 'h'  # freq for time features encoding,
    # options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly],
    # you can also use more detailed freq like 15min or 3h

    ### Encoder Decoder
    args.seq_len = 96  # input sequence length of Informer encoder
    args.label_len = 48  # start token length of Informer decoder
    args.pred_len = 24  # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 7  # encoder input size
    args.dec_in = 7  # decoder input size
    args.c_out = 7  # output size
    args.factor = 5  # probsparse attn factor
    args.d_model = 512  # dimension of model
    args.n_heads = 8  # num of heads
    args.e_layers = 2  # num of encoder layers
    args.d_layers = 1  # num of decoder layers
    args.d_ff = 2048  # dimension of fcn in model
    args.dropout = 0.05  # dropout
    args.attn = 'prob'  # attention used in encoder, options:[prob, full]
    args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu'  # activation
    args.distil = True  # whether to use distilling in encoder
    args.output_attention = False  # whether to output attention in ecoder
    args.mix = True
    args.padding = 0

    ### Training
    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False  # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 6
    args.patience = 3
    args.des = 'exp'

    # args.use_gpu = False  # True if torch.cuda.is_available() else False
    # args.gpu = 0
    #
    # args.use_multi_gpu = False
    # args.devices = '0,1,2,3'

    args.detail_freq = args.freq  # the actual freq
    args.freq = args.freq[-1:]  # Not important

    return args


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_informer_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pth"""
    exp = Exp_Informer(args=_create_informer_args())
    sd = torch.load(checkpoint_path, map_location="cpu")
    exp.model.load_state_dict(sd)
    return exp.model


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# @torch.no_grad()
# def convert_informer_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
#     """
#     Copy/paste/tweak model's weights to our BERT structure.
#     """
#     informer = load_informer_checkpoint(checkpoint_path)
#
#     informer.model.upgrade_state_dict(informer.model.state_dict())
#     if hf_checkpoint_name is None:
#         hf_checkpoint_name = checkpoint_path.replace(".", "-")
#     config = BartConfig.from_pretrained(hf_checkpoint_name)
#
#     if checkpoint_path == "bart.large.mnli":
#         state_dict = bart.state_dict()
#         remove_ignore_keys_(state_dict)
#         state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
#         for src, dest in mnli_rename_keys:
#             rename_key(state_dict, src, dest)
#         model = BartForSequenceClassification(config).eval()
#         model.load_state_dict(state_dict)
#         fairseq_output = bart.predict("mnli", tokens, return_logits=True)
#         new_model_outputs = model(tokens)[0]  # logits
#     else:  # no classification heads to worry about
#         state_dict = bart.model.state_dict()
#         remove_ignore_keys_(state_dict)
#         state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
#         fairseq_output = bart.extract_features(tokens)
#         if hf_checkpoint_name == "facebook/bart-large":
#             model = BartModel(config).eval()
#             model.load_state_dict(state_dict)
#             new_model_outputs = model(tokens).model[0]
#         else:
#             model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
#             model.model.load_state_dict(state_dict)
#             if hasattr(model, "lm_head"):
#                 model.lm_head = make_linear_from_emb(model.model.shared)
#             new_model_outputs = model.model(tokens)[0]
#
#     # Check results
#     assert fairseq_output.shape == new_model_outputs.shape
#     assert (fairseq_output == new_model_outputs).all().item()
#     Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
#     model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    informer_checkpoint_default_path = "./Informer2020/informer_checkpoints/informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0/checkpoint.pth"

    # parser = argparse.ArgumentParser()
    # parser.add_argument("informer_path", default=None, type=str, help="a path to a model.pth on local filesystem.")
    # parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # args = parser.parse_args()

    # convert_informer_checkpoint(args.informer_path, args.pytorch_dump_folder_path)
    informer = load_informer_checkpoint(informer_checkpoint_default_path)
    print(informer)
