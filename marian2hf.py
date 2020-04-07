import numpy as np
import yaml
import argparse

from transformers import BertModel, BertConfig
from durbango import pickle_save, lmap
from transformers.modeling_bert import BertLayer


def desc_str_to_config(desc_str: np.ndarray):
    cfg_str = ''.join(lmap(chr, ))
    yaml_cfg = yaml.load(cfg_str[:-1])



import torch
from .marian_constants import *
def convert_layer(opus_dict, layer_tag):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_tag): continue
        stripped = k.lstrip(layer_tag)
        v = opus_dict[k]
        if needs_transpose[stripped]:
            v = v.T
        sd[LAYER_CONVERTER[stripped]] = torch.tensor(v).squeeze()
        #remaining_keys.remove(k)
    return sd
def load_encoder_layers_(bmodel, opus_dict):
    for i, layer in enumerate(bmodel.encoder.layer):
        layer_tag = f'encoder_l{i+1}'
        sd = convert_layer(opus_dict, layer_tag)
        layer.load_state_dict(sd, strict=True)


#remaining_keys = set(opus_dict.keys())




def convert_encoder(opus_dict):
    n_enc_layers = max(int(k.split('_')[1].lstrip('l')) for k in opus_dict if k.startswith('encoder'))
    vocab_size = opus_dict['Wemb'].shape[0]
    hidden_size, intermediate_shape =opus_dict['encoder_l1_ffn_W1'].shape
    cfg_str = ''.join(lmap(chr, opus_dict['special:model.yml']))
    yaml_cfg = yaml.load(cfg_str[:-1])
    encoder_cfg  = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=n_enc_layers,
        num_attention_heads=yaml_cfg['transformer-heads'], #getattr(yaml_cfg, 'transformer-heads', 'swish'),
        intermediate_size=intermediate_shape,
        hidden_act=yaml_cfg['transformer-aan-activation'],
    )
    bmodel = BertModel(encoder_cfg)
    load_encoder_layers_(bmodel, opus_dict)
    return bmodel


def _rename_embs(bmodel, opus_dict):
    sd_new = {"word_embeddings.weight": opus_dict['Wemb']}
    if 'Wpos' in opus_dict:
            sd_new["position_embeddings.weight"] = opus_dict['Wpos']



DEFAULT_PATH = '/Users/shleifer/marian/opus_en_fr/opus.bpe32k-bpe32k.transformer.model1.npz.best-perplexity.npz'
def main(opus_path=DEFAULT_PATH):
    opus_dict = np.load(opus_path)
    encoder: BertModel = convert_encoder(opus_dict)
    BertModel.embeddings.load_state_dict(_rename_embs(opus_dict))
