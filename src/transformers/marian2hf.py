import numpy as np
import yaml
import argparse

from transformers import BertModel, BertConfig
from durbango import pickle_save, lmap, remove_prefix
import torch
from transformers.marian_constants import *


def convert_encoder_layer(opus_dict, layer_tag):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_tag): continue
        stripped = remove_prefix(k, layer_tag)
        v = opus_dict[k].T  # besides embeddings, everything must be transposed.
        sd[BERT_LAYER_CONVERTER[stripped]] = torch.tensor(v).squeeze()
    return sd

def load_layers_(bmodel: BertModel, opus_state: dict):
    for i, layer in enumerate(bmodel.encoder.layer):
        layer_tag = f'decoder_l{i+1}_' if bmodel.config.is_decoder else f'encoder_l{i+1}_'
        sd = convert_encoder_layer(opus_state, layer_tag)
        layer.load_state_dict(sd, strict=True)

def convert_embeddings_(opus_state) -> dict:
    sd = {}
    for k in EMBED_CONVERTER:
        if k in opus_state:
            sd[EMBED_CONVERTER[k]] = opus_state[k]
    return sd




def convert_to_berts(opus_state: dict):
    marian_cfg: dict = load_model_yaml(opus_state)
    vocab_size = opus_state['Wemb'].shape[0]
    hidden_size, intermediate_shape = opus_state['encoder_l1_ffn_W1'].shape
    encoder_cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=int(marian_cfg['enc-depth']),
        num_attention_heads=int(marian_cfg['transformer-heads']), #getattr(yaml_cfg, 'transformer-heads', 'swish'),
        intermediate_size=intermediate_shape,
        hidden_act=marian_cfg['transformer-aan-activation'],
    )
    encoder = BertModel(encoder_cfg)
    decoder_cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=int(marian_cfg['dec-depth']),
        num_attention_heads=int(marian_cfg['transformer-heads']), #getattr(yaml_cfg, 'transformer-heads', 'swish'),
        intermediate_size=intermediate_shape,
        hidden_act=marian_cfg['transformer-aan-activation'],
        is_decoder=True
    )
    bdecoder = BertModel(decoder_cfg)
    load_layers_(bdecoder, opus_state)
    embs_state: dict = convert_embeddings_(opus_state)
    result = encoder.load_state_dict(embs_state)
    print(f'Embeddings: {result})')  # TODO(SS): logger
    return encoder, bdecoder


def load_model_yaml(opus_dict):
    cfg_str = ''.join(lmap(chr, opus_dict['special:model.yml']))
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    for k in ['dec-depth', 'enc-depth', 'transformer-heads']:
        yaml_cfg[k] = int(yaml_cfg[k])
    return yaml_cfg


DEFAULT_PATH = '/Users/shleifer/marian/opus_en_fr/opus.bpe32k-bpe32k.transformer.model1.npz.best-perplexity.npz'
def main(opus_path=DEFAULT_PATH):
    opus_state = np.load(opus_path)
    enc, dec = convert_to_berts(opus_state)




