import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import torch
import wget
import yaml

from durbango import lmap, pickle_save, remove_prefix
from transformers import BertConfig, BertModel, MarianConfig, MarianModel, BartConfig
from transformers.marian_constants import BERT_LAYER_CONVERTER, EMBED_CONVERTER, BART_CONVERTER, assume_vals
from transformers.tokenization_marian import MarianSPTokenizer
from transformers.tokenization_utils import ADDED_TOKENS_FILE, TOKENIZER_CONFIG_FILE

OPUS_MODELS_PATH = "/Users/shleifer/OPUS-MT-train/models"  # git clone git@github.com:Helsinki-NLP/Opus-MT.git


def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T  # besides embeddings, everything must be transposed.
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd


def load_layers_(layer_lst: torch.nn.ModuleList, opus_state: dict, converter=BERT_LAYER_CONVERTER, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=True)





CONFIG_KEY = "special:model.yml"
import warnings
import numpy as np
def add_emb_entries(wemb, n_special_tokens=1):
    vsize, d_model = wemb.shape
    new_shit = np.zeros((n_special_tokens, d_model))
    return np.concatenate([wemb, new_shit])




from transformers import BartConfig, BartForConditionalGeneration


def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)

def _cast_obj(v):
    bool_dct = {'true': True, 'false': False}
    if not isinstance(v,str): return v
    elif v in bool_dct: return bool_dct[v]
    try: return int(v)
    except (TypeError, ValueError): return v
def cast_marian_config(raw_cfg: Dict[str, str]) -> Dict: return {k: _cast_obj(v) for k,v in raw_cfg.items()}


def load_config_from_state_dict(opus_dict):
    cfg_str = "".join(lmap(chr, opus_dict[CONFIG_KEY]))
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    return cast_marian_config(yaml_cfg)


def find_model_file(dest_dir):  # this one better
    model_files = list(Path(dest_dir).glob("*.npz"))
    assert len(model_files) == 1, model_files
    model_file = model_files[0]
    return model_file


def parse_readmes(repo_path=OPUS_MODELS_PATH):
    results = {}
    for p in Path(repo_path).ls():
        n_dash = p.name.count("-")
        if n_dash == 0:
            continue
        else:
            lns = list(open(p / "README.md").readlines())
            results[p.name] = parse_readme(lns)
    return results


def parse_readme(lns):
    subres = {}
    for ln in [x.strip() for x in lns]:
        if not ln.startswith("*"):
            continue
        ln = ln[1:].strip()

        for k in ["download", "dataset", "models", "model", "pre-processing"]:
            if ln.startswith(k):
                break
        else:
            continue
        if k in ["dataset", "model", "pre-processing"]:
            splat = ln.split(":")
            _, v = splat
            subres[k] = v
        elif k == "download":
            v = ln.split("(")[-1][:-1]
            subres[k] = v
    return subres


def unzip(zip_path, dest_dir):
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(dest_dir)


def download_unzip(url, dest_dir):
    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_metadata(dest_dir):
    dest = Path(dest_dir)
    dname = dest.name.split("-")
    dct = dict(target_lang=dname[-1], source_lang="-".join(dname[:-1]))
    save_json(dct, dest_dir / TOKENIZER_CONFIG_FILE)


def add_to_vocab_(vocab: Dict[str, int], special_tokens: List[str]):
    start = max(vocab.values()) + 1
    for tok in special_tokens:
        if tok in vocab:
            continue
        vocab[tok] = start
        start += 1


def add_special_tokens_to_vocab(model_dir: Path = "en-de"):
    vocab = yaml.load(open("en-de/opus.spm32k-spm32k.vocab.yml"), Loader=yaml.BaseLoader)
    vocab = {k: int(v) for k, v in vocab.items()}
    add_to_vocab_(vocab, ["<pad>"])
    save_json(vocab, model_dir / "vocab.json")


def save_tokenizer(self, save_directory):
    # FIXME, what if you add tokens?
    dest = Path(save_directory)
    src_path = Path(self.init_kwargs["source_spm"])

    for dest_name in {"source.spm", "target.spm", "tokenizer_config.json"}:
        shutil.copyfile(src_path.parent / dest_name, dest / dest_name)
    save_json(self.encoder, dest / "vocab.json")


def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    assert v1 == v2, f'hparams {k1},{k2} differ: {v1} != {v2}'


def convert_v2(marian_cfg, pad_token_id, eos_token_id, bos_token_id):
    for k,v in assume_vals.items():
        actual = marian_cfg[k]
        assert actual == v, f'Unexpected config value for {k} expected {v} got {actual}'

    check_equal(marian_cfg, "transformer-ffn-activation", "transformer-aan-activation")
    check_equal(marian_cfg, "transformer-ffn-depth", "transformer-aan-depth")
    check_equal(marian_cfg, "transformer-dim-ffn", "transformer-dim-aan")

    bart_cfg = BartConfig(
        vocab_size=marian_cfg['vocab_size'],
        decoder_layers=marian_cfg['dec-depth'],
        encoder_layers=marian_cfg['enc-depth'],
        decoder_attention_heads=marian_cfg["transformer-heads"],
        encoder_attention_heads=marian_cfg["transformer-heads"],
        static_position_embeddings=marian_cfg["transformer-train-position-embeddings"],
        decoder_ffn_dim=marian_cfg["transformer-dim-ffn"],
        encoder_ffn_dim=marian_cfg["transformer-dim-ffn"],
        d_model=marian_cfg["dim-emb"],
        activation_function=marian_cfg["transformer-aan-activation"],
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        max_position_embeddings=marian_cfg["dim-emb"],
        scale_embedding=True,
    )
    return bart_cfg


class OpusState:
    def __init__(self, npz_path):
        self.state_dict = np.load(npz_path)
        cfg = load_config_from_state_dict(self.state_dict)
        self.state_dict = dict(self.state_dict)
        self.state_dict['Wemb'] = add_emb_entries(self.state_dict['Wemb'], 1)
        pad_token_id = bos_token_id =  self.state_dict['Wemb'].shape[0] - 1

        #self.cfg['vocab_size'] = cfg['dim-vocabs'][0]
        assert cfg['dim-vocabs'][0] == cfg['dim-vocabs'][1]
        cfg['vocab_size'] = pad_token_id + 1
        #self.state_dict['Wemb'].sha
        self.state_keys = list(self.state_dict.keys())
        if "Wtype" in self.state_dict:
            raise ValueError('found Wtype key')
        self.encoder_l1 = self.sub_keys('encoder_l1')
        self.decoder_l1 = self.sub_keys('decoder_l1')
        self.decoder_l2 = self.sub_keys('decoder_l2')
        if len(self.encoder_l1) != 16:
            warnings.warn(f'Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}')
        if len(self.decoder_l1) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')

        if len(self.decoder_l2) != 26:
            warnings.warn(f'Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}')

        self.cfg = cfg

    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if k.startswith("encoder_l") or k.startswith("decoder_l") or k == CONFIG_KEY:
                continue
            else:
                extra.append(k)
        return extra

    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    def load_marian_model(self,  pad_token_id, eos_token_id) -> MarianModel:
        state_dict, marian_cfg = self.state_dict, self.cfg

        hidden_size, intermediate_shape = state_dict["encoder_l1_ffn_W1"].shape
        assert hidden_size == marian_cfg["dim-emb"]
        cfg = convert_v2(marian_cfg, pad_token_id, eos_token_id, None)
        model = BartForConditionalGeneration(cfg)
        print("loaded empty marian model")
        load_layers_(model.model.encoder.layers, state_dict, converter=BART_CONVERTER, )
        print("loaded encoder")
        load_layers_(model.model.decoder.layers, state_dict, converter=BART_CONVERTER, is_decoder=True)

        embs_state: dict = convert_embeddings_(state_dict)
        wemb_tensor = torch.nn.Parameter(torch.FloatTensor(state_dict['Wemb']))
        model.model.shared.weight = wemb_tensor
        model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared

        if 'Wpos' in state_dict:
            wpos_tensor = torch.tensor(state_dict['Wpos'])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        #result = model.model.encoder.load_state_dict(embs_state, strict=False)
        #print(f"encoder.embeddings: {result})")  # TODO(SS): logger
        #result2 = model.model.decoder.load_state_dict(embs_state, strict=False)
        #model.shared = model.model.decoder.embed_tokens
        print(f'extra bart keys: {self.get_extra_bart_keys(model)}, extra_opus: {self.extra_keys}')
        assert model.model.shared.padding_idx == pad_token_id


        return model


    def get_extra_bart_keys(self, model) -> list:
        extra_bart_keys = []
        for k in model.model.state_dict():
            if k.startswith('encoder.layers') or k.startswith('decoder.layers'):
                continue
            elif k.startswith('encoder.embed') or k.startswith('decoder.embed'):
                continue
            else:
                extra_bart_keys.append(k)

        return extra_bart_keys

def convert_embeddings_(state_dict) -> dict:
    sd = {}
    for k in EMBED_CONVERTER:
        if k in state_dict:
            sd[EMBED_CONVERTER[k]] = torch.tensor(state_dict[k])
    return sd

def main(source_dir, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    add_special_tokens_to_vocab(source_dir)
    tokenizer = MarianSPTokenizer.from_pretrained(str(source_dir))
    save_tokenizer(tokenizer, dest_dir)

    model_path = find_model_file(source_dir)
    opus_state = OpusState(model_path)  # not actually a dict
    assert opus_state.cfg['vocab_size'] == len(tokenizer.encoder)
    decoder_yml = cast_marian_config(load_yaml(source_dir / 'decoder.yml'))
    decoder_remap = {"beam-size": 'num_beams', "normalize": 'len_pen??', 'word-penalty': 'len_pen??'}

    save_json(opus_state.cfg, dest_dir / "marian_original_config.json")

    model = opus_state.load_marian_model(tokenizer.pad_token_id,
                              tokenizer.eos_token_id)
    #model.resize_token_embeddings(tokenizer.vocab_size)  # account for added pad token
    #model.config.vocab_size = tokenizer.vocab_size
    #model.config.eos_token_id = tokenizer.eos_token_id
    #model.config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src", type=str, help="path to marian model dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    source_dir = Path(args.src)
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    main(source_dir, dest_dir)


