# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from transformers import MarianConfig, MarianMTModel, MarianTokenizer
from transformers.hf_api import HfApi


def remove_suffix(text: str, suffix: str):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # or whatever


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T  # besides embeddings, everything must be transposed.
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd


def load_layers_(layer_lst: nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=True)


def find_pretrained_model(src_lang: str, tgt_lang: str) -> List[str]:
    """Find models that can accept src_lang as input and return tgt_lang as output."""
    prefix = "Helsinki-NLP/opus-mt-"
    api = HfApi()
    model_list = api.model_list()
    model_ids = [x.modelId for x in model_list if x.modelId.startswith("Helsinki-NLP")]
    src_and_targ = [
        remove_prefix(m, prefix).lower().split("-") for m in model_ids if "+" not in m
    ]  # + cant be loaded.
    matching = [f"{prefix}{a}-{b}" for (a, b) in src_and_targ if src_lang in a and tgt_lang in b]
    return matching


def add_emb_entries(wemb, final_bias, n_special_tokens=1):
    vsize, d_model = wemb.shape
    embs_to_add = np.zeros((n_special_tokens, d_model))
    new_embs = np.concatenate([wemb, embs_to_add])
    bias_to_add = np.zeros((n_special_tokens, 1))
    new_bias = np.concatenate((final_bias, bias_to_add), axis=1)
    return new_embs, new_bias


def _cast_yaml_str(v):
    bool_dct = {"true": True, "false": False}
    if not isinstance(v, str):
        return v
    elif v in bool_dct:
        return bool_dct[v]
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def cast_marian_config(raw_cfg: Dict[str, str]) -> Dict:
    return {k: _cast_yaml_str(v) for k, v in raw_cfg.items()}


CONFIG_KEY = "special:model.yml"


def load_config_from_state_dict(opus_dict):
    import yaml

    cfg_str = "".join([chr(x) for x in opus_dict[CONFIG_KEY]])
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    return cast_marian_config(yaml_cfg)


def find_model_file(dest_dir):  # this one better
    model_files = list(Path(dest_dir).glob("*.npz"))
    assert len(model_files) == 1, model_files
    model_file = model_files[0]
    return model_file


# Group Names Logic: change long opus model names to something shorter, like opus-mt-en-ROMANCE
ROM_GROUP = (
    "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT"
    "+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co"
    "+nap+scn+vec+sc+ro+la"
)
GROUPS = [
    ("cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh", "ZH"),
    (ROM_GROUP, "ROMANCE"),
    ("de+nl+fy+af+da+fo+is+no+nb+nn+sv", "NORTH_EU"),
    ("da+fo+is+no+nb+nn+sv", "SCANDINAVIA"),
    ("se+sma+smj+smn+sms", "SAMI"),
    ("nb_NO+nb+nn_NO+nn+nog+no_nb+no", "NORWAY"),
    ("ga+cy+br+gd+kw+gv", "CELTIC"),  # https://en.wikipedia.org/wiki/Insular_Celtic_languages
]
GROUP_TO_OPUS_NAME = {
    "opus-mt-ZH-de": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-de",
    "opus-mt-ZH-fi": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
    "opus-mt-ZH-sv": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-sv",
    "opus-mt-SCANDINAVIA-SCANDINAVIA": "da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv",
    "opus-mt-NORTH_EU-NORTH_EU": "de+nl+fy+af+da+fo+is+no+nb+nn+sv-de+nl+fy+af+da+fo+is+no+nb+nn+sv",
    "opus-mt-de-ZH": "de-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-en_el_es_fi-en_el_es_fi": "en+el+es+fi-en+el+es+fi",
    "opus-mt-en-ROMANCE": "en-fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
    "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
    "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la",
    "opus-mt-en-CELTIC": "en-ga+cy+br+gd+kw+gv",
    "opus-mt-es-NORWAY": "es-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-fi_nb_no_nn_ru_sv_en-SAMI": "fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms",
    "opus-mt-fi-ZH": "fi-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-fi-NORWAY": "fi-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-ROMANCE-en": "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
    "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
    "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la-en",
    "opus-mt-CELTIC-en": "ga+cy+br+gd+kw+gv-en",
    "opus-mt-sv-ZH": "sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-sv-NORWAY": "sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
}
OPUS_GITHUB_URL = "https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/"
ORG_NAME = "Helsinki-NLP/"


def convert_opus_name_to_hf_name(x):
    """For OPUS-MT-Train/ DEPRECATED"""
    for substr, grp_name in GROUPS:
        x = x.replace(substr, grp_name)
    return x.replace("+", "_")


def convert_hf_name_to_opus_name(hf_model_name):
    """
    Relies on the assumption that there are no language codes like pt_br in models that are not in GROUP_TO_OPUS_NAME.
    """
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    if hf_model_name in GROUP_TO_OPUS_NAME:
        opus_w_prefix = GROUP_TO_OPUS_NAME[hf_model_name]
    else:
        opus_w_prefix = hf_model_name.replace("_", "+")
    return remove_prefix(opus_w_prefix, "opus-mt-")


def get_system_metadata(repo_root):
    import git

    return dict(
        helsinki_git_sha=git.Repo(path=repo_root, search_parent_directories=True).head.object.hexsha,
        transformers_git_sha=git.Repo(path=".", search_parent_directories=True).head.object.hexsha,
        port_machine=socket.gethostname(),
        port_time=time.strftime("%Y-%m-%d-%H:%M"),
    )


# docstyle-ignore
FRONT_MATTER_TEMPLATE = """---
language:
{}
tags:
- translation

license: apache-2.0
---
"""
DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")


def write_model_card(
    hf_model_name: str,
    repo_root=DEFAULT_REPO,
    save_dir=Path("marian_converted"),
    dry_run=False,
    extra_metadata={},
) -> str:
    """
    Copy the most recent model's readme section from opus, and add metadata. upload command: aws s3 sync model_card_dir
    s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
    """
    import pandas as pd

    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    opus_name: str = convert_hf_name_to_opus_name(hf_model_name)
    assert repo_root in ("OPUS-MT-train", "Tatoeba-Challenge")
    opus_readme_path = Path(repo_root).joinpath("models", opus_name, "README.md")
    assert opus_readme_path.exists(), f"Readme file {opus_readme_path} not found"

    opus_src, opus_tgt = [x.split("+") for x in opus_name.split("-")]

    readme_url = f"https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md"

    s, t = ",".join(opus_src), ",".join(opus_tgt)
    metadata = {
        "hf_name": hf_model_name,
        "source_languages": s,
        "target_languages": t,
        "opus_readme_url": readme_url,
        "original_repo": repo_root,
        "tags": ["translation"],
    }
    metadata.update(extra_metadata)
    metadata.update(get_system_metadata(repo_root))

    # combine with opus markdown

    extra_markdown = (
        f"### {hf_model_name}\n\n* source group: {metadata['src_name']} \n* target group: "
        f"{metadata['tgt_name']} \n*  OPUS readme: [{opus_name}]({readme_url})\n"
    )

    content = opus_readme_path.open().read()
    content = content.split("\n# ")[-1]  # Get the lowest level 1 header in the README -- the most recent model.
    splat = content.split("*")[2:]
    print(splat[3])
    content = "*".join(splat)
    content = (
        FRONT_MATTER_TEMPLATE.format(metadata["src_alpha2"])
        + extra_markdown
        + "\n* "
        + content.replace("download", "download original weights")
    )

    items = "\n\n".join([f"- {k}: {v}" for k, v in metadata.items()])
    sec3 = "\n### System Info: \n" + items
    content += sec3
    if dry_run:
        return content, metadata
    sub_dir = save_dir / f"opus-mt-{hf_model_name}"
    sub_dir.mkdir(exist_ok=True)
    dest = sub_dir / "README.md"
    dest.open("w").write(content)
    pd.Series(metadata).to_json(sub_dir / "metadata.json")

    # if dry_run:
    return content, metadata


def make_registry(repo_path="Opus-MT-train/models"):
    if not (Path(repo_path) / "fr-en" / "README.md").exists():
        raise ValueError(
            f"repo_path:{repo_path} does not exist: "
            "You must run: git clone git@github.com:Helsinki-NLP/Opus-MT-train.git before calling."
        )
    results = {}
    for p in Path(repo_path).iterdir():
        n_dash = p.name.count("-")
        if n_dash == 0:
            continue
        else:
            lns = list(open(p / "README.md").readlines())
            results[p.name] = _parse_readme(lns)
    return [(k, v["pre-processing"], v["download"], v["download"][:-4] + ".test.txt") for k, v in results.items()]


def convert_all_sentencepiece_models(model_list=None, repo_path=None, dest_dir=Path("marian_converted")):
    """Requires 300GB"""
    save_dir = Path("marian_ckpt")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    save_paths = []
    if model_list is None:
        model_list: list = make_registry(repo_path=repo_path)
    for k, prepro, download, test_set_url in tqdm(model_list):
        if "SentencePiece" not in prepro:  # dont convert BPE models.
            continue
        if not os.path.exists(save_dir / k):
            download_and_unzip(download, save_dir / k)
        pair_name = convert_opus_name_to_hf_name(k)
        convert(save_dir / k, dest_dir / f"opus-mt-{pair_name}")

        save_paths.append(dest_dir / f"opus-mt-{pair_name}")
    return save_paths


def lmap(f, x) -> List:
    return list(map(f, x))


def fetch_test_set(test_set_url):
    import wget

    fname = wget.download(test_set_url, "opus_test.txt")
    lns = Path(fname).open().readlines()
    src = lmap(str.strip, lns[::4])
    gold = lmap(str.strip, lns[1::4])
    mar_model = lmap(str.strip, lns[2::4])
    assert (
        len(gold) == len(mar_model) == len(src)
    ), f"Gold, marian and source lengths {len(gold)}, {len(mar_model)}, {len(src)} mismatched"
    os.remove(fname)
    return src, mar_model, gold


def convert_whole_dir(path=Path("marian_ckpt/")):
    for subdir in tqdm(list(path.ls())):
        dest_dir = f"marian_converted/{subdir.name}"
        if (dest_dir / "pytorch_model.bin").exists():
            continue
        convert(source_dir, dest_dir)


def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
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


def save_tokenizer_config(dest_dir: Path):
    dname = dest_dir.name.split("-")
    dct = dict(target_lang=dname[-1], source_lang="-".join(dname[:-1]))
    save_json(dct, dest_dir / "tokenizer_config.json")


def add_to_vocab_(vocab: Dict[str, int], special_tokens: List[str]):
    start = max(vocab.values()) + 1
    added = 0
    for tok in special_tokens:
        if tok in vocab:
            continue
        vocab[tok] = start + added
        added += 1
    return added


def find_vocab_file(model_dir):
    return list(model_dir.glob("*vocab.yml"))[0]


def add_special_tokens_to_vocab(model_dir: Path) -> None:
    vocab = load_yaml(find_vocab_file(model_dir))
    vocab = {k: int(v) for k, v in vocab.items()}
    num_added = add_to_vocab_(vocab, ["<pad>"])
    print(f"added {num_added} tokens to vocab")
    save_json(vocab, model_dir / "vocab.json")
    save_tokenizer_config(model_dir)


def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    assert v1 == v2, f"hparams {k1},{k2} differ: {v1} != {v2}"


def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {
        "tied-embeddings-all": True,
        "layer-normalization": False,
        "right-left": False,
        "transformer-ffn-depth": 2,
        "transformer-aan-depth": 2,
        "transformer-no-projection": False,
        "transformer-postprocess-emb": "d",
        "transformer-postprocess": "dan",  # Dropout, add, normalize
        "transformer-preprocess": "",
        "type": "transformer",
        "ulr-dim-emb": 0,
        "dec-cell-base-depth": 2,
        "dec-cell-high-depth": 1,
        "transformer-aan-nogate": False,
    }
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        assert actual == v, f"Unexpected config value for {k} expected {v} got {actual}"
    check_equal(marian_cfg, "transformer-ffn-activation", "transformer-aan-activation")
    check_equal(marian_cfg, "transformer-ffn-depth", "transformer-aan-depth")
    check_equal(marian_cfg, "transformer-dim-ffn", "transformer-dim-aan")


BIAS_KEY = "decoder_ff_logit_out_b"
BART_CONVERTER = {  # for each encoder and decoder layer
    "self_Wq": "self_attn.q_proj.weight",
    "self_Wk": "self_attn.k_proj.weight",
    "self_Wv": "self_attn.v_proj.weight",
    "self_Wo": "self_attn.out_proj.weight",
    "self_bq": "self_attn.q_proj.bias",
    "self_bk": "self_attn.k_proj.bias",
    "self_bv": "self_attn.v_proj.bias",
    "self_bo": "self_attn.out_proj.bias",
    "self_Wo_ln_scale": "self_attn_layer_norm.weight",
    "self_Wo_ln_bias": "self_attn_layer_norm.bias",
    "ffn_W1": "fc1.weight",
    "ffn_b1": "fc1.bias",
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    # Decoder Cross Attention
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    "context_Wo_ln_scale": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.bias",
}


class OpusState:
    def __init__(self, source_dir):
        npz_path = find_model_file(source_dir)
        self.state_dict = np.load(npz_path)
        cfg = load_config_from_state_dict(self.state_dict)
        assert cfg["dim-vocabs"][0] == cfg["dim-vocabs"][1]
        assert "Wpos" not in self.state_dict, "Wpos key in state dictionary"
        self.state_dict = dict(self.state_dict)
        self.wemb, self.final_bias = add_emb_entries(self.state_dict["Wemb"], self.state_dict[BIAS_KEY], 1)
        self.pad_token_id = self.wemb.shape[0] - 1
        cfg["vocab_size"] = self.pad_token_id + 1
        # self.state_dict['Wemb'].sha
        self.state_keys = list(self.state_dict.keys())
        assert "Wtype" not in self.state_dict, "Wtype key in state dictionary"
        self._check_layer_entries()
        self.source_dir = source_dir
        self.cfg = cfg
        hidden_size, intermediate_shape = self.state_dict["encoder_l1_ffn_W1"].shape
        assert (
            hidden_size == cfg["dim-emb"] == 512
        ), f"Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched or not 512"

        # Process decoder.yml
        decoder_yml = cast_marian_config(load_yaml(source_dir / "decoder.yml"))
        check_marian_cfg_assumptions(cfg)
        self.hf_config = MarianConfig(
            vocab_size=cfg["vocab_size"],
            decoder_layers=cfg["dec-depth"],
            encoder_layers=cfg["enc-depth"],
            decoder_attention_heads=cfg["transformer-heads"],
            encoder_attention_heads=cfg["transformer-heads"],
            decoder_ffn_dim=cfg["transformer-dim-ffn"],
            encoder_ffn_dim=cfg["transformer-dim-ffn"],
            d_model=cfg["dim-emb"],
            activation_function=cfg["transformer-aan-activation"],
            pad_token_id=self.pad_token_id,
            eos_token_id=0,
            bos_token_id=0,
            max_position_embeddings=cfg["dim-emb"],
            scale_embedding=True,
            normalize_embedding="n" in cfg["transformer-preprocess"],
            static_position_embeddings=not cfg["transformer-train-position-embeddings"],
            dropout=0.1,  # see opus-mt-train repo/transformer-dropout param.
            # default: add_final_layer_norm=False,
            num_beams=decoder_yml["beam-size"],
            decoder_start_token_id=self.pad_token_id,
            bad_words_ids=[[self.pad_token_id]],
            max_length=512,
        )

    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys("encoder_l1")
        self.decoder_l1 = self.sub_keys("decoder_l1")
        self.decoder_l2 = self.sub_keys("decoder_l2")
        if len(self.encoder_l1) != 16:
            warnings.warn(f"Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}")
        if len(self.decoder_l1) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")
        if len(self.decoder_l2) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")

    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if (
                k.startswith("encoder_l")
                or k.startswith("decoder_l")
                or k in [CONFIG_KEY, "Wemb", "Wpos", "decoder_ff_logit_out_b"]
            ):
                continue
            else:
                extra.append(k)
        return extra

    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    def load_marian_model(self) -> MarianMTModel:
        state_dict, cfg = self.state_dict, self.hf_config

        assert cfg.static_position_embeddings, "config.static_position_embeddings should be True"
        model = MarianMTModel(cfg)

        assert "hidden_size" not in cfg.to_dict()
        load_layers_(
            model.model.encoder.layers,
            state_dict,
            BART_CONVERTER,
        )
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)

        # handle tensors not associated with layers
        wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
        bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
        model.model.shared.weight = wemb_tensor
        model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared

        model.final_logits_bias = bias_tensor

        if "Wpos" in state_dict:
            print("Unexpected: got Wpos")
            wpos_tensor = torch.tensor(state_dict["Wpos"])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        if cfg.normalize_embedding:
            assert "encoder_emb_ln_scale_pre" in state_dict
            raise NotImplementedError("Need to convert layernorm_embedding")

        assert not self.extra_keys, f"Failed to convert {self.extra_keys}"
        assert (
            model.model.shared.padding_idx == self.pad_token_id
        ), f"Padding tokens {model.model.shared.padding_idx} and {self.pad_token_id} mismatched"
        return model


def download_and_unzip(url, dest_dir):
    try:
        import wget
    except ImportError:
        raise ImportError("you must pip install wget")

    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)


def convert(source_dir: Path, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    add_special_tokens_to_vocab(source_dir)
    tokenizer = MarianTokenizer.from_pretrained(str(source_dir))
    tokenizer.save_pretrained(dest_dir)

    opus_state = OpusState(source_dir)
    assert opus_state.cfg["vocab_size"] == len(
        tokenizer.encoder
    ), f"Original vocab size {opus_state.cfg['vocab_size']} and new vocab size {len(tokenizer.encoder)} mismatched"
    # save_json(opus_state.cfg, dest_dir / "marian_original_config.json")
    # ^^ Uncomment to save human readable marian config for debugging

    model = opus_state.load_marian_model()
    model = model.half()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


def load_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


def save_json(content: Union[Dict, List], path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f)


def unzip(zip_path: str, dest_dir: str) -> None:
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(dest_dir)


if __name__ == "__main__":
    """
    Tatoeba conversion instructions in scripts/tatoeba/README.md
    """
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src", type=str, help="path to marian model sub dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    source_dir = Path(args.src)
    assert source_dir.exists(), f"Source directory {source_dir} not found"
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    convert(source_dir, dest_dir)
