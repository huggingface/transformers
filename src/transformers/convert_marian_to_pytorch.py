import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile

import numpy as np
import torch
from tqdm import tqdm

from transformers import MarianConfig, MarianMTModel, MarianSentencePieceTokenizer


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


def load_layers_(layer_lst: torch.nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=True)


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


def parse_readmes(repo_path):
    results = {}
    for p in Path(repo_path).ls():
        n_dash = p.name.count("-")
        if n_dash == 0:
            continue
        else:
            lns = list(open(p / "README.md").readlines())
            results[p.name] = _parse_readme(lns)
    return results


def download_all_sentencepiece_models(repo_path="Opus-MT-train/models"):
    """Requires 300GB"""
    save_dir = Path("marian_ckpt")
    if not Path(repo_path).exists():
        raise ValueError("You must run: git clone git@github.com:Helsinki-NLP/Opus-MT-train.git")
    results: dict = parse_readmes(repo_path)
    for k, v in tqdm(list(results.items())):
        if os.path.exists(save_dir / k):
            print(f"already have path {k}")
            continue
        if "SentencePiece" not in v["pre-processing"]:
            continue
        download_and_unzip(v["download"], save_dir / k)


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


def write_metadata(dest_dir: Path):
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


def add_special_tokens_to_vocab(model_dir: Path) -> None:
    vocab = load_yaml(model_dir / "opus.spm32k-spm32k.vocab.yml")
    vocab = {k: int(v) for k, v in vocab.items()}
    num_added = add_to_vocab_(vocab, ["<pad>"])
    print(f"added {num_added} tokens to vocab")
    save_json(vocab, model_dir / "vocab.json")
    write_metadata(model_dir)


def save_tokenizer(self, save_directory):
    dest = Path(save_directory)
    src_path = Path(self.init_kwargs["source_spm"])

    for dest_name in {"source.spm", "target.spm", "tokenizer_config.json"}:
        shutil.copyfile(src_path.parent / dest_name, dest / dest_name)
    save_json(self.encoder, dest / "vocab.json")


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
        assert "Wpos" not in self.state_dict
        self.state_dict = dict(self.state_dict)
        self.wemb, self.final_bias = add_emb_entries(self.state_dict["Wemb"], self.state_dict[BIAS_KEY], 1)
        self.pad_token_id = self.wemb.shape[0] - 1
        cfg["vocab_size"] = self.pad_token_id + 1
        # self.state_dict['Wemb'].sha
        self.state_keys = list(self.state_dict.keys())
        if "Wtype" in self.state_dict:
            raise ValueError("found Wtype key")
        self._check_layer_entries()
        self.source_dir = source_dir
        self.cfg = cfg
        hidden_size, intermediate_shape = self.state_dict["encoder_l1_ffn_W1"].shape
        assert hidden_size == cfg["dim-emb"] == 512

        # Process decoder.yml
        decoder_yml = cast_marian_config(load_yaml(source_dir / "decoder.yml"))
        # TODO: what are normalize and word-penalty?
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

        assert cfg.static_position_embeddings
        model = MarianMTModel(cfg)

        assert "hidden_size" not in cfg.to_dict()
        load_layers_(
            model.model.encoder.layers, state_dict, BART_CONVERTER,
        )
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)

        # handle tensors not associated with layers
        wemb_tensor = torch.nn.Parameter(torch.FloatTensor(self.wemb))
        bias_tensor = torch.nn.Parameter(torch.FloatTensor(self.final_bias))
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
        assert model.model.shared.padding_idx == self.pad_token_id
        return model


def download_and_unzip(url, dest_dir):
    try:
        import wget
    except ImportError:
        raise ImportError("you must pip install wget")

    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)


def main(source_dir, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    add_special_tokens_to_vocab(source_dir)
    tokenizer = MarianSentencePieceTokenizer.from_pretrained(str(source_dir))
    save_tokenizer(tokenizer, dest_dir)

    opus_state = OpusState(source_dir)
    assert opus_state.cfg["vocab_size"] == len(tokenizer.encoder)
    # save_json(opus_state.cfg, dest_dir / "marian_original_config.json")
    # ^^ Save human readable marian config for debugging

    model = opus_state.load_marian_model()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src", type=str, help="path to marian model dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    source_dir = Path(args.src)
    assert source_dir.exists()
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    main(source_dir, dest_dir)


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
