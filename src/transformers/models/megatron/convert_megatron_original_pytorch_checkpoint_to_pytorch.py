# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert fairseq Megatron checkpoint."""


import argparse
import glob
import os
from pathlib import Path

import fairseq
import torch
from fairseq import tasks
from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq.data.encoders.gpt2_bpe_utils import Encoder
from fairseq.dataclass.utils import convert_namespace_to_omegaconf, overwrite_args_by_name
from fairseq.models.transformer_lm import TransformerLanguageModel
from packaging import version

from transformers import MegatronConfig, MegatronForCausalLM, MegatronTokenizer
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! 💁 (╯°□°）╯︵ ┻━┻)"


def load_fairseq_checkpoint(checkpoint_path):
    """
    Load a model-parallel checkpoint to CPU.
    """
    # the checkpoint is partitioned into N_GPUS model-parallel shards
    shard_paths = sorted(glob.glob(os.path.join(checkpoint_path, "model-model_part-*.pt")))

    # load all checkpoints into memory
    state_shards = [torch.load(p, map_location="cpu") for p in shard_paths]

    # the model config is copied across shards, so we can use any one of them
    output_state = {"args": state_shards[0]["args"]}
    print(output_state["args"])
    output_model = {}

    # group model parameters by key
    model_shards = {k: [st["model"][k] for st in state_shards] for k in state_shards[0]["model"].keys()}

    for k, params in model_shards.items():
        if ".version" in k:
            continue
        elif "_float_tensor" in k:
            # embed_positions are copied across shards
            output_model[k] = params[0]
        elif "out_proj.weight" in k or "fc2.weight" in k:
            # row parallel weights
            output_model[k] = torch.cat(params, dim=1)
        elif "out_proj.bias" in k or "fc2.bias" in k:
            # biases are copied across shards
            output_model[k] = params[0]
        elif "_proj" in k or "fc1" in k:
            # column parallel weights
            output_model[k] = torch.cat(params, dim=0)
        elif "_norm" in k:
            # norms are copied across shards
            output_model[k] = params[0]
        elif "embed_tokens" in k:
            output_model[k] = torch.cat(params, dim=0)
        else:
            raise ValueError(f"We don't know how to merge the parameter {k}")

    output_state["model"] = output_model
    return output_state


def load_fairseq_model(checkpoint_path):
    logger.info("Converting fairseq's model_parallel checkpoint into a single model...")
    state = load_fairseq_checkpoint(checkpoint_path)
    cfg = convert_namespace_to_omegaconf(state["args"])
    # specify the path to `dict.txt` which is needed for embeddings initialization
    overwrite_args_by_name(cfg, {"task": {"data": checkpoint_path}})
    task = tasks.setup_task(cfg.task)
    logger.info("Initializing faiseq model from state dict...")
    model = TransformerLanguageModel.build_model(cfg.model, task).eval()
    model.load_state_dict(state["model"])
    model.upgrade_state_dict(model.state_dict())
    return cfg, model, task


def fairseq_tokenize(text, tokenizer, task):
    # The raw tokenizer does not contain special tokens
    # and the bpe indices are not the ones that model expects.
    # Fairseq rearranges them according to frequency and saves
    # the id mappings into dict.txt
    tokens = " ".join([str(t) for t in tokenizer.encode(text)])
    bpe_sentence = "<s> " + tokens + " </s>"
    # encode bpe tokens into model's input ids
    tokens = task.source_dictionary.encode_line(bpe_sentence, append_eos=False)
    return tokens.long()


@torch.no_grad()
def convert_fairseq_checkpoint(checkpoint_path, pytorch_dump_path):
    """
    Convert tokenizer files. Copy/paste/tweak model's weights to our BART structure.
    """
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    encoder_path = os.path.join(checkpoint_path, "encoder.json")
    bpe_path = os.path.join(checkpoint_path, "vocab.bpe")

    logger.info("Loading the fairseq model...")
    fs_cfg, fs_model, fs_task = load_fairseq_model(checkpoint_path)
    fs_tokenizer: Encoder = get_encoder(encoder_path, bpe_path)

    fs_tokens = fairseq_tokenize(SAMPLE_TEXT, fs_tokenizer, fs_task)
    fs_tokens = fs_tokens.unsqueeze(0)

    tokenizer = MegatronTokenizer.from_pretrained("megatron-11b")
    hf_tokens = tokenizer.encode(SAMPLE_TEXT, return_tensors="pt")
    assert torch.eq(fs_tokens, hf_tokens).all()

    state_dict = fs_model.state_dict()
    state_dict.pop("decoder.version", None)
    # megatron uses fixed sinusoidal embeddings, this is just a dummy tensor
    state_dict.pop("decoder.embed_positions._float_tensor", None)
    # equivalent to lm_head with tied embeddings
    state_dict.pop("decoder.output_projection.weight", None)

    config = MegatronConfig(
        vocab_size=51200,
        max_position_embeddings=fs_cfg.model.max_target_positions,
        decoder_layers=fs_cfg.model.decoder_layers,
        decoder_ffn_dim=fs_cfg.model.decoder_ffn_embed_dim,
        decoder_attention_heads=fs_cfg.model.decoder_attention_heads,
        decoder_layerdrop=fs_cfg.model.decoder_layerdrop,
        is_decoder=True,
        activation_function=fs_cfg.model.activation_fn,
        d_model=fs_cfg.model.decoder_embed_dim,
        dropout=fs_cfg.model.dropout,
        attention_dropout=fs_cfg.model.attention_dropout,
        decoder_start_token_id=0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
        scale_embedding=not fs_cfg.model.no_scale_embedding,
    )
    logger.info("Initializing the 🤗 Megatron model")
    model = MegatronForCausalLM(config).eval()
    logger.info("Loading fairseq's state disct into Megatron")
    missing, extra = model.model.load_state_dict(state_dict, strict=False)
    unexpected_missing = [k for k in missing if k != "decoder.embed_positions.weight"]
    assert unexpected_missing == [], f"Missing key(s) in state_dict: {unexpected_missing}"
    assert extra == [], f"Extra keys in the original state_dict: {extra}"

    fairseq_outputs = fs_model.extract_features(fs_tokens, encoder_out=None)[0]
    new_model_outputs = model.model(hf_tokens)[0]

    # Check results
    assert fairseq_outputs.shape == new_model_outputs.shape
    assert torch.eq(fairseq_outputs, new_model_outputs).all()
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--fairseq_path", type=str, help="Path to the unpacked fairseq Megatron model checkpoint.")
    parser.add_argument("--pytorch_dump_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_fairseq_checkpoint(args.fairseq_path, args.pytorch_dump_path)
