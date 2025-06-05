# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import shutil
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging
from pytorch_lightning import Trainer

from transformers import LlamaTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import LlamaConverter


"""
Script to convert a nemotron checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_nemotron_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin

2) Generate the full HF model folder

    python convert_nemotron_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder \

    Use the --cpu-only flag if the model cannot fit in the GPU (e.g. Nemotron4 340b).
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to .nemo file or extracted folder",
    )
    parser.add_argument("--output_path", type=str, default=None, required=False, help="Path to HF .bin file")
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, e.g. a folder containing https://huggingface.co/nvidia/Minitron-8B-Base",
    )
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.cfg.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def convert_hf_config(nemo_config, tokenizer, vocab_size, dtype, hf_output_path, hf_url="nvidia/Minitron-8B-Base"):
    """
    Convert NeMo config to HF config
    """
    NEMO_ACT2HF = {
        "squared-relu": "relu2",
        "fast-swiglu": "silu",
    }
    DTYPE2HF = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    hf_config = {
        "_name_or_path": hf_url,
        "architectures": ["NemotronForCausalLM"],
        "bos_token_id": tokenizer.bos_id,
        "eos_token_id": tokenizer.eos_id,
        "hidden_act": NEMO_ACT2HF[nemo_config.activation],
        "hidden_size": nemo_config.hidden_size,
        "initializer_range": nemo_config.init_method_std,
        "intermediate_size": nemo_config.ffn_hidden_size,
        "max_position_embeddings": nemo_config.max_position_embeddings,
        "model_type": "nemotron",
        "num_attention_heads": nemo_config.num_attention_heads,
        "num_hidden_layers": nemo_config.num_layers,
        "num_key_value_heads": nemo_config.get("num_query_groups", nemo_config.num_attention_heads),
        "norm_eps": nemo_config.layernorm_epsilon,
        "rope_theta": nemo_config.get("rotary_base", 10000),
        "partial_rotary_factor": nemo_config.get("rotary_percentage", 1.0),
        "tie_word_embeddings": False,
        "torch_dtype": DTYPE2HF[dtype],
        "transformers_version": "4.32.0.dev0",  # TODO
        "use_cache": True,
        "vocab_size": vocab_size,
    }
    if nemo_config.kv_channels is not None:
        hf_config["kv_channels"] = nemo_config.kv_channels
    json.dump(hf_config, open(f"{hf_output_path}/config.json", "w"), indent=2)


def convert(input_nemo_file, output_hf_file, precision=None, cpu_only=False) -> None:
    """
    Convert NeMo weights to HF weights
    """
    dummy_trainer = Trainer(devices=1, accelerator="cpu", strategy=NLPDDPStrategy())
    model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
    model_config.tensor_model_parallel_size = 1
    model_config.pipeline_model_parallel_size = 1
    model_config.sequence_parallel = False
    model_config.transformer_engine = True
    if cpu_only:
        map_location = torch.device("cpu")
        model_config.use_cpu_initialization = True
        model_config.dist_ckpt_load_on_device = False
    else:
        map_location = None

    if cpu_only:
        logging.info("******** Loading model on CPU. This will take a significant amount of time.")

    model = MegatronGPTModel.restore_from(
        input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    )

    vocab_size = model.padded_vocab_size

    if precision is None:
        precision = model.cfg.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(f"Precision string {precision} is not recognized, falling back to fp32")
        dtype = torch.float32  # fallback
    logging.info(f"Using precision {dtype}")

    def param_to_weights(param):
        return param.to(dtype)

    checkpoint = OrderedDict()

    hidden_size = model.cfg.hidden_size
    head_num = model.cfg.num_attention_heads
    num_layers = model.cfg.num_layers
    ffn_hidden_size = model.cfg.ffn_hidden_size
    num_query_groups = model.cfg.get("num_query_groups", head_num)  # different num_query_groups for 70B
    if num_query_groups is None:
        num_query_groups = head_num
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model.state_dict()["model.embedding.word_embeddings.weight"]
    embed_weights_base_name = "model.embed_tokens.weight"
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    for l in range(int(num_layers)):
        print(f"converting layer {l}")

        qkv_weights = model.state_dict()[f"model.decoder.layers.{l}.self_attention.linear_qkv.weight"]
        qkv_weights = qkv_weights.reshape([qkv_total_dim, -1, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
        ## Example of slices
        ## (without GQA): num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]

        q_weights_base_name = f"model.layers.{l}.self_attn.q_proj.weight"
        k_weights_base_name = f"model.layers.{l}.self_attn.k_proj.weight"
        v_weights_base_name = f"model.layers.{l}.self_attn.v_proj.weight"

        checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        # attention dense
        o_weight = model.state_dict()[f"model.decoder.layers.{l}.self_attention.linear_proj.weight"]
        o_weight_base_name = f"model.layers.{l}.self_attn.o_proj.weight"
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model.state_dict()[f"model.decoder.layers.{l}.mlp.linear_fc1.weight"]
        mlp_up_proj_weight = model.state_dict()[f"model.decoder.layers.{l}.mlp.linear_fc2.weight"]

        if mlp_weights.shape[0] != mlp_up_proj_weight.shape[1]:
            # Has projection (used for swi-glu)
            logging.warning(
                "Gated projection layers detected in NeMo checkpoint. Currently Nemotron HF does not support gated MLP."
            )
            assert mlp_weights.shape[0] == 2 * mlp_up_proj_weight.shape[1]

            mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
            mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

            mlp_down_proj_base_name = f"model.layers.{l}.mlp.gate_proj.weight"
            mlp_gate_proj_base_name = f"model.layers.{l}.mlp.up_proj.weight"

            checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
            checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)
        else:
            mlp_down_proj_weight = mlp_weights
            mlp_down_proj_base_name = f"model.layers.{l}.mlp.up_proj.weight"
            checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)

        mlp_up_proj_base_name = f"model.layers.{l}.mlp.down_proj.weight"
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model.state_dict()[f"model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_weight"]
        input_ln_base_name = f"model.layers.{l}.input_layernorm.weight"
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)
        if (
            model.state_dict().get(f"model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_bias", None)
            is not None
        ):
            input_ln_bias = model.state_dict()[f"model.decoder.layers.{l}.self_attention.linear_qkv.layer_norm_bias"]
            input_ln_bias_name = f"model.layers.{l}.input_layernorm.bias"
            checkpoint[input_ln_bias_name] = param_to_weights(input_ln_bias)

        post_attn_ln_weight = model.state_dict()[f"model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_weight"]
        post_attn_ln_base_name = f"model.layers.{l}.post_attention_layernorm.weight"
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)
        if model.state_dict().get(f"model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_bias", None) is not None:
            post_attn_ln_bias = model.state_dict()[f"model.decoder.layers.{l}.mlp.linear_fc1.layer_norm_bias"]
            post_attn_ln_bias_name = f"model.layers.{l}.post_attention_layernorm.bias"
            checkpoint[post_attn_ln_bias_name] = param_to_weights(post_attn_ln_bias)

        print(f"done layer {l}")

    final_ln_weight = model.state_dict()["model.decoder.final_layernorm.weight"]
    final_ln_base_name = "model.norm.weight"
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)
    if model.state_dict().get("model.decoder.final_layernorm.bias", None) is not None:
        final_ln_bias = model.state_dict()["model.decoder.final_layernorm.bias"]
        final_ln_bias_name = "model.norm.bias"
        checkpoint[final_ln_bias_name] = param_to_weights(final_ln_bias)

    output_layer_weight = model.state_dict()["model.output_layer.weight"]
    output_layer_base_name = "lm_head.weight"
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights saved to {output_hf_file}")

    return model_config, model.tokenizer, dtype, vocab_size


def extract_nemotron_tokenizer(nemo_file, model_config, output_hf_path, nemo_tokenizer):
    tokenizer_cfg = model_config.tokenizer
    if tokenizer_cfg.library == "sentencepiece":
        # For sentencepiece tokenizer, we are wrapping with HF's LlamaTokenizer
        # and convert it to a PreTrainedTokenizerFast
        tokenizer_fn = tokenizer_cfg.model[5:]
        output_tokenizer = f"{output_hf_path}/tokenizer.model"
        if nemo_file.endswith(".nemo"):
            import tarfile

            archive = tarfile.open(nemo_file, "r")
            tokenizer_filename = "./" + tokenizer_fn  # exclude 'nemo:' prefix
            archive.extract(tokenizer_filename, output_hf_path)
            archive.close()
            os.rename(f"{output_hf_path}/{tokenizer_fn}", output_tokenizer)
        elif os.path.isdir(nemo_file):
            shutil.copy(f"{nemo_file}/{tokenizer_fn}", output_tokenizer)
        # We use LlamaTokenizer for sentencepiece based tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(output_hf_path, legacy=False)
        # Convert the LlamaTokenizer to a PreTrainedTokenizerFast instance
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=LlamaConverter(tokenizer).converted(), model_input_names=["input_ids", "token_type_ids"]
        )
        tokenizer.save_pretrained(output_hf_path)
        logging.info(f"Setencepiece tokenizer has been saved to {output_tokenizer}")
    elif isinstance(nemo_tokenizer, AutoTokenizer):
        nemo_tokenizer.tokenizer.save_pretrained(output_hf_path)
        logging.info(f"HF AutoTokenizer has been saved to {output_hf_path}")
    else:
        raise ValueError(f"Unsupported tokenizer type: library: {tokenizer_cfg.library}, type: {tokenizer_cfg.type}")


if __name__ == "__main__":
    args = get_args()
    if not args.hf_output_path:
        assert args.output_path is not None, "Need to provide either output_path or hf_output_path"
    else:
        args.output_path = f"{args.hf_output_path}/pytorch_model.bin"
        logging.info(f"weight will be saved to {args.output_path}")

    nemo_config, nemo_tokenizer, dtype, vocab_size = convert(
        args.input_name_or_path, args.output_path, precision=args.precision, cpu_only=args.cpu_only
    )
    if args.hf_input_path and args.hf_output_path:
        convert_hf_config(nemo_config, nemo_tokenizer, vocab_size, dtype, args.hf_output_path, args.hf_input_path)
        extract_nemotron_tokenizer(args.input_name_or_path, nemo_config, args.hf_output_path, nemo_tokenizer)
    else:
        logging.info("`hf_input_path` and/or `hf_output_path` not provided, not generating full HF model.")
        logging.info(f".bin file is saved to {args.output_path}")
