####################################################################################################

# Copyright (c) 2023-, NVIDIA CORPORATION and eagle705.  All rights reserved.
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

####################################################################################################

import argparse
import json
import os

import torch
from omegaconf.omegaconf import OmegaConf
from transformers import T5Config


def get_hf_t5_base_config():
    config = {
        "apply_query_key_layer_scaling": True,
        "architectures": ["MegatronT5ForConditionalGeneration"],
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 768,
        "decoder_start_token_id": 0,
        "dense_act_fn": "gelu_new",
        "dropout_rate": 0.1,
        "eos_token_id": 3,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "is_gated_act": True,
        "layer_norm_epsilon": 1e-05,
        "max_position_embeddings": 512,
        "model_type": "t5",
        "normalization": "layernorm",
        "num_decoder_layers": 12,
        "num_heads": 12,
        "num_layers": 12,
        "output_past": True,
        "pad_token_id": 0,
        "position_embedding_type": "learned_absolute",
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": False,
        "use_bias": True,
        "use_cache": True,
        "use_rescale_tie_word_embeddings": False,
        "vocab_size": 51200,
    }
    return config


def megatron_to_hf_config(megatron_cfg):
    hf_model_config = get_hf_t5_base_config()

    print(hf_model_config)

    if megatron_cfg.encoder.activation == "swiglu":
        act_fn = "silu"
    elif megatron_cfg.encoder.activation == "geglu":
        act_fn = "gelu_new"
    # FLAN-T5 models have things configured this way. it is not supproted yet.
    elif megatron_cfg.encoder.activation == "geglu" and megatron_cfg.data.dataset_type == "flan-t5":
        act_fn = "gelu"
        hf_model_config["is_gated_act"] = True
    else:
        raise ValueError(f"Unknown activation: {megatron_cfg.encoder.activation}")

    hf_model_config["num_layers"] = megatron_cfg.encoder.num_layers
    hf_model_config["d_model"] = megatron_cfg.encoder.hidden_size
    hf_model_config["d_ff"] = megatron_cfg.encoder.ffn_hidden_size
    hf_model_config["d_kv"] = megatron_cfg.encoder.kv_channels
    hf_model_config["num_heads"] = megatron_cfg.encoder.num_attention_heads
    hf_model_config["dense_act_fn"] = act_fn
    hf_model_config["relative_attention_max_distance"] = megatron_cfg.encoder.relative_attention_max_distance
    hf_model_config["relative_attention_num_buckets"] = megatron_cfg.encoder.relative_attention_num_buckets
    hf_model_config["num_decoder_layers"] = megatron_cfg.decoder.num_layers
    hf_model_config["position_embedding_type"] = megatron_cfg.encoder.position_embedding_type
    hf_model_config["max_position_embeddings"] = megatron_cfg.max_position_embeddings
    hf_model_config["use_bias"] = all(
        [
            megatron_cfg.encoder.bias_activation_fusion,
            megatron_cfg.encoder.bias_dropout_add_fusion,
            megatron_cfg.encoder.bias,
        ]
    )
    hf_model_config["layer_norm_epsilon"] = megatron_cfg.encoder.layernorm_epsilon
    # Check tie_word_embeddings, share_token_embeddings, share_decoder_tokens_head_embeddings, vocab_size

    return hf_model_config


####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        config.tokenizer_type = ds_args.tokenizer_type
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = ds_args.ffn_hidden_size if "ffn_hidden_size" in ds_args else 4 * ds_args.hidden_size

    # # The number of heads.
    heads = config.num_heads
    # # The hidden_size per head.
    # hidden_size_per_head = config.hidden_size // heads
    hidden_size_per_head = config.d_model // heads

    attention_dim = config.d_kv * config.num_heads
    print(
        f"attention_dim:{attention_dim}, config.d_kv:{config.d_kv}, config.num_heads:{config.num_heads}, hidden_size_per_head:{hidden_size_per_head}, config.d_model:{config.d_model}"
    )

    for nn_type in ["encoder", "decoder"]:
        """
        input_key: NeMo-Megatron ckpt
        output_key: Huggingface ckpt
        """

        #################################################
        ###### Enc-Dec Embeddings and Output Layer ######
        #################################################
        word_embeddings = input_state_dict["enc_dec_model.{}_embedding.word_embeddings.weight".format(nn_type)]
        output_state_dict["{}.embed_tokens.weight".format(nn_type)] = word_embeddings

        if nn_type == "encoder":
            output_state_dict["shared.weight"] = word_embeddings
        else:
            # sharing weights
            output_state_dict["lm_head.weight"] = word_embeddings
            output_state_dict["lm_head.bias"] = input_state_dict["enc_dec_model.tokens_head.bias"]

        #################################################
        ################# RPE Weights ###################
        #################################################
        # convert pos embed
        pos_embeddings = input_state_dict["enc_dec_model.{}_embedding.position_embeddings.weight".format(nn_type)]
        output_state_dict["{}.position_embeddings.weight".format(nn_type)] = pos_embeddings
        output_state_dict["position_embeddings.weight"] = pos_embeddings

        # use it in the case of relative_position_embedding
        # pos_embeddings = input_state_dict["enc_dec_model.{}_relative_position_embedding.relative_position_embedding.weight".format(nn_type)]
        # output_state_dict["{}.block.0.layer.0.SelfAttention.relative_attention_bias.weight".format(nn_type)] = pos_embeddings

        for i in range(0, config.num_layers):
            # Block in HF corresponds to layer in NeMo.
            # Layer in HF does not correspond to anything in NeMo. Layer 0 is self attn, layer 1 is cross-attn.

            #################################################
            ############### Attention Layers ################
            #################################################
            for weight_bias in ["weight", "bias"]:
                if nn_type == "decoder":
                    # convert weights for decoder
                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.inter_attention.query.{}".format(
                        nn_type, i, weight_bias
                    )
                    attention_qkv_weight = input_state_dict[input_key]
                    output_key = "{}.block.{}.layer.1.EncDecAttention.q.{}".format(nn_type, i, weight_bias)
                    if weight_bias == "weight":
                        output_state_dict[output_key] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim, :]
                    else:
                        output_state_dict[output_key] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.inter_attention.key_value.{}".format(
                        nn_type, i, weight_bias
                    )
                    attention_qkv_weight = input_state_dict[input_key]
                    output_key = "{}.block.{}.layer.1.EncDecAttention.k.{}".format(nn_type, i, weight_bias)
                    if weight_bias == "weight":
                        output_state_dict[output_key] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim, :]
                    else:
                        output_state_dict[output_key] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim]
                    output_key = "{}.block.{}.layer.1.EncDecAttention.v.{}".format(nn_type, i, weight_bias)
                    if weight_bias == "weight":
                        output_state_dict[output_key] = attention_qkv_weight[1 * attention_dim : 2 * attention_dim, :]
                    else:
                        output_state_dict[output_key] = attention_qkv_weight[1 * attention_dim : 2 * attention_dim]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.inter_attention.dense.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.EncDecAttention.o.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.self_attention.query_key_value.{}".format(
                    nn_type, i, weight_bias
                )

                attention_qkv_weight = input_state_dict[input_key]
                # Store the tensor as we need the bias as well to interleave QKV and biases.

                output_key_q = "{}.block.{}.layer.0.SelfAttention.q.{}".format(nn_type, i, weight_bias)
                output_key_k = "{}.block.{}.layer.0.SelfAttention.k.{}".format(nn_type, i, weight_bias)
                output_key_v = "{}.block.{}.layer.0.SelfAttention.v.{}".format(nn_type, i, weight_bias)

                if weight_bias == "weight":
                    output_state_dict[output_key_q] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim, :]
                    output_state_dict[output_key_k] = attention_qkv_weight[1 * attention_dim : 2 * attention_dim, :]
                    output_state_dict[output_key_v] = attention_qkv_weight[2 * attention_dim : 3 * attention_dim, :]
                else:
                    output_state_dict[output_key_q] = attention_qkv_weight[0 * attention_dim : 1 * attention_dim]
                    output_state_dict[output_key_k] = attention_qkv_weight[1 * attention_dim : 2 * attention_dim]
                    output_state_dict[output_key_v] = attention_qkv_weight[2 * attention_dim : 3 * attention_dim]

                input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.self_attention.dense.{}".format(
                    nn_type, i, weight_bias
                )
                output_key = "{}.block.{}.layer.0.SelfAttention.o.{}".format(nn_type, i, weight_bias)
                output_state_dict[output_key] = input_state_dict[input_key]

                #################################################
                ################## FFN Layers ###################
                #################################################

                if nn_type == "encoder":
                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_h_to_4h.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.DenseReluDense.wi_0.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_h_to_4h_2.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.DenseReluDense.wi_1.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_4h_to_h.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.DenseReluDense.wo.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]
                else:
                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_h_to_4h.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.2.DenseReluDense.wi_0.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_h_to_4h_2.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.2.DenseReluDense.wi_1.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.mlp.dense_4h_to_h.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.2.DenseReluDense.wo.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                #################################################
                ################## LayerNorm ####################
                #################################################

                # convert ln
                input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.input_layernorm.{}".format(
                    nn_type, i, weight_bias
                )
                output_key = "{}.block.{}.layer.0.layer_norm.{}".format(nn_type, i, weight_bias)
                output_state_dict[output_key] = input_state_dict[input_key]

                if nn_type == "decoder":
                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.post_attention_layernorm.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.layer_norm.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                    input_key = (
                        "enc_dec_model.enc_dec_model.decoder.model.layers.{}.post_inter_attention_layernorm.{}".format(
                            i, weight_bias
                        )
                    )
                    output_key = "{}.block.{}.layer.2.layer_norm.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                else:
                    input_key = "enc_dec_model.enc_dec_model.{}.model.layers.{}.post_attention_layernorm.{}".format(
                        nn_type, i, weight_bias
                    )
                    output_key = "{}.block.{}.layer.1.layer_norm.{}".format(nn_type, i, weight_bias)
                    output_state_dict[output_key] = input_state_dict[input_key]

                input_key = "enc_dec_model.enc_dec_model.{}.model.final_layernorm.{}".format(nn_type, weight_bias)
                output_key = "{}.final_layer_norm.{}".format(nn_type, weight_bias)
                output_state_dict[output_key] = input_state_dict[input_key]

    # It should be done!
    return output_state_dict


####################################################################################################


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument("--path_to_checkpoint", type=str, help="Path to the ZIP file containing the checkpoint")
    parser.add_argument("--output_dir", type=str, help="output_dir")
    parser.add_argument(
        "--hparams_yaml_file",
        default="hparams.yaml",
        type=str,
        help="An optional hparams yaml file describing the pre-trained model from NeMo-Megatron.",
    )
    parser.add_argument(
        "--config_file",
        default="config.json",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    args = parser.parse_args()
    print(f"args:{args}")

    # Extract the hparams.
    megatron_cfg = OmegaConf.load(args.hparams_yaml_file)
    config = megatron_to_hf_config(megatron_cfg.cfg)
    with open(args.config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=4, ensure_ascii=False))

    # Extract the basename.
    output_dir = args.output_dir

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")
    if "state_dict" in input_state_dict:
        input_state_dict = input_state_dict["state_dict"]
    config = T5Config.from_json_file(args.config_file)

    # Convert.
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    save_dir = os.path.join(output_dir, "output")
    os.makedirs(save_dir, exist_ok=True)
    # Store the config to file.
    print("Saving config")
    # config.save_pretrained(basename)
    config.save_pretrained(save_dir)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(save_dir, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
