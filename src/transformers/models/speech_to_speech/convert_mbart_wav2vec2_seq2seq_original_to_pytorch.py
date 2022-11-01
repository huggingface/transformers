# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert a Wav2Vec2Conformer-2-MBart checkpoint from fairseq to Transformers."""


import argparse

import fairseq
import torch
from torch import nn

from transformers import (
    MBartConfig,
    MBartForCausalLM,
    SpeechToSpeechConfig,
    SpeechToSpeechModel,
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerModel,
    Wav2Vec2FeatureExtractor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.linear_k": "encoder.layers.*.self_attn.linear_k",
    "self_attn.linear_v": "encoder.layers.*.self_attn.linear_v",
    "self_attn.linear_q": "encoder.layers.*.self_attn.linear_q",
    "self_attn.pos_bias_u": "encoder.layers.*.self_attn.pos_bias_u",
    "self_attn.pos_bias_v": "encoder.layers.*.self_attn.pos_bias_v",
    "self_attn.linear_out": "encoder.layers.*.self_attn.linear_out",
    "self_attn.linear_pos": "encoder.layers.*.self_attn.linear_pos",
    "self_attn.rotary_emb": "encoder.embed_positions",
    "self_attn_layer_norm": "encoder.layers.*.self_attn_layer_norm",
    "conv_module.pointwise_conv1": "encoder.layers.*.conv_module.pointwise_conv1",
    "conv_module.pointwise_conv2": "encoder.layers.*.conv_module.pointwise_conv2",
    "conv_module.depthwise_conv": "encoder.layers.*.conv_module.depthwise_conv",
    "conv_module.batch_norm": "encoder.layers.*.conv_module.batch_norm",
    "conv_module.layer_norm": "encoder.layers.*.conv_module.layer_norm",
    "ffn1.w_1": "encoder.layers.*.ffn1.intermediate_dense",
    "ffn1.w_2": "encoder.layers.*.ffn1.output_dense",
    "ffn1.layer_norm": "encoder.layers.*.ffn1_layer_norm",
    "ffn2.w_1": "encoder.layers.*.ffn2.intermediate_dense",
    "ffn2.w_2": "encoder.layers.*.ffn2.output_dense",
    "ffn2.layer_norm": "encoder.layers.*.ffn2_layer_norm",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]

# Copied from transformers.models.wav2vec2_conformer.convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.set_recursively
def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    elif weight_type == "inv_freq":
        hf_pointer.inv_freq.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


# Adapted from transformers.models.wav2vec2_conformer.convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.recursively_load_weights
def recursively_load_weights(fairseq_model, hf_model):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.feature_extractor
    adapter = hf_model.adapter

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        elif any(x in name for x in ["adaptor", "w2v_encoder.proj.", "w2v_proj_ln."]):
            load_adapter(name, value, adapter, unused_weights)
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "pos_bias_u" in name:
                        weight_type = None
                    elif "pos_bias_v" in name:
                        weight_type = None
                    elif "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: don't match quantizer.weight_proj
                        weight_type = "weight"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "inv_freq" in name:
                        weight_type = "inv_freq"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


# Copied from transformers.models.wav2vec2_conformer.convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.load_conv_layer
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


# Copied from transformers.models.speech_encoder_decoder.convert_mbart_wav2vec2_seq2seq_original_to_pytorch.load_adapter
def load_adapter(full_name, value, adapter, unused_weights):
    name = full_name.split("adaptor.")[-1]
    items = name.split(".")

    if items[1].isdigit():
        layer_id = int(items[1])
    else:
        layer_id = None

    if "adaptor" not in full_name:
        if "proj_ln" in full_name:
            # has to be layer norm
            if "bias" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.bias.data.shape} was found."
                adapter.proj_layer_norm.bias.data = value
                logger.info(f"Adapter proj layer norm bias was initialized from {full_name}.")
            if "weight" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.weight.data.shape} was found."
                adapter.proj_layer_norm.weight.data = value
        else:
            # has to be projection layer
            if "bias" in name:
                assert (
                    value.shape == adapter.proj.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.bias.data.shape} was found."
                adapter.proj.bias.data = value
                logger.info(f"Adapter proj layer bias was initialized from {full_name}.")
            if "weight" in name:
                assert (
                    value.shape == adapter.proj.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.weight.data.shape} was found."
                adapter.proj.weight.data = value
                logger.info(f"Adapter proj layer weight was initialized from {full_name}.")
    elif isinstance(layer_id, int):
        if "bias" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.bias.data.shape} was found."
            adapter.layers[layer_id].conv.bias.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
        elif "weight" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.weight.data.shape} was found."
            adapter.layers[layer_id].conv.weight.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


# Copied from transformers.models.speech_encoder_decoder.convert_mbart_wav2vec2_seq2seq_original_to_pytorch.make_linear_from_emb
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    dict_path,
    config_yaml_path,
    encoder_config_path,
    decoder_config_path,
    decoder_output_dim,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # load model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides={
            "config_yaml": config_yaml_path,
            "data": "/".join(dict_path.split("/")[:-1]),
            "task": "speech_to_text",
        },
    )
    model = model[0].eval()
    fairseq_config = cfg["model"]

    # load configs
    encoder_config = Wav2Vec2ConformerConfig.from_pretrained(
        encoder_config_path,
        add_adapter=True,
        num_adapter_layers=fairseq_config.adaptor_n_layers,
        adapter_stride=fairseq_config.adaptor_stride,
        adapter_kernel_size=fairseq_config.adaptor_kernel_size,
        use_auth_token=True,
        output_hidden_size=fairseq_config.w2v_args.model.encoder_embed_dim,
    )
    decoder_config = MBartConfig.from_pretrained(
        decoder_config_path,
        decoder_layers=fairseq_config.decoder_layers,
        vocab_size=decoder_output_dim,
        max_position_embeddings=4002,
        learned_embedding=False,
        layernorm_embedding=False,
    )

    # load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_config_path, use_auth_token=True)

    # set weights for wav2vec2 encoder
    hf_encoder = Wav2Vec2ConformerModel(encoder_config)

    recursively_load_weights(model.encoder, hf_encoder)

    # load decoder weights
    hf_decoder = MBartForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    hf_wav2vec = SpeechToSpeechModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False

    # tokenizer = MBart50Tokenizer(dict_path)
    # tokenizer.save_pretrained(pytorch_dump_folder_path)

    config = hf_wav2vec.config.to_dict()
    # config["tokenizer_class"] = "mbart50"
    config["feature_extractor_type"] = "wav2vec2"

    hf_wav2vec.config = SpeechToSpeechConfig.from_dict(config)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_yaml_path", default=None, type=str, help="Path to yaml file of fine-tuned model")
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-conformer-rel-pos-large",
        type=str,
        help="Path to hf encoder wav2vec2 conformer checkpoint config",
    )
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/mbart-large-50-one-to-many-mmt",
        type=str,
        help="Path to hf decoder checkpoint config",
    )
    parser.add_argument("--decoder_output_dim", default=1007, type=int, help="decoder output dim (=vocab size)")

    args = parser.parse_args()
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.dict_path,
        args.config_yaml_path,
        encoder_config_path=args.encoder_config_path,
        decoder_config_path=args.decoder_config_path,
        decoder_output_dim=args.decoder_output_dim,
    )
