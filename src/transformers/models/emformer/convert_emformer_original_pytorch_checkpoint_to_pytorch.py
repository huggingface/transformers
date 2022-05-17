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
"""Convert Emformer checkpoint."""


import argparse
import json
import os

import torch
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH, RNNTBundle
from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_MUSTC, EMFORMER_RNNT_BASE_TEDLIUM3

from transformers import EmformerConfig, EmformerFeatureExtractor, EmformerForRNNT, logging


NAME2BUNDLE = {
    "base_librispeech": EMFORMER_RNNT_BASE_LIBRISPEECH,
    "base_mustc": EMFORMER_RNNT_BASE_MUSTC,
    "base_tedlium": EMFORMER_RNNT_BASE_TEDLIUM3,
}


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def act2name(activation):
    if isinstance(activation, torch.nn.ReLU):
        return "relu"
    elif isinstance(activation, torch.nn.GELU):
        return "gelu"
    elif isinstance(activation, torch.nn.SiLU):
        return "silu"
    elif isinstance(activation, torch.nn.Tanh):
        return "tanh"
    else:
        raise ValueError(f"Unsupported activation {activation}")


def convert_config(bundle: RNNTBundle):
    cfg = EmformerConfig()
    decoder = bundle.get_decoder().model
    transcriber = decoder.transcriber
    transcriber_layer = transcriber.transformer.emformer_layers[0]
    predictor = decoder.predictor

    cfg.input_dim = transcriber.input_linear.in_features
    cfg.time_reduction_input_dim = transcriber.input_linear.out_features
    cfg.time_reduction_stride = decoder.transcriber.time_reduction.stride

    cfg.num_attention_heads = transcriber_layer.attention.num_heads
    cfg.ffn_dim = transcriber_layer.pos_ff[1].out_features
    cfg.num_hidden_layers = len(transcriber.transformer.emformer_layers)
    cfg.segment_length = transcriber.transformer.segment_length * cfg.time_reduction_stride
    cfg.hidden_dropout = transcriber_layer.dropout.p
    cfg.hidden_act = act2name(transcriber_layer.pos_ff[2])
    cfg.left_context_length = transcriber.transformer.left_context_length
    cfg.right_context_length = transcriber.transformer.right_context_length * cfg.time_reduction_stride

    cfg.output_dim = transcriber.output_linear.out_features

    cfg.vocab_size = predictor.embedding.num_embeddings
    cfg.symbol_embedding_dim = predictor.embedding.embedding_dim
    cfg.num_lstm_layers = len(predictor.lstm_layers)
    cfg.lstm_hidden_dim = predictor.lstm_layers[0].p2g.out_features // 4
    cfg.lstm_layer_norm = predictor.lstm_layers[0].c_norm.eps
    cfg.lstm_dropout = predictor.dropout.p

    cfg.joiner_activation = act2name(decoder.joiner.activation)

    return cfg


def convert_feature_extractor(bundle: RNNTBundle):
    extractor_pipeline = bundle.get_streaming_feature_extractor().pipeline
    feature_extractor = EmformerFeatureExtractor(
        sampling_rate=extractor_pipeline[0].sample_rate,
        n_fft=extractor_pipeline[0].n_fft,
        n_mels=extractor_pipeline[0].n_mels,
        hop_length=extractor_pipeline[0].hop_length,
        global_mean=extractor_pipeline[3].mean,
        global_invstddev=extractor_pipeline[3].invstddev,
        feature_size=extractor_pipeline[0].n_mels,
        padding_value=0.0,
    )

    return feature_extractor


def convert_weights(bundle: RNNTBundle, config: EmformerConfig) -> EmformerForRNNT:
    model = EmformerForRNNT(config)

    decoder = bundle.get_decoder().model
    transcriber = decoder.transcriber
    predictor = decoder.predictor

    model.transcriber.input_linear.load_state_dict(transcriber.input_linear.state_dict())
    for i in range(config.num_hidden_layers):
        src_layer = transcriber.transformer.emformer_layers[i]
        dst_layer = model.transcriber.encoder.emformer_layers[i]
        dst_layer.attention.load_state_dict(src_layer.attention.state_dict())
        dst_layer.pos_ff.load_state_dict(src_layer.pos_ff.state_dict())
        dst_layer.layer_norm_input.load_state_dict(src_layer.layer_norm_input.state_dict())
        dst_layer.layer_norm_output.load_state_dict(src_layer.layer_norm_output.state_dict())
    model.transcriber.output_linear.load_state_dict(transcriber.output_linear.state_dict())
    model.transcriber.layer_norm.load_state_dict(transcriber.layer_norm.state_dict())

    model.predictor.embedding.load_state_dict(predictor.embedding.state_dict())
    model.predictor.input_layer_norm.load_state_dict(predictor.input_layer_norm.state_dict())
    for i in range(config.num_lstm_layers):
        src_layer = predictor.lstm_layers[i]
        dst_layer = model.predictor.lstm_layers[i]
        dst_layer.x2g.load_state_dict(src_layer.x2g.state_dict())
        dst_layer.p2g.load_state_dict(src_layer.p2g.state_dict())
        dst_layer.c_norm.load_state_dict(src_layer.c_norm.state_dict())
        dst_layer.g_norm.load_state_dict(src_layer.g_norm.state_dict())
    model.predictor.linear.load_state_dict(predictor.linear.state_dict())
    model.predictor.output_layer_norm.load_state_dict(predictor.output_layer_norm.state_dict())

    model.joiner.linear.load_state_dict(decoder.joiner.linear.state_dict())

    return model


@torch.no_grad()
def convert_emformer_checkpoint(model_name: str, model_output_dir: str):
    """
    Copy/paste/tweak model's weights to the Transformers design.
    """
    bundle = NAME2BUNDLE[model_name]
    config = convert_config(bundle)
    feature_extractor = convert_feature_extractor(bundle)
    model = convert_weights(bundle, config)
    model.eval()

    waveform = torch.load("/home/anton/repos/audio/examples/asr/emformer_rnnt/librispeech_waveform_0.pt")
    features = feature_extractor(waveform, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, help="Path to the Emformer source model name")
    parser.add_argument("--model_output_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_emformer_checkpoint(args.model_name, args.model_output_dir)
