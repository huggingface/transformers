# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert X-MOD checkpoint."""

import argparse
from pathlib import Path

import fairseq
import torch
from fairseq.models.xmod import XMODModel as FairseqXmodModel
from packaging import version

from transformers import XmodConfig, XmodForMaskedLM, XmodForSequenceClassification
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.12.2"):
    raise Exception("requires fairseq >= 0.12.2")
if version.parse(fairseq.__version__) > version.parse("2"):
    raise Exception("requires fairseq < v2")

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello, World!"
SAMPLE_LANGUAGE = "en_XX"


def convert_xmod_checkpoint_to_pytorch(
    xmod_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    data_dir = Path("data_bin")
    xmod = FairseqXmodModel.from_pretrained(
        model_name_or_path=str(Path(xmod_checkpoint_path).parent),
        checkpoint_file=Path(xmod_checkpoint_path).name,
        _name="xmod_base",
        arch="xmod_base",
        task="multilingual_masked_lm",
        data_name_or_path=str(data_dir),
        bpe="sentencepiece",
        sentencepiece_model=str(Path(xmod_checkpoint_path).parent / "sentencepiece.bpe.model"),
        src_dict=str(data_dir / "dict.txt"),
    )
    xmod.eval()  # disable dropout
    print(xmod)

    xmod_sent_encoder = xmod.model.encoder.sentence_encoder
    config = XmodConfig(
        vocab_size=xmod_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=xmod.cfg.model.encoder_embed_dim,
        num_hidden_layers=xmod.cfg.model.encoder_layers,
        num_attention_heads=xmod.cfg.model.encoder_attention_heads,
        intermediate_size=xmod.cfg.model.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
        pre_norm=xmod.cfg.model.encoder_normalize_before,
        adapter_reduction_factor=getattr(xmod.cfg.model, "bottleneck", 2),
        adapter_layer_norm=xmod.cfg.model.adapter_layer_norm,
        adapter_reuse_layer_norm=xmod.cfg.model.adapter_reuse_layer_norm,
        ln_before_adapter=xmod.cfg.model.ln_before_adapter,
        languages=xmod.cfg.model.languages,
    )
    if classification_head:
        config.num_labels = xmod.model.classification_heads["mnli"].out_proj.weight.shape[0]

    print("Our X-MOD config:", config)

    model = XmodForSequenceClassification(config) if classification_head else XmodForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight = xmod_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = xmod_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c xmod doesn't use them.

    model.roberta.embeddings.LayerNorm.weight = xmod_sent_encoder.layernorm_embedding.weight
    model.roberta.embeddings.LayerNorm.bias = xmod_sent_encoder.layernorm_embedding.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer = model.roberta.encoder.layer[i]
        xmod_layer = xmod_sent_encoder.layers[i]

        # self attention
        self_attn = layer.attention.self
        if not (
            xmod_layer.self_attn.k_proj.weight.data.shape
            == xmod_layer.self_attn.q_proj.weight.data.shape
            == xmod_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        ):
            raise AssertionError("Dimensions of self-attention weights do not match.")

        self_attn.query.weight.data = xmod_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = xmod_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = xmod_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = xmod_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = xmod_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = xmod_layer.self_attn.v_proj.bias

        # self-attention output
        self_output = layer.attention.output
        if self_output.dense.weight.shape != xmod_layer.self_attn.out_proj.weight.shape:
            raise AssertionError("Dimensions of self-attention output weights do not match.")
        self_output.dense.weight = xmod_layer.self_attn.out_proj.weight
        self_output.dense.bias = xmod_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = xmod_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = xmod_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate = layer.intermediate
        if intermediate.dense.weight.shape != xmod_layer.fc1.weight.shape:
            raise AssertionError("Dimensions of intermediate weights do not match.")
        intermediate.dense.weight = xmod_layer.fc1.weight
        intermediate.dense.bias = xmod_layer.fc1.bias

        # output
        bert_output = layer.output
        if bert_output.dense.weight.shape != xmod_layer.fc2.weight.shape:
            raise AssertionError("Dimensions of feed-forward weights do not match.")
        bert_output.dense.weight = xmod_layer.fc2.weight
        bert_output.dense.bias = xmod_layer.fc2.bias
        bert_output.LayerNorm.weight = xmod_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = xmod_layer.final_layer_norm.bias
        if bert_output.adapter_layer_norm is not None:
            bert_output.adapter_layer_norm.weight = xmod_layer.adapter_layer_norm.weight
            bert_output.adapter_layer_norm.bias = xmod_layer.adapter_layer_norm.bias

        if list(sorted(bert_output.adapter_modules.keys())) != list(sorted(xmod_layer.adapter_modules.keys())):
            raise AssertionError("Lists of language adapters do not match.")
        for lang_code, adapter in xmod_layer.adapter_modules.items():
            to_adapter = bert_output.adapter_modules[lang_code]
            from_adapter = xmod_layer.adapter_modules[lang_code]
            to_adapter.dense1.weight = from_adapter.fc1.weight
            to_adapter.dense1.bias = from_adapter.fc1.bias
            to_adapter.dense2.weight = from_adapter.fc2.weight
            to_adapter.dense2.bias = from_adapter.fc2.bias

        # end of layer

    if xmod_sent_encoder.layer_norm is not None:
        model.roberta.encoder.LayerNorm.weight = xmod_sent_encoder.layer_norm.weight
        model.roberta.encoder.LayerNorm.bias = xmod_sent_encoder.layer_norm.bias

    if classification_head:
        model.classifier.dense.weight = xmod.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = xmod.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = xmod.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = xmod.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = xmod.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = xmod.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = xmod.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = xmod.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = xmod.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = xmod.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids = xmod.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1
    model.roberta.set_default_language(SAMPLE_LANGUAGE)

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = xmod.model.classification_heads["mnli"](xmod.extract_features(input_ids))
    else:
        their_output = xmod.model(input_ids, lang_id=[SAMPLE_LANGUAGE])[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--xmod_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_xmod_checkpoint_to_pytorch(
        args.xmod_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
