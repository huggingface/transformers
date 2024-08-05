# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

"""
This script converts fairseq/fairseq2 SONAR text encoder and decoder checkpoints to transformers.
The reference architectures are given in https://github.com/facebookresearch/SONAR/blob/main/sonar/models/sonar_text/builder.py.
The checkpoints for conversion can be found in:
- encoder: https://github.com/facebookresearch/SONAR/blob/main/sonar/cards/text_sonar_basic_encoder.yaml
- decoder: https://github.com/facebookresearch/SONAR/blob/main/sonar/cards/text_sonar_basic_decoder.yaml
"""

import argparse

import torch
from torch import nn

from transformers import M2M100Config, M2M100DecoderModel, M2M100EncoderModel


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def get_parameter_renames(model, state_dict, is_decoder=True):
    parameter_renames = {}
    trf_names = {name for name, value in model.named_parameters()}
    fs2_names = set(state_dict.keys())

    for trf_name in trf_names:
        fs2_name = trf_name
        if trf_name == "shared.weight":
            fs2_name = "decoder_frontend.embed.weight" if is_decoder else "encoder_frontend.embed.weight"
        if trf_name == "lm_head.weight":
            fs2_name = "final_proj.weight"

        if trf_name.startswith("layers."):
            fs2_name = "decoder." + trf_name if is_decoder else "encoder." + trf_name
        if trf_name.startswith("layer_norm.") and is_decoder:
            fs2_name = "decoder." + trf_name
        if trf_name.startswith("encoder.layer_norm.") and not is_decoder:
            fs2_name = trf_name.split(".", 1)[1]

        if ".encoder_attn." in fs2_name:
            fs2_name = fs2_name.replace(".encoder_attn.", ".encoder_decoder_attn.")
        if ".encoder_attn_layer_norm." in fs2_name:
            fs2_name = fs2_name.replace(".encoder_attn_layer_norm.", ".encoder_decoder_attn_layer_norm.")
        if ".out_proj." in fs2_name:
            fs2_name = fs2_name.replace(".out_proj.", ".output_proj.")
        if ".fc1." in fs2_name:
            fs2_name = fs2_name.replace(
                ".fc1.",
                ".ffn.inner_proj.",
            )
        if ".fc2." in fs2_name:
            fs2_name = fs2_name.replace(
                ".fc2.",
                ".ffn.output_proj.",
            )
        if ".final_layer_norm." in fs2_name:
            fs2_name = fs2_name.replace(
                ".final_layer_norm.",
                ".ffn_layer_norm.",
            )

        if fs2_name in fs2_names:
            parameter_renames[trf_name] = fs2_name
        else:
            raise ValueError(f"Parameter {trf_name} could not be mapped from transformers to fairseq2 state dict.")
    return parameter_renames


def reorder_special_tokens(new_state_dict):
    """
    In fairseq2, special tokens are ['<pad>', '<unk>', '<s>', '</s>'].
    In transformers (NLLB) they are ['<s>', '<pad>', '</s>', '<unk>'].
    We want to reuse the NLLB tokenizer, so we reorder the embeddings.
    """
    special_token_embs = new_state_dict["shared.weight"].data[[2, 0, 3, 1]].clone()
    for param_name in [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
        "shared.weight",
    ]:
        if param_name in new_state_dict:
            new_state_dict[param_name].data[[0, 1, 2, 3]] = special_token_embs


def convert_sonar_checkpoint_from_disk(checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint_dict["model"]

    # In Fairseq2 SONAR checkpoints, there are no configs (they are supposed to be in a yaml model card elsewhere).
    # Thus, we just assume the "basic" hyperparameters
    # see the arhc registry at https://github.com/facebookresearch/SONAR/blob/main/sonar/models/sonar_text/builder.py

    config = M2M100Config(
        vocab_size=256206,
        max_position_embeddings=1024,
        encoder_layers=24,
        decoder_layers=24,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=1024 * 8,
        decoder_ffn_dim=1024 * 8,
        d_model=1024,
        encoder_layerdrop=0,
        decoder_layerdrop=0,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_function="relu",
    )

    if any(parameter_name.startswith("encoder.") for parameter_name in state_dict):
        is_decoder = False
        model = M2M100EncoderModel(config)
    elif any(parameter_name.startswith("decoder.") for parameter_name in state_dict):
        is_decoder = True
        model = M2M100DecoderModel(config)
    else:
        raise ValueError("The state dict does not seem to contain SONAR encoder or decoder.")

    parameter_renames = get_parameter_renames(model, state_dict, is_decoder)
    new_state_dict = {trf_name: state_dict[fs2_name] for trf_name, fs2_name in parameter_renames.items()}
    reorder_special_tokens(new_state_dict)

    if is_decoder:
        new_state_dict["decoder.embed_tokens.weight"] = new_state_dict["shared.weight"]
    else:
        new_state_dict["encoder.embed_tokens.weight"] = new_state_dict["shared.weight"]

    model.load_state_dict(new_state_dict, strict=True)
    model.tie_weights()

    return model


def test_conversion_accuracy(fairseq2_encoder_path, fairseq2_decoder_path):
    """
    This test is not directly invoked, because the encoder and decoder paths should be provided explicitly,
    and these checkpoints are too heavy to download them by default, just for a test.
    Please run the test from your code like below:
    ```
    from transformers.models.m2m_100.convert_sonar_original_checkpoint_to_transformers import test_conversion_accuracy
    test_conversion_accuracy(PATH_TO_ENCODER, PATH_TO_DECODER)
    ```
    The fairseq2 checkpoints can be downloaded from the urls indicated in the following cards:
        - https://github.com/facebookresearch/SONAR/blob/main/sonar/cards/text_sonar_basic_encoder.yaml
        - https://github.com/facebookresearch/SONAR/blob/main/sonar/cards/text_sonar_basic_decoder.yaml

    The reference embeddings were obtained with:
    ```
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
    ref_embeddings = t2vec_model.predict(sentences, source_lang="eng_Latn")[:, :5]
    ```
    """
    from transformers import NllbTokenizer
    from transformers.modeling_outputs import BaseModelOutput

    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", clean_up_tokenization_spaces=True)
    sentences = ["My name is SONAR.", "I can embed the sentences into vectorial space."]
    tokenizer.src_lang = "eng_Latn"
    batch = tokenizer(sentences, padding=True, return_tensors="pt")

    print("Converting the encoder...")
    enc = convert_sonar_checkpoint_from_disk(fairseq2_encoder_path).eval()
    assert isinstance(enc, M2M100EncoderModel)

    print("Conversion completed, testing the embedding accuracy...")
    with torch.inference_mode():
        enc_out = enc(**batch, pool_last_hidden_state=True)
    assert enc_out.last_hidden_state.shape == (2, 1, 1024)
    embeddings = enc_out.last_hidden_state.squeeze(1)

    ref_embeddings = torch.tensor(
        [[-0.005286, 0.002008, -0.000562, 0.006344, 0.006329], [-0.000330, -0.007055, 0.007644, 0.001841, 0.003727]]
    )
    assert torch.allclose(embeddings[:, :5], ref_embeddings, rtol=1e-3)
    print("The embedding accuracy test has passed!")

    print("Converting the decoder...")
    dec = convert_sonar_checkpoint_from_disk(fairseq2_decoder_path).eval()
    assert isinstance(dec, M2M100DecoderModel)

    print("Conversion completed, testing the decoding accuracy...")
    gen_out = dec.generate(
        # passing encoder_outputs is not recommended, because beam search decoding modifies them in place, which is ugly
        # encoder_outputs=enc_out,
        encoder_outputs=BaseModelOutput(last_hidden_state=enc_out.last_hidden_state.clone()),
        num_beams=5,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
    )
    text_out = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    assert text_out == ["My name is SONAR.", "I can embed the sentences into vector space."]
    print("The decoding accuracy test has passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("dump_folder_path", default=None, type=str, help="Path to the output transformers model.")
    args = parser.parse_args()
    model = convert_sonar_checkpoint_from_disk(args.fairseq_path)
    model.save_pretrained(args.dump_folder_path)
