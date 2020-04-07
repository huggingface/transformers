# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Convert BertExtAbs's checkpoints.

The script looks like it is doing something trivial but it is not. The "weights"
proposed by the authors are actually the entire model pickled. We need to load
the model within the original codebase to be able to only save its `state_dict`.
"""

import argparse
import logging
from collections import namedtuple

import torch

from model_bertabs import BertAbsSummarizer
from models.model_builder import AbsSummarizer  # The authors' implementation
from transformers import BertTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_TEXT = "Hello world! cécé herlolip"


BertAbsConfig = namedtuple(
    "BertAbsConfig",
    [
        "temp_dir",
        "large",
        "use_bert_emb",
        "finetune_bert",
        "encoder",
        "share_emb",
        "max_pos",
        "enc_layers",
        "enc_hidden_size",
        "enc_heads",
        "enc_ff_size",
        "enc_dropout",
        "dec_layers",
        "dec_hidden_size",
        "dec_heads",
        "dec_ff_size",
        "dec_dropout",
    ],
)


def convert_bertabs_checkpoints(path_to_checkpoints, dump_path):
    """ Copy/paste and tweak the pre-trained weights provided by the creators
    of BertAbs for the internal architecture.
    """

    # Instantiate the authors' model with the pre-trained weights
    config = BertAbsConfig(
        temp_dir=".",
        finetune_bert=False,
        large=False,
        share_emb=True,
        use_bert_emb=False,
        encoder="bert",
        max_pos=512,
        enc_layers=6,
        enc_hidden_size=512,
        enc_heads=8,
        enc_ff_size=512,
        enc_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
    )
    checkpoints = torch.load(path_to_checkpoints, lambda storage, loc: storage)
    original = AbsSummarizer(config, torch.device("cpu"), checkpoints)
    original.eval()

    new_model = BertAbsSummarizer(config, torch.device("cpu"))
    new_model.eval()

    # -------------------
    # Convert the weights
    # -------------------

    logging.info("convert the model")
    new_model.bert.load_state_dict(original.bert.state_dict())
    new_model.decoder.load_state_dict(original.decoder.state_dict())
    new_model.generator.load_state_dict(original.generator.state_dict())

    # ----------------------------------
    # Make sure the outpus are identical
    # ----------------------------------

    logging.info("Make sure that the models' outputs are identical")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # prepare the model inputs
    encoder_input_ids = tokenizer.encode("This is sample éàalj'-.")
    encoder_input_ids.extend([tokenizer.pad_token_id] * (512 - len(encoder_input_ids)))
    encoder_input_ids = torch.tensor(encoder_input_ids).unsqueeze(0)
    decoder_input_ids = tokenizer.encode("This is sample 3 éàalj'-.")
    decoder_input_ids.extend([tokenizer.pad_token_id] * (512 - len(decoder_input_ids)))
    decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze(0)

    # failsafe to make sure the weights reset does not affect the
    # loaded weights.
    assert torch.max(torch.abs(original.generator[0].weight - new_model.generator[0].weight)) == 0

    # forward pass
    src = encoder_input_ids
    tgt = decoder_input_ids
    segs = token_type_ids = None
    clss = None
    mask_src = encoder_attention_mask = None
    mask_tgt = decoder_attention_mask = None
    mask_cls = None

    # The original model does not apply the geneator layer immediatly but rather in
    # the beam search (where it combines softmax + linear layer). Since we already
    # apply the softmax in our generation process we only apply the linear layer here.
    # We make sure that the outputs of the full stack are identical
    output_original_model = original(src, tgt, segs, clss, mask_src, mask_tgt, mask_cls)[0]
    output_original_generator = original.generator(output_original_model)

    output_converted_model = new_model(
        encoder_input_ids, decoder_input_ids, token_type_ids, encoder_attention_mask, decoder_attention_mask
    )[0]
    output_converted_generator = new_model.generator(output_converted_model)

    maximum_absolute_difference = torch.max(torch.abs(output_converted_model - output_original_model)).item()
    print("Maximum absolute difference beween weights: {:.2f}".format(maximum_absolute_difference))
    maximum_absolute_difference = torch.max(torch.abs(output_converted_generator - output_original_generator)).item()
    print("Maximum absolute difference beween weights: {:.2f}".format(maximum_absolute_difference))

    are_identical = torch.allclose(output_converted_model, output_original_model, atol=1e-3)
    if are_identical:
        logging.info("all weights are equal up to 1e-3")
    else:
        raise ValueError("the weights are different. The new model is likely different from the original one.")

    # The model has been saved with torch.save(model) and this is bound to the exact
    # directory structure. We save the state_dict instead.
    logging.info("saving the model's state dictionary")
    torch.save(
        new_model.state_dict(), "./bertabs-finetuned-cnndm-extractive-abstractive-summarization/pytorch_model.bin"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bertabs_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()

    convert_bertabs_checkpoints(
        args.bertabs_checkpoint_path, args.pytorch_dump_folder_path,
    )
