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
"""Convert mLUKE checkpoint."""

import argparse
import json
import os
from collections import OrderedDict

import torch

from transformers import LukeConfig, LukeForMaskedLM, MLukeTokenizer, XLMRobertaTokenizer
from transformers.tokenization_utils_base import AddedToken


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):
    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(use_entity_aware_attention=True, **metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["module"]

    # Load the entity vocab file
    entity_vocab = load_original_entity_vocab(entity_vocab_path)
    # add an entry for [MASK2]
    entity_vocab["[MASK2]"] = max(entity_vocab.values()) + 1
    config.entity_vocab_size += 1

    tokenizer = XLMRobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # Add special tokens to the token vocabulary for downstream tasks
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [entity_token_1, entity_token_2]})
    config.vocab_size += 2

    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "r") as f:
        tokenizer_config = json.load(f)
    tokenizer_config["tokenizer_class"] = "MLukeTokenizer"
    with open(os.path.join(pytorch_dump_folder_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    with open(os.path.join(pytorch_dump_folder_path, MLukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # Initialize the embeddings of the special tokens
    ent_init_index = tokenizer.convert_tokens_to_ids(["@"])[0]
    ent2_init_index = tokenizer.convert_tokens_to_ids(["#"])[0]

    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[ent_init_index].unsqueeze(0)
    ent2_emb = word_emb[ent2_init_index].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])
    # add special tokens for 'entity_predictions.bias'
    for bias_name in ["lm_head.decoder.bias", "lm_head.bias"]:
        decoder_bias = state_dict[bias_name]
        ent_decoder_bias = decoder_bias[ent_init_index].unsqueeze(0)
        ent2_decoder_bias = decoder_bias[ent2_init_index].unsqueeze(0)
        state_dict[bias_name] = torch.cat([decoder_bias, ent_decoder_bias, ent2_decoder_bias])

    # Initialize the query layers of the entity-aware self-attention mechanism
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            prefix = f"encoder.layer.{layer_index}.attention.self."
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # Initialize the embedding of the [MASK2] entity using that of the [MASK] entity for downstream tasks
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_mask_emb = entity_emb[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb, entity_mask_emb])
    # add [MASK2] for 'entity_predictions.bias'
    entity_prediction_bias = state_dict["entity_predictions.bias"]
    entity_mask_bias = entity_prediction_bias[entity_vocab["[MASK]"]].unsqueeze(0)
    state_dict["entity_predictions.bias"] = torch.cat([entity_prediction_bias, entity_mask_bias])

    model = LukeForMaskedLM(config=config).eval()

    state_dict.pop("entity_predictions.decoder.weight")
    state_dict.pop("lm_head.decoder.weight")
    state_dict.pop("lm_head.decoder.bias")
    state_dict_for_hugging_face = OrderedDict()
    for key, value in state_dict.items():
        if not (key.startswith("lm_head") or key.startswith("entity_predictions")):
            state_dict_for_hugging_face[f"luke.{key}"] = state_dict[key]
        else:
            state_dict_for_hugging_face[key] = state_dict[key]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_for_hugging_face, strict=False)

    if set(unexpected_keys) != {"luke.embeddings.position_ids"}:
        raise ValueError(f"Unexpected unexpected_keys: {unexpected_keys}")
    if set(missing_keys) != {
        "lm_head.decoder.weight",
        "lm_head.decoder.bias",
        "entity_predictions.decoder.weight",
    }:
        raise ValueError(f"Unexpected missing_keys: {missing_keys}")

    model.tie_weights()
    assert (model.luke.embeddings.word_embeddings.weight == model.lm_head.decoder.weight).all()
    assert (model.luke.entity_embeddings.entity_embeddings.weight == model.entity_predictions.decoder.weight).all()

    # Check outputs
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path, task="entity_classification")

    text = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
    span = (0, 9)
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")

    outputs = model(**encoding)

    # Verify word hidden states
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 33, 768))
        expected_slice = torch.tensor([[0.0892, 0.0596, -0.2819], [0.0134, 0.1199, 0.0573], [-0.0169, 0.0927, 0.0644]])

    if not (outputs.last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.last_hidden_state.shape is {outputs.last_hidden_state.shape}, Expected shape is {expected_shape}"
        )
    if not torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # Verify entity hidden states
    if model_size == "large":
        raise NotImplementedError
    else:  # base
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[-0.1482, 0.0609, 0.0322]])

    if not (outputs.entity_last_hidden_state.shape == expected_shape):
        raise ValueError(
            f"Outputs.entity_last_hidden_state.shape is {outputs.entity_last_hidden_state.shape}, Expected shape is"
            f" {expected_shape}"
        )
    if not torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4):
        raise ValueError

    # Verify masked word/entity prediction
    tokenizer = MLukeTokenizer.from_pretrained(pytorch_dump_folder_path)
    text = "Tokyo is the capital of <mask>."
    span = (24, 30)
    encoding = tokenizer(text, entity_spans=[span], return_tensors="pt")

    outputs = model(**encoding)

    input_ids = encoding["input_ids"][0].tolist()
    mask_position_id = input_ids.index(tokenizer.convert_tokens_to_ids("<mask>"))
    predicted_id = outputs.logits[0][mask_position_id].argmax(dim=-1)
    assert "Japan" == tokenizer.decode(predicted_id)

    predicted_entity_id = outputs.entity_logits[0][0].argmax().item()
    multilingual_predicted_entities = [
        entity for entity, entity_id in tokenizer.entity_vocab.items() if entity_id == predicted_entity_id
    ]
    assert [e for e in multilingual_predicted_entities if e.startswith("en:")][0] == "en:Japan"

    # Finally, save our PyTorch model and tokenizer
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)


def load_original_entity_vocab(entity_vocab_path):
    SPECIAL_TOKENS = ["[MASK]", "[PAD]", "[UNK]"]

    data = [json.loads(line) for line in open(entity_vocab_path)]

    new_mapping = {}
    for entry in data:
        entity_id = entry["id"]
        for entity_name, language in entry["entities"]:
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            new_entity_name = f"{language}:{entity_name}"
            new_mapping[new_entity_name] = entity_id
    return new_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument(
        "--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration."
    )
    parser.add_argument(
        "--entity_vocab_path",
        default=None,
        type=str,
        help="Path to an entity_vocab.tsv file, containing the entity vocabulary.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model."
    )
    parser.add_argument(
        "--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted."
    )
    args = parser.parse_args()
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
