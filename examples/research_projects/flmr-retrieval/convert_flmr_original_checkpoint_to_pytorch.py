# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import argparse
import collections
from pathlib import Path

import torch
from torch.serialization import default_restore_location

from transformers import (
    FLMRConfig,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRTextConfig,
    FLMRVisionConfig,
)


list_of_keys_to_load = ["epoch", "global_step", "state_dict", "optimizer_states"]
CheckpointState = collections.namedtuple("CheckpointState", list_of_keys_to_load)


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    print(f"Reading saved model from {model_file}")
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    state_dict = {k: v for k, v in state_dict.items() if k in list_of_keys_to_load}
    return CheckpointState(**state_dict)


class FLMRState:
    def __init__(self, src_file: Path, vision_model_version: str = "openai/clip-vit-base-patch32"):
        self.src_file = src_file
        self.vision_model_version = vision_model_version

    def load_flmr_model(self):
        raise NotImplementedError

    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> "FLMRState":
        if comp_type.startswith("FLMR"):
            return FLMRModelForRetrievalState(*args, **kwargs)
        if comp_type.startswith("PreFLMR"):
            return PreFLMRModelForRetrievalState(*args, **kwargs)
        else:
            raise ValueError("Component type must be either 'ctx_encoder', 'question_encoder' or 'reader'.")


class FLMRModelForRetrievalState(FLMRState):
    def load_flmr_model(self):
        query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained("bert-base-uncased", query_maxlen=512)
        context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained("bert-base-uncased")
        # FLMR uses some special tokens that are not in the original BERT tokenizer
        special_tokens_to_add = ["<BOV>", "<SOV>", "<EOV>", "<BOQ>", "<EOQ>", "<BOC>", "<EOC>", "<BOK>", "<EOK>"]
        query_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        context_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        text_config = FLMRTextConfig.from_pretrained("bert-base-uncased")
        vision_config = FLMRVisionConfig.from_pretrained(self.vision_model_version)

        flmr_config = FLMRConfig.from_text_vision_configs(
            text_config=text_config,
            vision_config=vision_config,
            query_concat_output_from_vision_encoder=True,
            query_concat_output_from_text_encoder=True,
            context_concat_output_from_vision_encoder=False,
            context_concat_output_from_text_encoder=True,
            vision_model_version=self.vision_model_version,
        )

        model = FLMRModelForRetrieval(
            config=flmr_config, query_tokenizer=query_tokenizer, context_tokenizer=context_tokenizer
        )

        print(f"Loading FLMR biencoder from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)

        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        # state_dict = {"context_text_encoder.bert_model.embeddings.position_ids": model.context_text_encoder.bert_model.embeddings.position_ids}
        state_dict = {}
        from pprint import pprint

        # pprint(model.state_dict().keys())
        # print("=================================")
        for key, value in saved_state.state_dict.items():
            # if key in ['model.vision_projection.model.0.weight',
            #            'model.model.linear.weight',
            #            'model.model.bert.encoder.layer.0.attention.self.query.weight']:
            #     print(key, value[0])

            if key.startswith("model.model.bert"):
                new_key = key.replace("model.model.bert.", "context_text_encoder.bert_model.")
            elif key.startswith("model.model.linear"):
                new_key = key.replace("model.model.linear.", "context_text_encoder_linear.")
            elif key.startswith("model.vision_projection."):
                new_key = key.replace("model.vision_projection.", "context_vision_projection.")

            if new_key.startswith("context_text_encoder.bert_model.embeddings.position_ids"):
                # This is to fix the bug that came with a recent update in Huggingface
                # position_ids are no longer used in the model, and thus we need to ignore it from the checkpoint that was trained using an older version of Huggingface.
                continue

            if new_key.startswith("context_text_encoder.bert_model.embeddings.word_embeddings.weight"):
                # check the shape of the embedding and then update the vocab size
                print(value.shape)
                vocab_size = value.shape[0]
                model.context_text_encoder.bert_model.resize_token_embeddings(vocab_size)

            state_dict[new_key] = value

        if model.config.use_vision_encoder:
            # Need to load vision model parameters into the checkpoint
            from transformers import CLIPVisionModel

            vision_model = CLIPVisionModel.from_pretrained(model.config.vision_model_version)

            vision_model_state_dict = vision_model.state_dict()
            for key, value in vision_model_state_dict.items():
                new_key = "context_vision_encoder.vision_model." + key
                state_dict[new_key] = value

        if not model.config.separate_query_and_context_text_encoder:
            # Need to copy the context_text_encoder parameters to query_text_encoder
            temp_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("context_text_encoder"):
                    new_key = key.replace("context_text_encoder", "query_text_encoder")
                    temp_state_dict[new_key] = value

            state_dict.update(temp_state_dict)

        if not model.config.separate_query_and_context_vision_encoder:
            # Need to copy the context_vision_encoder parameters to query_vision_encoder
            temp_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("context_vision_encoder"):
                    new_key = key.replace("context_vision_encoder", "query_vision_encoder")
                    temp_state_dict[new_key] = value
                if key.startswith("context_vision_projection"):
                    new_key = key.replace("context_vision_projection", "query_vision_projection")
                    temp_state_dict[new_key] = value

            state_dict.update(temp_state_dict)

        # Compare the keys in state_dict, and the keys in model.state_dict()
        # Report parameters in state_dict but not in model.state_dict()
        missing_keys = []
        for key in state_dict.keys():
            if key not in model.state_dict().keys():
                missing_keys.append(key)
        print("=================================")
        print("Keys in the checkpoint but not in the model:")
        pprint(missing_keys)
        print("=================================")
        # Report parameters in model.state_dict() but not in state_dict
        extra_keys = []
        for key in model.state_dict().keys():
            if key not in state_dict.keys():
                extra_keys.append(key)
        print("=================================")
        print("Keys in the model but not in the checkpoint:")
        pprint(extra_keys)
        print("=================================")

        model.load_state_dict(state_dict)
        return model


class PreFLMRModelForRetrievalState(FLMRState):
    def load_flmr_model(self, **kwargs):
        query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained("bert-base-uncased")
        context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained("bert-base-uncased")

        text_config = FLMRTextConfig.from_pretrained("bert-base-uncased")
        vision_config = FLMRVisionConfig.from_pretrained(self.vision_model_version)

        flmr_config = FLMRConfig.from_text_vision_configs(
            text_config=text_config,
            vision_config=vision_config,
            query_concat_output_from_vision_encoder=True,
            query_concat_output_from_text_encoder=True,
            context_concat_output_from_vision_encoder=False,
            context_concat_output_from_text_encoder=True,
            use_transformer_mapping_network=True,
            transformer_mapping_config_base="bert-base-uncased",
            transformer_mapping_num_hidden_layers=1,
            separate_query_and_context_text_encoder=True,
            mask_instruction_token=":",
            vision_model_version=self.vision_model_version,
        )

        model = FLMRModelForRetrieval(
            config=flmr_config, query_tokenizer=query_tokenizer, context_tokenizer=context_tokenizer
        )

        print(f"Loading PreFLMR biencoder from {self.src_file}")
        saved_state = load_states_from_checkpoint(self.src_file)

        # Fix changes from https://github.com/huggingface/transformers/commit/614fef1691edb806de976756d4948ecbcd0c0ca3
        # state_dict = {"context_text_encoder.bert_model.embeddings.position_ids": model.context_text_encoder.bert_model.embeddings.position_ids}
        state_dict = {}
        from pprint import pprint

        # pprint(list(model.state_dict().keys()))
        # print("=================================")
        # pprint(list(saved_state.state_dict.keys()))
        # input()
        for key, value in saved_state.state_dict.items():
            if key.startswith("model.model.bert"):
                new_key = key.replace("model.model.bert.", "context_text_encoder.bert_model.")
            elif key.startswith("model.model.linear."):
                new_key = key.replace("model.model.linear.", "context_text_encoder_linear.")
            elif key.startswith("model.vision_model."):
                new_key = key.replace("model.vision_model.", "context_vision_encoder.vision_model.")
            elif key.startswith("model.vision_projection_linear."):
                new_key = key.replace("model.vision_projection_linear.", "transformer_mapping_output_linear.")
            elif key.startswith("model.vision_projection_input_linear2."):
                new_key = key.replace("model.vision_projection_input_linear2.", "transformer_mapping_input_linear.")
            elif key.startswith("model.vision_projection_input_linear."):
                new_key = key.replace("model.vision_projection_input_linear.", "context_vision_projection.")
            elif key.startswith("model.vision_projection."):  # In PreFLMR, this is transformer_mapping_network
                new_key = key.replace("model.vision_projection.", "transformer_mapping_network.")
            elif key.startswith("model.query_encoder."):
                new_key = key.replace("model.query_encoder.", "query_text_encoder.bert_model.")
            elif key.startswith("model.query_linear."):
                new_key = key.replace("model.query_linear.", "query_text_encoder_linear.")
            else:
                new_key = key

            if new_key.startswith("context_text_encoder.bert_model.embeddings.position_ids"):
                # This is to fix the bug that came with a recent update in Huggingface
                # position_ids are no longer used in the model, and thus we need to ignore it from the checkpoint that was trained using an older version of Huggingface.
                continue

            if new_key.startswith("context_text_encoder.bert_model.embeddings.word_embeddings.weight"):
                # check the shape of the embedding and then update the vocab size
                print(value.shape)
                vocab_size = value.shape[0]
                model.context_text_encoder.bert_model.resize_token_embeddings(vocab_size)

            # print(f"{key}\t -> \t{new_key} \t {value.shape}")
            state_dict[new_key] = value

        if model.config.use_vision_encoder:
            # Need to load vision model parameters into the checkpoint
            from transformers import CLIPVisionModel

            vision_model = CLIPVisionModel.from_pretrained(self.vision_model_version)

            vision_model_state_dict = vision_model.state_dict()
            for key, value in vision_model_state_dict.items():
                new_key = "context_vision_encoder.vision_model." + key
                state_dict[new_key] = value

        if not model.config.separate_query_and_context_text_encoder:
            print("Copying context_text_encoder parameters to query_text_encoder")
            # Need to copy the context_text_encoder parameters to query_text_encoder
            temp_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("context_text_encoder"):
                    new_key = key.replace("context_text_encoder", "query_text_encoder")
                    temp_state_dict[new_key] = value

            state_dict.update(temp_state_dict)

        if not model.config.separate_query_and_context_vision_encoder:
            print("Copying context_vision_encoder parameters to query_vision_encoder")
            # Need to copy the context_vision_encoder parameters to query_vision_encoder
            temp_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("context_vision_encoder"):
                    new_key = key.replace("context_vision_encoder", "query_vision_encoder")
                    temp_state_dict[new_key] = value
                if key.startswith("context_vision_projection"):
                    new_key = key.replace("context_vision_projection", "query_vision_projection")
                    temp_state_dict[new_key] = value

            state_dict.update(temp_state_dict)

        # Compare the keys in state_dict, and the keys in model.state_dict()
        # Report parameters in state_dict but not in model.state_dict()
        missing_keys = []
        for key in state_dict.keys():
            if key not in model.state_dict().keys():
                missing_keys.append(key)
        print("=================================")
        print("Keys in the checkpoint but not in the model:")
        pprint(missing_keys)
        print("=================================")
        # Report parameters in model.state_dict() but not in state_dict
        extra_keys = []
        for key in model.state_dict().keys():
            if key not in state_dict.keys():
                extra_keys.append(key)
        print("=================================")
        print("Keys in the model but not in the checkpoint:")
        pprint(extra_keys)
        print("=================================")

        model.load_state_dict(state_dict)
        return model


def convert(args, src_file: Path, dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    flmr_state = FLMRState.from_type(args.type, src_file=src_file, vision_model_version=args.vision_model_version)
    model = flmr_state.load_flmr_model()
    model.save_pretrained(dest_dir)
    model.query_tokenizer.save_pretrained(dest_dir / "query_tokenizer")
    model.context_tokenizer.save_pretrained(dest_dir / "context_tokenizer")

    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(dest_dir / "query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(dest_dir / "context_tokenizer")
    model = model.from_pretrained(
        dest_dir, query_tokenizer=query_tokenizer, context_tokenizer=context_tokenizer
    )  # sanity check
    # print parameters
    # for name, param in model.named_parameters():
    #     if name in ["context_vision_encoder.vision_model.encoder.layers.0.self_attn.k_proj.weight", "context_vision_projection.model.0.weight", "context_text_encoder_linear.weight", "context_text_encoder.bert_model.encoder.layer.0.attention.self.query.weight"]:
    #         print(name, param[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--type", type=str, help="Type of the FLMR model: 'FLMR' or 'PreFLMR'.")
    parser.add_argument(
        "--src",
        type=str,
        help=(
            "Path to the flmr checkpoint file. They can be downloaded from the official FLMR repo"
            " https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering."
        ),
    )
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model directory.")
    parser.add_argument(
        "--vision_model_version",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Version of the CLIP vision model to use.",
    )
    args = parser.parse_args()

    src_file = Path(args.src)
    dest_dir = f"converted-{src_file.name}" if args.dest is None else args.dest
    dest_dir = Path(dest_dir)
    assert src_file.exists()
    assert args.type is not None, "Please specify the type of the FLMR model to convert: 'FLMR' or 'PreFLMR'."
    convert(args, src_file, dest_dir)
