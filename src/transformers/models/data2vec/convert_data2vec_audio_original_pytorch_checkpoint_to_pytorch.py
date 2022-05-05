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
"""Convert Wav2Vec2 checkpoint."""


import argparse
import os
from functools import reduce

import fairseq
import torch
from datasets import load_dataset

from transformers import Wav2Vec2Processor, logging
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig

# Copied from https://github.com/pytorch/fairseq/blob/main/examples/data2vec/models/data2vec_audio.py
from transformers.models.data2vec.data2vec_audio import Data2VecAudioModel as Dummy  # noqa: F401
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioForCTC, Data2VecAudioModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "models.0.layer_norm": "feature_projection.layer_norm",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
TOP_LEVEL_KEYS = [
    "lm_head",
]


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def recursively_load_weights(fairseq_model, hf_model, is_headless):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    if not is_headless:
        feature_extractor = hf_model.data2vec_audio.feature_extractor
        pos_conv_embedding = hf_model.data2vec_audio.encoder.pos_conv_embed

    else:
        feature_extractor = hf_model.feature_extractor
        pos_conv_embedding = hf_model.encoder.pos_conv_embed

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
            )
            is_used = True
        elif "pos_conv" in name:
            load_pos_conv_layer(
                name,
                value,
                pos_conv_embedding,
                unused_weights,
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if not is_headless:
                    mapped_key = "data2vec_audio." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: don't match quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def access_by_string(module, path):
    names = path.split(".")
    return reduce(getattr, names, module)


def set_weights(full_name, module, fsq_value, hf_weight_path):
    hf_weight = access_by_string(module, hf_weight_path)
    hf_value = hf_weight.data

    if fsq_value.shape != hf_value.shape:
        raise ValueError(f"{full_name} has size {fsq_value.shape}, but {hf_value.shape} was found.")
    hf_weight.data = fsq_value
    logger.info(f"{full_name} was correctly initialized from {hf_weight_path}.")


def load_conv_layer(full_name, value, feature_extractor, unused_weights):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    weight_type = name.split(".")[-1]
    if type_id == 0:
        layer_type = "conv"
    elif type_id == 2:
        layer_type = "layer_norm"
    else:
        unused_weights.append(full_name)
        return

    set_weights(full_name, feature_extractor, value, f"conv_layers.{layer_id}.{layer_type}.{weight_type}")


def load_pos_conv_layer(full_name, value, pos_conv_embeddings, unused_weights):
    name = full_name.split("pos_conv.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    weight_type = name.split(".")[-1]
    if type_id != 0:
        unused_weights.append(full_name)
        return
    else:
        layer_type = "conv"

    set_weights(full_name, pos_conv_embeddings, value, f"layers.{layer_id}.{layer_type}.{weight_type}")


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = Data2VecAudioConfig.from_pretrained(config_path)
    else:
        config = Data2VecAudioConfig()

    if not is_finetuned:
        # Modify final_proj layer name
        hf_wav2vec = Data2VecAudioModel(config)
        data2vec_checkpoint_dir = os.path.dirname(checkpoint_path)

        state_dict = torch.load(checkpoint_path)
        state_dict["model"]["final_proj.weight"] = state_dict["model"].pop("final_proj.0.weight")
        state_dict["model"]["final_proj.bias"] = state_dict["model"].pop("final_proj.0.bias")
        converted_ckpt = os.path.join(data2vec_checkpoint_dir, "converted.pt")
        torch.save(state_dict, converted_ckpt)
    else:
        hf_wav2vec = Data2VecAudioForCTC(config)
        converted_ckpt = checkpoint_path

    def load_data2vec(path):
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
        return model[0].eval()

    model = load_data2vec(converted_ckpt)

    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")

    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    input_audio = [x["array"] for x in ds[:4]["audio"]]

    inputs = processor(input_audio, return_tensors="pt", padding=True)

    input_values = inputs.input_values
    attention_mask = inputs.attention_mask
    #    input_values = inputs.input_values[:, :-1]
    #    attention_mask = inputs.attention_mask[:, :-1]

    hf_wav2vec.eval()
    model.eval()
    if is_finetuned:
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)[
            "encoder_out"
        ].transpose(0, 1)
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["logits"]

        pred_ids = torch.argmax(our_output, dim=-1)
        output_string = processor.batch_decode(pred_ids)

        print(f"Expected Output: {ds[:4]['text']}, Pred: {output_string}")
    else:
        their_output = model(source=input_values, padding_mask=(1 - attention_mask), mask=False, features_only=True)[
            "layer_results"
        ][-1][0].transpose(0, 1)
        our_output = hf_wav2vec(input_values, attention_mask=attention_mask)["last_hidden_state"]

    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)

    if is_finetuned:
        processor.save_pretrained(pytorch_dump_folder_path)
    else:
        processor.feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    args = parser.parse_args()
    convert_wav2vec2_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, not args.not_finetuned
    )
