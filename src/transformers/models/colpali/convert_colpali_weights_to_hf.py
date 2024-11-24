# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert ColPali weights."""

import argparse
from pathlib import Path
from typing import Any, Dict, cast

import torch
from PIL import Image

from transformers.models.colpali import ColPaliForRetrieval, ColPaliProcessor
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_DTYPE = torch.bfloat16
TOLERANCE = 1e-3


# Copied from https://huggingface.co/vidore/colpali-v1.2-merged/blob/main/config.json
ORIGINAL_CONFIG: Dict[str, Any] = {
    "image_token_index": 257152,
    "_vocab_size": 257152,
    "projection_dim": 2048,
    "hidden_size": 2048,
    "vision_config": {
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
        "torchscript": False,
        "torch_dtype": None,
        "use_bfloat16": False,
        "tf_legacy_loss": False,
        "pruned_heads": {},
        "tie_word_embeddings": True,
        "chunk_size_feed_forward": 0,
        "is_encoder_decoder": False,
        "is_decoder": False,
        "cross_attention_hidden_size": None,
        "add_cross_attention": False,
        "tie_encoder_decoder": False,
        "max_length": 20,
        "min_length": 0,
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "num_return_sequences": 1,
        "output_scores": False,
        "return_dict_in_generate": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "architectures": None,
        "finetuning_task": None,
        "id2label": {0: "LABEL_0", 1: "LABEL_1"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "tokenizer_class": None,
        "prefix": None,
        "bos_token_id": None,
        "pad_token_id": None,
        "eos_token_id": None,
        "sep_token_id": None,
        "decoder_start_token_id": None,
        "task_specific_params": None,
        "problem_type": None,
        "_name_or_path": "",
        "_attn_implementation_autoset": False,
        "model_type": "siglip_vision_model",
        "num_image_tokens": 1024,
        "projection_dim": 2048,
        "projector_hidden_act": "gelu_fast",
        "vision_use_head": False,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_channels": 3,
        "patch_size": 14,
        "image_size": 448,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-06,
        "hidden_act": "gelu_pytorch_tanh",
    },
    "is_encoder_decoder": False,
    "text_config": {
        "vocab_size": 257216,
        "max_position_embeddings": 8192,
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "head_dim": 256,
        "num_key_value_heads": 1,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": None,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-06,
        "use_cache": True,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
        "torchscript": False,
        "torch_dtype": "float32",
        "use_bfloat16": False,
        "tf_legacy_loss": False,
        "pruned_heads": {},
        "tie_word_embeddings": True,
        "chunk_size_feed_forward": 0,
        "is_encoder_decoder": False,
        "is_decoder": False,
        "cross_attention_hidden_size": None,
        "add_cross_attention": False,
        "tie_encoder_decoder": False,
        "max_length": 20,
        "min_length": 0,
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "num_return_sequences": 1,
        "output_scores": False,
        "return_dict_in_generate": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "architectures": None,
        "finetuning_task": None,
        "id2label": {0: "LABEL_0", 1: "LABEL_1"},
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "tokenizer_class": None,
        "prefix": None,
        "bos_token_id": 2,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "sep_token_id": None,
        "decoder_start_token_id": None,
        "task_specific_params": None,
        "problem_type": None,
        "_name_or_path": "",
        "_attn_implementation_autoset": False,
        "model_type": "gemma",
        "num_image_tokens": 1024,
    },
    "return_dict": True,
    "output_hidden_states": False,
    "output_attentions": False,
    "torchscript": False,
    "torch_dtype": "bfloat16",
    "use_bfloat16": False,
    "tf_legacy_loss": False,
    "pruned_heads": {},
    "tie_word_embeddings": True,
    "chunk_size_feed_forward": 0,
    "is_decoder": False,
    "cross_attention_hidden_size": None,
    "add_cross_attention": False,
    "tie_encoder_decoder": False,
    "max_length": 20,
    "min_length": 0,
    "do_sample": False,
    "early_stopping": False,
    "num_beams": 1,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "num_return_sequences": 1,
    "output_scores": False,
    "return_dict_in_generate": False,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "remove_invalid_values": False,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "architectures": ["ColPali"],
    "finetuning_task": None,
    "id2label": {0: "LABEL_0", 1: "LABEL_1"},
    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
    "tokenizer_class": None,
    "prefix": None,
    "bos_token_id": 2,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "sep_token_id": None,
    "decoder_start_token_id": None,
    "task_specific_params": None,
    "problem_type": None,
    "_name_or_path": "vidore/colpali-v1.2-merged",
    "_attn_implementation_autoset": True,
    "transformers_version": "4.47.0.dev0",
    "model_type": "paligemma",
}

TEST_IMAGES = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
TEST_QUERIES = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

ORIGINAL_IMAGE_OUTPUTS_SLICE = {
    "slice": (slice(None), slice(3), slice(3)),
    "value": torch.tensor(
        [
            [
                [-0.0610, 0.0850, 0.1943],
                [-0.0520, 0.0859, 0.1250],
                [-0.0874, 0.0703, 0.1895],
            ],
            [
                [0.0432, 0.0211, 0.0669],
                [0.0461, 0.0142, 0.1416],
                [-0.0742, 0.1035, 0.1670],
            ],
        ],
        dtype=ORIGINAL_DTYPE,
    ),
}
ORIGINAL_QUERY_OUTPUTS_SLICE = {
    "slice": (slice(None), slice(3), slice(3)),
    "value": torch.tensor(
        [
            [
                [0.1621, -0.0206, 0.0972],
                [-0.1074, -0.1162, 0.0281],
                [-0.0459, -0.1123, -0.0559],
            ],
            [
                [0.1650, -0.0198, 0.0967],
                [-0.0923, -0.1118, 0.0640],
                [-0.1299, -0.0640, 0.1172],
            ],
        ],
        dtype=ORIGINAL_DTYPE,
    ),
}


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


def rename_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("custom_text_proj"):
            new_key = key.replace("custom_text_proj", "embedding_proj_layer")
        new_state_dict[new_key] = value
    return new_state_dict


@torch.no_grad()
def convert_colpali_weights_to_hf(output_dir: str, push_to_hub: bool):
    # Get the device
    device = get_torch_device("auto")
    print(f"Device: {device}")

    # Load the original model's state_dict
    original_state_dict: Dict[str, torch.Tensor] = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/vidore/colpali-v1.2-merged-state_dict/resolve/main/colpali_v1_2_merged_state_dict.pth",
        map_location="cpu",
    )

    # Format the state_dict keys
    original_state_dict = rename_state_dict_keys(original_state_dict)

    # Add the extra attributes for the new model
    new_config = {
        "vlm_backbone_config": ORIGINAL_CONFIG.copy(),
        "model_type": "colpali",
        "is_composition": False,
        "embedding_dim": 128,
        "initializer_range": 0.02,  # unused as initialized weights will be replaced
    }

    # Create the new config
    config = cast(ColPaliConfig, ColPaliConfig.from_dict(new_config))

    # Load the untrained model
    model = ColPaliForRetrieval(config=config).to(device).eval()
    print("Created model with new config and randomly initialized weights")

    # NOTE: The model was initialized with float32 weights. We need to convert it to the desired precision.
    # There are two ways to set the model's dtype:
    # - Using `model.from_pretrained(..., torch_dtype=dtype_precision)` doesn't convert the hyperparameters to the desired precision.
    # - Using `model.to(dtype_precision)` converts all values - including the hyperparameters - to the desired precision.
    # The following snippet allows a fine-grained control over the model's dtype, making sure that all
    # the new weights' dtypes match the original model.
    for param in model.parameters():
        param.data = param.data.to(ORIGINAL_DTYPE)
    print(f"Converted the new model weights to `{ORIGINAL_DTYPE}`")

    # Load the original weights
    model.load_state_dict(original_state_dict)
    print("Loaded original model weights")

    # Tie the weights (following ColPali's `__init__`` step)
    if model.model.language_model._tied_weights_keys is not None:
        model._tied_weights_keys = [f"model.language_model.{k}" for k in model.model.language_model._tied_weights_keys]

    # Sanity check: ensure all keys are the same
    state_dict_keys_old = set(original_state_dict.keys())
    state_dict_keys_new = set(model.state_dict().keys())
    disjoint_keys = state_dict_keys_old.symmetric_difference(state_dict_keys_new)
    if disjoint_keys:
        raise ValueError(f"Incompatible keys: {disjoint_keys}")

    # Sanity checks: forward pass with images and queries
    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained("vidore/colpali-v1.2-merged"))

    batch_images = processor.process_images(images=TEST_IMAGES).to(device)
    batch_queries = processor.process_queries(text=TEST_QUERIES).to(device)

    # Predict with the new model
    with torch.no_grad():
        outputs_images_new = model(**batch_images, return_dict=True).embeddings
        outputs_queries_new = model(**batch_queries, return_dict=True).embeddings

    # Compare the outputs with the original model
    mae_images = torch.mean(
        torch.abs(
            outputs_images_new[ORIGINAL_IMAGE_OUTPUTS_SLICE["slice"]]
            - ORIGINAL_IMAGE_OUTPUTS_SLICE["value"].to(outputs_images_new.device).to(ORIGINAL_DTYPE)
        )
    )
    mae_queries = torch.mean(
        torch.abs(
            outputs_queries_new[ORIGINAL_QUERY_OUTPUTS_SLICE["slice"]]
            - ORIGINAL_QUERY_OUTPUTS_SLICE["value"].to(outputs_queries_new.device).to(ORIGINAL_DTYPE)
        )
    )

    # Sanity checks
    print(f"Mean Absolute Error (MAE) for images: {mae_images}")
    print(f"Mean Absolute Error (MAE) for queries: {mae_queries}")  # FIXME: MAE ≈ 0.0017
    if mae_images > TOLERANCE or mae_queries > TOLERANCE:
        raise ValueError("Mean Absolute Error (MAE) is greater than the tolerance")

    if not torch.allclose(
        outputs_images_new[ORIGINAL_IMAGE_OUTPUTS_SLICE["slice"]],
        ORIGINAL_IMAGE_OUTPUTS_SLICE["value"].to(outputs_images_new.device).to(ORIGINAL_DTYPE),
        rtol=TOLERANCE,
    ):
        raise ValueError("Outputs for images do not match the original model's outputs")
    if not torch.allclose(
        outputs_queries_new[ORIGINAL_QUERY_OUTPUTS_SLICE["slice"]],
        ORIGINAL_QUERY_OUTPUTS_SLICE["value"].to(outputs_queries_new.device).to(ORIGINAL_DTYPE),
        rtol=TOLERANCE,
    ):
        raise ValueError("Outputs for queries do not match the original model's outputs")

    breakpoint()

    # Save the model
    if push_to_hub:
        model.push_to_hub(output_dir, private=True)
        print(f"Model pushed to the hub at `{output_dir}`")
    else:
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model.save_pretrained(output_dir)
        print(f"Model saved to `{output_dir}`")


CLI_HELP = """
This script converts the original ColPali model to the HF model format.\n

Example usage: "python src/transformers/models/colpali/convert_colpali_weights_to_hf.py --output_dir vidore/colpali-v1.2-hf --push_to_hub".
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_HELP)
    parser.add_argument(
        "--output_dir",
        default="vidore/colpali-v1.2-hf",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    convert_colpali_weights_to_hf(output_dir=args.output_dir, push_to_hub=args.push_to_hub)
