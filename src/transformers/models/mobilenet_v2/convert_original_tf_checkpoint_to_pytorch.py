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
"""Convert MobileNetV2 checkpoints from the tensorflow/models library."""

import argparse
import json
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    MobileNetV2Config,
    MobileNetV2ForImageClassification,
    MobileNetV2ForSemanticSegmentation,
    MobileNetV2ImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """

    tf_to_pt_map = {}

    if isinstance(model, (MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation)):
        backbone = model.mobilenet_v2
    else:
        backbone = model

    # Use the EMA weights if available
    def ema(x):
        return x + "/ExponentialMovingAverage" if x + "/ExponentialMovingAverage" in tf_weights else x

    prefix = "MobilenetV2/Conv/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.first_conv.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.first_conv.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.first_conv.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.first_conv.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.first_conv.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/depthwise/"
    tf_to_pt_map[ema(prefix + "depthwise_weights")] = backbone.conv_stem.conv_3x3.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.conv_3x3.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.conv_3x3.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.conv_3x3.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.conv_3x3.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/project/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.reduce_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.reduce_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.reduce_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.reduce_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.reduce_1x1.normalization.running_var

    for i in range(16):
        tf_index = i + 1
        pt_index = i
        pointer = backbone.layer[pt_index]

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/expand/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.expand_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.expand_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.expand_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.expand_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.expand_1x1.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/depthwise/"
        tf_to_pt_map[ema(prefix + "depthwise_weights")] = pointer.conv_3x3.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.conv_3x3.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.conv_3x3.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.conv_3x3.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.conv_3x3.normalization.running_var

        prefix = f"MobilenetV2/expanded_conv_{tf_index}/project/"
        tf_to_pt_map[ema(prefix + "weights")] = pointer.reduce_1x1.convolution.weight
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.reduce_1x1.normalization.bias
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.reduce_1x1.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.reduce_1x1.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.reduce_1x1.normalization.running_var

    prefix = "MobilenetV2/Conv_1/"
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_1x1.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_1x1.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_1x1.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_1x1.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_1x1.normalization.running_var

    if isinstance(model, MobileNetV2ForImageClassification):
        prefix = "MobilenetV2/Logits/Conv2d_1c_1x1/"
        tf_to_pt_map[ema(prefix + "weights")] = model.classifier.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.classifier.bias

    if isinstance(model, MobileNetV2ForSemanticSegmentation):
        prefix = "image_pooling/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_pool.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_pool.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_pool.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_pool.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_pool.normalization.running_var
        )

        prefix = "aspp0/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_aspp.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_aspp.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_aspp.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_aspp.normalization.running_mean
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_aspp.normalization.running_var
        )

        prefix = "concat_projection/"
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_projection.convolution.weight
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_projection.normalization.bias
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_projection.normalization.weight
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = (
            model.segmentation_head.conv_projection.normalization.running_mean
        )
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = (
            model.segmentation_head.conv_projection.normalization.running_var
        )

        prefix = "logits/semantic/"
        tf_to_pt_map[ema(prefix + "weights")] = model.segmentation_head.classifier.convolution.weight
        tf_to_pt_map[ema(prefix + "biases")] = model.segmentation_head.classifier.convolution.bias

    return tf_to_pt_map


def load_tf_weights_in_mobilenet_v2(model, config, tf_checkpoint_path):
    """Load TensorFlow checkpoints in a PyTorch model."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # Build TF to PyTorch weights loading map
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        array = tf_weights[name]

        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        elif "weights" in name:
            logger.info("Transposing")
            if len(pointer.shape) == 2:  # copying into linear layer
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        pointer.data = torch.from_numpy(array)

        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)
        tf_weights.pop(name + "/Momentum", None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


def get_mobilenet_v2_config(model_name):
    config = MobileNetV2Config(layer_norm_eps=0.001)

    if "quant" in model_name:
        raise ValueError("Quantized models are not supported.")

    matches = re.match(r"^.*mobilenet_v2_([^_]*)_([^_]*)$", model_name)
    if matches:
        config.depth_multiplier = float(matches[1])
        config.image_size = int(matches[2])

    if model_name.startswith("deeplabv3_"):
        config.output_stride = 8
        config.num_labels = 21
        filename = "pascal-voc-id2label.json"
    else:
        # The TensorFlow version of MobileNetV2 predicts 1001 classes instead
        # of the usual 1000. The first class (index 0) is "background".
        config.num_labels = 1001
        filename = "imagenet-1k-id2label.json"

    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))

    if config.num_labels == 1001:
        id2label = {int(k) + 1: v for k, v in id2label.items()}
        id2label[0] = "background"
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileNetV2 structure.
    """
    config = get_mobilenet_v2_config(model_name)

    # Load 🤗 model
    if model_name.startswith("deeplabv3_"):
        model = MobileNetV2ForSemanticSegmentation(config).eval()
    else:
        model = MobileNetV2ForImageClassification(config).eval()

    # Load weights from TensorFlow checkpoint
    load_tf_weights_in_mobilenet_v2(model, config, checkpoint_path)

    # Check outputs on an image, prepared by MobileNetV2ImageProcessor
    image_processor = MobileNetV2ImageProcessor(
        crop_size={"width": config.image_size, "height": config.image_size},
        size={"shortest_edge": config.image_size + 32},
    )
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits

    if model_name.startswith("deeplabv3_"):
        assert logits.shape == (1, 21, 65, 65)

        if model_name == "deeplabv3_mobilenet_v2_1.0_513":
            expected_logits = torch.tensor(
                [
                    [[17.5790, 17.7581, 18.3355], [18.3257, 18.4230, 18.8973], [18.6169, 18.8650, 19.2187]],
                    [[-2.1595, -2.0977, -2.3741], [-2.4226, -2.3028, -2.6835], [-2.7819, -2.5991, -2.7706]],
                    [[4.2058, 4.8317, 4.7638], [4.4136, 5.0361, 4.9383], [4.5028, 4.9644, 4.8734]],
                ]
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        assert logits.shape == (1, 1001)

        if model_name == "mobilenet_v2_1.4_224":
            expected_logits = torch.tensor([0.0181, -1.0015, 0.4688])
        elif model_name == "mobilenet_v2_1.0_224":
            expected_logits = torch.tensor([0.2445, -1.1993, 0.1905])
        elif model_name == "mobilenet_v2_0.75_160":
            expected_logits = torch.tensor([0.2482, 0.4136, 0.6669])
        elif model_name == "mobilenet_v2_0.35_96":
            expected_logits = torch.tensor([0.1451, -0.4624, 0.7192])
        else:
            expected_logits = None

        if expected_logits is not None:
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        repo_id = "google/" + model_name
        image_processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="mobilenet_v2_1.0_224",
        type=str,
        help="Name of the MobileNetV2 model you'd like to convert. Should in the form 'mobilenet_v2_<depth>_<size>'.",
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )

    args = parser.parse_args()
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
