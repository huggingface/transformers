# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert VitPose checkpoints from the original repository.

URL: https://github.com/vitae-transformer/vitpose

Notebook to get the original logits: https://colab.research.google.com/drive/1QDX_2POTpl6JaZAV2WIFjuiqDsDwiqMZ?usp=sharing.
"""

import argparse
import os
import re

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"patch_embed.proj": "embeddings.patch_embeddings.projection",
    r"pos_embed": "embeddings.position_embeddings",
    r"blocks": "encoder.layer",
    r"attn.proj": "attention.output.dense",
    r"attn": "attention.self",
    r"norm1": "layernorm_before",
    r"norm2": "layernorm_after",
    r"last_norm": "layernorm",
    r"keypoint_head": "head",
    r"final_layer": "conv",
}

MODEL_TO_FILE_NAME_MAPPING = {
    # VitPose models, simple decoder
    "vitpose-base-simple": "vitpose-b-simple.pth",
    # VitPose models, classic decoder
    "vitpose-base": "vitpose-b.pth",
    # VitPose models, COCO-AIC-MPII
    "vitpose-base-coco-aic-mpii": "vitpose_base_coco_aic_mpii.pth",
    # VitPose+ models
    "vitpose-plus-small": "vitpose+_small.pth",
    "vitpose-plus-base": "vitpose+_base.pth",
    "vitpose-plus-large": "vitpose+_large.pth",
    "vitpose-plus-huge": "vitpose+_huge.pth",
}


def get_config(model_name):
    if "plus" in model_name:
        num_experts = 6
        if "small" in model_name:
            part_features = 96
            out_indices = [12]
        elif "base" in model_name:
            part_features = 192
            out_indices = [12]
        elif "large" in model_name:
            part_features = 256
            out_indices = [24]
        elif "huge" in model_name:
            part_features = 320
            out_indices = [32]
        else:
            raise ValueError(f"Model {model_name} not supported")
    else:
        num_experts = 1
        part_features = 0

    # size of the architecture
    if "small" in model_name:
        hidden_size = 384
        num_hidden_layers = 12
        num_attention_heads = 12
    elif "large" in model_name:
        hidden_size = 1024
        num_hidden_layers = 24
        num_attention_heads = 16
    elif "huge" in model_name:
        hidden_size = 1280
        num_hidden_layers = 32
        num_attention_heads = 16

    backbone_config = VitPoseBackboneConfig(
        out_indices=out_indices,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_experts=num_experts,
        part_features=part_features,
    )

    use_simple_decoder = "simple" in model_name

    edges = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]
    id2label = {
        0: "Nose",
        1: "L_Eye",
        2: "R_Eye",
        3: "L_Ear",
        4: "R_Ear",
        5: "L_Shoulder",
        6: "R_Shoulder",
        7: "L_Elbow",
        8: "R_Elbow",
        9: "L_Wrist",
        10: "R_Wrist",
        11: "L_Hip",
        12: "R_Hip",
        13: "L_Knee",
        14: "R_Knee",
        15: "L_Ankle",
        16: "R_Ankle",
    }

    label2id = {v: k for k, v in id2label.items()}

    config = VitPoseConfig(
        backbone_config=backbone_config,
        num_labels=17,
        use_simple_decoder=use_simple_decoder,
        edges=edges,
        id2label=id2label,
        label2id=label2id,
    )

    return config


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@torch.no_grad()
def write_model(model_name, model_path, push_to_hub, check_logits=True):
    # ------------------------------------------------------------
    # Vision model params and config
    # ------------------------------------------------------------

    # params from config
    config = get_config(model_name)

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    # load original state_dict
    filename = MODEL_TO_FILE_NAME_MAPPING[model_name]
    print(f"Fetching all parameters from the checkpoint at {filename}...")

    checkpoint_path = hf_hub_download(
        repo_id="nielsr/vitpose-original-checkpoints", filename=filename, repo_type="model"
    )

    print("Converting model...")
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    all_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    dim = config.backbone_config.hidden_size

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        value = original_state_dict[key]

        if re.search("associate_heads", new_key) or re.search("backbone.cls_token", new_key):
            # This associated_heads is concept of auxiliary head so does not require in inference stage.
            # backbone.cls_token is optional forward function for dynamically change of size, see detail in https://github.com/ViTAE-Transformer/ViTPose/issues/34
            pass
        elif re.search("qkv", new_key):
            state_dict[new_key.replace("self.qkv", "attention.query")] = value[:dim]
            state_dict[new_key.replace("self.qkv", "attention.key")] = value[dim : dim * 2]
            state_dict[new_key.replace("self.qkv", "attention.value")] = value[-dim:]
        elif re.search("head", new_key) and not config.use_simple_decoder:
            # Pattern for deconvolution layers
            deconv_pattern = r"deconv_layers\.(0|3)\.weight"
            new_key = re.sub(deconv_pattern, lambda m: f"deconv{int(m.group(1)) // 3 + 1}.weight", new_key)
            # Pattern for batch normalization layers
            bn_patterns = [
                (r"deconv_layers\.(\d+)\.weight", r"batchnorm\1.weight"),
                (r"deconv_layers\.(\d+)\.bias", r"batchnorm\1.bias"),
                (r"deconv_layers\.(\d+)\.running_mean", r"batchnorm\1.running_mean"),
                (r"deconv_layers\.(\d+)\.running_var", r"batchnorm\1.running_var"),
                (r"deconv_layers\.(\d+)\.num_batches_tracked", r"batchnorm\1.num_batches_tracked"),
            ]

            for pattern, replacement in bn_patterns:
                if re.search(pattern, new_key):
                    # Convert the layer number to the correct batch norm index
                    layer_num = int(re.search(pattern, key).group(1))
                    bn_num = layer_num // 3 + 1
                    new_key = re.sub(pattern, replacement.replace(r"\1", str(bn_num)), new_key)
            state_dict[new_key] = value
        else:
            state_dict[new_key] = value

    print("Loading the checkpoint in a Vitpose model.")
    model = VitPoseForPoseEstimation(config)
    model.eval()
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    # create image processor
    image_processor = VitPoseImageProcessor()

    # verify image processor
    image = prepare_img()
    boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]
    pixel_values = image_processor(images=image, boxes=boxes, return_tensors="pt").pixel_values

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="vitpose_batch_data.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")["img"]
    # we allow for a small difference in the pixel values due to the original repository using cv2
    assert torch.allclose(pixel_values, original_pixel_values, atol=1e-1)

    dataset_index = torch.tensor([0])

    with torch.no_grad():
        print("Shape of original_pixel_values: ", original_pixel_values.shape)
        print("First values of original_pixel_values: ", original_pixel_values[0, 0, :3, :3])

        # first forward pass
        outputs = model(original_pixel_values, dataset_index=dataset_index)
        output_heatmap = outputs.heatmaps

        print("Shape of output_heatmap: ", output_heatmap.shape)
        print("First values: ", output_heatmap[0, 0, :3, :3])

        # second forward pass (flipped)
        # this is done since the model uses `flip_test=True` in its test config
        original_pixel_values_flipped = torch.flip(original_pixel_values, [3])
        outputs_flipped = model(
            original_pixel_values_flipped,
            dataset_index=dataset_index,
            flip_pairs=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]),
        )
        output_flipped_heatmap = outputs_flipped.heatmaps

    outputs.heatmaps = (output_heatmap + output_flipped_heatmap) * 0.5

    # Verify pose_results
    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]

    if check_logits:
        # Simple decoder checkpoints
        if model_name == "vitpose-base-simple":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([3.98180511e02, 1.81808380e02]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor([8.66642594e-01]),
                atol=5e-2,
            )
        # Classic decoder checkpoints
        elif model_name == "vitpose-base":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([3.9807913e02, 1.8182812e02]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor([8.8235235e-01]),
                atol=5e-2,
            )
        # COCO-AIC-MPII checkpoints
        elif model_name == "vitpose-base-coco-aic-mpii":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([3.98305542e02, 1.81741592e02]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor([8.69966745e-01]),
                atol=5e-2,
            )
        # VitPose+ models
        elif model_name == "vitpose-plus-small":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([398.1597, 181.6902]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor(0.9051),
                atol=5e-2,
            )
        elif model_name == "vitpose-plus-base":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([3.98201294e02, 1.81728302e02]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor([8.75046968e-01]),
                atol=5e-2,
            )
        elif model_name == "vitpose-plus-large":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([398.1409, 181.7412]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor(0.8746),
                atol=5e-2,
            )
        elif model_name == "vitpose-plus-huge":
            assert torch.allclose(
                pose_results[1]["keypoints"][0],
                torch.tensor([398.2079, 181.8026]),
                atol=5e-2,
            )
            assert torch.allclose(
                pose_results[1]["scores"][0],
                torch.tensor(0.8693),
                atol=5e-2,
            )
        else:
            raise ValueError("Model not supported")
    print("Conversion successfully done.")

    if model_path is not None:
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        image_processor.save_pretrained(model_path)

    if push_to_hub:
        print(f"Pushing model and image processor for {model_name} to hub")
        # we created a community organization on the hub for this model
        # maintained by the Transformers team
        model.push_to_hub(f"usyd-community/{model_name}")
        image_processor.push_to_hub(f"usyd-community/{model_name}")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitpose-base-simple",
        choices=MODEL_TO_FILE_NAME_MAPPING.keys(),
        type=str,
        help="Name of the VitPose model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to store the converted model."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--check_logits", action="store_false", help="Whether or not to verify the logits of the converted model."
    )

    args = parser.parse_args()
    write_model(
        model_path=args.pytorch_dump_folder_path,
        model_name=args.model_name,
        push_to_hub=args.push_to_hub,
        check_logits=args.check_logits,
    )


if __name__ == "__main__":
    main()
