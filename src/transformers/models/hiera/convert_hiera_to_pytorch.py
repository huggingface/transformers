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

import argparse
from PIL import Image

import torch

# from transformers import HieraConfig, HieraModel
from transformers import HieraConfig, HieraModel
from transformers import BeitImageProcessor
from transformers.image_utils import PILImageResampling, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import requests

def rename_key(name):
    if ".proj." in name:
        name = name.replace(".proj.", ".projection.")
    if "attn" in name:
        name = name.replace("attn", "attention")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "patch_embedding")
    return name


def convert_state_dict(orig_state_dict, config):
    updated_model_state = {rename_key(k): v for k, v in orig_state_dict.items()}
    return updated_model_state


def convert_Hiera_checkpoint(checkpoint_url, pytorch_dump_folder_path, **kwargs):
    strict = True
    pretrained_models_links = {
        "hiera_tiny_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
        },
        "hiera_small_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
        },
        "hiera_base_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
        },
        "hiera_base_plus_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
        },
        "hiera_large_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
        },
        "hiera_huge_224": {
            "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
            "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
        },
        "hiera_base_16x224": {
            "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
            "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
        },
        "hiera_base_plus_16x224": {
            "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
            "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
        },
        "hiera_large_16x224": {
            "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
            "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
        },
        "hiera_huge_16x224": {
            "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
            "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
        },
    }

    if "hiera_tiny_224" in checkpoint_url:
        config = HieraConfig(
            embedding_dimension=96,
            number_of_heads=1,
            stages=(1, 2, 7, 2),
        )
        checkpoints = pretrained_models_links["hiera_tiny_224"]
        checkpoint = pretrained_models_links["hiera_tiny_224"]["mae_in1k_ft_in1k"]

    elif "hiera_small_224" in checkpoint_url:
        config = HieraConfig(
            embedding_dimension=96,
            number_of_heads=1,
            stages=(1, 2, 11, 2),
        )
        checkpoints = pretrained_models_links["hiera_small_224"]
        checkpoint = pretrained_models_links["hiera_small_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=96, number_of_heads=1, stages=(2, 3, 16, 3))

        checkpoints = pretrained_models_links["hiera_base_224"]
        checkpoint = pretrained_models_links["hiera_base_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_plus_224" in checkpoint_url:
        config = HieraConfig(
            embedding_dimension=112,
            number_of_heads=2,
            stages=(2, 3, 16, 3),
        )
        checkpoints = pretrained_models_links["hiera_base_plus_224"]
        checkpoint = pretrained_models_links["hiera_base_plus_224"]["mae_in1k_ft_in1k"]

    elif "hiera_large_224" in checkpoint_url:
        config = HieraConfig(
            embedding_dimension=144,
            number_of_heads=2,
            stages=(2, 6, 36, 4),
        )
        checkpoints = pretrained_models_links["hiera_large_224"]
        checkpoint = pretrained_models_links["hiera_large_224"]["mae_in1k_ft_in1k"]

    elif "hiera_huge_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4))
        checkpoints = pretrained_models_links["hiera_huge_224"]
        checkpoint = pretrained_models_links["hiera_huge_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_16x224" in checkpoint_url:
        config = HieraConfig(
            input_size=(16, 224, 224),
            q_stride=(1, 2, 2),
            mask_unit_size=(1, 8, 8),
            patch_kernel=(3, 7, 7),
            patch_stride=(2, 4, 4),
            patch_padding=(1, 3, 3),
            sep_position_embeddings=True,
        )
        checkpoints = pretrained_models_links["hiera_base_16x224"]
        checkpoint = pretrained_models_links["hiera_base_16x224"]["mae_k400_ft_k400"]

    elif "hiera_base_plus_16x224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=112, number_of_heads=2, stages=(2, 3, 16, 3))
        checkpoints = pretrained_models_links["hiera_base_plus_16x224"]
        checkpoint = pretrained_models_links["hiera_base_plus_16x224"]["mae_k400_ft_k400"]

    elif "hiera_large_16x224" in checkpoint_url:
        config = HieraConfig(
            embedding_dimension=144,
            number_of_heads=2,
            stages=(2, 6, 36, 4),
        )
        checkpoints = pretrained_models_links["hiera_large_16x224"]
        checkpoint = pretrained_models_links["hiera_large_16x224"]["mae_k400_ft_k400"]

    elif "hiera_huge_16x224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4))
        checkpoints = pretrained_models_links["hiera_huge_16x224"]
        checkpoint = pretrained_models_links["hiera_huge_16x224"]["mae_k400_ft_k400"]
    elif checkpoint not in checkpoints:
        raise RuntimeError(f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}.")

    pretrained = True
    if pretrained:
        if checkpoints is None:
            raise RuntimeError("This model currently doesn't have pretrained weights available.")
        elif checkpoint is None:
            raise RuntimeError("No checkpoint specified.")

        state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
        state_dict["model_state"] = convert_state_dict(state_dict["model_state"], {})
        if "head.projection.weight" in state_dict["model_state"]:
            # Set the number of classes equal to the state_dict only if the user doesn't want to overwrite it
            if config.num_classes is None:
                config.num_classes = state_dict["model_state"]["head.projection.weight"].shape[0]
            # If the user specified a different number of classes, remove the projection weights or else we'll error out
            elif config.num_classes != state_dict["model_state"]["head.projection.weight"].shape[0]:
                del state_dict["model_state"]["head.projection.weight"]
                del state_dict["model_state"]["head.projection.bias"]

    model = HieraModel(config=config)
    if pretrained:
        # Disable being strict when trying to load a encoder-decoder model into an encoder-only model
        if "decoder_position_embeddings" in state_dict["model_state"] and not hasattr(
            model, "decoder_position_embeddings"
        ):
            strict = False

        model.load_state_dict(state_dict["model_state"], strict=strict)


    image_processor = BeitImageProcessor(
                                        size = {"height":256,"width":256},
                                        do_rescale=True,
                                        do_center_crop=True,
                                        crop_size = {"height":224,"width":224},
                                        do_normalize=True,
                                        do_reduce_labels=False,
                                        do_resize=True,
                                        image_std=IMAGENET_DEFAULT_STD,
                                        image_mean=IMAGENET_DEFAULT_MEAN,
                                        resample = PILImageResampling.BICUBIC)  
    
    
    url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processed_image = image_processor(images=image, return_tensors="pt")  
    model.load_state_dict(state_dict["model_state"], strict=strict)
    expected_slice = torch.tensor(
    [ 0.1825,  0.8655,  0.5779,  1.1550,  1.1025,  0.6381,  1.0288, -0.0624, 0.1455]
        )
    # If you also want intermediate feature maps
    out = model(processed_image.pixel_values)
    out.last_hidden_state.argmax(dim=-1).item()
    assert torch.allclose(out.last_hidden_state[0, :9], expected_slice, atol=1e-4)


    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path, safe_serialization=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    checkpoint_url = "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth"
    convert_Hiera_checkpoint(checkpoint_url, pytorch_dump_folder_path="/home/ubuntu/home/hiera/hiera_base_224")
