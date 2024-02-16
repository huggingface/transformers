import argparse

import requests
import torch
from PIL import Image
# from .configuration_hiera import HieraConfig
# from .hiera import Hiera
# from transformers import HieraConfig, Hiera
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def rename_key(name):
    # if "patch_embed.proj" in name:
    #     name = name.replace("patch_embed.proj", "patch_embed.projection")
    # # elif "block.proj" in name:
    # #     name = name.replace("block.proj", "block.projection")
    # elif "attn.proj" in name:
    #     name = name.replace("attn.proj", "attn.projection")
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



class HieraImageProcessor:
    def __init__(self, size):
        self.size = size
        self.transform_list = [
            transforms.Resize(int((256 / 224) * self.size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.size)
        ]
        self.transform_vis = transforms.Compose(self.transform_list)
        self.transform_norm = transforms.Compose(self.transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    
    def process_image(self, image_url):
        # Load the image
        img = Image.open(requests.get(image_url, stream=True).raw)
        
        # Apply transformations
        img_vis = self.transform_vis(img)
        img_norm = self.transform_norm(img)
        
        return img_norm



def convert_Hiera_checkpoint( checkpoint_url, pytorch_dump_folder_path):
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
        }
    }


    if "hiera_tiny_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=96, 
                            number_of_heads=1, 
                            stages=(1, 2, 7, 2),)
        checkpoints = pretrained_models_links["hiera_tiny_224"]
        checkpoint = pretrained_models_links["hiera_tiny_224"]["mae_in1k_ft_in1k"]

    elif "hiera_small_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=96, 
                            number_of_heads=1, 
                            stages=(1, 2, 11, 2),)
        checkpoints = pretrained_models_links["hiera_small_224"]
        checkpoint = pretrained_models_links["hiera_small_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=96, 
                            number_of_heads=1, 
                            stages=(2, 3, 16, 3),)
        checkpoints = pretrained_models_links["hiera_base_224"]
        checkpoint = pretrained_models_links["hiera_base_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_plus_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=112, 
                            number_of_heads=2, 
                            stages=(2, 3, 16, 3),)
        checkpoints = pretrained_models_links["hiera_base_plus_224"]
        checkpoint = pretrained_models_links["hiera_base_plus_224"]["mae_in1k_ft_in1k"]

    elif "hiera_large_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=144, 
                            number_of_heads=2, 
                            stages=(2, 6, 36, 4),)
        checkpoints = pretrained_models_links["hiera_large_224"]
        checkpoint = pretrained_models_links["hiera_large_224"]["mae_in1k_ft_in1k"]

    elif "hiera_huge_224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=256, 
                            number_of_heads=4, 
                            stages=(2, 6, 36, 4))
        checkpoints = pretrained_models_links["hiera_huge_224"]
        checkpoint = pretrained_models_links["hiera_huge_224"]["mae_in1k_ft_in1k"]

    elif "hiera_base_16x224" in checkpoint_url:
        config = HieraConfig(num_classes=num_classes,  # Assuming num_classes is defined elsewhere
                            input_size=(16, 224, 224),
                            q_stride=(1, 2, 2),
                            mask_unit_size=(1, 8, 8),
                            patch_kernel=(3, 7, 7),
                            patch_stride=(2, 4, 4),
                            patch_padding=(1, 3, 3),
                            sep_position_embeddings=True,)
        checkpoints = pretrained_models_links["hiera_base_16x224"]
        checkpoint = pretrained_models_links["hiera_base_16x224"]["mae_k400_ft_k400"]

    elif "hiera_base_plus_16x224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=112, 
                            number_of_heads=2, 
                            stages=(2, 3, 16, 3))
        checkpoints = pretrained_models_links["hiera_base_plus_16x224"]
        checkpoint = pretrained_models_links["hiera_base_plus_16x224"]["mae_k400_ft_k400"]

    elif "hiera_large_16x224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=144, 
                            number_of_heads=2, 
                            stages=(2, 6, 36, 4), )
        checkpoints = pretrained_models_links["hiera_large_16x224"]
        checkpoint = pretrained_models_links["hiera_large_16x224"]["mae_k400_ft_k400"]

    elif "hiera_huge_16x224" in checkpoint_url:
        config = HieraConfig(embedding_dimension=256, 
                            number_of_heads=4, 
                            stages=(2, 6, 36, 4) )
        checkpoints = pretrained_models_links["hiera_huge_16x224"]
        checkpoint = pretrained_models_links["hiera_huge_16x224"]["mae_k400_ft_k400"]


    pretrained = True
    if pretrained:
        if checkpoints is None:
            raise RuntimeError("This model currently doesn't have pretrained weights available.")
        elif checkpoint is None:
            raise RuntimeError("No checkpoint specified.")
        elif checkpoint not in checkpoints:
            raise RuntimeError(f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}.")

        state_dict = torch.hub.load_state_dict_from_url(checkpoints[checkpoint], map_location="cpu")
        state_dict["model_state"] = convert_state_dict(state_dict["model_state"],{})
        if "head.projection.weight" in state_dict["model_state"]:
            # Set the number of classes equal to the state_dict only if the user doesn't want to overwrite it
            if config.num_classes is None:
                config.num_classes = state_dict["model_state"]["head.projection.weight"].shape[0]
            # If the user specified a different number of classes, remove the projection weights or else we'll error out
            elif config.num_classes != state_dict["model_state"]["head.projection.weight"].shape[0]:
                del state_dict["model_state"]["head.projection.weight"]
                del state_dict["model_state"]["head.projection.bias"]

    model = Hiera(config)
    if pretrained:
        # Disable being strict when trying to load a encoder-decoder model into an encoder-only model
        if "decoder_position_embeddings" in state_dict["model_state"] and not hasattr(model, "decoder_position_embeddings"):
            strict = False

        model.load_state_dict(state_dict["model_state"], strict=strict)
    



    url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"

    image = Image.open(requests.get(url, stream=True).raw)

    
    image_processor = HieraImageProcessor(size=config.image_size)
    inputs = image_processor.process_image(images=image, return_tensors="pt")

    # forward pass
    out = model(inputs[None, ...])

    # 207: golden retriever  (imagenet-1k)
    out.argmax(dim=-1).item()


    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    checkpoint_url = "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth"
    convert_Hiera_checkpoint(checkpoint_url, pytorch_dump_folder_path="~/")

