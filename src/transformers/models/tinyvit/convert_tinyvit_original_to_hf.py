import argparse
import json

import requests
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import AutoImageProcessor, TinyVitConfig, TinyVitForImageClassification
from torchvision.transforms import Normalize, Resize, CenterCrop, Compose, ToTensor, InterpolationMode
import torch

def get_tinyvit_config(model_name):
    config = TinyVitConfig()

    if model_name == "tinyvit-21m-224":
        hidden_sizes = [96, 192, 384, 576]
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 18]
        window_sizes = [7, 7, 14, 7]
    else:
        raise NotImplementedError("To do")

    # if "in22k" in tinyvit_name:
    #     num_classes = 21841
    # else:
    #     num_classes = 1000
    #     repo_id = "huggingface/label-files"
    #     filename = "imagenet-1k-id2label.json"
    #     id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    #     id2label = {int(k): v for k, v in id2label.items()}
    #     config.id2label = id2label
    #     config.label2id = {v: k for k, v in id2label.items()}

    config.num_labels = 1000
    config.hidden_sizes = hidden_sizes
    config.depths = depths
    config.num_heads = num_heads
    config.window_sizes = window_sizes

    return config



def convert_tinyvit_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_tinyvit_config(model_name)
    model = TinyVitForImageClassification(config)

    name_to_checkpoint_path = {
        "tinyvit-21m-224": "/Users/nielsrogge/Documents/TinyViT/model.pth",
    }

    checkpoint_path = name_to_checkpoint_path[model_name]
    state_dict = torch.load(checkpoint_path)
    # new_state_dict = convert_state_dict(state_dict)

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "patch_embed" in key:
            key = key.replace("patch_embed", "tinyvit.embeddings")
        if "layers" in key:
            key = key.replace("layers", "tinyvit.encoder.stages")
        if "blocks" in key:
            key = key.replace("blocks", "layers")
        if "norm_head" in key:
            key = key.replace("norm_head", "tinyvit.layernorm")
        if "head" in key:
            key = key.replace("head", "classifier")
        state_dict[key] = val

    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open("/Users/nielsrogge/Documents/python_projecten/transformers/tests/fixtures/tests_samples/COCO/000000039769.png").convert("RGB")

    transforms = Compose([
        Resize(size=256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size=224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    pixel_values = transforms(image).unsqueeze(0)

    print("Loading batch...")
    pixel_values = torch.load("/Users/nielsrogge/Documents/TinyViT/batch.pth")
    print("Shape of pixel_values:", pixel_values.shape)

    # TODO assert values
    # image_processor = AutoImageProcessor.from_pretrained("microsoft/{}".format(tinyvit_name.replace("_", "-")))
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = image_processor(images=image, return_tensors="pt")

    logits = model(pixel_values).logits
    print("First values of logits:", logits[0, :3])
    print("Shape of logits:", logits.shape)
    print(logits.argmax(-1).item())
    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        raise NotImplementedError("To do")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="tinyvit-21m-224",
        type=str,
        choices=["tinyvit-21m-224", "tinyvit-21m-384", "tinyvit-21m-512"],
        help="Name of the model.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_tinyvit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
