import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from transformers import CLIPSegConfig, CLIPSegForImageSegmentation


def get_clipseg_config():
    config = CLIPSegConfig()
    config.vision_config.patch_size = 16
    return config


def rename_key(name):
    # update prefixes
    if "clip_model" in name:
        name = name.replace("clip_model", "clipseg")
    if "transformer" in name:
        if "visual" in name:
            name = name.replace("visual.transformer", "vision_model")
        else:
            name = name.replace("transformer", "text_model")
    if "resblocks" in name:
        name = name.replace("resblocks", "encoder.layers")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "attn" in name:
        name = name.replace("attn", "self_attn")
    # text encoder
    if "token_embedding" in name:
        name = name.replace("token_embedding", "text_model.embeddings.token_embedding")
    if "positional_embedding" in name and "visual" not in name:
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # vision encoder
    if "visual.class_embedding" in name:
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    if "visual.positional_embedding" in name:
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layrnorm")
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    # projection layers
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if key.startswith("clip_model") and "attn.in_proj" in key:
            key_split = key.split(".")
            if "visual" in key:
                layer_num = int(key_split[4])
                dim = config.vision_config.hidden_size
                prefix = "vision_model"
            else:
                layer_num = int(key_split[3])
                dim = config.text_config.hidden_size
                prefix = "text_model"

            if "weight" in key:
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"clipseg.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            new_name = rename_key(key)
            if "visual_projection" in new_name or "text_projection" in new_name:
                val = val.T
            orig_state_dict[new_name] = val

    return orig_state_dict


image_transforms = Compose(
    [
        ToTensor(),
        Resize((224, 224)),
    ]
)


def convert_clipseg_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    config = get_clipseg_config()
    model = CLIPSegForImageSegmentation(config)
    model.eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    for key in state_dict.copy().keys():
        if key.startswith("model"):
            state_dict.pop(key, None)

    print("ORIGINAL STATE DICT")
    for name, param in state_dict.items():
        print(name, param.shape)

    state_dict = convert_state_dict(state_dict, config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # TODO create feature extractor
    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    image = Image.open("/Users/nielsrogge/Documents/cats.jpg").convert("RGB")
    pixel_values = image_transforms(image).unsqueeze(0).repeat(4, 1, 1, 1)

    # prompts = ["a glass", "something to fill", "wood", "a jar"]
    # tokenizer = CLIPTokenizer.from_pretrained("openai/")
    # input_ids = CLIPTokenizer(prompts, padding="max_length", return_tensors="pt")
    input_ids = torch.tensor([[1, 2] + [9] * 75]).repeat(4, 1)

    print("Shape of pixel values:", pixel_values.shape)
    print("Shape of input ids:", input_ids.shape)

    outputs = model(input_ids, pixel_values)
    print(outputs.keys())

    # assert torch.allclose(outputs, expected_slice, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/CLIPSeg/test.pth",
        type=str,
        help="Path to the original checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_clipseg_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)
