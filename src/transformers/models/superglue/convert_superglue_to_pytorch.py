import argparse

import torch

from transformers import SuperGlueConfig, SuperGlueModel


def get_superglue_config(checkpoint_url):
    config = SuperGlueConfig(
        descriptor_dim=256,
        keypoint_encoder_sizes=[32, 64, 128, 256],
        gnn_layers_types=['self', 'cross'] * 9,
        sinkhorn_iterations=100,
        matching_threshold=0.2,
    )

    if "superglue_indoor" in checkpoint_url:
        config.model_version = "indoor"
    elif "superglue_outdoor" in checkpoint_url:
        config.model_version = "outdoor"

    return config


def create_rename_keys(config, state_dict):
    rename_keys = []

    # keypoint encoder
    n = len([3] + config.keypoint_encoder_sizes + [config.descriptor_dim])
    for i in range(n * 2 + 1):
        if ((i + 1) % 3) != 0:
            rename_keys.append((f"kenc.encoder.{i}.weight", f"keypoint_encoder.encoder.layers.{i}.weight"))
            rename_keys.append((f"kenc.encoder.{i}.bias", f"keypoint_encoder.encoder.layers.{i}.bias"))
            if ((i % 3) - 1) == 0:
                rename_keys.append((f"kenc.encoder.{i}.running_mean",
                                    f"keypoint_encoder.encoder.layers.{i}.running_mean"))
                rename_keys.append((f"kenc.encoder.{i}.running_var",
                                    f"keypoint_encoder.encoder.layers.{i}.running_var"))
                rename_keys.append((f"kenc.encoder.{i}.num_batches_tracked",
                                    f"keypoint_encoder.encoder.layers.{i}.num_batches_tracked"))

    # gnn
    for i in range(len(config.gnn_layers_types)):
        rename_keys.append((f"gnn.layers.{i}.attn.merge.weight", f"gnn.layers.{i}.attention.merge.weight"))
        rename_keys.append((f"gnn.layers.{i}.attn.merge.bias", f"gnn.layers.{i}.attention.merge.bias"))
        for j in range(3):
            rename_keys.append((f"gnn.layers.{i}.attn.proj.{j}.weight", f"gnn.layers.{i}.attention.proj.{j}.weight"))
            rename_keys.append((f"gnn.layers.{i}.attn.proj.{j}.bias", f"gnn.layers.{i}.attention.proj.{j}.bias"))
        for j in range(len([config.descriptor_dim * 2, config.descriptor_dim * 2, config.descriptor_dim]) + 1):
            if j != 2 :
                rename_keys.append((f"gnn.layers.{i}.mlp.{j}.weight", f"gnn.layers.{i}.mlp.layers.{j}.weight"))
                rename_keys.append((f"gnn.layers.{i}.mlp.{j}.bias", f"gnn.layers.{i}.mlp.layers.{j}.bias"))
                if j == 1:
                    rename_keys.append((f"gnn.layers.{i}.mlp.{j}.running_mean",
                                        f"gnn.layers.{i}.mlp.layers.{j}.running_mean"))
                    rename_keys.append((f"gnn.layers.{i}.mlp.{j}.running_var",
                                        f"gnn.layers.{i}.mlp.layers.{j}.running_var"))
                    rename_keys.append((f"gnn.layers.{i}.mlp.{j}.num_batches_tracked",
                                        f"gnn.layers.{i}.mlp.layers.{j}.num_batches_tracked"))
    return rename_keys


# Copied from transformers.models.dinov2.convert_dinov2_to_hf
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_superglue_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    TODO docs
    """

    print("Downloading original model from checkpoint...")
    config = get_superglue_config(checkpoint_url)

    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    print(original_state_dict)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(config, original_state_dict)
    new_state_dict = original_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    for key in new_state_dict.copy().keys():
        val = new_state_dict.pop(key)
        if not key.startswith("superglue"):
            key = "superglue." + key
        new_state_dict[key] = val

    model = SuperGlueModel(config)
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Successfully loaded weights in the model")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth",
        type=str,
        help="URL of the original SuperGlue checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_superglue_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
