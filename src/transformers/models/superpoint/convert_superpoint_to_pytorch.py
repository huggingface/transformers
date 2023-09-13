import argparse

import torch

from transformers import SuperPointConfig, SuperPointModel


def get_superglue_config():
    config = SuperPointConfig(
        conv_layers_sizes=[64, 64, 128, 128, 256],
        descriptor_dim=256,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        nms_radius=4,
        border_removal_distance=4,
        initializer_range=0.02,
    )

    return config


def create_rename_keys(config, state_dict):
    rename_keys = []

    # Encoder weights
    rename_keys.append((f"conv1a.weight", f"encoder.conv1a.weight"))
    rename_keys.append((f"conv1b.weight", f"encoder.conv1b.weight"))
    rename_keys.append((f"conv2a.weight", f"encoder.conv2a.weight"))
    rename_keys.append((f"conv2b.weight", f"encoder.conv2b.weight"))
    rename_keys.append((f"conv3a.weight", f"encoder.conv3a.weight"))
    rename_keys.append((f"conv3b.weight", f"encoder.conv3b.weight"))
    rename_keys.append((f"conv4a.weight", f"encoder.conv4a.weight"))
    rename_keys.append((f"conv4b.weight", f"encoder.conv4b.weight"))
    rename_keys.append((f"conv1a.bias", f"encoder.conv1a.bias"))
    rename_keys.append((f"conv1b.bias", f"encoder.conv1b.bias"))
    rename_keys.append((f"conv2a.bias", f"encoder.conv2a.bias"))
    rename_keys.append((f"conv2b.bias", f"encoder.conv2b.bias"))
    rename_keys.append((f"conv3a.bias", f"encoder.conv3a.bias"))
    rename_keys.append((f"conv3b.bias", f"encoder.conv3b.bias"))
    rename_keys.append((f"conv4a.bias", f"encoder.conv4a.bias"))
    rename_keys.append((f"conv4b.bias", f"encoder.conv4b.bias"))

    # Keypoint Decoder weights
    rename_keys.append((f"convPa.weight", f"keypoint_decoder.convSa.weight"))
    rename_keys.append((f"convPb.weight", f"keypoint_decoder.convSb.weight"))
    rename_keys.append((f"convPa.bias", f"keypoint_decoder.convSa.bias"))
    rename_keys.append((f"convPb.bias", f"keypoint_decoder.convSb.bias"))

    # Descriptor Decoder weights
    rename_keys.append((f"convDa.weight", f"descriptor_decoder.convDa.weight"))
    rename_keys.append((f"convDb.weight", f"descriptor_decoder.convDb.weight"))
    rename_keys.append((f"convDa.bias", f"descriptor_decoder.convDa.bias"))
    rename_keys.append((f"convDb.bias", f"descriptor_decoder.convDb.bias"))

    return rename_keys


# Copied from transformers.models.dinov2.convert_dinov2_to_hf
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


@torch.no_grad()
def convert_superpoint_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    TODO docs
    """

    print("Downloading original model from checkpoint...")
    config = get_superglue_config()

    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    print(original_state_dict)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(config, original_state_dict)
    new_state_dict = original_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    model = SuperPointModel(config)
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Successfully loaded weights in the model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth",
        type=str,
        help="URL of the original SuperPoint checkpoint you'd like to convert.",
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
    convert_superpoint_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub
    )
