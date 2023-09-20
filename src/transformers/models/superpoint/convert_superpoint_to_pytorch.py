import argparse
import os

import requests
import torch
from PIL import Image

from transformers import SuperPointConfig, SuperPointModel, SuperPointImageProcessor


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


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_superpoint_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    TODO docs
    """

    print("Downloading original model from checkpoint...")
    config = get_superglue_config()

    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(config, original_state_dict)
    new_state_dict = original_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_state_dict, src, dest)

    model = SuperPointModel(config)
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Successfully loaded weights in the model")

    preprocessor = SuperPointImageProcessor()
    inputs = preprocessor(images=prepare_img(), return_tensors="pt")
    outputs = model(**inputs)

    expected_keypoints_shape = (568, 2)
    expected_scores_shape = (568,)
    expected_descriptors_shape = (256, 568)

    expected_keypoints_values = torch.tensor([[480.0, 9.0], [494.0, 9.0], [489.0, 16.0]])
    expected_scores_values = torch.tensor([0.0064, 0.0140, 0.0595, 0.0728, 0.5170, 0.0175, 0.1523, 0.2055, 0.0336])
    expected_descriptors_value = torch.tensor(-0.1096)

    assert outputs.keypoints.shape == expected_keypoints_shape
    assert outputs.scores.shape == expected_scores_shape
    assert outputs.descriptors.shape == expected_descriptors_shape

    expected_keypoints = outputs.keypoints[:3]
    expected_scores = outputs.scores[:9]

    assert torch.allclose(outputs.keypoints[:3], expected_keypoints_values, atol=1e-3)
    assert torch.allclose(outputs.scores[:9], expected_scores_values, atol=1e-3)
    assert torch.allclose(outputs.descriptors[0, 0], expected_descriptors_value, atol=1e-3)
    print("Model outputs match the original results!")

    if save_model:
        print("Saving model to local...")
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

        model_name = "superpoint"
        if push_to_hub:
            print(f"Pushing {model_name} to the hub...")
        model.push_to_hub(model_name)
        preprocessor.push_to_hub(model_name)


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
