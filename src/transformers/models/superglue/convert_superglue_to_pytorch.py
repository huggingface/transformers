import argparse

import numpy as np
import requests
import torch
from PIL import Image
import cv2

from transformers import SuperGlueConfig, SuperGlueModel, SuperPointImageProcessor, AutoImageProcessor, AutoModel, \
    AutoConfig, SuperPointConfig, AutoModelForKeypointDetection
from transformers.models.superglue.modeling_superglue import ImageMatchingOutput


def get_superglue_config(checkpoint_url):
    keypoint_detection_config = AutoConfig.from_pretrained("magic-leap-community/superpoint")
    config = SuperGlueConfig(
        descriptor_dim=256,
        keypoint_encoder_sizes=[32, 64, 128, 256],
        gnn_layers_types=['self', 'cross'] * 9,
        sinkhorn_iterations=100,
        matching_threshold=0.2,
        keypoint_detector_config=keypoint_detection_config
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


def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return inp


def prepare_imgs_as_original():
    url = "tests/fixtures/tests_samples/image_matching/london_bridge_78916675_4568141288.jpg"
    im1 = read_image(url, 'cuda', [640, 480], 0, False)
    url = "tests/fixtures/tests_samples/image_matching/london_bridge_19481797_2295892421.jpg"
    im2 = read_image(url, 'cuda', [640, 480], 0, False)
    return torch.cat([im1, im2], dim=0)


def prepare_imgs_for_image_processor():
    url = "tests/fixtures/tests_samples/image_matching/london_bridge_78916675_4568141288.jpg"
    im1 = Image.open(url)
    url = "tests/fixtures/tests_samples/image_matching/london_bridge_19481797_2295892421.jpg"
    im2 = Image.open(url)
    return [im1, im2]

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

    model = SuperGlueModel(config)
    model.load_state_dict(new_state_dict, strict=False)
    model.to("cuda")
    model.eval()
    print("Successfully loaded weights in the model")

    ## USE REGULAR IMAGE PROCESSOR FOR INFERENCE
    images = prepare_imgs_for_image_processor()
    preprocessor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    inputs = preprocessor(images=images, return_tensors="pt")
    inputs.to("cuda")
    output: ImageMatchingOutput = model(**inputs, return_dict=True)
    print("Number of matching keypoints using image processor")
    print(torch.sum(output.image0_matches != -1))
    print(torch.sum(output.image1_matches != -1))

    images = prepare_imgs_as_original()
    output: ImageMatchingOutput = model(pixel_values=images, return_dict=True)
    print("Number of matching keypoints using original reading code")
    print(torch.sum(output.image0_matches != -1))
    print(torch.sum(output.image1_matches != -1))
    # output: ImageMatchingOutput = model(pixel_values=images, return_dict=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth",
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
