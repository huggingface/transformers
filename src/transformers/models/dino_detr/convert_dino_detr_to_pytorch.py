"""Convert Dino DETR checkpoints."""

import argparse
import json
import re
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    DinoDetrConfig,
    DinoDetrForObjectDetection,
    DinoDetrImageProcessor,
)
from transformers.utils import logging


torch.manual_seed(42)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"(encoder\..*)norm1\.(weight|bias)": r"\1self_attn_layer_norm.\2",
    r"(encoder\..*)norm2\.(weight|bias)": r"\1final_layer_norm.\2",
    r"(encoder\..*)norm3\.(weight|bias)": r"\1final_layer_norm.\2",
    r"(encoder\..*)linear1\.(weight|bias)": r"\1fc1.\2",
    r"(encoder\..*)linear2\.(weight|bias)": r"\1fc2.\2",
    r"backbone\.0\.body": r"backbone.conv_encoder.model",
    r"tgt_embed": r"content_query_embeddings",
    r"^(.*)$": r"model.\1",
}


@torch.no_grad()
def convert_dino_detr_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    push_to_hub,
):
    """
    Copy/paste/tweak model's weights to our Deformable DETR structure.
    """

    # load default config
    config = DinoDetrConfig()
    # set labels
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]

    for original_key, converted_key in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        for key in list(state_dict.copy().keys()):
            new_key = re.sub(original_key, converted_key, key)
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    processor = DinoDetrImageProcessor()
    model = DinoDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, nms_iou_threshold=10, conf_threshold=0
    )[0]

    expected_scores = torch.tensor(
        [
            0.7475,
            0.7341,
            0.7229,
            0.4707,
            0.4449,
            0.3086,
            0.1927,
            0.1794,
            0.1334,
            0.1251,
            0.1113,
            0.1111,
            0.1110,
            0.1097,
            0.1039,
            0.1038,
            0.1015,
            0.0963,
            0.0962,
        ]
    )

    print("Scores:", results["scores"][:19])
    print("Expected Scores:", expected_scores)

    assert torch.allclose(results["scores"][:19], expected_scores, atol=1e-4)

    print("Everything ok!")

    # Save model and image processor
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )
    args = parser.parse_args()
    convert_dino_detr_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
