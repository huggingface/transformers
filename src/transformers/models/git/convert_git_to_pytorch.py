# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert GIT checkpoints from the original repository.

URL: https://github.com/microsoft/GenerativeImage2Text/tree/main"""


import argparse
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    GitConfig,
    GitForCausalLM,
    GitProcessor,
    GitVisionConfig,
    VideoMAEImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_git_config(model_name):
    if "base" in model_name and "vqa" in model_name:
        image_size = 480
    elif "large" in model_name and "vqa" in model_name:
        image_size = 420
    else:
        image_size = 224

    vision_config = GitVisionConfig(image_size=image_size)

    if "large" in model_name:
        vision_config.patch_size = 14
        vision_config.hidden_size = 1024
        vision_config.intermediate_size = 4096
        vision_config.num_hidden_layers = 24
        vision_config.num_attention_heads = 16

    is_video = "vatex" in model_name or "msrvtt" in model_name
    num_image_with_embedding = 6 if is_video else None
    config = GitConfig(vision_config=vision_config.to_dict(), num_image_with_embedding=num_image_with_embedding)

    return config, image_size, is_video


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, prefix=""):
    rename_keys = []

    # image encoder
    # ftm: off
    rename_keys.append(
        (f"{prefix}image_encoder.class_embedding", "git.image_encoder.vision_model.embeddings.class_embedding")
    )
    rename_keys.append(
        (
            f"{prefix}image_encoder.positional_embedding",
            "git.image_encoder.vision_model.embeddings.position_embedding.weight",
        )
    )
    rename_keys.append(
        (f"{prefix}image_encoder.conv1.weight", "git.image_encoder.vision_model.embeddings.patch_embedding.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_pre.weight", "git.image_encoder.vision_model.pre_layrnorm.weight"))
    rename_keys.append((f"{prefix}image_encoder.ln_pre.bias", "git.image_encoder.vision_model.pre_layrnorm.bias"))
    rename_keys.append(
        (f"{prefix}image_encoder.ln_post.weight", "git.image_encoder.vision_model.post_layernorm.weight")
    )
    rename_keys.append((f"{prefix}image_encoder.ln_post.bias", "git.image_encoder.vision_model.post_layernorm.bias"))
    # fmt: on
    rename_keys.append((f"{prefix}image_encoder.proj", "git.image_encoder.visual_projection.weight"))

    # fmt: off
    for i in range(config.vision_config.num_hidden_layers):
        # image encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.attn.out_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_1.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_fc.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.mlp.c_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"{prefix}image_encoder.transformer.resblocks.{i}.ln_2.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias"))
    # fmt: on

    # text decoder
    # fmt: off
    rename_keys.append((f"{prefix}textual.embedding.words.weight", "git.embeddings.word_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.embedding.positions.weight", "git.embeddings.position_embeddings.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.weight", "git.visual_projection.visual_projection.0.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.0.bias", "git.visual_projection.visual_projection.0.bias"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.weight", "git.visual_projection.visual_projection.1.weight"))
    rename_keys.append((f"{prefix}textual.visual_projection.1.bias", "git.visual_projection.visual_projection.1.bias"))

    rename_keys.append((f"{prefix}textual.embedding.layer_norm.weight", "git.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{prefix}textual.embedding.layer_norm.bias", "git.embeddings.LayerNorm.bias"))
    rename_keys.append((f"{prefix}textual.output.weight", "output.weight"))
    rename_keys.append((f"{prefix}textual.output.bias", "output.bias"))
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.weight", f"git.encoder.layer.{i}.attention.self.query.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.query.bias", f"git.encoder.layer.{i}.attention.self.query.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.weight", f"git.encoder.layer.{i}.attention.self.key.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.key.bias", f"git.encoder.layer.{i}.attention.self.key.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.weight", f"git.encoder.layer.{i}.attention.self.value.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.self.value.bias", f"git.encoder.layer.{i}.attention.self.value.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.weight", f"git.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.dense.bias", f"git.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.weight", f"git.encoder.layer.{i}.attention.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.bias", f"git.encoder.layer.{i}.attention.output.LayerNorm.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.weight", f"git.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.intermediate.dense.bias", f"git.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.weight", f"git.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.dense.bias", f"git.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.weight", f"git.encoder.layer.{i}.output.LayerNorm.weight"))
        rename_keys.append((f"{prefix}textual.transformer.encoder.layer.{i}.output.LayerNorm.bias", f"git.encoder.layer.{i}.output.LayerNorm.bias"))
    # fmt: on

    if config.num_image_with_embedding is not None:
        rename_keys.append(("img_temperal_embedding.0", "git.img_temperal_embedding.0"))
        rename_keys.append(("img_temperal_embedding.1", "git.img_temperal_embedding.1"))
        rename_keys.append(("img_temperal_embedding.2", "git.img_temperal_embedding.2"))
        rename_keys.append(("img_temperal_embedding.3", "git.img_temperal_embedding.3"))
        rename_keys.append(("img_temperal_embedding.4", "git.img_temperal_embedding.4"))
        rename_keys.append(("img_temperal_embedding.5", "git.img_temperal_embedding.5"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val.T if "image_encoder.visual_projection" in new else val


# we split up the matrix of each CLIP encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, prefix=""):
    dim = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}image_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[
            :dim, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:dim]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            dim : dim * 2, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[
            dim : dim * 2
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[
            -dim:, :
        ]
        state_dict[f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-dim:]


# We will verify our results on an image
def prepare_img(model_name):
    if "textvqa" in model_name:
        filepath = hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        image = Image.open(filepath).convert("RGB")
    else:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    return image


def prepare_video():
    from decord import VideoReader, cpu

    # set seed for reproducability
    np.random.seed(0)

    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        """
        Sample a given number of frame indices from the video.

        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.

        Returns:
            indices (`List[int]`): List of sampled frame indices
        """
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    # video clip consists of 300 frames (10 seconds at 30 FPS)
    file_path = hf_hub_download(repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset")
    videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

    # sample 6 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=6, frame_sample_rate=4, seg_len=len(videoreader))
    video = videoreader.get_batch(indices).asnumpy()

    return video


@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """

    model_name_to_url = {
        "git-base": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt",
        "git-base-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_COCO/snapshot/model.pt",
        "git-base-textcaps": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTCAPS/snapshot/model.pt",
        "git-base-vqav2": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VQAv2/snapshot/model.pt",
        "git-base-textvqa": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_TEXTVQA/snapshot/model.pt",  # todo
        "git-base-vatex": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_VATEX/snapshot/model.pt",
        "git-base-msrvtt-qa": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_BASE_MSRVTT_QA/snapshot/model.pt"
        ),
        "git-large": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE/snapshot/model.pt",
        "git-large-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_COCO/snapshot/model.pt",
        "git-large-textcaps": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTCAPS/snapshot/model.pt"
        ),
        "git-large-vqav2": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VQAv2/snapshot/model.pt",
        "git-large-textvqa": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_TEXTVQA/snapshot/model.pt",
        "git-large-vatex": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_VATEX/snapshot/model.pt",
        "git-large-msrvtt-qa": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_MSRVTT_QA/snapshot/model.pt"
        ),
        "git-large-r": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R/snapshot/model.pt",
        "git-large-r-coco": "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_COCO/snapshot/model.pt",
        "git-large-r-textcaps": (
            "https://publicgit.blob.core.windows.net/data/output/GIT_LARGE_R_TEXTCAPS/snapshot/model.pt"
        ),
    }

    model_name_to_path = {
        "git-large": "/Users/nielsrogge/Documents/GIT/git_large_model.pt",
        "git-large-coco": "/Users/nielsrogge/Documents/GIT/git_large_coco_model.pt",
        "git-large-textcaps": "/Users/nielsrogge/Documents/GIT/git_large_textcaps_model.pt",
        "git-large-vqav2": "/Users/nielsrogge/Documents/GIT/git_large_vqav2_model.pt",
        "git-large-textvqa": "/Users/nielsrogge/Documents/GIT/git_large_textvqa_model.pt",
    }

    # define GIT configuration based on model name
    config, image_size, is_video = get_git_config(model_name)
    if "large" in model_name and not is_video and "large-r" not in model_name:
        # large checkpoints take way too long to download
        checkpoint_path = model_name_to_path[model_name]
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    else:
        checkpoint_url = model_name_to_url[model_name]
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", file_name=model_name)[
            "model"
        ]
    # rename keys
    prefix = "module." if model_name == "git-base" else ""
    rename_keys = create_rename_keys(config, prefix=prefix)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, prefix=prefix)

    # load HuggingFace model
    model = GitForCausalLM(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    assert missing_keys == ["git.embeddings.position_ids", "git.image_encoder.vision_model.embeddings.position_ids"]
    assert unexpected_keys == ["git.image_encoder.visual_projection.weight"]

    # verify results
    image_processor = (
        VideoMAEImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
        if is_video
        else CLIPImageProcessor(
            size={"shortest_edge": image_size}, crop_size={"height": image_size, "width": image_size}
        )
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_input_names=["input_ids", "attention_mask"])
    processor = GitProcessor(tokenizer=tokenizer, image_processor=image_processor)

    if is_video:
        video = prepare_video()
        pixel_values = processor(images=list(video), return_tensors="pt").pixel_values
    else:
        image = prepare_img(model_name)
        image_transforms = Compose(
            [
                Resize(image_size, interpolation=Image.BICUBIC),
                CenterCrop(image_size),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        original_pixel_values = image_transforms(image).unsqueeze(0)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        assert torch.allclose(pixel_values, original_pixel_values)

    input_ids = torch.tensor([[101]])
    outputs = model(input_ids, pixel_values=pixel_values)
    logits = outputs.logits
    print("Logits:", logits[0, -1, :3])

    if model_name == "git-base":
        expected_slice_logits = torch.tensor([-1.2832, -1.2835, -1.2840])
    elif model_name == "git-base-coco":
        expected_slice_logits = torch.tensor([-0.9925, -0.9930, -0.9935])
    elif model_name == "git-base-textcaps":
        expected_slice_logits = torch.tensor([-1.2980, -1.2983, -1.2985])
    elif model_name == "git-base-vqav2":
        expected_slice_logits = torch.tensor([-0.8570, -0.8568, -0.8561])
    elif model_name == "git-base-textvqa":
        expected_slice_logits = torch.tensor([-1.4085, -1.4083, -1.4082])
    elif model_name == "git-base-vatex":
        expected_slice_logits = torch.tensor([-1.3451, -1.3447, -1.3447])
    elif model_name == "git-base-msrvtt-qa":
        expected_slice_logits = torch.tensor([-0.8554, -0.8550, -0.8540])
    elif model_name == "git-large":
        expected_slice_logits = torch.tensor([-1.1708, -1.1707, -1.1705])
    elif model_name == "git-large-coco":
        expected_slice_logits = torch.tensor([-1.0425, -1.0423, -1.0422])
    elif model_name == "git-large-textcaps":
        expected_slice_logits = torch.tensor([-1.2705, -1.2708, -1.2706])
    elif model_name == "git-large-vqav2":
        expected_slice_logits = torch.tensor([-0.7042, -0.7043, -0.7043])
    elif model_name == "git-large-textvqa":
        expected_slice_logits = torch.tensor([-0.8590, -0.8592, -0.8590])
    elif model_name == "git-large-vatex":
        expected_slice_logits = torch.tensor([-1.0113, -1.0114, -1.0113])
    elif model_name == "git-large-msrvtt-qa":
        expected_slice_logits = torch.tensor([0.0130, 0.0134, 0.0131])
    elif model_name == "git-large-r":
        expected_slice_logits = torch.tensor([-1.1283, -1.1285, -1.1286])
    elif model_name == "git-large-r-coco":
        expected_slice_logits = torch.tensor([-0.9641, -0.9641, -0.9641])
    elif model_name == "git-large-r-textcaps":
        expected_slice_logits = torch.tensor([-1.1121, -1.1120, -1.1124])

    assert torch.allclose(logits[0, -1, :3], expected_slice_logits, atol=1e-4)
    print("Looks ok!")

    prompt = ""
    if "textvqa" in model_name:
        prompt = "what does the front of the bus say at the top?"
    elif "msrvtt-qa" in model_name:
        prompt = "what does the woman eat?"
    elif "vqa" in model_name:
        prompt = "what are the cats doing?"
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    input_ids = [processor.tokenizer.cls_token_id] + input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    print("Generating caption...")
    generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
    print("Generated caption:", processor.batch_decode(generated_ids, skip_special_tokens=True))

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and processor of {model_name} to the hub...")
        model.push_to_hub(f"microsoft/{model_name}")
        processor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="git-base",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    args = parser.parse_args()
    convert_git_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
