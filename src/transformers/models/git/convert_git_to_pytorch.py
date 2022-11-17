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

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

import requests
from transformers import AutoTokenizer, GITConfig, GITForCausalLM, GITVisionConfig, GITProcessor, CLIPImageProcessor, BertTokenizerFast
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_git_config(model_name):
    config = GITConfig(vision_config=GITVisionConfig())

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # image encoder
    # ftm: off
    rename_keys.append(
        ("module.image_encoder.class_embedding", "git.image_encoder.vision_model.embeddings.class_embedding")
    )
    rename_keys.append(
        (
            "module.image_encoder.positional_embedding",
            "git.image_encoder.vision_model.embeddings.position_embedding.weight",
        )
    )
    rename_keys.append(
        ("module.image_encoder.conv1.weight", "git.image_encoder.vision_model.embeddings.patch_embedding.weight")
    )
    rename_keys.append(("module.image_encoder.ln_pre.weight", "git.image_encoder.vision_model.pre_layrnorm.weight"))
    rename_keys.append(("module.image_encoder.ln_pre.bias", "git.image_encoder.vision_model.pre_layrnorm.bias"))
    rename_keys.append(("module.image_encoder.ln_post.weight", "git.image_encoder.vision_model.post_layernorm.weight"))
    rename_keys.append(("module.image_encoder.ln_post.bias", "git.image_encoder.vision_model.post_layernorm.bias"))
    # fmt: on
    rename_keys.append(("module.image_encoder.proj", "git.image_encoder.visual_projection.weight"))

    # fmt: off
    for i in range(config.vision_config.num_hidden_layers):
        # image encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.attn.out_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.attn.out_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.ln_1.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.ln_1.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.mlp.c_fc.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.mlp.c_fc.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.mlp.c_proj.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.mlp.c_proj.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.ln_2.weight", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"module.image_encoder.transformer.resblocks.{i}.ln_2.bias", f"git.image_encoder.vision_model.encoder.layers.{i}.layer_norm2.bias"))
    # fmt: on

    # text decoder
    # fmt: off
    rename_keys.append(("module.textual.embedding.words.weight", "git.embeddings.word_embeddings.weight"))
    rename_keys.append(("module.textual.embedding.positions.weight", "git.embeddings.position_embeddings.weight"))
    rename_keys.append(("module.textual.visual_projection.0.weight", "git.visual_projection.visual_projection.0.weight"))
    rename_keys.append(("module.textual.visual_projection.0.bias", "git.visual_projection.visual_projection.0.bias"))
    rename_keys.append(("module.textual.visual_projection.1.weight", "git.visual_projection.visual_projection.1.weight"))
    rename_keys.append(("module.textual.visual_projection.1.bias", "git.visual_projection.visual_projection.1.bias"))

    # rename_keys.append(("module.textual.embedding.token_type.weight", "git.embeddings.token_type_embeddings.weight"))
    rename_keys.append(("module.textual.embedding.layer_norm.weight", "git.embeddings.LayerNorm.weight"))
    rename_keys.append(("module.textual.embedding.layer_norm.bias", "git.embeddings.LayerNorm.bias"))
    rename_keys.append(("module.textual.output.weight", "output.weight"))
    rename_keys.append(("module.textual.output.bias", "output.bias"))
    for i in range(config.num_hidden_layers):
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.query.weight", f"git.encoder.layer.{i}.attention.self.query.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.query.bias", f"git.encoder.layer.{i}.attention.self.query.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.key.weight", f"git.encoder.layer.{i}.attention.self.key.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.key.bias", f"git.encoder.layer.{i}.attention.self.key.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.value.weight", f"git.encoder.layer.{i}.attention.self.value.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.self.value.bias", f"git.encoder.layer.{i}.attention.self.value.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.output.dense.weight", f"git.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.output.dense.bias", f"git.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.weight", f"git.encoder.layer.{i}.attention.output.LayerNorm.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.attention.output.LayerNorm.bias", f"git.encoder.layer.{i}.attention.output.LayerNorm.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.intermediate.dense.weight", f"git.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.intermediate.dense.bias", f"git.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.output.dense.weight", f"git.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.output.dense.bias", f"git.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.output.LayerNorm.weight", f"git.encoder.layer.{i}.output.LayerNorm.weight"))
        rename_keys.append((f"module.textual.transformer.encoder.layer.{i}.output.LayerNorm.bias", f"git.encoder.layer.{i}.output.LayerNorm.bias"))
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val.T if "image_encoder.visual_projection" in new else val


# we split up the matrix of each CLIP encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    dim = config.vision_config.hidden_size
    for i in range(config.vision_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"module.image_encoder.transformer.resblocks.{i}.attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"module.image_encoder.transformer.resblocks.{i}.attn.in_proj_bias")
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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


image_transforms = Compose(
    [
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)


@torch.no_grad()
def convert_git_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our GIT structure.
    """

    model_name_to_url = {
        "git-base": "https://publicgit.blob.core.windows.net/data/output/GIT_BASE/snapshot/model.pt",
    }

    # define GIT configuration based on model name
    config = get_git_config(model_name)
    # load original state_dict from URL
    checkpoint_url = model_name_to_url[model_name]
    # state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]
    # TODO remove line below
    state_dict = torch.load("/Users/nielsrogge/Documents/GIT/model.pt", map_location="cpu")["model"]
    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = GITForCausalLM(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()

    assert missing_keys == ["git.embeddings.position_ids", "git.image_encoder.vision_model.embeddings.position_ids"]
    assert len(unexpected_keys) == 0

    # verify results
    image_processor = CLIPImageProcessor()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    processor = GITProcessor(tokenizer=tokenizer, feature_extractor=image_processor)

    image = prepare_img()
    pixel_values = image_transforms(image).unsqueeze(0)
    input_ids = torch.tensor([[101]])

    logits = model(input_ids, pixel_values=pixel_values).logits
    assert torch.allclose(logits[0, -1, :3], torch.tensor([-1.2832, -1.2835, -1.2840]), atol=1e-4)
    print("Looks ok!")

    print("Generating caption...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    predicted_ids = []
    for i in range(20):
        outputs = model(pixel_values=pixel_values, input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        # perform argmax on the last dimension (i.e. greedy decoding)
        predicted_id = logits.argmax(-1)
        predicted_ids.append(predicted_id.item())
        print(tokenizer.decode([predicted_id.squeeze()]))
        # print("Predicted id:", predicted_id)
        # add predicted id to input_ids
        input_ids = torch.cat([input_ids, predicted_id.unsqueeze(0)], dim=1)
    # generated_ids = model.generate(pixel_values, max_length=20)
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # print("Generated caption:", tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # TODO create feature extractor
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model to the hub...")
        model.push_to_hub(f"microsoft/{model_name}")


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
