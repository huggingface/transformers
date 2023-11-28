# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert OWLv2 checkpoints from the original repository.

URL: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit"""

import argparse
import collections
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.training import checkpoints
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    CLIPTokenizer,
    Owlv2Config,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Processor,
    Owlv2TextConfig,
    Owlv2VisionConfig,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_owlv2_config(model_name):
    if "large" in model_name:
        image_size = 1008
        patch_size = 14
        vision_hidden_size = 1024
        vision_intermediate_size = 4096
        vision_num_hidden_layers = 24
        vision_num_attention_heads = 16
        projection_dim = 768
        text_hidden_size = 768
        text_intermediate_size = 3072
        text_num_attention_heads = 12
        text_num_hidden_layers = 12
    else:
        image_size = 960
        patch_size = 16
        vision_hidden_size = 768
        vision_intermediate_size = 3072
        vision_num_hidden_layers = 12
        vision_num_attention_heads = 12
        projection_dim = 512
        text_hidden_size = 512
        text_intermediate_size = 2048
        text_num_attention_heads = 8
        text_num_hidden_layers = 12

    vision_config = Owlv2VisionConfig(
        patch_size=patch_size,
        image_size=image_size,
        hidden_size=vision_hidden_size,
        num_hidden_layers=vision_num_hidden_layers,
        intermediate_size=vision_intermediate_size,
        num_attention_heads=vision_num_attention_heads,
    )
    text_config = Owlv2TextConfig(
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_attention_heads=text_num_attention_heads,
        num_hidden_layers=text_num_hidden_layers,
    )

    config = Owlv2Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=projection_dim,
    )

    return config


def flatten_nested_dict(params, parent_key="", sep="/"):
    items = []

    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, model_name):
    rename_keys = []

    # fmt: off
    # CLIP vision encoder
    rename_keys.append(("backbone/clip/visual/class_embedding", "owlv2.vision_model.embeddings.class_embedding"))
    rename_keys.append(("backbone/clip/visual/conv1/kernel", "owlv2.vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/positional_embedding", "owlv2.vision_model.embeddings.position_embedding.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/scale", "owlv2.vision_model.pre_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_pre/bias", "owlv2.vision_model.pre_layernorm.bias"))

    for i in range(config.vision_config.num_hidden_layers):
        if "v2" in model_name:
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_0/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_0/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        else:
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_1/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_2/scale", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/ln_2/bias", f"owlv2.vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_fc/kernel", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_fc/bias", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_proj/kernel", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/mlp/c_proj/bias", f"owlv2.vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/query/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/query/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/key/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/key/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/value/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/value/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/out/kernel", f"owlv2.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"backbone/clip/visual/transformer/resblocks.{i}/attn/out/bias", f"owlv2.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("backbone/clip/visual/ln_post/scale", "owlv2.vision_model.post_layernorm.weight"))
    rename_keys.append(("backbone/clip/visual/ln_post/bias", "owlv2.vision_model.post_layernorm.bias"))

    # CLIP text encoder
    rename_keys.append(("backbone/clip/text/token_embedding/embedding", "owlv2.text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("backbone/clip/text/positional_embedding", "owlv2.text_model.embeddings.position_embedding.weight"))

    for i in range(config.text_config.num_hidden_layers):
        if "v2" in model_name:
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_0/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_0/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.bias"))
        else:
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_1/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm1.bias"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_2/scale", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.weight"))
            rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/ln_2/bias", f"owlv2.text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_fc/kernel", f"owlv2.text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_fc/bias", f"owlv2.text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_proj/kernel", f"owlv2.text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/mlp/c_proj/bias", f"owlv2.text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/query/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/query/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/key/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/key/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/value/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/value/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/out/kernel", f"owlv2.text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"backbone/clip/text/transformer/resblocks.{i}/attn/out/bias", f"owlv2.text_model.encoder.layers.{i}.self_attn.out_proj.bias"))

    rename_keys.append(("backbone/clip/text/ln_final/scale", "owlv2.text_model.final_layer_norm.weight"))
    rename_keys.append(("backbone/clip/text/ln_final/bias", "owlv2.text_model.final_layer_norm.bias"))

    # logit scale
    rename_keys.append(("backbone/clip/logit_scale", "owlv2.logit_scale"))

    # projection heads
    rename_keys.append(("backbone/clip/text/text_projection/kernel", "owlv2.text_projection.weight"))

    # class and box heads
    rename_keys.append(("backbone/merged_class_token/scale", "layer_norm.weight"))
    rename_keys.append(("backbone/merged_class_token/bias", "layer_norm.bias"))
    rename_keys.append(("class_head/Dense_0/kernel", "class_head.dense0.weight"))
    rename_keys.append(("class_head/Dense_0/bias", "class_head.dense0.bias"))
    rename_keys.append(("class_head/logit_shift/kernel", "class_head.logit_shift.weight"))
    rename_keys.append(("class_head/logit_scale/kernel", "class_head.logit_scale.weight"))
    rename_keys.append(("class_head/logit_scale/bias", "class_head.logit_scale.bias"))
    rename_keys.append(("class_head/logit_shift/bias", "class_head.logit_shift.bias"))
    rename_keys.append(("obj_box_head/Dense_0/kernel", "box_head.dense0.weight"))
    rename_keys.append(("obj_box_head/Dense_0/bias", "box_head.dense0.bias"))
    rename_keys.append(("obj_box_head/Dense_1/kernel", "box_head.dense1.weight"))
    rename_keys.append(("obj_box_head/Dense_1/bias", "box_head.dense1.bias"))
    rename_keys.append(("obj_box_head/Dense_2/kernel", "box_head.dense2.weight"))
    rename_keys.append(("obj_box_head/Dense_2/bias", "box_head.dense2.bias"))

    # objectness head (only for v2)
    if "v2" in model_name:
        rename_keys.append(("objectness_head/Dense_0/kernel", "objectness_head.dense0.weight"))
        rename_keys.append(("objectness_head/Dense_0/bias", "objectness_head.dense0.bias"))
        rename_keys.append(("objectness_head/Dense_1/kernel", "objectness_head.dense1.weight"))
        rename_keys.append(("objectness_head/Dense_1/bias", "objectness_head.dense1.bias"))
        rename_keys.append(("objectness_head/Dense_2/kernel", "objectness_head.dense2.weight"))
        rename_keys.append(("objectness_head/Dense_2/bias", "objectness_head.dense2.bias"))

    # fmt: on

    return rename_keys


def rename_and_reshape_key(dct, old, new, config):
    val = dct.pop(old)

    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    if "patch_embedding" in new:
        print("Reshaping patch embedding... for", new)
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    if new.endswith("bias"):
        val = val.reshape(-1)

    dct[new] = torch.from_numpy(np.array(val))


@torch.no_grad()
def convert_owlv2_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our OWL-ViT structure.
    """
    config = get_owlv2_config(model_name)

    # see available checkpoints at https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit#pretrained-checkpoints
    variables = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    variables = variables["params"] if "v2" in model_name else variables["optimizer"]["target"]
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    state_dict = flatten_nested_dict(flax_params)

    # Rename keys
    rename_keys = create_rename_keys(config, model_name)
    for src, dest in rename_keys:
        rename_and_reshape_key(state_dict, src, dest, config)

    # load HuggingFace model
    model = Owlv2ForObjectDetection(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ["owlv2.visual_projection.weight"]
    assert unexpected_keys == []
    model.eval()

    # Initialize image processor
    size = {"height": config.vision_config.image_size, "width": config.vision_config.image_size}
    image_processor = Owlv2ImageProcessor(size=size)
    # Initialize tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!", model_max_length=16)
    # Initialize processor
    processor = Owlv2Processor(image_processor=image_processor, tokenizer=tokenizer)

    # Verify pixel_values and input_ids
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="owlvit_pixel_values_960.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath).permute(0, 3, 1, 2)

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="owlv2_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath).squeeze()

    filepath = hf_hub_download(repo_id="adirik/OWL-ViT", repo_type="space", filename="assets/astronaut.png")
    image = Image.open(filepath)
    texts = [["face", "rocket", "nasa badge", "star-spangled banner"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")

    if "large" not in model_name:
        assert torch.allclose(inputs.pixel_values, original_pixel_values.float(), atol=1e-6)
    assert torch.allclose(inputs.input_ids[:4, :], original_input_ids[:4, :], atol=1e-6)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        objectness_logits = outputs.objectness_logits

    if verify_logits:
        if model_name == "owlv2-base-patch16":
            expected_logits = torch.tensor(
                [[-10.0043, -9.0226, -8.0433], [-12.4569, -14.0380, -12.6153], [-21.0731, -22.2705, -21.8850]]
            )
            expected_boxes = torch.tensor(
                [[0.0136, 0.0223, 0.0269], [0.0406, 0.0327, 0.0797], [0.0638, 0.1539, 0.1255]]
            )
            expected_objectness_logits = torch.tensor(
                [[-5.6589, -7.7702, -16.3965]],
            )
        elif model_name == "owlv2-base-patch16-finetuned":
            expected_logits = torch.tensor(
                [[-9.2391, -9.2313, -8.0295], [-14.5498, -16.8450, -14.7166], [-15.1278, -17.3060, -15.7169]],
            )
            expected_boxes = torch.tensor(
                [[0.0103, 0.0094, 0.0207], [0.0483, 0.0729, 0.1013], [0.0629, 0.1396, 0.1313]]
            )
            expected_objectness_logits = torch.tensor(
                [[-6.5234, -13.3788, -14.6627]],
            )
        elif model_name == "owlv2-base-patch16-ensemble":
            expected_logits = torch.tensor(
                [[-8.6353, -9.5409, -6.6154], [-7.9442, -9.6151, -6.7117], [-12.4593, -15.3332, -12.1048]]
            )
            expected_boxes = torch.tensor(
                [[0.0126, 0.0090, 0.0238], [0.0387, 0.0227, 0.0754], [0.0582, 0.1058, 0.1139]]
            )
            expected_objectness_logits = torch.tensor(
                [[-6.0628, -5.9507, -10.4486]],
            )
        elif model_name == "owlv2-large-patch14":
            expected_logits = torch.tensor(
                [[-12.6662, -11.8384, -12.1880], [-16.0599, -16.5835, -16.9364], [-21.4957, -26.7038, -25.1313]],
            )
            expected_boxes = torch.tensor(
                [[0.0136, 0.0161, 0.0256], [0.0126, 0.0135, 0.0202], [0.0498, 0.0948, 0.0915]],
            )
            expected_objectness_logits = torch.tensor(
                [[-6.7196, -9.4590, -13.9472]],
            )
        elif model_name == "owlv2-large-patch14-finetuned":
            expected_logits = torch.tensor(
                [[-9.5413, -9.7130, -7.9762], [-9.5731, -9.7277, -8.2252], [-15.4434, -19.3084, -16.5490]],
            )
            expected_boxes = torch.tensor(
                [[0.0089, 0.0080, 0.0175], [0.0112, 0.0098, 0.0179], [0.0375, 0.0821, 0.0528]],
            )
            expected_objectness_logits = torch.tensor(
                [[-6.2655, -6.5845, -11.3105]],
            )
        elif model_name == "owlv2-large-patch14-ensemble":
            expected_logits = torch.tensor(
                [[-12.2037, -12.2070, -11.5371], [-13.4875, -13.8235, -13.1586], [-18.2007, -22.9834, -20.6816]],
            )
            expected_boxes = torch.tensor(
                [[0.0126, 0.0127, 0.0222], [0.0107, 0.0113, 0.0164], [0.0482, 0.1162, 0.0885]],
            )
            expected_objectness_logits = torch.tensor(
                [[-7.7572, -8.3637, -13.0334]],
            )

        print("Objectness logits:", objectness_logits[:3, :3])
        print("Logits:", logits[0, :3, :3])
        print("Pred boxes:", pred_boxes[0, :3, :3])

        assert torch.allclose(logits[0, :3, :3], expected_logits, atol=1e-3)
        assert torch.allclose(pred_boxes[0, :3, :3], expected_boxes, atol=1e-3)
        assert torch.allclose(objectness_logits[:3, :3], expected_objectness_logits, atol=1e-3)
        print("Looks ok!")
    else:
        print("Model converted without verifying logits")

    if pytorch_dump_folder_path is not None:
        print("Saving model and processor locally...")
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)

        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing {model_name} to the hub...")
        model.push_to_hub(f"google/{model_name}")
        processor.push_to_hub(f"google/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="owlv2-base-patch16",
        choices=[
            "owlv2-base-patch16",
            "owlv2-base-patch16-finetuned",
            "owlv2-base-patch16-ensemble",
            "owlv2-large-patch14",
            "owlv2-large-patch14-finetuned",
            "owlv2-large-patch14-ensemble",
        ],
        type=str,
        help="Name of the Owlv2 model you'd like to convert from FLAX to PyTorch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the original Flax checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify_logits",
        action="store_false",
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_owlv2_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub, args.verify_logits
    )
