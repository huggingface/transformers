# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Convert LLaVa 1.6 checkpoints from the original repository.

URL: https://github.com/haotian-liu/LLaVA/tree/main.


The command used to obtain original logits is the following:
python llava/eval/run_llava.py --model-path "liuhaotian/llava-v1.6-mistral-7b" --image-file "images/llava_v1_5_radar.jpg" --query "What is shown in this image?" --max_new_tokens 100 --temperature 0
"""

import argparse
import glob
import json
from pathlib import Path

import requests
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaImageProcessor,
    LlavaProcessor,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value.to(torch.float16)
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        text_model_id = data["_name_or_path"]
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        text_model_id = "lmsys/vicuna-7b-v1.5"
    vision_model_id = data["mm_vision_tower"]

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = LlavaImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = LlavaConfig(
        text_config=text_config.to_dict(),
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
    )
    config.pad_token_id = 32001

    with init_empty_weights():
        model = LlavaForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)
    model.eval()

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    pad_shape = 64
    model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )

    device = "cuda:2"
    model.to(device)

    # prepare inputs
    image = load_image()
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath, map_location="cpu")
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")

    # verify inputs
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        # replace -200 by 32000 (since we use token ID = 32000 for the image token)
        original_input_ids[original_input_ids == -200] = 32000
        print(tokenizer.decode([id for id in original_input_ids.tolist()[0] if id != -200]))

        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()
        assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    # verify single forward pass
    image_sizes = torch.tensor([[899, 1024]])
    assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of logits:", outputs.logits.shape)
        print("First values of logits:", outputs.logits[0, :3, :3])

        if model_id == "liuhaotian/llava-v1.6-mistral-7b":
            expected_slice = torch.tensor(
                [[-4.8555, -4.6992, -0.1996], [-10.5703, -10.7344, -2.7246], [-7.0391, -7.3672, -0.2634]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
            expected_slice = torch.tensor(
                [[1.4883, 0.9976, -0.6992], [-9.7031, -5.7031, -1.5557], [-5.1328, -5.5586, 8.8281]],
                dtype=torch.float32,
                device=device,
            )
        else:
            raise ValueError(f"Model {model_id} not supported")

        assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
        print("Logits are ok!")

    # verify generation
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        expected_text = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular radar chart, there are several axes labeled with different metrics or benchmarks, such as "MMM-Vet," "MMM-Bench," "LLaVA-Bench," "SLED-Bench," "'
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        expected_text = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER:  \nWhat is shown in this image? ASSISTANT: The image appears to be a graphical representation of a benchmarking study comparing the performance of various models or systems. It\'s a scatter plot with a circular layout, where each point represents a different model or system, and the axes represent different metrics or dimensions of comparison.\n\nThe metrics are likely related to machine learning or artificial intelligence performance, as indicated by the terms like "BLIP-2," "Instruct BLIP," "POE," "QWA," "V"""
    else:
        raise ValueError(f"Model {model_id} not supported")

    assert generated_text == expected_text
    print("Generated text is ok!")

    # verify batched generation
    inputs = processor(images=[image, image], text=[prompt, prompt], return_tensors="pt").to(device)

    print("Batched generation...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=3,
        use_cache=True,
    )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = model_id.split("/")[-1]
        model.push_to_hub(f"llava-hf/{repo_id}-hf")
        processor.push_to_hub(f"llava-hf/{repo_id}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="liuhaotian/llava-v1.6-mistral-7b",
        choices=["liuhaotian/llava-v1.6-mistral-7b", "liuhaotian/llava-v1.6-vicuna-7b"],
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
