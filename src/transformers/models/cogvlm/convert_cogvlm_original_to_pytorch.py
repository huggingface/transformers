# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert CogVLM checkpoints from the original repository.
"""

import argparse

import requests
import torch
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CogVLMConfig,
    CogVLMForCausalLM,
    CogVLMProcessor,
    LlamaTokenizer,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


original_device = "cuda:0"
hf_device = "cuda:2"


@torch.no_grad()
def convert_cogvlm_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # load original model
    revision="refs/pr/3" if model_name == "THUDM/cogvlm-chat-hf" else None
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # load_in_4bit=True,
        revision=revision,
    )
    original_model.to(original_device)

    print("Original config:", original_model.config)

    # verify chat example
    query = "Describe this image"
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "https://raw.githubusercontent.com/THUDM/CogVLM/main/assets/metrics-min.png"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    tokenizer = LlamaTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5", model_input_names=["input_ids", "attention_mask", "token_type_ids"]
    )

    if model_name == "THUDM/cogvlm-chat-hf":
        # chat mode
        inputs = original_model.build_conversation_input_ids(
            tokenizer, query=query, history=[], images=[image], template_version="chat",
        ) 
    elif model_name == "THUDM/cogvlm-base-224-hf":
        # base mode
        inputs = original_model.build_conversation_input_ids(
            tokenizer, query=query, history=[], images=[image], template_version="base",
        )

    def gather_inputs(inputs, device, use_bfloat16=True):
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
            "images": [[inputs["images"][0].to(device).to(dtype)]],
        }
        return inputs

    original_inputs = gather_inputs(inputs, device=original_device)
    gen_kwargs = {"max_length": 2048, "do_sample": False}  # TODO set to max_length: 2048

    print("Original input_ids:", tokenizer.decode(original_inputs["input_ids"][0]))

    with torch.no_grad():
        outputs = original_model.generate(**original_inputs, **gen_kwargs)
        outputs = outputs[:, original_inputs["input_ids"].shape[1] :]
        original_generated_text = tokenizer.decode(outputs[0])

    # load HF model
    # rename in_channels to num_channels for sake of consistency
    original_model.config.vision_config["num_channels"] = original_model.config.vision_config.pop("in_channels")

    config = CogVLMConfig(**original_model.config.to_dict())
    model = CogVLMForCausalLM(config)

    # load state dict
    model.load_state_dict(original_model.state_dict())
    model.to(hf_device)
    model.eval()

    # cast all parameters to bfloat16
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)

    # create processor
    image_size = original_model.config.vision_config["image_size"]
    image_processor = CLIPImageProcessor(
        size={"height": image_size, "width": image_size},
        do_center_crop=False,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    processor = CogVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)

    original_inputs = gather_inputs(inputs, device=hf_device)
    original_inputs["pixel_values"] = torch.stack(original_inputs.pop("images")[0])

    if model_name == "THUDM/cogvlm-chat-hf":
        # chat history template
        prompt = f"Question: {query} Answer:"
    elif model_name == "THUDM/cogvlm-base-224-hf":
        # base history template
        prompt = f"{query}"
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(hf_device, torch.bfloat16)

    for k, v in inputs.items():
        print(k, v.shape)

    print("HF input_ids:", tokenizer.decode(inputs.input_ids[0]))

    # verify inputs
    # for k, v in inputs.items():
    #     assert torch.allclose(v, original_inputs[k].to(v.device))

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(outputs[0])

    print("Original text:", original_generated_text)
    print("HF text:", generated_text)
    assert original_generated_text == generated_text
    # with torch.no_grad():
    #      original_logits = original_model({"image": original_pixel_values, "text_input": [""]}).logits
    #      logits = model(pixel_values, input_ids).logits

    # assert original_logits.shape == logits.shape
    # print("First values of original logits:", original_logits[0, :3, :3])
    # print("First values of HF logits:", logits[0, :3, :3])

    # # assert values
    # assert torch.allclose(original_logits.to(logits.device), logits, atol=1e-4)
    # print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        processor.push_to_hub(f"nielsr/{model_name.split('/')[-1]}")
        model.push_to_hub(f"nielsr/{model_name.split('/')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="THUDM/cogvlm-chat-hf",
        type=str,
        choices=["THUDM/cogvlm-chat-hf", "THUDM/cogvlm-base-490-hf", "THUDM/cogvlm-base-224-hf"],
        help="Name of the model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_cogvlm_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
