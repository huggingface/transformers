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

import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, CogVLMForCausalLM, LlamaTokenizer, CogVLMConfig, CogVLMImageProcessor, CogVLMProcessor
from PIL import Image
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


@torch.no_grad()
def convert_cogvlm_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # load original model
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    original_model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # load_in_4bit=True,
        revision="refs/pr/3",
    )
    original_model.to("cuda:0")

    # load HF model
    config = CogVLMConfig()
    model = CogVLMForCausalLM(config)

    # verify chat example
    query = 'Describe this image'
    image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream=True).raw).convert('RGB')
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0]))

    # create processor
    image_size = 224
    image_processor = CogVLMImageProcessor(
        size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    processor = CogVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(model_device)

    # make sure processor creates exact same pixel values
    # assert torch.allclose(pixel_values, original_pixel_values.to(pixel_values.device))

    # original_model.to(lavis_device)
    # model.to(model_device)
    # with torch.no_grad():
    #     if "opt" in model_name:
    #         original_logits = original_model({"image": original_pixel_values, "text_input": [""]}).logits
    #         logits = model(pixel_values, input_ids).logits
    #     else:
    #         original_logits = original_model(
    #             {"image": original_pixel_values, "text_input": ["\n"], "text_output": ["\n"]}
    #         ).logits
    #         labels = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
    #         logits = model(pixel_values, input_ids, labels=labels).logits

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
        processor.push_to_hub(f"nielsr/{model_name}")
        model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="blip2-opt-2.7b",
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_cogvlm_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
