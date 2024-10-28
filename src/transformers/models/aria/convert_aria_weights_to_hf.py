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
import argparse
import glob
import time

import requests
import torch
from huggingface_hub import login, snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AriaConfig,
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoConfig,
    AutoTokenizer,
    Idefics3VisionConfig,
)


login("token")

EPILOG_TXT = """Example:
    python transformers/src/transformers/models/aria/convert_aria_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14-336 --output_hub_path org/aria-v1.5-7b-conv --old_state_dict_id liuhaotian/aria-v1.5-7b

Example for creating the old state dict file with Python:

    import torch
    from aria.model.language_model.aria_llama import AriaLlamaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = AriaLlamaForCausalLM.from_pretrained("liuhaotian/aria-v1.5-7b", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/aria-v1.5-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    "vision_tower.vision_model": "vision_tower",
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


# used only for aria-interlave
# for ex: Qwen/Qwen1.5-0.5B-Chat google/siglip-so400m-patch14-384 lmms-lab/aria-next-interleave-qwen-0.5b
def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    new_state_dict['vision_tower.post_layernorm.weight'] = torch.zeros((1152,))
    new_state_dict['vision_tower.post_layernorm.bias'] = torch.zeros((1152,))

    return new_state_dict


def convert_aria_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id).text_config

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:  # qwen already has a pad token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    processor = AriaProcessor.from_pretrained(
        text_model_id, tokenizer_path=text_model_id,
    )

    config = AutoConfig.from_pretrained(text_model_id)

    vision_config = Idefics3VisionConfig(
        hidden_size=1152,
        image_size=980,
        intermediate_size=4304,
        num_attention_heads=16,
        num_hidden_layers=27,
        patch_size=14,
        torch_dtype="bfloat16",
    ).to_dict()

    config = AriaConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    # llms-lab interleeave models do not use any selection startegy except for last hidden state
    if "Qwen" in text_model_id:
        config.image_token_index = 151646
        if "siglip" in vision_model_id:
            config.vision_feature_select_strategy = "full"
            config.vision_feature_layer = -1
    else:
        config.pad_token_id = 32001
        config.image_token_index = 32000

    with torch.device("meta"):
        model = AriaForConditionalGeneration(config)

    state_dict = load_original_state_dict(old_state_dict_id)

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=False, assign=True)

    # pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    # mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    # n = pre_expansion_embeddings.size()[0]
    # sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # # We add an image token so we resize the model and pad to 64 for performance reasons
    # pad_shape = 64
    # vocab_size = config.text_config.vocab_size
    # model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    # model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
    #     tuple(
    #         (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
    #     ),
    #     dim=0,
    # )
    # model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
    #     tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
    #     dim=0,
    # )

    ### Test generation
    t1 = time.time()
    image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
    # image2 = Image.open("bird.jpg")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": "What is the color of the bird's beak?", "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=8,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=False,
    )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(output_ids, skip_special_tokens=True)

    t2 = time.time()
    print(response)
    print(f"Generation time: {(t2-t1):.3f}s")

    ### Push
    model.save_pretrained(output_hub_path)
    processor.save_pretrained(output_hub_path)
    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        default="rhymes-ai/Aria",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        default="rhymes-ai/Aria",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        default="m-ric/Aria_hf",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        default="rhymes-ai/Aria",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()
    convert_aria_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)


if __name__ == "__main__":
    main()
