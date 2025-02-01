import argparse

import requests
import torch
from PIL import Image

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    InternVL2_5Config,
    InternVL2_5ForConditionalGeneration,
    InternVL2_5ImageProcessor,
    InternVL2_5Processor,
)


DTYPE2HF = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def prepare_input(messages, image_url, processor):
    # Load image
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda", torch.bfloat16)
    return inputs


def compare_models(original_model, converted_model, processor, messages, image_url):
    # Prepare identical inputs
    inputs = prepare_input(messages, image_url, processor)

    # Get predictions from converted model
    with torch.no_grad():
        converted_outputs = converted_model.generate(**inputs)
        converted_output = converted_outputs[0][len(inputs["input_ids"][0]) :]
        converted_result = processor.tokenizer.decode(converted_output, skip_special_tokens=True)

    inputs.pop("num_patches")
    # Get predictions from original model
    with torch.no_grad():
        original_outputs = original_model.generate(**inputs)
        original_result = processor.tokenizer.decode(original_outputs[0], skip_special_tokens=True)

    # Compare outputs
    print("Original Model Output:", original_result)
    print("Converted Model Output:", converted_result)

    if original_result == converted_result:
        print("The outputs are identical.")
        return True
    else:
        print("The outputs are different.")
        return False


def convert_hf_config(model_name, output_dir, push_to_hub=False):
    # Construct the HF URL
    hf_url = f"OpenGVLab/{model_name}"

    internvl_config = AutoConfig.from_pretrained(hf_url, trust_remote_code=True)
    internvl_tokenizer = AutoTokenizer.from_pretrained(hf_url, trust_remote_code=True, use_fast=True)

    image_size = internvl_config.force_image_size or internvl_config.vision_config.image_size
    patch_size = internvl_config.vision_config.patch_size
    num_image_token = int((image_size // patch_size) ** 2 * (internvl_config.downsample_ratio**2))

    hf_config = {
        "model_type": "internvl2_5",
        "architectures": ["InternVL2_5ForConditionalGeneration"],
        "_name_or_path": "",
        "vision_config": {
            "architectures": ["InternVL2_5VisionModel"],
            "num_channels": internvl_config.vision_config.num_channels,
            "patch_size": internvl_config.vision_config.patch_size,
            "image_size": internvl_config.vision_config.image_size,
            "hidden_size": internvl_config.vision_config.hidden_size,
            "num_attention_heads": internvl_config.vision_config.num_attention_heads,
            "num_hidden_layers": internvl_config.vision_config.num_hidden_layers,
            "qkv_bias": internvl_config.vision_config.qkv_bias,
            "hidden_act": internvl_config.vision_config.hidden_act,
            "norm_type": internvl_config.vision_config.norm_type,
            "layer_norm_eps": internvl_config.vision_config.layer_norm_eps,
            "dropout": internvl_config.vision_config.dropout,
            "drop_path_rate": internvl_config.vision_config.drop_path_rate,
            "attention_dropout": internvl_config.vision_config.attention_dropout,
            "initializer_range": internvl_config.vision_config.initializer_range,
            "initializer_factor": internvl_config.vision_config.initializer_factor,
            "qk_normalization": internvl_config.vision_config.qk_normalization,
            "intermediate_size": internvl_config.vision_config.intermediate_size,
            "torch_dtype": DTYPE2HF[internvl_config.vision_config.torch_dtype],
            "_attn_implementation": "flash_attention_2" if internvl_config.vision_config.use_flash_attn else "eager",
        },
        "text_config": internvl_config.llm_config.to_dict(),
        "max_dynamic_patch": internvl_config.max_dynamic_patch,
        "min_dynamic_patch": internvl_config.min_dynamic_patch,
        "use_thumbnail": internvl_config.use_thumbnail,
        "dynamic_image_size": internvl_config.dynamic_image_size,
        "force_image_size": internvl_config.force_image_size,
        "pixel_shuffle_version": internvl_config.ps_version,
        "select_layer": internvl_config.select_layer,
        "downsample_ratio": internvl_config.downsample_ratio,
        "torch_dtype": DTYPE2HF[internvl_config.torch_dtype],
        "num_image_token": num_image_token,
        "bos_token_id": internvl_tokenizer.bos_token_id,
        "eos_token_id": internvl_tokenizer.eos_token_id,
        "image_token_id": internvl_tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>"),
        "image_start_token_id": internvl_tokenizer.convert_tokens_to_ids("<img>"),
        "image_end_token_id": internvl_tokenizer.convert_tokens_to_ids("</img>"),
    }

    config = InternVL2_5Config(**hf_config)
    model = (
        InternVL2_5ForConditionalGeneration.from_pretrained(hf_url, config=config, torch_dtype="auto").eval().cuda()
    )

    # Load original model
    internvl_model = AutoModel.from_pretrained(hf_url, trust_remote_code=True, torch_dtype="auto").eval().cuda()

    internvl_model.img_context_token_id = model.config.image_token_id

    image_processor = InternVL2_5ImageProcessor(
        size={"height": image_size, "width": image_size},
        min_tiles=internvl_config.min_dynamic_patch,
        max_tiles=internvl_config.max_dynamic_patch,
        use_thumbnail=internvl_config.use_thumbnail,
        num_image_token=num_image_token,
    )

    internvl_tokenizer.chat_template = (
        "{% set system_message = 'You are a helpful assistant.' %}"
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}"
        "{% for message in loop_messages %}"
        "{% set content = message['content'] %}"
        "{% if message['role'] == 'user' %}"
        "{% if content is iterable and content is sequence and content[0]['type'] == 'image' %}"
        "{{ '<|im_start|>user\\n<image>\\n' + content[1]['text'] + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if content is iterable and content is sequence %}"
        "{% for part in content %}"
        "{% if part['type'] == 'text' %}"
        "{{ part['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{{ '<|im_end|>\\n' }}"
        "{% else %}"
        "{{ content + '<|im_end|>\\n' }}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
    )

    processor = InternVL2_5Processor(
        image_processor=image_processor, tokenizer=internvl_tokenizer, chat_template=internvl_tokenizer.chat_template
    )

    # Compare models before saving
    messages = [{"role": "user", "content": "<image>\nWhat kind of dog is this?"}]
    image_url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"

    print("\nComparing model outputs before saving...")
    models_match = compare_models(internvl_model, model, processor, messages, image_url)

    if not models_match:
        raise ValueError("Model outputs do not match.")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    internvl_tokenizer.save_pretrained(output_dir)

    if push_to_hub:
        model.push_to_hub(f"thisisiron/{model_name}")
        processor.push_to_hub(f"thisisiron/{model_name}")
        internvl_tokenizer.push_to_hub(f"thisisiron/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert InternVL2.5 checkpoints configuration")

    # Model configuration
    choices = [
        "InternVL2_5-78B-MPO",
        "InternVL2_5-26B-MPO",
        "InternVL2_5-4B-MPO",
        "InternVL2_5-1B-MPO",
        "InternVL2_5-38B-MPO",
        "InternVL2_5-8B-MPO",
        "InternVL2_5-2B-MPO",
        "InternVL2_5-78B",
        "InternVL2_5-38B",
        "InternVL2_5-26B",
        "InternVL2_5-8B",
        "InternVL2_5-4B",
        "InternVL2_5-2B",
        "InternVL2_5-1B",
    ]

    parser.add_argument(
        "--model_name",
        default="InternVL2_5-1B",
        choices=choices,
        type=str,
        help="Name of the model to convert",
    )

    # Input/Output paths
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path to the output directory for the converted model.",
    )

    # Hub options
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted model to the hub",
    )

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"./{args.model_name}"

    # Convert the checkpoint
    convert_hf_config(args.model_name, args.output_dir, args.push_to_hub)
