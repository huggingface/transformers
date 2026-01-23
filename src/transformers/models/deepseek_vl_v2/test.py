from transformers.models.deepseek_vl_v2.modeling_deepseek_vl_v2 import (
    DeepseekVLV2ForCausalLM,
)
from transformers.models.deepseek_vl_v2.processing_deepseek_vl_v2 import (
    DeepseekVLV2Processor,
)

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

image = Image.open("./test.png")

if __name__ == "__main__":
    model_path = "deepseek-ai/deepseek-vl2-tiny"

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = vl_gpt.to(torch.bfloat16).cuda().eval()

    processor: DeepseekVLV2Processor = AutoProcessor.from_pretrained(model_path)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": "<image>"},
                {"type": "text", "text": "Please describe this image in detail."},
            ],
        },
        {"role": "assistant", "content": []},
    ]

    inputs = processor(
        conversation=conversation, images=[image], return_tensors="pt"
    ).to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Assistant:", response)
