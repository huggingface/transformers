from transformers.models.deepseek_vl_v2.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM
from transformers.models.deepseek_vl_v2.processing_deepseek_vl_v2 import (
    DeepseekVLV2Processor,
)
from PIL import Image

image = Image.open("./test.png")

if __name__ == "__main__":
    model = DeepseekVLV2ForCausalLM.from_pretrained("deepseek-ai/deepseek-vl2")
    processor = DeepseekVLV2Processor.from_pretrained("deepseek-ai/deepseek-vl2")

    inputs = processor(text="a photo of a cat", images=image, return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(processor.batch_decode(outputs, skip_special_tokens=True))
