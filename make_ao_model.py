import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig, Qwen3VLMoeForConditionalGeneration, AutoProcessor

# Configure logging to see warnings and debug information
logging.basicConfig(
    level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s"
)

# Enable specific loggers that might contain the serialization warnings
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("torchao").setLevel(logging.INFO)
logging.getLogger("safetensors").setLevel(logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.INFO)

model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    IntxWeightOnlyConfig,
    PerRow,
    PerAxis,
    FqnToConfig,
    Float8Tensor,
    Int4TilePackedTo4dTensor,
    IntxUnpackedToInt8Tensor,
    Float8Tensor,
)
from torchao.quantization.quantize_.common import KernelPreference


float8dyn = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow(), kernel_preference=KernelPreference.FBGEMM)

qconfig_dict = {
    r"re:model\.language_model\.layers\.3.mlp.experts.gate_up_proj": float8dyn,
    # "model.language_model.layers.3.mlp.gate": float8dyn,
}
quant_config = FqnToConfig(qconfig_dict)
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)

quantized_model.model.language_model.layers[3].mlp.experts.gate_up_proj.qdata = quantized_model.model.language_model.layers[3].mlp.experts.gate_up_proj.qdata.transpose(1, 2).contiguous()

quantized_model.save_pretrained("Qwen3-VL-30B-A3B-Instruct-quantized")
del quantized_model


quantized_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen3-VL-30B-A3B-Instruct-quantized", 
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# print(quantized_model.quantization_config)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Inference: Generation of the output
generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
