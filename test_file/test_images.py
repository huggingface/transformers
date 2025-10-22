from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

MODEL_PATH = "/opensource/GLM-4V-9B-HF-1023"
# MODEL_PATH = "/cloud/oss_checkpoints/zai-org/GLM-4.1V-9B-Thinking"
print(MODEL_PATH)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test_file/test.png",
            },
            {"type": "text", "text": "这是什么"},
            # {"type": "text", "text": "你好"},
        ],
    },
]
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, repetition_penalty=1.0, max_new_tokens=512, temperature=1.0)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
print(output_text)
breakpoint()
