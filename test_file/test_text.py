from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/mnt/GLM-4.5V-0804"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."},
        ],
    }
]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
)
breakpoint()
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
inputs.pop("token_type_ids", None)
inputs.pop("attention_mask", None)
inputs = inputs.to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print(output_text)
