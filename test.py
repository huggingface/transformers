import torch

from transformers import AutoProcessor
from transformers import PerceptionLMForConditionalGeneration

processor = AutoProcessor.from_pretrained("/checkpoint/vision_encoder/smhu/debug/plm_hf_1b", use_fast=True)
print(type(processor))

model = PerceptionLMForConditionalGeneration.from_pretrained("/checkpoint/vision_encoder/smhu/debug/plm_hf_1b").to(torch.bfloat16).to("cuda")
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "/home/smhu/code/occhi/apps/plm/dummy_datasets/image/images/14496_0.PNG",
            },
            {"type": "text", "text": "Describe the bar plot in the image."},
        ],
    }
]

print(model.config)


inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
original_token_ids = inputs["input_ids"].cpu().numpy().tolist()
token_ids = torch.load("/checkpoint/vision_encoder/smhu/debug/0/token_values_dump_0.pt")
desired_token_ids = token_ids.cpu().numpy().tolist()

assert original_token_ids == desired_token_ids

inputs = inputs.to(model.device)
torch.save(inputs['pixel_values'], "/checkpoint/vision_encoder/smhu/debug/0/pixel_values_dump_0.pt")
generate_ids = model.generate(**inputs, max_new_tokens=256)
# Remove input_ids from generate_ids to get only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generate_ids_without_inputs = generate_ids[:, input_length:]

print(generate_ids_without_inputs.cpu().numpy().tolist())
for output in processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True):
    print(output)
