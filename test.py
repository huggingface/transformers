from transformers import AutoProcessor
from transformers import PerceptionLMForConditionalGeneration


MODEL_PATH = "/checkpoint/vision_encoder/smhu/debug/plm_hf_1b"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = PerceptionLMForConditionalGeneration.from_pretrained(MODEL_PATH).to("cuda")
conversation1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "/home/smhu/code/occhi/apps/plm/dummy_datasets/image/images/89_0.PNG",
            },
            {"type": "text", "text": "Describe the bar plot in the image."},
        ],
    }
]
conversation2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "/home/smhu/code/occhi/apps/plm/dummy_datasets/video/videos/GUWR5TyiY-M_000012_000022.mp4",
            },
            {"type": "text", "text": "Can you describe the video in detail?"},
        ],
    }
]
# print(model.config)
inputs = processor.apply_chat_template(
    [conversation1, conversation2],
    num_frames=32,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_load_backend="decord",
    padding=True,
    padding_side="left",
)
inputs = inputs.to(model.device)
# torch.save(inputs['pixel_values'], "/checkpoint/vision_encoder/smhu/debug/0/pixel_values_dump_0.pt")
generate_ids = model.generate(**inputs, max_new_tokens=256)
# Remove input_ids from generate_ids to get only the newly generated tokens
input_length = inputs["input_ids"].shape[1]
generate_ids_without_inputs = generate_ids[:, input_length:]

# print(generate_ids_without_inputs.cpu().numpy().tolist())
for output in processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True):
    print(output)
