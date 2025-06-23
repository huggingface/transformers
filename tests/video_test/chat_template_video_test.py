import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import image_utils
from transformers import video_utils

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# conversation = [
#     {
#         "role": "user",
#         "content":[
#             # {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
#             {"type": "image", "image": image_utils.load_image("http://images.cocodataset.org/val2017/000000039769.jpg")},
#             {"type": "text", "text": "Describe this image."}
#         ]
#     }
# ]

# inputs = processor.apply_chat_template(
#     conversation,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device, dtype=torch.bfloat16)

# output_ids = model.generate(**inputs, max_new_tokens=128)
# generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
# print(generated_texts)

video, _ = video_utils.load_video("./test_video.mp4")
# print(f"Video shape: {video.shape}")  # Should be (T, H, W, C) where T is the number of frames

# Video
conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "video", "path": "./test_video.mp4"},
            {"type": "video", "video": video},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])