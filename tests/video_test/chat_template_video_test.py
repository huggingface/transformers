import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import image_utils
from transformers import video_utils

# test with llava-one-vission

# from transformers import LlavaOnevisionForConditionalGeneration

# model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
# processor = AutoProcessor.from_pretrained(model_id)

# messages = [
#     {
#       "role": "system",
#       "content": [{"type": "text", "text": "You are a friendly chatbot who always responds in the style of a pirate"}],
#     },
#     {
#       "role": "user",
#       "content": [
#             {"type": "video", "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"},
#             {"type": "text", "text": "What do you see in this video?"},
#         ],
#     },
# ]

# processed_chat = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     num_frames=1,
#     video_load_backend="decord",
#     truncation=True,
#     max_length=1024,
# )
# # print(processed_chat.keys())
# print(processed_chat['pixel_values_videos'].shape)
# processed_chat = {k: v.to(model.device) for k, v in processed_chat.items()}
# # # torch.cuda.empty_cache()
# generated_ids = model.generate(**processed_chat, do_sample=False, max_new_tokens=32)
# generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
# print(generated_texts[0])
# exit(0)

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

video, metadata = video_utils.load_video("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4")
# print(f"Video shape: {video.shape}")  # Should be (T, H, W, C) where T is the number of frames

# Video
conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "video", "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"},
            {"type": "video", "video": {'frames':video, 'metadata': metadata}},
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
    truncation=True,
    max_length=32768,
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])