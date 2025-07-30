from transformers import AutoProcessor
import time

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")

message = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/raid/raushan/Cooking_cake.mp4"},
            {"type": "text", "text": "What is shown in this video?"},
        ],
    },
]

start = time.perf_counter()
for _ in range(5):
    out_dict_with_video = processor.apply_chat_template(
        [message, message, message],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        num_frames=10,
        return_tensors="pt",
        video_load_backend="torchvision",
    )
print(time.perf_counter() - start)
