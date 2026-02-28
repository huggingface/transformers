from transformers import AutoModel, AutoProcessor


model_path = "SreyanG-NVIDIA/omnivinci-hf"

model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    load_audio_in_video=True,
    num_video_frames=128,
    audio_chunk_length="max_3600",
).eval()
processor = AutoProcessor.from_pretrained(model_path, padding_side="left", use_fast=False)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "nvidia.mp4"},
            {
                "type": "text",
                "text": "Assess the video, followed by a detailed description of it's video and audio contents.",
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=False,
)

generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
