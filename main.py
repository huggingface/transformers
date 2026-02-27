import torch

from transformers.models.omnivinci.modeling_omnivinci import OmniVinciForCausalLM
from transformers.models.omnivinci.processing_omnivinci import OmniVinciProcessor


model_path = "/fs/nexus-projects/JSALT_workshop/lasha/Dev/comni"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = OmniVinciForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    device_map="auto",
    load_audio_in_video=True,
    num_video_frames=128,
    audio_chunk_length="max_3600",
).eval()
processor = OmniVinciProcessor.from_pretrained(model_path, config=model.config, padding_side="left", use_fast=False)

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

inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_dict=True)

inputs["input_ids"] = inputs["input_ids"].to(model.device)
inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    do_sample=False,
)

generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
