import torch
from PIL import Image
from transformers import SmolVLMProcessor, AutoModelForVision2Seq
from transformers.models.smolvlm.video_processing_smolvlm import load_smolvlm_video

processor = SmolVLMProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", torch_dtype=torch.bfloat16, device_map="cuda")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "/raid/raushan/karate.mp4"},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    video_fps=1,
    num_frames=64,
    skip_secs=1.0,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
torch.save(inputs, "inputs_new.pt")

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])

out_text =  "Assistant: The video showcases a person in a white martial arts uniform lying on a mat in a gymnasium, performing a series of movements. The person is seen lying on their back with their arms extended, and their legs are bent at the knees, suggesting a relaxed posture. The gymnasium has large windows with blinds, and the walls are adorned with a pattern of vertical lines. The person's attire is casual, consisting of a white shirt and pants, and they are wearing a black belt."
new_out_text = "Assistant: The video showcases a sequence of images from a martial arts training session, focusing on a man in a white gi, likely a karate practitioner, who is seen in a training room with large windows and a gymnasium-like environment. The man is seen walking through the gym, with the camera following him, and then moving towards the camera, suggesting a shift in focus or direction. The environment is well-lit, with natural light streaming in from the windows, creating a bright and airy"
