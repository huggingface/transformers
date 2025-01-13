import av
import torch
import decord
from decord import VideoReader, cpu

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, SiglipImageProcessor

model_id = "/raid/raushan/llava-next-video-qwen-7b"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id, torch_dtype=torch.bfloat16)
img_proc = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

image = Image.open("/raid/raushan/image.png")


def load_video(video_path, max_frames_num,fps=1,force_sample=False):

    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    print(spare_frames.shape)
    return spare_frames,frame_time,video_time


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# <image>Time farmes are this moments and we ahev 64 frames
# Please describe this video in detail.<|im_end|>
# <|im_start|>assistant

conversation = [
    {

        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."},
            ],
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "The video lasts for 19.97 seconds, and 64 frames are uniformly sampled from it. These frames are located at 0.00s,0.30s,0.60s,0.93s,1.23s,1.57s,1.87s,2.20s,2.50s,2.83s,3.13s,3.47s,3.77s,4.10s,4.40s,4.73s,5.03s,5.37s,5.67s,6.00s,6.30s,6.63s,6.93s,7.27s,7.57s,7.90s,8.20s,8.53s,8.83s,9.17s,9.47s,9.80s,10.10s,10.43s,10.73s,11.07s,11.37s,11.70s,12.00s,12.33s,12.63s,12.97s,13.27s,13.60s,13.90s,14.23s,14.53s,14.87s,15.17s,15.50s,15.80s,16.13s,16.43s,16.77s,17.07s,17.40s,17.70s,18.03s,18.33s,18.67s,18.97s,19.30s,19.60s,19.93s.Please answer the following questions related to this video.\nPlease describe this video in detail."},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<video>The video lasts for 19.97 seconds, and 64 frames are uniformly sampled from it. These frames are located at 0.00s,0.30s,0.60s,0.93s,1.23s,1.57s,1.87s,2.20s,2.50s,2.83s,3.13s,3.47s,3.77s,4.10s,4.40s,4.73s,5.03s,5.37s,5.67s,6.00s,6.30s,6.63s,6.93s,7.27s,7.57s,7.90s,8.20s,8.53s,8.83s,9.17s,9.47s,9.80s,10.10s,10.43s,10.73s,11.07s,11.37s,11.70s,12.00s,12.33s,12.63s,12.97s,13.27s,13.60s,13.90s,14.23s,14.53s,14.87s,15.17s,15.50s,15.80s,16.13s,16.43s,16.77s,17.07s,17.40s,17.70s,18.03s,18.33s,18.67s,18.97s,19.30s,19.60s,19.93s.Please answer the following questions related to this video.\nPlease describe this video in detail.<|im_end|>\n<|im_start|>assistant"

video_path = "/raid/raushan/karate.mp4" # hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video, can sample more for longer videos
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 64).astype(int)
clip = read_video_pyav(container, indices)

clip, frame_time,video_time = load_video(video_path, max_frames_num=64, force_sample=True)
inputs_video = processor(text=prompt, videos=clip, return_tensors="pt").to(device=model.device, dtype=torch.bfloat16)

output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
