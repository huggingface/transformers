# from PIL import Image
# import requests
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# 
# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# 
# prompt = "USER: <image>\nWhat's the content of the image? <image> ASSISTANT:"
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# 
# inputs = processor(text=[prompt], images=[image, image], return_tensors="pt")
# for k, v in inputs.items():
#     print(k, v.shape)
# 
# processor.vision_feature_select_strategy = "default"
# processor.patch_size = 14
# inputs_expanded = processor(text=[prompt], images=[image, image], return_tensors="pt")
# for k, v in inputs_expanded.items():
#     print(k, v.shape)
# 
# # Generate
# generate_ids = model.generate(**inputs, max_new_tokens=15)
# generate_ids_expanded = model.generate(**inputs_expanded, max_new_tokens=15)
# assert(generate_ids_expanded == generate_ids)



# from PIL import Image
# import requests
# from transformers import AutoProcessor, LlavaNextForConditionalGeneration
# 
# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# 
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
# cats_image = Image.open(requests.get(url, stream=True).raw)
# lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)
# 
# very_wide_url = "https://as2.ftcdn.net/v2/jpg/02/84/07/89/1000_F_284078927_VW1YQ7jCb7Xz8cWkd7nUytk1j3KCCHcY.jpg"
# very_wide_image = Image.open(requests.get(very_wide_url, stream=True).raw)
# 
# prompts = [
#     "[INST] <image>\nWhat is shown in this image? [/INST]",
#     "[INST] <image>\nDescribe the image. [/INST]",
#     "[INST] <image>\nHow many cats do you see in this image? [/INST]",
#     ]
# processor.vision_feature_select_strategy = "default"
# processor.patch_size = 14
# inputs = processor(text=prompts, images=[very_wide_image, lowres_img, cats_image], padding=True, return_tensors="pt")
# for k, v in inputs.items():
#     print(k, v.shape)
# 
# # Generate
# generate_ids = model.generate(**inputs, max_new_tokens=30)
# out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(out)
# 
# 
# import torch
# from PIL import Image
# import requests
# from transformers import AutoProcessor, VipLlavaForConditionalGeneration
# 
# model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", device_map="auto", torch_dtype=torch.float16)
# processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")
# 
# prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
# question = "Can you please describe this image?"
# prompt = prompt.format(question)
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"
# image = Image.open(requests.get(url, stream=True).raw)
# 
# inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
# 
# # Generate
# generate_ids = model.generate(**inputs, max_new_tokens=20)
# out = processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
# print(out)
# 

from PIL import Image
import requests
import numpy as np
import av
from huggingface_hub import hf_hub_download
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


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

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

prompt = "USER: <video>Why is this video funny? ASSISTANT:"
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

processor.vision_feature_select_strategy = "default"
processor.patch_size = 14

inputs = processor(text=prompt, videos=clip, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)


# to generate from image and video mix
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = [
#         "USER: <image> How many cats are there in the image? ASSISTANT:",
#         "USER: <video>Why is this video funny? ASSISTANT:"
#      ]
# inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")
# 
# # Generate
# generate_ids = model.generate(**inputs, max_new_tokens=50)
# out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)