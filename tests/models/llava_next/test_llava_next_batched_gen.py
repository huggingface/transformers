import torch
from huggingface_hub import hf_hub_download
import requests
from PIL import Image

from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from transformers.models.llava_next.modeling_better_llava_next import BetterLlavaNextForConditionalGeneration


device = "cuda:0"
model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_path)
processor.tokenizer.padding_side = "left"

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda",
)


# ! Differnt images, same prompt

cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)

prompts = [
    "[INST] <image>\nWhat is shown in this image? [/INST]",
    "[INST] <image>\nWhat is shown in this image? [/INST]",
]
inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
processor.tokenizer.padding_side = "left"
output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id, padding_side=processor.tokenizer.padding_side)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))

"""
[INST]  
What is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multivariate chart that displays values for multiple variables represented on axes starting from the same point. This particular radar chart is showing the performance of different models or systems across various metrics.

The axes represent different metrics or benchmarks, such as MM-Vet, MM-Vet-GQA, MM-Vet-GQA-VizWiz, LLaVa-Bench, SLED-Bench, and several others. Each axis is labeled with the name of the metric and a numerical value, which likely represents a score or a performance measure.

The colored areas within the chart represent different models or systems, such as MME, BLIP-2, InstructionBLIP, and others. The size of the area on each axis indicates the performance of the model or system on that particular metric.

The chart is color-coded to differentiate between the different models or systems, and it provides a visual comparison of their performance across the various metrics. This kind of chart is often used in machine learning and artificial intelligence to compare the performance of different models or algorithms. 
[INST]  
What is shown in this image? [/INST] The image shows two cats lying on a pink blanket. The cat on the left is curled up in a relaxed position, while the cat on the right is stretched out with its head resting on the blanket. There is a remote control next to the cat on the left, suggesting that this scene might be taking place in a living room or a similar space where people might watch television. The cats appear to be sleeping or resting. 
"""


# Differnt image, different prompt
prompts = [
    "[INST] <image>\nWhat is shown in this image? [/INST]",
    "[INST] <image>\nDescribe it [/INST]",
]
inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
processor.tokenizer.padding_side = "left"
output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id, padding_side=processor.tokenizer.padding_side)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))

"""
[INST]  
What is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multivariate chart that displays values for multiple variables represented on axes starting from the same point. This particular radar chart is showing the performance of different models or systems across various metrics.

The axes represent different metrics or benchmarks, such as MM-Vet, MM-Vet-GQA, MM-Vet-GQA-VizWiz, LLaVa-Bench, SLED-Bench, and several others. Each axis is labeled with the name of the metric and a numerical value, which likely represents a score or a performance measure.

The colored areas within the chart represent different models or systems, such as MME, BLIP-2, InstructionBLIP, and others. The size of the area on each axis indicates the performance of the model or system on that particular metric.

The chart is color-coded to differentiate between the different models or systems, and it provides a visual comparison of their performance across the various metrics. This kind of chart is often used in machine learning and artificial intelligence to compare the performance of different models or algorithms. 
[INST]  
Describe it [/INST] In Detail:

In the tranquil setting of a cozy living room, two feline companions are captured in a moment of serene slumber. The first cat, a tabby with a coat of gray and black stripes, is curled up on the left side of the pink blanket that adorns the red couch. Its head is comfortably nestled on the armrest of the couch, suggesting a sense of security and contentment.

On the right side of the blanket, the second cat, a calico with a coat of white, black, and orange, is also curled up. Its head is resting on the armrest of the couch, mirroring the first cat's position. The two cats are positioned in such a way that they are facing each other, creating a sense of companionship and mutual trust.

The red couch on which they are sleeping is adorned with a pink blanket, adding a touch of warmth and comfort to the scene. On the armrest of the couch, there's a remote control, perhaps indicating that the cats' human companion was recently watching television before drifting off to sleep.

The image captures a peaceful moment in the lives of these two cats, who seem to be enjoying a quiet afternoon nap on their favorite couch. 
"""




# ! same image, different prompt
scene_img = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)

prompts = [
    f"[INST] <image> What are the things I should be cautious about when I visit here? [/INST]",
    f"[INST] <image> Describe what you see. [/INST]",
]
inputs = processor(prompts, [scene_img] * 2, return_tensors='pt', padding=True).to("cuda")

output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id, padding_side=processor.tokenizer.padding_side)

for o in output:
    print(processor.decode(o, skip_special_tokens=True))
"""
[INST]   What are the things I should be cautious about when I visit here? [/INST] When visiting a location like the one shown in the image, which appears to be a serene lake with a dock and surrounded by forest, there are several things you should be cautious about:

1. **Water Safety**: If you plan to swim or engage in water activities, make sure you are aware of the water's depth and currents. Lakes can have unseen hazards like underwater rocks or sudden drop-offs.

2. **Wildlife**: Forested areas can be home to wildlife. Be aware of your surroundings and know what to do if you encounter animals. Do not feed or approach wildlife.

3. **Weather Conditions**: Mountain weather can change rapidly. Check the forecast before you go and be prepared for sudden changes in weather.

4. **Navigation**: If you plan to hike or explore the area, make sure you have a map and compass or a GPS device. It's easy to get lost in natural settings.

5. **Leave No Trace**: Be mindful of your impact on the environment. Take all your trash with you, stay on marked trails, and respect the natural habitat.

6. **Emergency Preparedness**: Have a basic first aid kit and know how to use it. Also, have a way to contact emergency services if needed.

7. **Dress Appropriately**: Wear layers and sturdy footwear to protect against the elements and potential hazards.

8. **Hydration and Nutrition**: Bring enough water and food for your trip, especially if you're planning to be out for an extended period.

9. **Local Regulations**: Familiarize yourself with any local regulations or restrictions, such as fishing licenses or campfire rules.

10. **Respect Other Visitors**: Be considerate of other people visiting the area. Keep noise to a minimum and give others space to enjoy the natural beauty.

Remember to always let someone know where you are going and when you expect to return, especially if you're venturing into remote areas. 
[INST]   Describe what you see. [/INST] The image shows a serene natural setting. In the foreground, there is a wooden dock extending into a calm body of water, which appears to be a lake. The dock is made of weathered planks and has a simple, rustic appearance. The water reflects the sky and the surrounding landscape, creating a mirror-like effect.

In the background, there is a range of mountains with snow-capped peaks, suggesting a high-altitude location. The mountains are lush with green vegetation, indicating that the photo was likely taken during a season when the vegetation is in full bloom.

The sky is overcast with a soft, diffused light, which gives the scene a tranquil and somewhat ethereal quality. The overall color palette is dominated by shades of blue, green, and gray, which contribute to the cool and serene atmosphere of the image.

There are no visible texts or human-made structures that provide additional context or information about the location. The image is a beautiful representation of a natural landscape, capturing the harmony between the man-made dock and the natural beauty of the mountains and the lake. 
"""


