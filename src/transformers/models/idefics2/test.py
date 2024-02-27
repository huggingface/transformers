import numpy as np
import requests
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.models.idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration

from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from transformers.utils import logging

logging.set_verbosity_info()


checkpoint = "HuggingFaceM4/idefics2"

device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")
# DEVICE = torch.device("cpu")
processor = AutoProcessor.from_pretrained(checkpoint)

# TODO - make loading work with eager attention implementation
model_0 = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device_0)

model_1 = Idefics2ForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
).to(device_1)

image_seq_len = model_0.config.perceiver_config.resampler_n_latents
BOS_TOKEN = processor.tokenizer.bos_token
BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

# The processor is the same as the Idefics processor except for the BILINEAR interpolation,
# so this is a hack in order to redefine ONLY the transform method
def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)

    height, width = x.shape[:2]
    aspect_ratio = width / height
    if width >= height and width > 980:
        width = 980
        height = int(width / aspect_ratio)
    elif height > width and height > 980:
        height = 980
        width = int(height * aspect_ratio)
    width = max(width, 378)
    height = max(height, 378)

    x = resize(x, (height, width), resample=PILImageResampling.BILINEAR)
    x = processor.image_processor.rescale(x, scale=1 / 255)
    x = processor.image_processor.normalize(
        x,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x

url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
image = processor.image_processor.fetch_images([url])[0]

inputs = processor.tokenizer(
    f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
    return_tensors="pt",
    add_special_tokens=False,
)
inputs["pixel_values"] = processor.image_processor([image], transform=custom_transform)
inputs_0 = {k: v.to(device_0) for k, v in inputs.items()}
inputs_1 = {k: v.to(device_1) for k, v in inputs.items()}

with torch.no_grad():
    outputs_0 = model_0(**inputs_0, output_hidden_states=True)
    outputs_1 = model_1(**inputs_1, output_hidden_states=True)


# prefix = "remote"
# np.save(f"{prefix}_logits", outputs_0.logits.to(torch.float32).cpu().numpy())
# for i in range(len(outputs_0.hidden_states)):
#     np.save(f"{prefix}_hidden_states_{str(i).zfill(2)}", outputs_0.hidden_states[i].to(torch.float32).cpu().numpy())

# prefix = "local"
# np.save(f"{prefix}_logits", outputs_1.logits.to(torch.float32).cpu().numpy())
# for i in range(len(outputs_1.hidden_states)):
#     np.save(f"{prefix}_hidden_states_{str(i).zfill(2)}", outputs_1.hidden_states[i].to(torch.float32).cpu().numpy())


def max_diff(a, b):
    if isinstance(a, torch.Tensor):
        a = a.to(torch.float32).cpu().numpy()

    if isinstance(b, torch.Tensor):
        b = b.to(torch.float32).cpu().numpy()

    return np.amax(np.abs(a - b))


print("Max diff logits", max_diff(outputs_0.logits, outputs_1.logits))

for i in range(len(outputs_0.hidden_states)):
    print(f"Max diff hidden_states_{i}", max_diff(outputs_0.hidden_states[i], outputs_1.hidden_states[i]))
