import numpy as np
import requests
from PIL import Image

from transformers import LlavaImageProcessor


processor = LlavaImageProcessor()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

result1 = processor.pad_to_square_original(image=image)
result1 = np.array(result1)

print("Shape of result1:", result1.shape)

result2 = processor.pad_to_square(np.array(image))

assert result1.shape == result2.shape

np.testing.assert_allclose(result1, result2)
