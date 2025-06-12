import gc
from io import BytesIO

import pytest
import requests
import torch
from PIL import Image

from transformers import AutoProcessor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA GPU")
def test_auto_processor_does_not_leak_gpu_memory():
    """Test whether AutoProcessor releases GPU memory properly."""

    torch.cuda.empty_cache()
    start_memory = torch.cuda.memory_allocated()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        use_fast=True,
        trust_remote_code=False,
        revision=None,
    )

    url = "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    result = processor(
        text=["<|im_start|>user\nWhat’s in this image?<|vision_start|><|image_pad|><|vision_end|>"],
        images=[image],
        return_tensors="pt",
        padding=True,
        device="cuda",
    )

    del result
    del processor
    torch.cuda.empty_cache()
    gc.collect()

    end_memory = torch.cuda.memory_allocated()

    # Allow up to 20MB tolerance due to internal caching
    assert (end_memory - start_memory) < 20 * 1024 * 1024, (
        f"Expected <20MB memory leak, but got: {(end_memory - start_memory) / (1024 * 1024):.2f}MB"
    )
