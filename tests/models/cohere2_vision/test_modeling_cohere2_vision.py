"""Testing suite for the PyTorch Cohere2Vision model."""

import unittest

import pytest
import torch
from PIL import Image

from transformers import AutoModel, AutoProcessor, Cohere2VisionConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    from transformers import Cohere2VisionForConditionalGeneration, Cohere2VisionPreTrainedModel


class Cohere2VisionModelTester:
    def __init__(self, parent):
        self.parent = parent

    def get_small_vision_config(self):
        return Cohere2VisionConfig(
            vision_config={
                "hidden_size": 16,
                "image_size": 32,
                "intermediate_size": 16,
                "num_attention_heads": 4,
                "num_hidden_layers": 4,
            },
            text_config={
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_small_vision_config()
        pixel_values = torch.rand(2, 3, 32, 32)
        input_ids = torch.randint(0, 100, (2, 8))
        image_num_patches = torch.tensor([1, 1])
        return config, pixel_values, input_ids, image_num_patches

    def create_and_check_model(self):
        config, pixel_values, input_ids, image_num_patches = self.prepare_config_and_inputs()
        model = Cohere2VisionForConditionalGeneration(config)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, pixel_values=pixel_values, image_num_patches=image_num_patches)
        assert result.logits is not None


@require_torch
class Cohere2VisionModelTest(unittest.TestCase):
    all_model_classes = (Cohere2VisionForConditionalGeneration, Cohere2VisionPreTrainedModel) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Cohere2VisionModelTester(self)

    def test_model(self):
        self.model_tester.create_and_check_model()


@pytest.fixture(scope="session")
def model_id():
    return "CohereLabs/command-a-vision-07-2025"


@pytest.fixture(scope="session")
def processor(model_id):
    return AutoProcessor.from_pretrained(model_id)


@pytest.fixture(scope="session")
def model(model_id):
    return AutoModel.from_pretrained(model_id, device_map="auto")


def open_img(img_path):
    return Image.open(img_path).convert("RGB")

def test_model_generation(processor, model):
    # Prepare conversations
    conversation1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": open_img("tests/test_images/image_resized.png")},
                {"type": "text", "text": "Describe this image"},
            ],
        },
    ]

    def test_fn1(x):
        return "golden retriever" in x.lower()

    conversation2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": open_img("tests/test_images/v1_93.jpg")},
                {"type": "text", "text": "Describe this image"},
            ],
        },
    ]

    def test_fn2(x):
        return "connecticut law of 1642" in x.lower()

    conversation3 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the capital of France?"},
            ],
        },
    ]

    def test_fn3(x):
        return "paris" in x.lower()

    # Batch of conversations to test
    batch_of_conversations = [
        conversation1,
        conversation2,
        conversation3,
    ]

    test_functions = [test_fn1, test_fn2, test_fn3]

    # Process the conversations
    inputs = processor.apply_chat_template(
        batch_of_conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    inputs["image_sizes"] = inputs["image_num_patches"]

    # Generate outputs
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, top_k=1, num_return_sequences=1, use_cache=False
        )

    # Decode the outputs
    decoded_outputs = []
    for i in range(len(output)):
        decoded_output = processor.tokenizer.decode(
            output[i][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        decoded_outputs.append(decoded_output)
        print(f"Output for conversation {i + 1}: {decoded_output}\n")

    # Assertions to verify the outputs
    assert len(decoded_outputs) == len(test_functions), "Number of outputs does not match expected."

    for idx, (generated, test_fn) in enumerate(zip(decoded_outputs, test_functions), start=1):
        assert test_fn(generated), f"Output for conversation {idx} does not match expected."
