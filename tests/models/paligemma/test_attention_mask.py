import unittest
import torch
import requests
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

@require_torch
@require_vision
class TestPaliGemmaAttentionMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
        cls.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-pt-224",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def test_pad_tokens_remain_masked(self):
        # Test with different length prompts to force padding
        prompts = ["<image>caption en", "<image>caption en " + "w" * 50]
        labels = ["short text", "longer text here"]
        
        # Use test image from HF test fixtures
        image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/shoe.png"
        raw_image = Image.open(requests.get(image_url, stream=True).raw)
        raw_images = [raw_image] * 2
        
        # Process with padding
        inputs = self.processor(
            text=prompts,
            images=raw_images,
            suffix=labels,
            return_tensors="pt",
            padding="longest"
        ).to(torch_device)
        
        # Get attention mask from model
        causal_mask = self.model._update_causal_mask(
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            past_key_values=None,
            cache_position=torch.tensor([0], device=torch_device),
            input_ids=inputs["input_ids"],
            inputs_embeds=None,
            is_training=True
        )
        
        # Get the padding positions from attention mask
        pad_positions = (inputs["attention_mask"] == 0).to(causal_mask.device)
        
        # For each position that should be padded
        for batch_idx in range(causal_mask.size(0)):
            for seq_idx in range(causal_mask.size(2)):
                # Check if this position should be padded
                if pad_positions[batch_idx, seq_idx]:
                    # Verify this position is properly masked in the causal mask
                    self.assertTrue(
                        torch.all(causal_mask[batch_idx, :, :, seq_idx] == torch.finfo(causal_mask.dtype).min),
                        f"Found unmasked padding token at batch {batch_idx}, sequence position {seq_idx}"
                    )
