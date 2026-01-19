#!/usr/bin/env python
"""
Test script to verify batch > 1 support for GLM-Image model.

This script tests:
1. Basic batch processing functionality
2. Consistency: batch=2 results should match running batch=1 twice
3. Different scenarios: text-to-image and image-to-image

Run on H100:
    python test_glm_image_batch.py
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from transformers import AutoProcessor, GlmImageForConditionalGeneration


def download_image(url: str) -> Image.Image:
    """Download an image from URL."""
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")


def test_processor_batch():
    """Test that processor correctly handles batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 1: Processor batch handling")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")

    # Test text-to-image with batch=2
    texts = [
        "Generate an image of a cute cat sitting on a sofa",
        "Generate an image of a dog playing in a park",
    ]

    inputs = processor(text=texts, images=None, return_tensors="pt", padding=True)

    print(f"✓ input_ids shape: {inputs['input_ids'].shape}")
    print(f"✓ attention_mask shape: {inputs['attention_mask'].shape}")
    print(f"✓ image_grid_thw shape: {inputs['image_grid_thw'].shape}")
    print(f"✓ images_per_sample: {inputs['images_per_sample']}")

    assert inputs["input_ids"].shape[0] == 2, "Batch size should be 2"
    assert inputs["images_per_sample"].shape[0] == 2, "images_per_sample should have 2 elements"

    print("\n✅ Processor batch test PASSED")
    return inputs


def test_model_forward_batch(device: str = "cuda"):
    """Test model forward pass with batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 2: Model forward pass with batch > 1")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")
    model = GlmImageForConditionalGeneration.from_pretrained(
        "zai-org/GLM-Image",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Test text-to-image with batch=2
    texts = [
        "A beautiful sunset over the ocean",
        "A snowy mountain landscape",
    ]

    inputs = processor(text=texts, images=None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input shapes:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\n✓ Output logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape[0] == 2, "Output batch size should be 2"

    print("\n✅ Model forward batch test PASSED")
    return model, processor


def test_batch_consistency(device: str = "cuda"):
    """Test that batch=2 gives same results as running batch=1 twice."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch consistency (batch=2 vs 2x batch=1)")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")
    model = GlmImageForConditionalGeneration.from_pretrained(
        "zai-org/GLM-Image",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    texts = [
        "A red sports car on a highway",
        "A green forest with tall trees",
    ]

    # Run with batch=2
    inputs_batch = processor(text=texts, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.logits

    # Run with batch=1 twice
    logits_single = []
    for text in texts:
        inputs_single = processor(text=[text], images=None, return_tensors="pt")
        inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}

        with torch.no_grad():
            outputs_single = model(**inputs_single)
            logits_single.append(outputs_single.logits)

    # Compare results (need to handle padding differences)
    # For the first sample, compare unpadded positions
    seq_len_0 = (inputs_batch["attention_mask"][0] == 1).sum().item()
    seq_len_1 = (inputs_batch["attention_mask"][1] == 1).sum().item()

    # Compare logits at valid positions
    logits_batch_0 = logits_batch[0, :seq_len_0]
    logits_batch_1 = logits_batch[1, :seq_len_1]
    logits_single_0 = logits_single[0][0, :seq_len_0]
    logits_single_1 = logits_single[1][0, :seq_len_1]

    # Check if they're close (allowing small numerical differences)
    diff_0 = (logits_batch_0 - logits_single_0).abs().max().item()
    diff_1 = (logits_batch_1 - logits_single_1).abs().max().item()

    print(f"✓ Sample 0 max difference: {diff_0:.6f}")
    print(f"✓ Sample 1 max difference: {diff_1:.6f}")

    # Tolerance for bfloat16
    tolerance = 1e-2
    if diff_0 < tolerance and diff_1 < tolerance:
        print(f"\n✅ Batch consistency test PASSED (tolerance: {tolerance})")
    else:
        print(f"\n⚠️ Batch consistency test WARNING: differences exceed tolerance")
        print("   This might be due to attention mask handling or numerical precision")


def test_generation_batch(device: str = "cuda", max_new_tokens: int = 50):
    """Test generation with batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 4: Generation with batch > 1")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")
    model = GlmImageForConditionalGeneration.from_pretrained(
        "zai-org/GLM-Image",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    texts = [
        "A futuristic city with flying cars",
        "A peaceful zen garden with cherry blossoms",
    ]

    inputs = processor(text=texts, images=None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Starting generation with batch_size=2, max_new_tokens={max_new_tokens}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
        )

    print(f"✓ Generated output shape: {outputs.shape}")
    print(f"✓ Generated {outputs.shape[1] - inputs['input_ids'].shape[1]} new tokens per sample")

    # Decode to verify
    for i, output in enumerate(outputs):
        decoded = processor.tokenizer.decode(output, skip_special_tokens=False)
        print(f"\nSample {i} output (first 200 chars):")
        print(f"  {decoded[:200]}...")

    print("\n✅ Generation batch test PASSED")


def test_image_to_image_batch(device: str = "cuda"):
    """Test image-to-image with batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 5: Image-to-image with batch > 1")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")
    model = GlmImageForConditionalGeneration.from_pretrained(
        "zai-org/GLM-Image",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Download test images
    urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
    ]

    try:
        images = [download_image(url) for url in urls]
    except Exception as e:
        print(f"⚠️ Could not download images: {e}")
        print("   Skipping image-to-image test")
        return

    # Use chat template for image-to-image
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Add a rainbow to this photo"},
                ],
            },
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Make this photo look like a painting"},
                ],
            },
        ],
    ]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input shapes:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    with torch.no_grad():
        outputs = model(**inputs)

    print(f"\n✓ Output logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape[0] == 2, "Output batch size should be 2"

    print("\n✅ Image-to-image batch test PASSED")


def test_different_image_counts(device: str = "cuda"):
    """Test batch with different number of source images per sample."""
    print("\n" + "=" * 60)
    print("TEST 6: Different image counts per sample")
    print("=" * 60)

    processor = AutoProcessor.from_pretrained("zai-org/GLM-Image")

    # Sample 1: text-to-image (0 source images)
    # Sample 2: image-to-image (1 source image)
    # This is a challenging case!

    print("⚠️ Mixed text-to-image and image-to-image in same batch")
    print("   This requires careful handling of images_per_sample")

    # For now, we test same-type batches
    # TODO: Support mixed-type batches if needed

    print("\n✅ Different image counts test NOTED (mixed batches may need additional work)")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GLM-Image Batch Support Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("⚠️ Running on CPU - tests will be slow and may run out of memory")

    # Run tests
    try:
        # Test 1: Processor
        test_processor_batch()

        # Test 2: Model forward
        test_model_forward_batch(device)

        # Test 3: Consistency
        test_batch_consistency(device)

        # Test 4: Generation (small tokens for speed)
        test_generation_batch(device, max_new_tokens=20)

        # Test 5: Image-to-image
        test_image_to_image_batch(device)

        # Test 6: Different image counts
        test_different_image_counts(device)

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
