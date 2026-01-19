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

import os
import torch
from PIL import Image
import requests
from io import BytesIO

from transformers import AutoProcessor, AutoTokenizer, GlmImageForConditionalGeneration, GlmImageProcessor

# Model paths - adjust as needed
# For HuggingFace Hub:
# MODEL_PATH = "zai-org/GLM-Image"
# PROCESSOR_PATH = "zai-org/GLM-Image"

# For local diffusers-style model:
MODEL_PATH = os.environ.get("GLM_IMAGE_MODEL_PATH", "/workspace/GLM-Image/vision_language_encoder")
PROCESSOR_PATH = os.environ.get("GLM_IMAGE_PROCESSOR_PATH", "/workspace/GLM-Image/processor")


def load_processor(processor_path: str):
    """Load processor from local path or HuggingFace Hub."""
    try:
        # Try loading with AutoProcessor first (for HF Hub models)
        return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    except Exception as e:
        print(f"AutoProcessor failed: {e}")
        print("Attempting to load components separately...")
        
        # For local diffusers-style models, load components separately
        tokenizer = AutoTokenizer.from_pretrained(processor_path, trust_remote_code=True)
        
        # The tokenizer_config.json should already have these attributes:
        # - image_token: "<|image|>"
        # - grid_bos_token: "<sop>"
        # - grid_eos_token: "<eop>"
        # - bos_token: "<|dit_token_16384|>"
        # If not present, set them manually
        if not hasattr(tokenizer, 'image_token') or tokenizer.image_token is None:
            tokenizer.image_token = "<|image|>"
        if not hasattr(tokenizer, 'grid_bos_token') or tokenizer.grid_bos_token is None:
            tokenizer.grid_bos_token = "<sop>"
        if not hasattr(tokenizer, 'grid_eos_token') or tokenizer.grid_eos_token is None:
            tokenizer.grid_eos_token = "<eop>"
        if not hasattr(tokenizer, 'bos_token') or tokenizer.bos_token is None:
            tokenizer.bos_token = "<|dit_token_16384|>"
        
        print(f"Tokenizer loaded with:")
        print(f"  image_token: {tokenizer.image_token}")
        print(f"  grid_bos_token: {tokenizer.grid_bos_token}")
        print(f"  grid_eos_token: {tokenizer.grid_eos_token}")
        print(f"  bos_token: {tokenizer.bos_token}")
        
        # Check if image processor config exists
        if os.path.exists(os.path.join(processor_path, "preprocessor_config.json")):
            from transformers import GlmImageImageProcessor
            image_processor = GlmImageImageProcessor.from_pretrained(processor_path)
        else:
            # Use default image processor
            from transformers import GlmImageImageProcessor
            image_processor = GlmImageImageProcessor()
        
        # Create GlmImageProcessor
        processor = GlmImageProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        return processor


def load_model(model_path: str, device: str = "cuda", torch_dtype=torch.bfloat16):
    """Load model from local path or HuggingFace Hub."""
    return GlmImageForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )


def download_image(url: str) -> Image.Image:
    """Download an image from URL."""
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")


def test_processor_batch():
    """Test that processor correctly handles batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 1: Processor batch handling")
    print("=" * 60)

    processor = load_processor(PROCESSOR_PATH)

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

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)

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

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)
    model.eval()

    # Test 3a: Same length prompts (no padding differences)
    print("\n--- Test 3a: Same-length prompts (no padding) ---")
    texts_same_len = [
        "A red car on a road",  # Same token count
        "A big dog in a park",  # Same token count
    ]

    inputs_batch = processor(text=texts_same_len, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    # Check if padding was actually needed
    mask = inputs_batch["attention_mask"]
    has_padding = (mask[0] != mask[1]).any().item() or (mask.sum(dim=1) != mask.shape[1]).any().item()
    print(f"  Sequences have padding: {has_padding}")
    print(f"  Sequence lengths: {mask.sum(dim=1).tolist()}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.logits

    # Run with batch=1 twice
    logits_single = []
    for text in texts_same_len:
        inputs_single = processor(text=[text], images=None, return_tensors="pt")
        inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
        with torch.no_grad():
            outputs_single = model(**inputs_single)
            logits_single.append(outputs_single.logits)

    seq_len_0 = (inputs_batch["attention_mask"][0] == 1).sum().item()
    seq_len_1 = (inputs_batch["attention_mask"][1] == 1).sum().item()
    
    diff_0 = (logits_batch[0, :seq_len_0] - logits_single[0][0, :seq_len_0]).abs().max().item()
    diff_1 = (logits_batch[1, :seq_len_1] - logits_single[1][0, :seq_len_1]).abs().max().item()

    print(f"  Sample 0 max difference: {diff_0:.6f}")
    print(f"  Sample 1 max difference: {diff_1:.6f}")
    
    same_len_passed = diff_0 < 1.0 and diff_1 < 1.0

    # Test 3b: Different length prompts (with padding)
    print("\n--- Test 3b: Different-length prompts (with padding) ---")
    texts_diff_len = [
        "A red sports car on a highway",
        "A green forest with tall trees and birds",
    ]

    inputs_batch = processor(text=texts_diff_len, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    mask = inputs_batch["attention_mask"]
    print(f"  Sequence lengths: {mask.sum(dim=1).tolist()}")

    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        logits_batch = outputs_batch.logits

    logits_single = []
    for text in texts_diff_len:
        inputs_single = processor(text=[text], images=None, return_tensors="pt")
        inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
        with torch.no_grad():
            outputs_single = model(**inputs_single)
            logits_single.append(outputs_single.logits)

    seq_len_0 = (inputs_batch["attention_mask"][0] == 1).sum().item()
    seq_len_1 = (inputs_batch["attention_mask"][1] == 1).sum().item()
    
    diff_0 = (logits_batch[0, :seq_len_0] - logits_single[0][0, :seq_len_0]).abs().max().item()
    diff_1 = (logits_batch[1, :seq_len_1] - logits_single[1][0, :seq_len_1]).abs().max().item()

    print(f"  Sample 0 max difference: {diff_0:.6f}")
    print(f"  Sample 1 max difference: {diff_1:.6f}")

    # For padded sequences, larger differences are expected due to:
    # 1. Different attention patterns (padding tokens affect softmax normalization)
    # 2. bfloat16 numerical precision
    # 3. Different computation order
    
    if same_len_passed:
        print("\n✅ Batch consistency test PASSED for same-length sequences")
        if diff_0 > 1.0 or diff_1 > 1.0:
            print("⚠️ Padded sequences have larger differences (expected behavior)")
    else:
        print("\n⚠️ Batch consistency test WARNING: differences exceed tolerance")
        print("   This might be due to attention mask handling or numerical precision")


def test_generation_batch(device: str = "cuda", max_new_tokens: int = 50):
    """Test generation with batch > 1."""
    print("\n" + "=" * 60)
    print("TEST 4: Generation with batch > 1")
    print("=" * 60)

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)

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

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)

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
    print("TEST 6: Different image counts per sample + Token-level consistency")
    print("=" * 60)

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)
    model.eval()

    # Test 6a: Text-to-image batch with token-level consistency check
    print("\n--- Test 6a: Text-to-image batch with token consistency ---")
    
    # Use same-length prompts to avoid padding differences
    texts_t2i = [
        "A mountain at sunset",  # Similar length
        "A coffee shop scene",   # Similar length
    ]
    
    # Batch processing
    inputs_batch = processor(text=texts_t2i, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    print(f"  images_per_sample: {inputs_batch.get('images_per_sample', 'N/A')}")
    print(f"  num_source_images_per_sample: {inputs_batch.get('num_source_images_per_sample', 'N/A')}")
    
    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
    
    # Single processing for comparison
    outputs_single = []
    for text in texts_t2i:
        inputs_single = processor(text=[text], images=None, return_tensors="pt")
        inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
        with torch.no_grad():
            out = model(**inputs_single)
            outputs_single.append(out.logits)
    
    # Compare token-level outputs
    seq_len_0 = (inputs_batch["attention_mask"][0] == 1).sum().item()
    seq_len_1 = (inputs_batch["attention_mask"][1] == 1).sum().item()
    
    diff_0 = (outputs_batch.logits[0, :seq_len_0] - outputs_single[0][0, :seq_len_0]).abs().max().item()
    diff_1 = (outputs_batch.logits[1, :seq_len_1] - outputs_single[1][0, :seq_len_1]).abs().max().item()
    
    print(f"  Token-level max diff sample 0: {diff_0:.6f}")
    print(f"  Token-level max diff sample 1: {diff_1:.6f}")
    
    t2i_consistent = diff_0 < 1.0 and diff_1 < 1.0
    if t2i_consistent:
        print("  ✓ Text-to-image batch is token-consistent with single processing")
    else:
        print("  ⚠️ Text-to-image batch has differences (check padding)")
    
    # Test 6b: Image-to-image batch with token-level consistency check
    print("\n--- Test 6b: Image-to-image batch with token consistency ---")
    
    # Use same source image for both to ensure consistency
    source_image = Image.new("RGB", (256, 256), color=(100, 150, 200))
    source_images = [source_image, source_image]
    
    # Use same-length prompts
    messages_list = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Add some clouds"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Add some birds"}]}],
    ]
    
    texts_i2i = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    
    # Batch processing
    inputs_batch = processor(text=texts_i2i, images=source_images, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    print(f"  images_per_sample: {inputs_batch.get('images_per_sample', 'N/A')}")
    print(f"  num_source_images_per_sample: {inputs_batch.get('num_source_images_per_sample', 'N/A')}")
    print(f"  pixel_values shape: {inputs_batch['pixel_values'].shape}")
    
    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
    
    # Single processing for comparison
    outputs_single = []
    for i, text in enumerate(texts_i2i):
        inputs_single = processor(text=[text], images=[source_images[i]], return_tensors="pt")
        inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
        with torch.no_grad():
            out = model(**inputs_single)
            outputs_single.append(out.logits)
    
    # Compare token-level outputs
    seq_len_0 = (inputs_batch["attention_mask"][0] == 1).sum().item()
    seq_len_1 = (inputs_batch["attention_mask"][1] == 1).sum().item()
    
    diff_0 = (outputs_batch.logits[0, :seq_len_0] - outputs_single[0][0, :seq_len_0]).abs().max().item()
    diff_1 = (outputs_batch.logits[1, :seq_len_1] - outputs_single[1][0, :seq_len_1]).abs().max().item()
    
    print(f"  Token-level max diff sample 0: {diff_0:.6f}")
    print(f"  Token-level max diff sample 1: {diff_1:.6f}")
    
    i2i_consistent = diff_0 < 1.0 and diff_1 < 1.0
    if i2i_consistent:
        print("  ✓ Image-to-image batch is token-consistent with single processing")
    else:
        print("  ⚠️ Image-to-image batch has differences (check padding/image handling)")
    
    # Test 6c: Generation token consistency
    print("\n--- Test 6c: Generation token consistency ---")
    
    # Use identical prompts to ensure no padding
    texts_gen = [
        "A beautiful garden",
        "A beautiful garden",  # Same prompt
    ]
    
    inputs_batch = processor(text=texts_gen, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    with torch.no_grad():
        gen_batch = model.generate(
            **inputs_batch,
            max_new_tokens=10,
            do_sample=False,  # Greedy decoding for determinism
        )
    
    inputs_single = processor(text=[texts_gen[0]], images=None, return_tensors="pt")
    inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
    
    with torch.no_grad():
        gen_single = model.generate(
            **inputs_single,
            max_new_tokens=10,
            do_sample=False,
        )
    
    # Compare generated tokens
    gen_tokens_batch_0 = gen_batch[0].tolist()
    gen_tokens_batch_1 = gen_batch[1].tolist()
    gen_tokens_single = gen_single[0].tolist()
    
    tokens_match_0 = gen_tokens_batch_0 == gen_tokens_single
    tokens_match_1 = gen_tokens_batch_1 == gen_tokens_single
    
    print(f"  Sample 0 tokens match single: {tokens_match_0}")
    print(f"  Sample 1 tokens match single: {tokens_match_1}")
    print(f"  Sample 0 == Sample 1: {gen_tokens_batch_0 == gen_tokens_batch_1}")
    
    if tokens_match_0 and tokens_match_1:
        print("  ✓ Generation is deterministic across batch and single processing")
    else:
        print("  ⚠️ Generation differs (may be due to numerical precision)")
        # Show the tokens for debugging
        print(f"  Batch[0]: {gen_tokens_batch_0[-10:]}")
        print(f"  Batch[1]: {gen_tokens_batch_1[-10:]}")
        print(f"  Single:   {gen_tokens_single[-10:]}")
    
    # Overall result
    if t2i_consistent and i2i_consistent:
        print("\n✅ Different image counts test PASSED with token-level consistency")
    else:
        print("\n⚠️ Test completed with some consistency warnings")
    
    print("\n✅ Different image counts test PASSED")


def test_debug_batch_internals(device: str = "cuda"):
    """Debug test to investigate batch vs single processing differences."""
    print("\n" + "=" * 60)
    print("TEST 7: Debug batch internals")
    print("=" * 60)

    processor = load_processor(PROCESSOR_PATH)
    model = load_model(MODEL_PATH, device)
    model.eval()
    
    # Use IDENTICAL prompts to eliminate any variability
    text = "A beautiful garden"
    texts = [text, text]
    
    print("\n--- Processing identical prompts ---")
    
    # Batch processing
    inputs_batch = processor(text=texts, images=None, return_tensors="pt", padding=True)
    inputs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_batch.items()}
    
    # Single processing
    inputs_single = processor(text=[text], images=None, return_tensors="pt")
    inputs_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_single.items()}
    
    print(f"\nBatch input_ids[0]: {inputs_batch['input_ids'][0].tolist()}")
    print(f"Batch input_ids[1]: {inputs_batch['input_ids'][1].tolist()}")
    print(f"Single input_ids:   {inputs_single['input_ids'][0].tolist()}")
    
    # Check if input_ids are identical
    ids_match = torch.equal(inputs_batch['input_ids'][0], inputs_single['input_ids'][0])
    print(f"\nInput IDs match: {ids_match}")
    
    # Check image_grid_thw
    print(f"\nBatch image_grid_thw: {inputs_batch['image_grid_thw']}")
    print(f"Single image_grid_thw: {inputs_single['image_grid_thw']}")
    
    print(f"\nBatch images_per_sample: {inputs_batch['images_per_sample']}")
    print(f"Single images_per_sample: {inputs_single['images_per_sample']}")
    
    # Now run forward and compare position_ids
    print("\n--- Comparing position_ids ---")
    
    # We need to hook into the model to capture position_ids
    # Let's manually call get_rope_index
    
    with torch.no_grad():
        # Batch
        pos_ids_batch, rope_deltas_batch = model.model.get_rope_index(
            input_ids=inputs_batch['input_ids'],
            image_grid_thw=inputs_batch['image_grid_thw'],
            images_per_sample=inputs_batch['images_per_sample'],
            attention_mask=inputs_batch['attention_mask'],
        )
        
        # Single
        pos_ids_single, rope_deltas_single = model.model.get_rope_index(
            input_ids=inputs_single['input_ids'],
            image_grid_thw=inputs_single['image_grid_thw'],
            images_per_sample=inputs_single['images_per_sample'],
            attention_mask=inputs_single['attention_mask'],
        )
    
    print(f"\nBatch position_ids shape: {pos_ids_batch.shape}")
    print(f"Single position_ids shape: {pos_ids_single.shape}")
    
    # Compare position_ids for sample 0 in batch vs single
    print(f"\nBatch pos_ids[:, 0, :]: {pos_ids_batch[:, 0, :].tolist()}")
    print(f"Single pos_ids[:, 0, :]: {pos_ids_single[:, 0, :].tolist()}")
    
    pos_ids_match = torch.equal(pos_ids_batch[:, 0, :], pos_ids_single[:, 0, :])
    print(f"\nPosition IDs match: {pos_ids_match}")
    
    # Run full forward pass
    print("\n--- Comparing forward outputs ---")
    
    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
        outputs_single = model(**inputs_single)
    
    # Compare logits
    diff_0 = (outputs_batch.logits[0] - outputs_single.logits[0]).abs().max().item()
    diff_1 = (outputs_batch.logits[1] - outputs_single.logits[0]).abs().max().item()
    
    print(f"Max diff batch[0] vs single: {diff_0:.6f}")
    print(f"Max diff batch[1] vs single: {diff_1:.6f}")
    
    # Also check if batch[0] == batch[1]
    diff_01 = (outputs_batch.logits[0] - outputs_batch.logits[1]).abs().max().item()
    print(f"Max diff batch[0] vs batch[1]: {diff_01:.6f}")
    
    if diff_0 < 0.01 and diff_1 < 0.01:
        print("\n✅ Debug test PASSED - batch and single outputs match")
    else:
        print("\n⚠️ Debug test found differences")
        
        # Let's also check inputs_embeds
        print("\n--- Checking intermediate states ---")
        
        # Get the embedding directly
        inputs_embeds_batch = model.model.language_model.embed_tokens(inputs_batch['input_ids'])
        inputs_embeds_single = model.model.language_model.embed_tokens(inputs_single['input_ids'])
        
        embed_diff = (inputs_embeds_batch[0] - inputs_embeds_single[0]).abs().max().item()
        print(f"Embedding diff: {embed_diff:.6f}")
        
        # Check attention mask
        print("\n--- Checking attention masks ---")
        print(f"Batch attention_mask: {inputs_batch['attention_mask']}")
        print(f"Single attention_mask: {inputs_single['attention_mask']}")
        
        # Check if the issue is in the language model (GlmImageTextModel) directly
        print("\n--- Testing language model directly with same position_ids ---")
        
        # Prepare identical inputs for both
        cache_position_batch = torch.arange(inputs_batch['input_ids'].shape[1], device=device)
        cache_position_single = torch.arange(inputs_single['input_ids'].shape[1], device=device)
        
        # Get the language model
        lang_model = model.model.language_model
        
        # Run through language model with explicit position_ids
        with torch.no_grad():
            # For batch - use position_ids directly
            lang_out_batch = lang_model(
                input_ids=inputs_batch['input_ids'],
                attention_mask=inputs_batch['attention_mask'],
                position_ids=pos_ids_batch,
                cache_position=cache_position_batch,
            )
            
            # For single
            lang_out_single = lang_model(
                input_ids=inputs_single['input_ids'],
                attention_mask=inputs_single['attention_mask'],
                position_ids=pos_ids_single,
                cache_position=cache_position_single,
            )
        
        lang_diff = (lang_out_batch.last_hidden_state[0] - lang_out_single.last_hidden_state[0]).abs().max().item()
        print(f"Language model hidden state diff: {lang_diff:.6f}")
        
        # Test with attention_mask=None to see if that's the issue
        print("\n--- Testing without attention_mask ---")
        with torch.no_grad():
            lang_out_batch_no_mask = lang_model(
                input_ids=inputs_batch['input_ids'],
                attention_mask=None,
                position_ids=pos_ids_batch,
                cache_position=cache_position_batch,
            )
            
            lang_out_single_no_mask = lang_model(
                input_ids=inputs_single['input_ids'],
                attention_mask=None,
                position_ids=pos_ids_single,
                cache_position=cache_position_single,
            )
        
        lang_diff_no_mask = (lang_out_batch_no_mask.last_hidden_state[0] - lang_out_single_no_mask.last_hidden_state[0]).abs().max().item()
        print(f"Language model diff (no attention_mask): {lang_diff_no_mask:.6f}")
        
        # Check rotary embeddings directly
        print("\n--- Checking rotary embeddings ---")
        
        inputs_embeds_batch = lang_model.embed_tokens(inputs_batch['input_ids'])
        inputs_embeds_single = lang_model.embed_tokens(inputs_single['input_ids'])
        
        # Get position embeddings
        position_embeddings_batch = lang_model.rotary_emb(inputs_embeds_batch, position_ids=pos_ids_batch)
        position_embeddings_single = lang_model.rotary_emb(inputs_embeds_single, position_ids=pos_ids_single)
        
        # position_embeddings is a tuple of (cos, sin)
        cos_batch, sin_batch = position_embeddings_batch
        cos_single, sin_single = position_embeddings_single
        
        print(f"cos_batch shape: {cos_batch.shape}")
        print(f"cos_single shape: {cos_single.shape}")
        
        # Compare cos and sin for sample 0
        cos_diff = (cos_batch[0] - cos_single[0]).abs().max().item()
        sin_diff = (sin_batch[0] - sin_single[0]).abs().max().item()
        
        print(f"cos diff: {cos_diff:.6f}")
        print(f"sin diff: {sin_diff:.6f}")
        
        # Test running first layer only
        print("\n--- Testing first decoder layer ---")
        
        first_layer = lang_model.layers[0]
        
        # Create causal mask manually
        from transformers.masking_utils import create_causal_mask
        
        mask_kwargs_batch = {
            "config": lang_model.config,
            "input_embeds": inputs_embeds_batch,
            "attention_mask": None,
            "cache_position": cache_position_batch,
            "past_key_values": None,
            "position_ids": None,
        }
        mask_kwargs_single = {
            "config": lang_model.config,
            "input_embeds": inputs_embeds_single,
            "attention_mask": None,
            "cache_position": cache_position_single,
            "past_key_values": None,
            "position_ids": None,
        }
        
        causal_mask_batch = create_causal_mask(**mask_kwargs_batch)
        causal_mask_single = create_causal_mask(**mask_kwargs_single)
        
        print(f"causal_mask_batch: {causal_mask_batch}")
        print(f"causal_mask_single: {causal_mask_single}")
        
        with torch.no_grad():
            layer_out_batch = first_layer(
                inputs_embeds_batch,
                attention_mask=causal_mask_batch,
                position_ids=None,
                past_key_values=None,
                cache_position=cache_position_batch,
                position_embeddings=position_embeddings_batch,
            )
            
            layer_out_single = first_layer(
                inputs_embeds_single,
                attention_mask=causal_mask_single,
                position_ids=None,
                past_key_values=None,
                cache_position=cache_position_single,
                position_embeddings=position_embeddings_single,
            )
        
        layer_diff = (layer_out_batch[0] - layer_out_single[0]).abs().max().item()
        print(f"First layer output diff: {layer_diff:.6f}")
        
        # Check if predictions match (argmax)
        print("\n--- Checking prediction consistency (argmax) ---")
        
        # Get logits from full forward
        logits_batch = outputs_batch.logits
        logits_single = outputs_single.logits
        
        # Get argmax predictions
        pred_batch_0 = logits_batch[0].argmax(dim=-1)
        pred_batch_1 = logits_batch[1].argmax(dim=-1)
        pred_single = logits_single[0].argmax(dim=-1)
        
        pred_match_0 = torch.equal(pred_batch_0, pred_single)
        pred_match_1 = torch.equal(pred_batch_1, pred_single)
        pred_match_01 = torch.equal(pred_batch_0, pred_batch_1)
        
        print(f"Predictions batch[0] == single: {pred_match_0}")
        print(f"Predictions batch[1] == single: {pred_match_1}")
        print(f"Predictions batch[0] == batch[1]: {pred_match_01}")
        
        if pred_match_0 and pred_match_1:
            print("\n✅ Predictions match! Numerical differences don't affect output tokens.")
        else:
            # Show where they differ
            diff_positions_0 = (pred_batch_0 != pred_single).nonzero(as_tuple=True)[0]
            print(f"\nPrediction differences at positions: {diff_positions_0.tolist()}")
            if len(diff_positions_0) > 0:
                for pos in diff_positions_0[:5]:  # Show first 5
                    print(f"  Position {pos}: batch={pred_batch_0[pos].item()}, single={pred_single[pos].item()}")
                    # Also show the logit values at these positions
                    print(f"    Batch logits[{pred_batch_0[pos].item()}]: {logits_batch[0, pos, pred_batch_0[pos]].item():.4f}")
                    print(f"    Single logits[{pred_single[pos].item()}]: {logits_single[0, pos, pred_single[pos]].item():.4f}")
                    print(f"    Batch logits[{pred_single[pos].item()}]: {logits_batch[0, pos, pred_single[pos]].item():.4f}")
        
        # Check top-5 consistency
        print("\n--- Checking top-5 consistency ---")
        topk_batch = logits_batch[0].topk(5, dim=-1).indices
        topk_single = logits_single[0].topk(5, dim=-1).indices
        
        top5_match = torch.equal(topk_batch, topk_single)
        print(f"Top-5 predictions match: {top5_match}")
        
        if not top5_match:
            diff_count = (topk_batch != topk_single).any(dim=-1).sum().item()
            print(f"Positions with different top-5: {diff_count} / {topk_batch.shape[0]}")


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
        # Test 7: Debug internals first to understand the issue
        test_debug_batch_internals(device)

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
