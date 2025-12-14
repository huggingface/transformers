# Image Processor Interpolation Method Verification Guide
To be checked (by comparing against original implementation):


beit

bit

clip - good

convnext

convnextv2

cvt

data2vec-vision

deit

dinat

dinov2

efficientformer

efficientnet

focalnet

imagegpt

levit

mobilenet_v1

mobilenet_v2

mobilevit

mobilevitv2

nat

perceiver

poolformer

pvt

regnet

resnet

segformer

siglip

swiftformer

swin

swinv2

van

vit

vit_hybrid

vit_msn






## Overview

This guide provides step-by-step instructions for verifying and potentially updating the default interpolation/resampling methods across all image processors in the transformers library, as outlined in [GitHub Issue #28180](https://github.com/huggingface/transformers/issues/28180.

## Background

### Why This Matters

Interpolation methods have a slight (often minimal) impact on model performance. The transformers library should use the same interpolation methods as the original model implementations to ensure consistency and optimal performance.

### The Problem

Some image processors in the transformers library may be using incorrect default interpolation methods. For example:
- **ViT** currently defaults to `BILINEAR` but should use `BICUBIC` (as seen in the [original implementation](https://github.com/google-research/vision_transformer))
- This affects inference quality and reproducibility

### Why We Can't Just Update Configs on the Hub

While we can update the default values in the image processor code, we **cannot** update the configs on the Hugging Face Hub because this would break people's fine-tuned models that rely on the current values.

## Technical Details

### Where Image Processors Are Located

All image processors are located in:
```
src/transformers/models/{model_name}/image_processing_{model_name}.py
```

For example:
- **ViT**: `src/transformers/models/vit/image_processing_vit.py`
- **CLIP**: `src/transformers/models/clip/image_processing_clip.py`
- **BEiT**: `src/transformers/models/beit/image_processing_beit.py`

### What to Look For

In each image processor file, look for the `__init__` method of the main image processor class. The default resample value is specified as a parameter:

```python
def __init__(
    self,
    do_resize: bool = True,
    size: Optional[dict[str, int]] = None,
    resample: PILImageResampling = PILImageResampling.BILINEAR,  # <-- THIS LINE
    # ... other parameters
):
```

### Available Interpolation Methods

The available PIL image resampling methods are (from `PIL.Image.Resampling`):
- `NEAREST` (0)
- `BOX` (4)
- `BILINEAR` (2)
- `HAMMING` (5)
- `BICUBIC` (3)
- `LANCZOS` (1)

Most models use either **BILINEAR** or **BICUBIC**.

---

## Step-by-Step Verification Process

### Step 1: Identify the Model to Check

Choose a model from the list below (see "Models to Verify" section).

### Step 2: Locate the Image Processor File

Navigate to:
```
src/transformers/models/{model_name}/image_processing_{model_name}.py
```

**Example**: For `vit`, open:
```
src/transformers/models/vit/image_processing_vit.py
```

### Step 3: Find the Current Default Resample Value

1. Open the image processor file
2. Look for the main image processor class (usually named `{ModelName}ImageProcessor`)
3. Find the `__init__` method
4. Locate the `resample` parameter and note its default value

**Example** (ViT):
```python
# Line 83 in src/transformers/models/vit/image_processing_vit.py
resample: PILImageResampling = PILImageResampling.BILINEAR,
```
Current default: **BILINEAR**

### Step 4: Find the Original Implementation

Search for the original implementation of the model. Common sources include:

1. **GitHub repositories** (official implementations)
2. **Research papers** (may specify in appendix/methodology)
3. **Model cards on Hugging Face Hub**
4. **Original framework code** (PyTorch, TensorFlow, JAX)

**Search strategies**:
- Google: `"{model_name} original implementation github"`
- Look in the model's README or documentation for references
- Check the model's docstring in transformers for references to original implementation

**Example sources**:
- **ViT**: https://github.com/google-research/vision_transformer
- **CLIP**: https://github.com/openai/CLIP
- **BEiT**: https://github.com/microsoft/unilm/tree/master/beit
- **DeiT**: https://github.com/facebookresearch/deit
- **Swin**: https://github.com/microsoft/Swin-Transformer

### Step 5: Identify the Correct Resample Method from Original Implementation

In the original implementation, look for:
- Image preprocessing code
- Data augmentation pipelines
- Dataset loaders
- Image transformation functions

**Common places to check**:
- `transforms.Resize()` calls
- `F.interpolate()` calls
- `cv2.resize()` calls
- Image loading utilities

**Mapping between frameworks**:

| **PyTorch Torchvision** | **PIL** | **OpenCV** | **TensorFlow** |
|-------------------------|---------|------------|----------------|
| `BILINEAR` (default) | `BILINEAR` | `INTER_LINEAR` | `bilinear` |
| `BICUBIC` | `BICUBIC` | `INTER_CUBIC` | `bicubic` |
| `NEAREST` | `NEAREST` | `INTER_NEAREST` | `nearest` |
| `LANCZOS` | `LANCZOS` | `INTER_LANCZOS4` | `lanczos` |

**Example**: ViT original implementation
- Repository: https://github.com/google-research/vision_transformer
- File to check: Look for `input_pipeline.py` or similar preprocessing files
- Expected finding: Uses `BICUBIC` interpolation

### Step 6: Compare and Document

1. **If they match**: Great! Mark the model as ✅ verified
2. **If they differ**: Document:
   - Current value in transformers
   - Expected value from original implementation
   - Source/link to original implementation
   - Any notes or special considerations

### Step 7: Create a GitHub Issue or PR (If Mismatch Found)

If you find a mismatch:

1. **Verify your findings** - Double-check the original implementation
2. **Create a GitHub issue** documenting:
   - Model name
   - Current default value
   - Expected default value
   - Link to original implementation showing the expected value
   - Impact assessment (if known)

3. **OR create a Pull Request** with:
   - Updated default value in the image processor `__init__` method
   - Updated docstring to reflect the new default
   - Reference to the original implementation in the PR description
   - Note that this doesn't update existing configs on the Hub

---

## Models to Verify

Below is the complete list of models that need verification, organized alphabetically:

### Model Checklist

- [ ] **beit**
  - File: `src/transformers/models/beit/image_processing_beit.py`
  - Original: https://github.com/microsoft/unilm/tree/master/beit
  - Current default: `BICUBIC` (Line 121)
  
- [ ] **bit**
  - File: `src/transformers/models/bit/image_processing_bit.py`
  - Original: https://github.com/google-research/big_transfer
  
- [ ] **clip**
  - File: `src/transformers/models/clip/image_processing_clip.py`
  - Original: https://github.com/openai/CLIP
  - Current default: `BICUBIC` (Line 95)
  
- [ ] **convnext**
  - File: `src/transformers/models/convnext/image_processing_convnext.py`
  - Original: https://github.com/facebookresearch/ConvNeXt
  
- [ ] **convnextv2**
  - File: `src/transformers/models/convnextv2/image_processing_convnextv2.py`
  - Original: https://github.com/facebookresearch/ConvNeXt-V2
  
- [ ] **cvt**
  - File: `src/transformers/models/cvt/image_processing_cvt.py`
  - Original: https://github.com/microsoft/CvT
  
- [ ] **data2vec-vision**
  - File: `src/transformers/models/data2vec/image_processing_data2vec.py`
  - Original: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
  
- [ ] **deit**
  - File: `src/transformers/models/deit/image_processing_deit.py`
  - Original: https://github.com/facebookresearch/deit
  
- [ ] **dinat**
  - File: `src/transformers/models/dinat/image_processing_dinat.py`
  - Original: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
  
- [ ] **dinov2**
  - File: `src/transformers/models/dinov2/image_processing_dinov2.py`
  - Original: https://github.com/facebookresearch/dinov2
  
- [ ] **efficientformer**
  - File: `src/transformers/models/efficientformer/image_processing_efficientformer.py`
  - Original: https://github.com/snap-research/EfficientFormer
  
- [ ] **efficientnet**
  - File: `src/transformers/models/efficientnet/image_processing_efficientnet.py`
  - Original: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
  
- [ ] **focalnet**
  - File: `src/transformers/models/focalnet/image_processing_focalnet.py`
  - Original: https://github.com/microsoft/FocalNet
  
- [ ] **imagegpt**
  - File: `src/transformers/models/imagegpt/image_processing_imagegpt.py`
  - Original: https://github.com/openai/image-gpt
  
- [ ] **levit**
  - File: `src/transformers/models/levit/image_processing_levit.py`
  - Original: https://github.com/facebookresearch/LeViT
  
- [ ] **mobilenet_v1**
  - File: `src/transformers/models/mobilenet_v1/image_processing_mobilenet_v1.py`
  - Original: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
  
- [ ] **mobilenet_v2**
  - File: `src/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py`
  - Original: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
  
- [ ] **mobilevit**
  - File: `src/transformers/models/mobilevit/image_processing_mobilevit.py`
  - Original: https://github.com/apple/ml-cvnets/tree/main/cvnets/models/classification/mobilevit.py
  
- [ ] **mobilevitv2**
  - File: `src/transformers/models/mobilevitv2/image_processing_mobilevitv2.py`
  - Original: https://github.com/apple/ml-cvnets
  
- [ ] **nat**
  - File: `src/transformers/models/nat/image_processing_nat.py`
  - Original: https://github.com/SHI-Labs/Neighborhood-Attention-Transformer
  
- [ ] **perceiver**
  - File: `src/transformers/models/perceiver/image_processing_perceiver.py`
  - Original: https://github.com/deepmind/deepmind-research/tree/master/perceiver
  
- [ ] **poolformer**
  - File: `src/transformers/models/poolformer/image_processing_poolformer.py`
  - Original: https://github.com/sail-sg/poolformer
  
- [ ] **pvt**
  - File: `src/transformers/models/pvt/image_processing_pvt.py`
  - Original: https://github.com/whai362/PVT
  
- [ ] **regnet**
  - File: `src/transformers/models/regnet/image_processing_regnet.py`
  - Original: https://github.com/facebookresearch/pycls
  
- [ ] **resnet**
  - File: `src/transformers/models/resnet/image_processing_resnet.py`
  - Original: https://github.com/pytorch/vision (torchvision.models.resnet)
  
- [ ] **segformer**
  - File: `src/transformers/models/segformer/image_processing_segformer.py`
  - Original: https://github.com/NVlabs/SegFormer
  
- [ ] **siglip**
  - File: `src/transformers/models/siglip/image_processing_siglip.py`
  - Original: https://github.com/google-research/big_vision
  
- [ ] **swiftformer**
  - File: `src/transformers/models/swiftformer/image_processing_swiftformer.py`
  - Original: https://github.com/Amshaker/SwiftFormer
  
- [ ] **swin**
  - File: `src/transformers/models/swin/image_processing_swin.py`
  - Original: https://github.com/microsoft/Swin-Transformer
  
- [ ] **swinv2**
  - File: `src/transformers/models/swinv2/image_processing_swinv2.py`
  - Original: https://github.com/microsoft/Swin-Transformer (V2 branch)
  
- [ ] **van**
  - File: `src/transformers/models/van/image_processing_van.py`
  - Original: https://github.com/Visual-Attention-Network/VAN-Classification
  
- [ ] **vit**
  - File: `src/transformers/models/vit/image_processing_vit.py`
  - Original: https://github.com/google-research/vision_transformer
  - Current default: `BILINEAR` (Line 83)
  - ⚠️ **Expected**: `BICUBIC` ([source](https://github.com/google-research/vision_transformer))
  
- [ ] **vit_hybrid**
  - File: `src/transformers/models/vit_hybrid/image_processing_vit_hybrid.py`
  - Original: https://github.com/google-research/vision_transformer
  
- [ ] **vit_msn**
  - File: `src/transformers/models/vit_msn/image_processing_vit_msn.py`
  - Original: https://github.com/facebookresearch/msn

---

## Example: Verifying ViT

Let me walk through a complete example for **ViT**:

### 1. Locate the file
```bash
src/transformers/models/vit/image_processing_vit.py
```

### 2. Find current default
Open the file and look at line 83:
```python
resample: PILImageResampling = PILImageResampling.BILINEAR,
```
**Current**: `BILINEAR`

### 3. Check original implementation
- Repository: https://github.com/google-research/vision_transformer
- Navigate to the data preprocessing code
- Check `vit_jax/input_pipeline.py` or similar files

Looking at the original ViT code, you'll find it uses **BICUBIC** interpolation for resizing images.

### 4. Document the finding
**Result**: ❌ Mismatch detected
- **Current**: `BILINEAR`
- **Expected**: `BICUBIC`
- **Source**: https://github.com/google-research/vision_transformer

### 5. Proposed change
Update line 83 in `src/transformers/models/vit/image_processing_vit.py` from:
```python
resample: PILImageResampling = PILImageResampling.BILINEAR,
```
to:
```python
resample: PILImageResampling = PILImageResampling.BICUBIC,
```

### 6. What NOT to change
- **Do NOT** modify `preprocessor_config.json` files on the Hub
- **Do NOT** change the preprocessor configs for existing model checkpoints
- Only update the default in the Python code

---

## Tips and Best Practices

### Finding Original Implementations

1. **Start with the model's docstring** in transformers - often contains links to the original paper and code
2. **Check the model card** on Hugging Face Hub
3. **Search GitHub** for official implementations
4. **Read the paper** - sometimes the preprocessing is detailed in the methodology section
5. **Look for `timm` implementations** - many vision models are in PyTorch Image Models (timm) which often preserves original preprocessing

### Common Pitfalls

1. **Default vs. Specified**: Some models might have BILINEAR as a general default but override it in specific training configs
2. **Different stages**: Training vs. inference might use different interpolation methods - check what was used for the final model weights
3. **Framework differences**: Be careful when translating between PyTorch, TensorFlow, and JAX - make sure you're comparing equivalent methods

### Verification Commands

To check current default values across all image processors:
```bash
# From the transformers root directory
grep -r "resample: PILImageResampling" src/transformers/models/*/image_processing_*.py | grep "def __init__"
```

To find all image processor files:
```bash
find src/transformers/models -name "image_processing_*.py" -type f
```

---

## Contributing

This is an excellent **first issue** for new contributors to the transformers library!

### Steps to contribute:

1. **Pick a model** from the unchecked list above
2. **Verify** the interpolation method following this guide
3. **Document your findings** in the checklist
4. **Create a PR** if you find a mismatch, or
5. **Update this document** to mark the model as verified

### PR Template

When creating a PR for an interpolation method fix:

```markdown
# Fix interpolation method for {ModelName}

## Summary
Updates the default interpolation method for {ModelName} image processor from {CURRENT} to {EXPECTED} to match the original implementation.

## Motivation
As discussed in #27742, some image processors use incorrect default interpolation methods. This ensures consistency with the original {ModelName} implementation.

## Changes
- Updated `resample` default in `image_processing_{model_name}.py` from `PILImageResampling.{CURRENT}` to `PILImageResampling.{EXPECTED}`
- Updated docstring to reflect the new default

## Verification
Original implementation: {LINK_TO_ORIGINAL_REPO}
Relevant code showing {EXPECTED} is used: {LINK_TO_SPECIFIC_FILE}

## Note
This only changes the default in new instantiations. Existing configs on the Hub are not affected to avoid breaking fine-tuned models.
```

---

## Questions?

If you have questions or need clarification:
1. Comment on [Issue #27742](https://github.com/huggingface/transformers/issues/27742)
2. Ask in the Hugging Face Discord
3. Tag relevant maintainers in your PR

---

## Appendix: Quick Reference

### File Structure
```
transformers/
├── src/
│   └── transformers/
│       ├── models/
│       │   ├── vit/
│       │   │   ├── __init__.py
│       │   │   ├── configuration_vit.py
│       │   │   ├── image_processing_vit.py  ← CHECK THIS
│       │   │   ├── modeling_vit.py
│       │   │   └── ...
│       │   ├── clip/
│       │   │   ├── image_processing_clip.py  ← CHECK THIS
│       │   │   └── ...
│       │   └── ...
│       └── ...
└── ...
```

### Interpolation Method Constants

From `transformers.image_utils.PILImageResampling`:
```python
class PILImageResampling:
    NEAREST = 0
    LANCZOS = 1
    BILINEAR = 2
    BICUBIC = 3
    BOX = 4
    HAMMING = 5
```

### Common Original Implementation Patterns

**PyTorch (torchvision)**:
```python
transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
```

**TensorFlow**:
```python
tf.image.resize(image, size, method='bicubic')
```

**PIL**:
```python
image.resize(size, resample=PIL.Image.BICUBIC)
```

---

**Last Updated**: December 2025
**Issue Reference**: transformers#27742
**Status**: In Progress - Community Verification Needed
