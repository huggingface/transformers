# Fix for Marian Model ONNX Export Issues

This directory contains a fix for the issue reported in [GitHub Issue #40122](https://github.com/huggingface/transformers/issues/40122) where OPUS Marian models exported to ONNX format produce incorrect or degraded output quality.

## Problem Description

The original issue involved exporting the decoder part of a Marian model (specifically OPUS en-ar) to ONNX format, which resulted in:

- **Original PyTorch output**: Proper Arabic translation with complete sentences
- **ONNX output**: Incomplete, degraded Arabic translation with missing words and poor quality

## Root Cause

The problem was caused by:

1. **Incorrect decoder isolation**: The original approach tried to export only the decoder without proper handling of the encoder-decoder coupling
2. **Missing attention mask handling**: Improper handling of attention masks and positional embeddings
3. **Incomplete input structure**: The custom `DecoderWithLMHead` wrapper didn't maintain the full model architecture

## Solution

We've implemented a proper solution by adding two new methods to the `MarianMTModel` class:

### 1. `export_encoder_to_onnx()`

Exports the encoder part of the Marian model with proper input/output handling.

### 2. `export_decoder_to_onnx()`

Exports the decoder part while maintaining the encoder-decoder coupling through `encoder_hidden_states`.

## Key Improvements

- **Proper architecture preservation**: Maintains the full encoder-decoder structure
- **Correct attention handling**: Properly handles attention masks and positional embeddings
- **Dynamic axes support**: Supports variable batch sizes and sequence lengths
- **Comprehensive testing**: Includes validation against the original PyTorch model

## Usage

### Basic Export

```python
from transformers import MarianMTModel

# Load model
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

# Export encoder and decoder
encoder_path = model.export_encoder_to_onnx("encoder.onnx")
decoder_path = model.export_decoder_to_onnx("decoder.onnx")
```

### Using the Example Script

```bash
# Export the OPUS en-ar model
python export_marian_onnx.py \
    --model_name "Helsinki-NLP/opus-mt-en-ar" \
    --output_dir "./onnx_export" \
    --opset_version 17

# Test with a custom sentence
python export_marian_onnx.py \
    --model_name "Helsinki-NLP/opus-mt-en-ar" \
    --test_sentence "Your custom sentence here"
```

## File Structure

```
examples/pytorch/translation/
├── export_marian_onnx.py          # Main export script
├── README_marian_onnx_fix.md      # This file
└── onnx_export/                   # Output directory (created after export)
    ├── encoder.onnx               # Exported encoder model
    └── decoder.onnx               # Exported decoder model
```

## Requirements

- `transformers` >= 4.28.0
- `torch` >= 1.9.0
- `onnxruntime` >= 1.4.0
- `numpy` >= 1.19.0

## Testing

The export script automatically tests the ONNX models by:

1. Running the same input through both PyTorch and ONNX models
2. Comparing the outputs for quality and correctness
3. Providing detailed feedback on the export success

## Expected Results

With this fix, you should see:

- ✅ **Identical output quality** between PyTorch and ONNX models
- ✅ **Proper Arabic translations** for OPUS en-ar models
- ✅ **Complete sentences** without missing words or degradation
- ✅ **Consistent performance** across different input lengths

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `input_shape` in the export methods
2. **ONNX runtime errors**: Ensure you have the correct ONNX opset version
3. **Output mismatch**: Check that the model is in evaluation mode before export

### Debug Mode

Enable verbose ONNX export by modifying the export methods:

```python
torch.onnx.export(
    # ... other parameters ...
    verbose=True,  # Enable verbose output
)
```

## Contributing

If you encounter issues with this fix or have improvements to suggest:

1. Test with the provided example script
2. Compare outputs between PyTorch and ONNX versions
3. Report any discrepancies with detailed error messages
4. Provide the specific model and input that causes issues

## Related Issues

- [GitHub Issue #40122](https://github.com/huggingface/transformers/issues/40122) - Original ONNX export issue
- [Marian Model Documentation](https://huggingface.co/docs/transformers/model_doc/marian)
- [ONNX Export Guide](https://huggingface.co/docs/transformers/serialization#export-to-onnx) 