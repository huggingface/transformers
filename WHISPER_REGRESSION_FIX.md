# Whisper Regression Fix - Issue #38378

## Problem Description

A regression was introduced in transformers version 4.52.0 (commit `da334bcfa`) that caused different inference results for fine-tuned Whisper models when using different versions of the transformers library. The issue specifically affected:

1. **Short-form transcription**: Better results on version 4.51.3+ but different from earlier versions
2. **Long-form transcription with timestamps**: Better results on version 4.46.0 than on recent versions
3. **Model hallucination**: Different hallucination patterns across versions

## Root Cause

The regression was caused by changes in the `_retrieve_avg_logprobs` method in `generation_whisper.py`. Specifically:

### Original Formula (transformers < 4.52.0)
```python
avg_logprobs = sum_logprobs / (length + 1)
```

### New Formula (transformers >= 4.52.0)
```python
avg_logprobs = sum_logprobs / len(tokens)
```

This change affected:
- **Confidence scoring**: Different average log probability calculations
- **Temperature fallback decisions**: Thresholds for when to use temperature fallback
- **Generation consistency**: Different transcription results for the same input

## Solution

We implemented a backward-compatible fix that:

1. **Adds a new configuration parameter**: `use_legacy_logprob_calculation`
2. **Defaults to legacy behavior**: Maintains backward compatibility by default
3. **Allows new behavior**: Users can opt into the new calculation method
4. **Preserves existing functionality**: No breaking changes to the API

### Configuration Options

```python
from transformers import WhisperConfig, WhisperForConditionalGeneration

# Default behavior (legacy mode for backward compatibility)
config = WhisperConfig()
# config.use_legacy_logprob_calculation = True (default)

# Explicit legacy mode (transformers < 4.52.0 behavior)
config = WhisperConfig(use_legacy_logprob_calculation=True)

# New mode (transformers >= 4.52.0 behavior)
config = WhisperConfig(use_legacy_logprob_calculation=False)

model = WhisperForConditionalGeneration(config)
```

### Usage Examples

#### For Backward Compatibility (Recommended)
```python
# This will use the same behavior as transformers < 4.52.0
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
# model.config.use_legacy_logprob_calculation is True by default

# Your existing code will work exactly as before
result = model.generate(input_features, return_timestamps=True)
```

#### For New Behavior
```python
# This will use the new behavior from transformers >= 4.52.0
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Explicitly enable new behavior
model.config.use_legacy_logprob_calculation = False

result = model.generate(input_features, return_timestamps=True)
```

## Impact on Different Use Cases

### Short-form Transcription
- **Legacy mode**: Consistent with transformers < 4.52.0
- **New mode**: May have slightly different confidence scores

### Long-form Transcription with Timestamps
- **Legacy mode**: Consistent with transformers < 4.52.0, may be better for some models
- **New mode**: Uses the updated calculation, may be better for newer fine-tuned models

### Temperature Fallback
- **Legacy mode**: Uses original thresholds for fallback decisions
- **New mode**: Uses updated thresholds, may trigger fallback differently

## Migration Guide

### For Existing Users (No Action Required)
If you want to maintain the exact same behavior as before the regression:
- **No changes needed**: The fix defaults to legacy behavior
- Your existing code will work exactly as before

### For New Users or Those Wanting Updated Behavior
If you want to use the new calculation method:
```python
# Set the configuration parameter
model.config.use_legacy_logprob_calculation = False
```

### For Model Developers
When fine-tuning new models, consider:
- **Testing both modes**: Compare results with both calculation methods
- **Documenting the choice**: Specify which mode works best for your model
- **Version compatibility**: Consider which transformers versions your users have

## Testing

The fix includes comprehensive tests to ensure:
1. **Backward compatibility**: Legacy mode produces the same results as before
2. **New functionality**: New mode works correctly
3. **Configuration handling**: Both modes can be set and used properly
4. **Deterministic behavior**: Results are consistent within each mode

Run the tests with:
```bash
python -m pytest tests/models/whisper/test_whisper_regression.py
```

## Technical Details

### Mathematical Difference
- **Legacy**: `avg_logprobs = sum_logprobs / (len(tokens) + 1)`
- **New**: `avg_logprobs = sum_logprobs / len(tokens)`
- **Relationship**: `new_logprobs = legacy_logprobs * (len(tokens) + 1) / len(tokens)`

### Code Changes
1. **Configuration**: Added `use_legacy_logprob_calculation` parameter to `WhisperConfig`
2. **Generation**: Modified `_retrieve_avg_logprobs` method to support both calculations
3. **Tests**: Added comprehensive test suite for regression scenarios

## Related Issues

- **GitHub Issue**: [#38378](https://github.com/huggingface/transformers/issues/38378)
- **Original Fix Commit**: `da334bcfa` - "🚨 Fix whisper decoding 🚨"
- **Regression Fix**: This implementation

## Future Considerations

- **Default behavior**: May change to new mode in a future major version
- **Deprecation**: Legacy mode may be deprecated in the future with proper notice
- **Model compatibility**: New models may be optimized for the new calculation method 