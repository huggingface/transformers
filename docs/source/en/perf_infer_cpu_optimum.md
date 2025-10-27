
# Optimum for CPU Inference: Quick Start Guide

[🤗 Optimum](https://huggingface.co/docs/optimum) is an extension of Transformers that provides hardware-specific optimizations. This guide shows you when and how to use Optimum for faster CPU inference.

---

## When to Use Optimum on CPU

Use **Optimum** when:
- ✅ You need **production-level performance** on CPU
- ✅ You can tolerate a **one-time model export step** (to ONNX format)
- ✅ Your model is **supported by ONNX Runtime** (most common models are)
- ✅ You want to use **quantization** (int8) for additional speedup

Stick with **vanilla Transformers** when:
- ⚠️ You're prototyping and need fast iteration
- ⚠️ You need model features not yet supported in ONNX export
- ⚠️ Your model architecture is not ONNX-compatible

---

## Performance Expectations

Typical speedups on CPU (depends on model and hardware):

| Optimization                      | Speedup vs PyTorch | Use Case              |
| --------------------------------- | ------------------ | --------------------- |
| ONNX Runtime (fp32)               | **2-3×**           | General inference     |
| ONNX Runtime + int8 quantization  | **4-6×**           | Latency-critical apps |
| ONNX Runtime + graph optimization | **2-4×**           | Batch processing      |

**Note**: Actual speedup varies by CPU generation, model size, and input length.

---

## Installation

```bash
# Install Optimum with ONNX Runtime support
pip install optimum[onnxruntime]

# Verify installation
python -c "from optimum.onnxruntime import ORTModelForSequenceClassification; print('Success!')"
```

---

## Quick Start: Text Classification

### Step 1: Export to ONNX (one-time setup)

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

# Export to ONNX and save locally
model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True  # Automatically export to ONNX
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save for reuse
save_path = "./onnx_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ ONNX model saved to {save_path}")
```

### Step 2: Run inference

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

# Load pre-exported ONNX model
model = ORTModelForSequenceClassification.from_pretrained("./onnx_model")
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# Use with pipeline (just like vanilla Transformers!)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = classifier("Optimum makes CPU inference fast!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## Text Generation Example

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_id = "gpt2"

# Export and load
model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

---

## Quantization for Extra Speed

### Dynamic Quantization (easiest)

Convert weights to int8 while keeping activations in float:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

# Create quantization config
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

# Load and quantize
model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True
)

# Apply quantization
model.quantize(
    save_directory="./quantized_model",
    quantization_config=qconfig
)
print("✅ Model quantized and saved!")
```

### Load and use quantized model

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import pipeline

model = ORTModelForSequenceClassification.from_pretrained("./quantized_model")
tokenizer = AutoTokenizer.from_pretrained("./quantized_model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This is incredibly fast!")
print(result)
```

---

## Benchmark Your Model

Compare Transformers vs Optimum performance:

```python
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
text = "Optimum is great for CPU inference!" * 10  # Longer input

# Vanilla Transformers
print("🐌 Transformers (PyTorch):")
model_pt = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe_pt = pipeline("text-classification", model=model_pt, tokenizer=tokenizer)

start = time.time()
for _ in range(100):
    _ = pipe_pt(text)
time_pt = time.time() - start
print(f"   Time: {time_pt:.2f}s")

# Optimum ONNX
print("🚀 Optimum (ONNX Runtime):")
model_ort = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
pipe_ort = pipeline("text-classification", model=model_ort, tokenizer=tokenizer)

start = time.time()
for _ in range(100):
    _ = pipe_ort(text)
time_ort = time.time() - start
print(f"   Time: {time_ort:.2f}s")

print(f"\n⚡ Speedup: {time_pt/time_ort:.2f}x faster with Optimum!")
```

**Expected output:**
```
🐌 Transformers (PyTorch):
   Time: 8.42s
🚀 Optimum (ONNX Runtime):
   Time: 3.15s

⚡ Speedup: 2.67x faster with Optimum!
```

---

## Supported Models

Most common architectures are supported. Check compatibility:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# This will tell you if your model is supported
try:
    model = ORTModelForSequenceClassification.from_pretrained(
        "your-model-name",
        export=True
    )
    print("✅ Model is supported!")
except Exception as e:
    print(f"❌ Not supported: {e}")
```

**Well-supported families:**
- BERT, RoBERTa, DistilBERT
- GPT-2, GPT-Neo
- T5, BART
- [Full list](https://huggingface.co/docs/optimum/onnxruntime/modeling_ort)

---

## Troubleshooting

### "Model not found" when loading ONNX model

**Problem**: ONNX files not saved correctly

**Solution**:
```python
# Make sure to save after export
model.save_pretrained("./my_onnx_model")
tokenizer.save_pretrained("./my_onnx_model")

# Check files exist
import os
assert os.path.exists("./my_onnx_model/model.onnx"), "ONNX file missing!"
```

### Quantization fails on Windows

**Problem**: AVX512 instructions not available

**Solution**: Use AVX2 quantization config
```python
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Use AVX2 instead of AVX512
qconfig = AutoQuantizationConfig.avx2(is_static=False)
```

### Slower than expected

**Problem**: Not using optimized ONNX Runtime build

**Solution**: Install the optimized version
```bash
pip install --upgrade onnxruntime
# For even better performance:
pip install onnxruntime-extensions
```

---

## CPU-Specific Tips

### Thread Configuration

Set optimal thread count for your CPU:

```python
import os

# Set before importing onnxruntime
cpu_count = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(cpu_count)
os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

# Then import and use
from optimum.onnxruntime import ORTModelForSequenceClassification
```

### Batch Processing

Process multiple inputs for better throughput:

```python
texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

# Efficient batch processing
results = classifier(texts, batch_size=4)
```

---

## Next Steps

- 📖 [Optimum documentation](https://huggingface.co/docs/optimum)
- 🔧 [Advanced quantization](https://huggingface.co/docs/optimum/onnxruntime/quantization)
- ⚡ [Graph optimization](https://huggingface.co/docs/optimum/onnxruntime/optimization)
- 🎯 [Benchmarking tools](https://huggingface.co/docs/optimum/benchmark)

---

## Comparison Table: Transformers vs Optimum

| Feature          | Transformers               | Optimum                               |
| ---------------- | -------------------------- | ------------------------------------- |
| **Setup**        | `pip install transformers` | `pip install optimum[onnxruntime]`    |
| **API**          | Identical for inference    | Identical for inference               |
| **Export step**  | None                       | One-time ONNX export                  |
| **CPU speed**    | Baseline                   | 2-6× faster                           |
| **Quantization** | Limited (dynamic only)     | Full support (static/dynamic)         |
| **GPU support**  | Full                       | ONNX Runtime or CUDAExecutionProvider |
| **New features** | Immediate                  | May lag behind                        |
| **Best for**     | Prototyping, research      | Production, deployment                |

---

## Summary

✅ **Use Optimum when**: You need fast CPU inference and can do one-time ONNX export  
✅ **Use quantization when**: You need maximum speed and can accept slight accuracy trade-off  
✅ **Start simple**: Export to ONNX first, optimize later  
✅ **Benchmark**: Always measure on your specific hardware and workload
