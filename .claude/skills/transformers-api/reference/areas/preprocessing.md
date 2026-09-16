# Preprocessing (tokenizers, processors, image/video processors, feature extractors)

## Contents
- [Scope](#scope)
- [Minimum questions (0–4)](#minimum-questions-04)
- [Choose the right preprocessor](#choose-the-right-preprocessor)
- [Text preprocessing: `AutoTokenizer`](#text-preprocessing-autotokenizer)
- [Chat templating: `apply_chat_template`](#chat-templating-apply_chat_template)
- [Vision preprocessing: `AutoImageProcessor`](#vision-preprocessing-autoimageprocessor)
- [Audio preprocessing: `AutoFeatureExtractor` and `AutoProcessor`](#audio-preprocessing-autofeatureextractor-and-autoprocessor)
- [Video preprocessing: `AutoVideoProcessor`](#video-preprocessing-autovideoprocessor)
- [Multimodal preprocessing: `AutoProcessor`](#multimodal-preprocessing-autoprocessor)
- [Batching + device sanity](#batching--device-sanity)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Repo hotspots](#repo-hotspots)

---

## Scope

Use this page when the user needs to convert **raw inputs** (text / images / audio / video / multimodal messages) into **model-ready tensors** for `transformers`.

---

## Minimum questions (0–4)

Ask only what’s needed to produce a runnable snippet:
1) **Modality + task + raw input format**  
   - Text / vision / audio / video / multimodal  
   - What you’re passing in (e.g., plain strings, chat messages, image URL/path/PIL, audio array + sampling rate, video frames)  
   - Desired output (logits, embeddings, generated tokens) / expected shapes if relevant
2) **Model id/path**  
   - Hugging Face Hub id or local path  
   - Optional but recommended for reproducibility/security: pinned `revision` (tag/branch/commit)
3) **Backend + device**  
   - PyTorch / TensorFlow / JAX  
   - CPU / CUDA / MPS (and which GPU index if CUDA)
4) If blocked: **full traceback + minimal repro**  
   - Smallest code sample that still fails + the exact error

---

## Choose the right preprocessor

Rule: **load preprocessing artifacts from the same checkpoint as the model**.

| Modality | Preferred class | Typical output keys |
|---|---|---|
| Text | `AutoTokenizer` | `input_ids`, `attention_mask` (maybe `token_type_ids`) |
| Image | `AutoImageProcessor` | `pixel_values` (maybe `pixel_mask`) |
| Audio | `AutoFeatureExtractor` *or* `AutoProcessor` (model-dependent) | `input_values` **or** `input_features` (sometimes `attention_mask`) |
| Video | `AutoVideoProcessor` **or** `AutoImageProcessor` (frame-based; model-dependent) | model-dependent video/frame tensors + optional metadata |
| Multimodal (text+image/audio/video) | `AutoProcessor` | combination (e.g., `input_ids` + `pixel_values`) |

If the model card/examples show `AutoProcessor`, prefer `AutoProcessor`.
Note: Some video classification models (e.g., VideoMAE) use a frame/image processor (`AutoImageProcessor` / `VideoMAEImageProcessor`) rather than `AutoVideoProcessor`.

---

## Text preprocessing: `AutoTokenizer`

### Minimal batch tokenization (PyTorch)
```python
from transformers import AutoTokenizer

model_id = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_id)

texts = ["hello world", "a much longer example sentence"]
batch = tok(
    texts,
    padding=True,        # pad to longest in batch
    truncation=True,     # truncate if needed
    return_tensors="pt",
)

print(batch.keys())
print(batch["input_ids"].shape)
```

### Practical padding/truncation defaults
- Safe batch default: `padding=True, truncation=True`
- Deterministic cap: add `max_length=...`
- Static shapes: `padding="max_length"` + `max_length=...`

### Decoder-only LMs: pad token + left padding for batching
Some causal LMs do not define a pad token. For batched inputs (esp. generation), set it explicitly.
```python
from transformers import AutoTokenizer

model_id = "gpt2"
tok = AutoTokenizer.from_pretrained(model_id)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "left"  # common for decoder-only batching

batch = tok(["hi", "hello there"], padding=True, return_tensors="pt")
print(batch["input_ids"].shape)
```

### Long inputs: sliding window with overlap (`stride`)
Use this when text exceeds context length and you want overlapping windows.
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "very long text " * 2000
enc = tok(
    text,
    truncation=True,
    max_length=512,
    stride=128,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,  # best with fast tokenizers
)

print("num_windows:", len(enc["input_ids"]))
```

### Token classification: word alignment (`is_split_into_words`)
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-cased")

words = ["New", "York", "City"]
enc = tok(words, is_split_into_words=True, return_tensors="pt")

# Fast tokenizers provide token->word alignment
word_ids = enc.word_ids(batch_index=0)
print(word_ids)
```

---

## Chat templating: `apply_chat_template`

Use chat templates when the model expects a specific conversation format.
If the user’s issue is decoding/stopping/streaming, route to `generation.md`.

```python
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# Access/auth note:
# - If this line fails with 401 Unauthorized / GatedRepoError, the repo is gated or private.
# - Fix: (1) request/accept access on the model page, then (2) authenticate:
#     * terminal: `huggingface-cli login`
#     * or set env var `HF_TOKEN=hf_...` and restart your kernel/session
# - Optional token examples:
#     * AutoTokenizer.from_pretrained(model_id, token=True)        # use cached login or HF_TOKEN
#     * AutoTokenizer.from_pretrained(model_id, token="hf_...")    # explicit token
# - Public demo alternative (no gating): "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a haiku about preprocessing."},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# If you later tokenize `prompt` yourself, set add_special_tokens=False to avoid duplicating special tokens.
```

To directly get token ids:
```python
enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(enc.shape)
```

---

## Vision preprocessing: `AutoImageProcessor`

### Minimal image preprocessing (PIL)
```python
from transformers import AutoImageProcessor
from PIL import Image

model_id = "google/vit-base-patch16-224"
imgp = AutoImageProcessor.from_pretrained(model_id)

image = Image.open("image.jpg")
inputs = imgp(images=image, return_tensors="pt")

print(inputs.keys())  # typically includes "pixel_values" (and sometimes "pixel_mask", model-dependent)
print(inputs["pixel_values"].shape)
```

### Fast image processors (if supported)
Some checkpoints provide a “fast” image processor path.
```python
from transformers import AutoImageProcessor

imgp = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    use_fast=True,
)
```

### What to change (typical; varies by checkpoint)
Prefer changing processor config/kwargs rather than writing ad-hoc transforms. Not every processor supports every knob below:
- resize/crop: `do_resize`, `size`, `do_center_crop`, `crop_size`
- normalize: `do_normalize`, `image_mean`, `image_std`
- rescale: `do_rescale`, `rescale_factor`

---

## Audio preprocessing: `AutoFeatureExtractor` and `AutoProcessor`

### Key rule: sampling rate must match
If audio outputs are “nonsense,” sampling-rate mismatch is a top cause. 
Prefer reading the expected sampling rate from the preprocessor rather than hardcoding it.

### Waveform models (e.g., wav2vec2): `AutoFeatureExtractor`
```python
import numpy as np
from transformers import AutoFeatureExtractor

model_id = "facebook/wav2vec2-base-960h"
fe = AutoFeatureExtractor.from_pretrained(model_id)
# 1 second of silence at the model's expected sampling rate (replace with real audio)
sr = fe.sampling_rate
waveform = np.zeros(sr, dtype=np.float32)
inputs = fe(
    waveform,
    sampling_rate=sr,
    padding=True,
    return_tensors="pt",
)
print(inputs.keys())  # typically includes "input_values" (+ "attention_mask" sometimes, model-dependent)
```

### Spectrogram-feature models (common for Whisper): `AutoProcessor`
Whisper-style models typically use a processor that returns `input_features`.
```python
import numpy as np
from transformers import AutoProcessor

model_id = "openai/whisper-small"
proc = AutoProcessor.from_pretrained(model_id)
sr = proc.feature_extractor.sampling_rate
waveform = np.zeros(sr, dtype=np.float32)
inputs = proc(
    waveform,
    sampling_rate=sr,
    return_tensors="pt",
)
print(inputs.keys())  # typically includes "input_features"
```

---

## Video preprocessing: `AutoVideoProcessor`

Video preprocessing may require a decoding backend depending on how you provide video.
Safest approach (no decoder dependency): **decode frames yourself** and pass frames.

### Option A (decoder-free): pass frames you already have
Example assumes you have a list of PIL images (frames) or numpy arrays.  
For a batch of videos, pass a list of frame-lists: `[[frame1, frame2, ...], [...]]`

```python
# VideoMAE uses an *image/frame* processor; AutoVideoProcessor is for certain VLM/video-chat model types.
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

mid = "MCG-NJU/videomae-base-finetuned-kinetics"
proc = AutoImageProcessor.from_pretrained(mid)
model = VideoMAEForVideoClassification.from_pretrained(mid)
num_frames = getattr(model.config, "num_frames", 16)
H, W = 224, 224
frames = [Image.fromarray(np.random.randint(0,256,(H,W,3),dtype=np.uint8))
          for _ in range(num_frames)]
inputs = proc(images=frames, return_tensors="pt")  # <-- key change
pred = model(**inputs).logits.argmax(-1).item()
print(model.config.id2label[pred])
```

### Option B: decode with TorchCodec, then pass frames to VideoMAE
```python
# Requirements:
#   pip install torch transformers torchcodec
#   + install FFmpeg (shared libs; on Windows this matters)

import torch
from torchcodec.decoders import VideoDecoder
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

video_path = "video.mp4"
model_id = "MCG-NJU/videomae-base-finetuned-kinetics"

proc = AutoImageProcessor.from_pretrained(model_id)   # VideoMAE is frame-based
model = VideoMAEForVideoClassification.from_pretrained(model_id).eval()

# TorchCodec decodes frames as uint8 tensors; use NHWC to get (N, H, W, C)
decoder = VideoDecoder(video_path, dimension_order="NHWC")
T = len(decoder)
if T == 0:
    raise RuntimeError(f"Video has 0 frames: {video_path}")

num = getattr(model.config, "num_frames", 16)
idx = torch.linspace(0, T - 1, num).round().long().clamp(0, T - 1)

fb = decoder.get_frames_at(indices=idx.tolist())  # FrameBatch; pixels in fb.data (uint8)
frames = [fb.data[i].cpu().numpy() for i in range(fb.data.shape[0])]  # list of HWC uint8 arrays

inputs = proc(images=frames, return_tensors="pt")
print(inputs.keys())

with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()

print(model.config.id2label[pred])
```
---

## Multimodal preprocessing: `AutoProcessor`

Use `AutoProcessor` for models that combine modalities (text + image/audio/video). 

### Recommended: chat template + image

```python
from transformers import AutoProcessor
from PIL import Image

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# For **LLaVA-OneVision**, it’s safest to build the prompt with the **chat template** (it inserts the required image placeholder token).
proc = AutoProcessor.from_pretrained(model_id)
image = Image.open("image.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
inputs = proc(text=prompt, images=image, return_tensors="pt")
print(inputs.keys())  # typically includes input_ids/attention_mask + pixel_values (and possibly others)
```
---

## Batching + device sanity
### Inspect keys, shapes, dtypes
```python
import torch
for k, v in inputs.items():
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v.dtype, v.device)
    else:
        print(k, type(v))
```

### Move tensors to device (PyTorch)
Some outputs support `.to(device)`; otherwise move per-tensor.
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    inputs = inputs.to(device)
except Exception:
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
```
---

## Pitfalls & fixes

### “Batch fails / shapes differ”
- Text: `padding=True` and usually `truncation=True`
- Audio: `padding=True` + consistent `sampling_rate`
- Vision/video: pass lists consistently; avoid mixing PIL paths/URLs/arrays in the same batch

### “Tokenizer has no pad token”
- Decoder-only: set `pad_token` (often to `eos_token`) and consider `padding_side="left"`

### “Output keys don’t match model forward”
Print `inputs.keys()` and confirm expected keys:
- text: `input_ids`, `attention_mask` (maybe `token_type_ids`)
- vision: `pixel_values` (maybe `pixel_mask`)
- audio: `input_values` or `input_features`
- multimodal: combinations

### “Audio outputs are wrong”
- Verify sampling rate, dtype, and that you’re passing a 1D waveform (not stereo without handling)

### “Double preprocessing (manual normalize + processor normalize)”
- Prefer processor config; if you must customize, disable the relevant processor steps (model-dependent)

---

## Repo hotspots 

### Tokenizers
- src/transformers/tokenization_utils_base.py
- src/transformers/tokenization_utils_fast.py
- src/transformers/tokenization_utils_tokenizers.py
- src/transformers/models/auto/tokenization_auto.py

### Processors
- src/transformers/processing_utils.py
- src/transformers/models/auto/processing_auto.py

### Image processors
- src/transformers/image_processing_utils.py
- src/transformers/image_processing_base.py
- src/transformers/models/auto/image_processing_auto.py
- model-specific: src/transformers/models/*/image_processing_*.py

### Feature extractors
- src/transformers/feature_extraction_utils.py
- src/transformers/models/auto/feature_extraction_auto.py
- model-specific: src/transformers/models/*/feature_extraction_*.py

### Video processors
- src/transformers/video_processing_utils.py
- src/transformers/models/auto/video_processing_auto.py
- src/transformers/video_utils.py
- model-specific: src/transformers/models/*/video_processing_*.py
  - example: src/transformers/models/videomae/video_processing_videomae.py

### Tests (entry points)
- tests/test_tokenization_common.py
- model-specific: tests/models/<model_name>/...