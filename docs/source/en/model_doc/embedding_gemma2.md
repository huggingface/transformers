<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-22.*


# EmbeddingGemma2

## Overview

EmbeddingGemma 2 is a multimodal embedding model from Google built on the [Gemma 4](./gemma4) architecture. It encodes text, images, video, and audio—individually or interleaved—into a shared 768-dimensional dense vector space for retrieval, semantic similarity, clustering, and classification.

Key features:

- **Multimodal bidirectional encoder.** Vision ([`Gemma4VisionModel`]) and audio ([`Gemma4AudioModel`]) towers feed into a bidirectional text encoder ([`EmbeddingGemma2TextModel`]) that interleaves full and sliding-window attention with context-aware Per-Layer Embeddings (PLE).
- **Matryoshka embedding head.** [`EmbeddingGemma2Model`] outputs token representations projected to `embedding_dim` (`768`) via a linear head and trained with Matryoshka Representation Learning (MRL), supporting prefix truncation to smaller dimensions.
- **Unified multimodal processing.** [`EmbeddingGemma2Processor`] processes text, images, audio, and video in a single call. By default, [`EmbeddingGemma2VideoProcessor`] samples video clips at 1 FPS up to 32 frames (`fps=1`, `max_frames=32`, `overflow_strategy="uniform"`, `add_timestamps=False`).

You can find all the original EmbeddingGemma checkpoints under the [EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma) collection. The examples below use the `google/embeddinggemma-2` identifier.

## Usage examples

A sentence embedding is obtained in two steps: mean pooling over the non-padded tokens of `last_hidden_state`, then L2 normalization. [Sentence Transformers](https://sbert.net) performs both steps for you and is the recommended entry point. The [`AutoModel`] tab writes the same pooling out by hand and produces identical embeddings.

<hfoptions id="usage">
<hfoption id="Sentence Transformers">

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]

# encode_query and encode_document apply the "query" and "document" task prompts
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)

# (1, 2) matrix of cosine similarities
print(model.similarity(query_embeddings, document_embeddings))
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")

# the task prompt is part of the input text, see the task prompts below
sentences = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    # (batch_size, sequence_length, config.text_config.embedding_dim)
    token_embeddings = model(**inputs).last_hidden_state

# mean pooling over the non-padded tokens, then L2 normalization
mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)

print(sentence_embeddings[0] @ sentence_embeddings[1])
```

</hfoption>
</hfoptions>

### Task prompts

The model supports optional task prompts prepended to the input (included in the pooled tokens), though prompts are not mandatory and the model also works without them. Because the optimal setup depends on the downstream domain and modality mix, we recommend evaluating both with and without task prompts on your specific task. Sentence Transformers ships the catalog below in `config_sentence_transformers.json`, so pass `prompt_name` (or use `encode_query` / `encode_document`, which map to `query` and `document`); with [`AutoModel`], prepend the prompt string yourself.

```python
embeddings = model.encode("How to train a neural network", prompt_name="Classification")
```

| `prompt_name` | prompt |
|---|---|
| `query`, `Retrieval-query`, `Retrieval`, `Reranking`, `BitextMining`, `SearchQuery` | `task: search result \| query: ` |
| `document`, `Document`, `Retrieval-document` | `title: none \| text: ` |
| `Classification`, `MultilabelClassification` | `task: classification \| query: ` |
| `Clustering` | `task: clustering \| query: ` |
| `CodeRetrieval`, `InstructionRetrieval` | `task: code retrieval \| query: ` |
| `FactChecking` | `task: fact checking \| query: ` |
| `QuestionAnswering` | `task: question answering \| query: ` |
| `STS`, `SentenceSimilarity`, `PairClassification`, `Summarization` | `task: sentence similarity \| query: ` |

For text, the prompt is prepended to the string. Inputs that carry media go through the chat template instead, where Sentence Transformers passes the prompt as a system message. The template emits system messages first right after `<bos>`, then renders each message's content items in the order supplied (`{"image": ..., "text": ...}` vs `{"text": ..., "image": ...}`), unless manual `<|image|>`, `<|video|>`, or `<|audio|>` placeholders are already present in the text. Pooling covers the prompt tokens in both cases.

### Matryoshka embeddings

The model is trained with Matryoshka Representation Learning, so a 768-dimensional embedding can be truncated to a shorter prefix and used as-is. Truncation drops the L2 normalization, so normalize again afterwards.

<hfoptions id="usage">
<hfoption id="Sentence Transformers">

```python
import torch


embeddings = model.encode(queries, truncate_dim=256, convert_to_tensor=True)
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
```

</hfoption>
<hfoption id="AutoModel">

```python
embeddings = F.normalize(sentence_embeddings[:, :256], p=2, dim=-1)
```

</hfoption>
</hfoptions>

## Multimodal embeddings

Text, images, video and audio are mapped into the same vector space, so embeddings from different modalities are directly comparable. Each modality can be embedded on its own or combined with other modalities in a single input.

### Single and combined modalities

Inputs in Sentence Transformers are dictionaries keyed by modality (`"text"`, `"image"`, `"audio"`, `"video"`). A key may hold a single item (PIL image, path, URL, or array) or a list of items of that modality, and several keys can be combined in one dictionary to produce a single joint embedding. A batch may also mix plain strings and multimodal dictionaries.

<hfoptions id="multimodal-basic">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# single modality per input
text_embedding = model.encode({"text": "A photo of a cat"})
image_embedding = model.encode({"image": IMAGE})
audio_embedding = model.encode({"audio": "path/to/audio.wav"})
video_embedding = model.encode({"video": "path/to/video.mp4"})

# multiple modalities combined into a single embedding
caption_embedding = model.encode({"image": IMAGE, "text": "A photo of a cat"})
scene_embedding = model.encode({"image": IMAGE, "audio": "path/to/audio.wav", "text": "A cat purring"})

# heterogeneous batch with a task prompt
embeddings = model.encode(
    [
        "A photo of a cat",
        {"image": IMAGE},
        {"image": IMAGE, "text": "A photo of a cat"},
        {"audio": "path/to/audio.wav"},
    ],
    prompt_name="document",
)
print(embeddings.shape)
# (4, 768)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# media-only (placeholder tokens are synthesized automatically when text is omitted)
inputs = processor(images=IMAGE, return_tensors="pt").to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
image_embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
image_embedding = F.normalize(image_embedding, p=2, dim=-1)
print(image_embedding.shape)
# torch.Size([1, 768])
```

</hfoption>
</hfoptions>

### Automatic ordering vs. manual placeholders

When no placeholder tokens (`<|image|>`, `<|video|>`, `<|audio|>`) are written in the text, modalities are placed in the exact order their keys appear in the dictionary (after any system/task prompt). To interleave text and media at specific positions, include `<|image|>`, `<|video|>`, or `<|audio|>` directly in the text — automatic placeholder insertion is then disabled for that input, and the number of placeholders in the text must match the number of passed multimodal items.

<hfoptions id="multimodal-placeholders">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

# 1. Automatic ordering: follows the dictionary's key order
#    -> <bos> [prompt] <image_tokens> <text_tokens> <eos>
image_first = model.encode({"image": IMAGE_1, "text": "A photo of a cat"}, prompt_name="document")

#    -> <bos> [prompt] <text_tokens> <image_tokens> <eos>
text_first = model.encode({"text": "A photo of a cat", "image": IMAGE_1}, prompt_name="document")

# 2. Manual placeholders: interleave media at exact positions in the text
#    No extra placeholders are inserted; counts must match the passed media inputs.
interleaved = model.encode(
    {
        "text": "A jacket similar to <|image|> or <|image|> featured in <|audio|>",
        "image": [IMAGE_1, IMAGE_2],
        "audio": "path/to/audio.wav",
    },
    prompt_name="query",
)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

# Explicit `<|image|>` placeholders directly in `processor(...)`
inputs = processor(
    text=["task: search result | query: A jacket similar to <|image|> or <|image|>"],
    images=[[IMAGE_1, IMAGE_2]],
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embedding = F.normalize((token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9), p=2, dim=-1)
```

</hfoption>
</hfoptions>

## Processor

[`EmbeddingGemma2Processor`] bundles the tokenizer, the image processor, the audio feature extractor and the video processor. When `text` is omitted, the processor automatically synthesizes `<|image|>`, `<|video|>`, and `<|audio|>` placeholders for each sample in the batch (including nested per-sample lists such as `audio=[[audio_1, audio_2], [audio_3]]` or combined modalities).

> [!TIP]
> For batched multimodal inputs, we recommend passing per-sample dictionaries through **Sentence Transformers** (`model.encode([{"image": ..., "text": ...}, ...])`). For maximum control over exact modality ordering and interleaving, include `<|image|>`, `<|video|>`, and `<|audio|>` placeholders manually in `text`.

```python
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# text only
inputs = processor(text=["task: search result | query: Which planet is the Red Planet?"], return_tensors="pt")

# media only (text=None), placeholders are synthesized per batch sample
inputs = processor(images=[IMAGE], return_tensors="pt")
inputs = processor(videos=["path/to/video.mp4"], return_tensors="pt")
inputs = processor(audio=[audio_array], return_tensors="pt")  # 1-D float array at 16 kHz

# nested per-sample lists or combined modalities with text=None
inputs = processor(images=[[IMAGE, IMAGE], [IMAGE]], audio=[[audio_array], [audio_array, audio_array]], return_tensors="pt")

# manual placeholders in text for full control over modality placement
inputs = processor(text=["<|image|> a photo of a cat"], images=[[IMAGE]], return_tensors="pt")
print(inputs.keys())
# dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_position_ids'])
```

The soft-token budget (`max_soft_tokens`) is configurable per call; by default, the checkpoint uses 280 soft tokens per image and 140 soft tokens per video frame.

```python
inputs = processor(images=[IMAGE], max_soft_tokens=70, return_tensors="pt")
print(inputs["input_ids"].shape, inputs["pixel_values"].shape)
# torch.Size([1, 68]) torch.Size([1, 630, 768])
```

By default, [`EmbeddingGemma2VideoProcessor`] samples frames at 1 FPS, caps a clip at 32 frames (`overflow_strategy="uniform"`), and leaves frame timestamps out of the prompt (`add_timestamps=False`). Every knob is overridable per call.

Rate-based sampling needs to know the source frame rate, which only comes from decoding a file. A pre-decoded array carries no `fps` or `duration`, so for those inputs the processor warns, skips FPS sampling, and applies the `max_frames` budget alone — pass a `VideoMetadata` with a valid `fps` and `duration` if you want the array sampled at a target rate. Timestamps have no such fallback: `add_timestamps=True` on an array with no `fps` raises, because a guessed rate would write wrong `mm:ss` labels into the prompt.

<hfoptions id="multimodal-video">
<hfoption id="Sentence Transformers">

```python
video_embedding = model.encode(
    {"video": "path/to/video.mp4"},
    processing_kwargs={"video": {"add_timestamps": True, "fps": 2, "max_frames": 16}},
)
```

</hfoption>
<hfoption id="AutoProcessor">

```python
inputs = processor(
    videos=["path/to/video.mp4"],
    add_timestamps=True,
    fps=2,
    max_frames=16,
    return_tensors="pt",
)
```

</hfoption>
</hfoptions>

Chat-style messages are also accepted, which is what Sentence Transformers (`>=6.1.0`) uses internally for media inputs. The template renders any `system` messages first (where Sentence Transformers places the task prompt), then emits the remaining content entries in the order provided (or expands manual `<|image|>`, `<|video|>`, and `<|audio|>` markers in-place when present in the text).

```python
messages = [
    [
        {"role": "system", "content": "title: none | text: "},
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE},
                {"type": "text", "text": "a photo of a cat"},
            ],
        },
    ]
]
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
```

## EmbeddingGemma2TextConfig

[[autodoc]] EmbeddingGemma2TextConfig

## EmbeddingGemma2Config

[[autodoc]] EmbeddingGemma2Config

## EmbeddingGemma2VideoProcessor

[[autodoc]] EmbeddingGemma2VideoProcessor
    - preprocess

## EmbeddingGemma2Processor

[[autodoc]] EmbeddingGemma2Processor
    - __call__

## EmbeddingGemma2PreTrainedModel

[[autodoc]] EmbeddingGemma2PreTrainedModel
    - forward

## EmbeddingGemma2TextModel

[[autodoc]] EmbeddingGemma2TextModel
    - forward

## EmbeddingGemma2Model

[[autodoc]] EmbeddingGemma2Model
    - forward
