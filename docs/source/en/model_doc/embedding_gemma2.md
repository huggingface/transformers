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
*This model was contributed to Hugging Face Transformers on 2026-10-06.*


# EmbeddingGemma2

## Overview

EmbeddingGemma 2 is a multimodal embedding model from Google built on the [Gemma 4](./gemma4) architecture. It encodes **text, images, audio, and video**—either individually or combined within the same input—into a shared 768-dimensional dense vector space for cross-modal retrieval, semantic similarity, clustering, and classification.

Key features:

- **Unified multimodal vector space.** Vision ([`Gemma4VisionModel`]) and audio ([`Gemma4AudioModel`]) towers project into a bidirectional text encoder ([`EmbeddingGemma2TextModel`]) that interleaves full and sliding-window attention with Per-Layer Embeddings (PLE). Any single modality (`text`, `image`, `audio`, `video`) or combination of modalities (`image + text`, `text + audio`, `image + audio`, multiple images, interleaved text and media) maps to a single comparable embedding.
- **Matryoshka Representation Learning (MRL).** [`EmbeddingGemma2Model`] outputs token representations projected to `embedding_dim` (`768`) via a linear head. Embeddings can be truncated to a smaller prefix (`512`, `256`, or `128`) and re-normalized with minimal quality loss.
- **Configurable visual & video budgets.** [`EmbeddingGemma2Processor`] lets you tune the soft-token budget per image or frame (`max_soft_tokens` in `{70, 140, 280, 560, 1120}`) and video sampling rate (`fps=1`, `max_frames=32`, `overflow_strategy="uniform"`, `add_timestamps=False` by default).
- **Selective modality tower loading.** Unused vision or audio towers can be disabled at load time (`vision_config=None`, `audio_config=None`) to reduce memory footprint for text-only or single-modality deployments.

You can find all the original EmbeddingGemma checkpoints under the [EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma) collection. The examples below use the `google/embeddinggemma-2` identifier.

## Usage examples

A sentence or multimodal embedding is obtained in two steps: mask-aware mean pooling over the non-padded tokens of `last_hidden_state`, followed by L2 normalization in `float32`. [Sentence Transformers](https://sbert.net) (`>=6.1.0`) performs preprocessing, prompt formatting, mean pooling, and normalization automatically and is the recommended entry point. With [`AutoModel`], use `processor.apply_chat_template` (or `AutoTokenizer` for plain text) and pool `last_hidden_state` directly.

### Text retrieval (`encode_query` / `encode_document`)

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
# (1, 768) (2, 768)

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

# Prepend the task prompt directly to plain text inputs (see Task prompts below)
sentences = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    # (batch_size, sequence_length, config.text_config.embedding_dim)
    token_embeddings = model(**inputs).last_hidden_state

# Mask-aware mean pooling over non-padded tokens, then L2 normalization in float32
mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
sentence_embeddings = F.normalize(sentence_embeddings.float(), p=2, dim=-1)

query_embedding, document_embeddings = sentence_embeddings[:1], sentence_embeddings[1:]
print(query_embedding @ document_embeddings.T)
```

</hfoption>
</hfoptions>

### Task prompts

The model supports optional task prompts prepended to the input (and included in mean pooling), though prompts are not mandatory and the model also works without them. Because the optimal setup depends on the downstream domain and modality mix, we recommend evaluating both with and without task prompts on your specific task. Sentence Transformers ships the catalog below in `config_sentence_transformers.json`, so pass `prompt_name` (or use `encode_query` / `encode_document`, which map to `query` and `document`); with [`AutoModel`], pass the prompt as a `system` message in `apply_chat_template` (or prepend it to plain text).

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")

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

### Matryoshka embeddings

The embedding head is trained with Matryoshka Representation Learning, so a 768-dimensional embedding can be sliced to a shorter prefix (`512`, `256`, or `128`) and re-normalized. This shrinks the index and speeds up retrieval while largely preserving ranking quality. Truncate queries and documents to the *same* dimension, and always normalize after slicing — in Sentence Transformers, `truncate_dim` slices the already-normalized output, so the prefix is no longer unit-length.

<hfoptions id="matryoshka">
<hfoption id="Sentence Transformers">

```python
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# truncate_dim can also be set once at load time: SentenceTransformer(..., truncate_dim=256)
model = SentenceTransformer("google/embeddinggemma-2")

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]

query_embeddings = model.encode_query(queries, truncate_dim=256, convert_to_tensor=True)
document_embeddings = model.encode_document(documents, truncate_dim=256, convert_to_tensor=True)

query_embeddings = F.normalize(query_embeddings.float(), p=2, dim=-1)
document_embeddings = F.normalize(document_embeddings.float(), p=2, dim=-1)

# (1, 256) and (2, 256): a 3x smaller index than the full 768 dimensions
print(query_embeddings.shape, document_embeddings.shape)

# the Mars document still ranks first
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

sentences = [
    "task: search result | query: Which planet is known as the Red Planet?",
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# Slice to the first 256 dimensions, then normalize
sentence_embeddings = F.normalize(sentence_embeddings[:, :256].float(), p=2, dim=-1)
print(sentence_embeddings.shape)
# torch.Size([3, 256])

query_embedding, document_embeddings = sentence_embeddings[:1], sentence_embeddings[1:]

# the Mars document still ranks first
print(query_embedding @ document_embeddings.T)
```

</hfoption>
</hfoptions>

## Multimodal embeddings

Text, images, audio, and video are mapped into the same 768-dimensional vector space, enabling **cross-modal retrieval** (e.g. searching images, audio, or videos with a text query) as well as **composed multimodal retrieval** (combining multiple modalities into a single query or document embedding).

In Sentence Transformers, multimodal inputs are passed as dictionaries keyed by `"text"`, `"image"`, `"audio"`, and `"video"` (where each value can be a single item or a list of items). With [`AutoModel`], pass chat-style message lists to `processor.apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`.

### 1. Single modalities & cross-modal retrieval

Each modality (`text`, `image`, `audio`, `video`) can be embedded on its own and compared directly against any other modality via cosine similarity:

<hfoptions id="multimodal-single">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# Embed each modality on its own into the shared (768,) space
candidates = model.encode(
    [
        {"text": "A photo of a cat"},
        {"image": IMAGE},
        {"audio": "path/to/audio.wav"},
        {"video": "path/to/video.mp4"},
    ]
)
print(candidates.shape)
# (4, 768)

# Cross-modal retrieval: rank text, image, audio, and video candidates against a text query
query = model.encode_query(["A fluffy cat outdoors"])
print(model.similarity(query, candidates))
# tensor([[...]]) of shape (1, 4)
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

conversations = [
    [{"role": "user", "content": [{"type": "text", "text": "A photo of a cat"}]}],
    [{"role": "user", "content": [{"type": "image", "url": IMAGE}]}],
    [{"role": "user", "content": [{"type": "audio", "url": "path/to/audio.wav"}]}],
    [{"role": "user", "content": [{"type": "video", "url": "path/to/video.mp4"}]}],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([4, 768])
```

</hfoption>
</hfoptions>

### 2. Composed multimodal embeddings (several modalities in one input)

Multiple modalities—or multiple items of the same modality—can be combined into a **single joint embedding**, including purely non-text combinations such as `image + audio`:

<hfoptions id="multimodal-composed">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE_1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
IMAGE_2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"

composed_embeddings = model.encode(
    [
        # text + image
        {"text": "A photo of a cat", "image": IMAGE_1},
        # text + audio
        {"text": "A song", "audio": "path/to/audio.wav"},
        # image + audio (no text required)
        {"image": IMAGE_1, "audio": "path/to/audio.wav"},
        # multiple images + audio + text in one embedding
        {"image": [IMAGE_1, IMAGE_2], "audio": "path/to/audio.wav", "text": "Compare both cats"},
    ]
)
print(composed_embeddings.shape)
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

conversations = [
    # text + image
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A photo of a cat"},
                {"type": "image", "url": IMAGE},
            ],
        }
    ],
    # text + audio
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A song"},
                {"type": "audio", "url": "path/to/audio.wav"},
            ],
        }
    ],
    # image + audio (no text required)
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE},
                {"type": "audio", "url": "path/to/audio.wav"},
            ],
        }
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([3, 768])
```

</hfoption>
</hfoptions>

### 3. Automatic ordering vs. manual placeholders

When no placeholder tokens (`<|image|>`, `<|video|>`, `<|audio|>`) appear in the text, modalities are emitted in the exact order their entries appear in the dictionary or chat message `content` (immediately after any `system` task prompt).

To interleave text and media at specific positions within a sentence, write `<|image|>`, `<|video|>`, or `<|audio|>` directly in the text—automatic placeholder insertion is then disabled for that input, and each placeholder is expanded in-order with the supplied media items:

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

# 1. Automatic ordering: follows the content list order after any system prompt
image_first_messages = [
    {"role": "system", "content": "title: none | text: "},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_1},
            {"type": "text", "text": "A photo of a cat"},
        ],
    },
]

# 2. Manual placeholders: interleave media at exact positions in the text
interleaved_messages = [
    {"role": "system", "content": "task: search result | query: "},
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_1},
            {"type": "image", "url": IMAGE_2},
            {"type": "audio", "url": "path/to/audio.wav"},
            {"type": "text", "text": "A jacket similar to <|image|> or <|image|> featured in <|audio|>"},
        ],
    },
]

inputs = processor.apply_chat_template(
    [image_first_messages, interleaved_messages],
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
```

</hfoption>
</hfoptions>

### 4. Heterogeneous (mixed-modality) batching

A single batch can mix plain text, single-modality media, and multi-modality inputs in one forward pass, producing the same embedding per row as encoding each item individually:

<hfoptions id="multimodal-batch">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

embeddings = model.encode(
    [
        "A photo of a cat",
        {"image": IMAGE},
        {"text": "A photo of a cat", "image": IMAGE},
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

conversations = [
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "text", "text": "A photo of a cat"}]},
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "image", "url": IMAGE}]},
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "A photo of a cat"},
                {"type": "image", "url": IMAGE},
            ],
        },
    ],
    [
        {"role": "system", "content": "title: none | text: "},
        {"role": "user", "content": [{"type": "audio", "url": "path/to/audio.wav"}]},
    ],
]

inputs = processor.apply_chat_template(
    conversations,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
print(embeddings.shape)
# torch.Size([4, 768])
```

</hfoption>
</hfoptions>

### 5. Disabling unused modality towers (memory optimization)

If your workload only embeds a subset of modalities (for example, **text-only** or **text + images** without audio), you can skip instantiating and loading the unused vision or audio towers by setting `vision_config=None` and/or `audio_config=None` on the config. The unused tower weights in the checkpoint are ignored cleanly without warnings, dropping the model from 744M to 439M parameters without the audio tower, or to 271M with neither tower.

<hfoptions id="selective-towers">
<hfoption id="Sentence Transformers">

Pass these through `config_kwargs` (which reaches `AutoConfig.from_pretrained`), not `model_kwargs`:

```python
from sentence_transformers import SentenceTransformer


# Text-only deployment (skips both vision and audio towers)
text_model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None, "audio_config": None},
)

# Vision + text deployment (skips audio tower)
vision_text_model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"audio_config": None},
)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModel


# Text-only deployment (skips both vision and audio towers)
text_model = AutoModel.from_pretrained(
    "google/embeddinggemma-2",
    vision_config=None,
    audio_config=None,
    device_map="auto",
)

# Vision + text deployment (skips audio tower)
vision_text_model = AutoModel.from_pretrained(
    "google/embeddinggemma-2",
    audio_config=None,
    device_map="auto",
)
```

</hfoption>
</hfoptions>

## Processor

[`EmbeddingGemma2Processor`] bundles the tokenizer, the image processor, the audio feature extractor, and the video processor.

> [!TIP]
> For batched multimodal inputs, we recommend passing per-sample dictionaries through **Sentence Transformers** (`model.encode([{"image": ..., "text": ...}, ...])`). For maximum control over exact modality ordering and interleaving, include `<|image|>`, `<|video|>`, and `<|audio|>` placeholders manually in `text`.

### Controlling visual token budget (`max_soft_tokens`)

The soft-token budget (`max_soft_tokens`, supported values `{70, 140, 280, 560, 1120}`) controls the resolution and number of visual tokens produced per image or video frame (defaulting to `280` per image and `140` per video frame). Lower values reduce sequence length and latency:

<hfoptions id="processor-soft-tokens">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

image_embedding = model.encode(
    {"image": IMAGE},
    processing_kwargs={"image": {"max_soft_tokens": 70}},
)
```

</hfoption>
<hfoption id="AutoProcessor">

```python
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")
IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

inputs = processor(images=[IMAGE], max_soft_tokens=70, return_tensors="pt")
print(inputs["input_ids"].shape, inputs["pixel_values"].shape)
# torch.Size([1, 67]) torch.Size([1, 630, 768])
```

</hfoption>
</hfoptions>

### Video frame sampling controls

By default, [`EmbeddingGemma2VideoProcessor`] samples frames at 1 FPS (`fps=1`), caps each clip at 32 frames (`max_frames=32`, `overflow_strategy="uniform"`), and omits frame timestamps from the prompt (`add_timestamps=False`). Each parameter can be overridden per call:

- `fps` (`int | float | None`): target sampling rate in frames per second. Requires `VideoMetadata` with valid `fps` and `duration` (automatically populated when decoding from a video file/URL). For pre-decoded frame arrays without metadata, FPS sampling is skipped with a warning and `max_frames` is applied directly.
- `max_frames` (`int | None`): maximum number of frames retained per video.
- `overflow_strategy` (`"uniform" | "truncate" | None`): how excess frames above `max_frames` are reduced (`"uniform"` resamples evenly across the clip; `"truncate"` keeps the leading `max_frames` frames).
- `add_timestamps` (`bool`): whether to prefix each frame's soft-token block with its `mm:ss` timestamp. Unlike `fps`, this has no fallback: `add_timestamps=True` on a video whose metadata has no `fps` raises, because a guessed rate would write wrong `mm:ss` labels into the prompt.

<hfoptions id="multimodal-video">
<hfoption id="Sentence Transformers">

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")

video_embedding = model.encode(
    {"video": "path/to/video.mp4"},
    processing_kwargs={"video": {"add_timestamps": True, "fps": 2, "max_frames": 16, "overflow_strategy": "truncate"}},
)
```

</hfoption>
<hfoption id="AutoProcessor">

```python
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

inputs = processor(
    videos=["path/to/video.mp4"],
    add_timestamps=True,
    fps=2,
    max_frames=16,
    overflow_strategy="truncate",
    return_tensors="pt",
)
```

</hfoption>
</hfoptions>

### Direct `processor(...)` calls without chat template

While `processor.apply_chat_template` is the primary entry point for multimodal conversations, `processor(...)` can also be called directly. When `text` is omitted (`text=None`), `<|image|>`, `<|video|>`, and `<|audio|>` placeholders are synthesized automatically for each sample in the batch:

```python
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
AUDIO = "path/to/audio.wav"  # a path, a URL, or a 1-D float array sampled at 16 kHz

# Text only
inputs = processor(text=["task: search result | query: Which planet is the Red Planet?"], return_tensors="pt")

# Media only (text=None): placeholders are synthesized per sample
inputs = processor(images=[IMAGE], return_tensors="pt")
inputs = processor(videos=["path/to/video.mp4"], return_tensors="pt")
inputs = processor(audio=[AUDIO], return_tensors="pt")

# Nested per-sample lists or combined modalities with text=None
inputs = processor(
    images=[[IMAGE, IMAGE], [IMAGE]],
    audio=[[AUDIO], [AUDIO, AUDIO]],
    return_tensors="pt",
)

# Manual placeholders in text for direct processor calls
inputs = processor(text=["<|image|> a photo of a cat"], images=[[IMAGE]], return_tensors="pt")
print(list(inputs.keys()))
# ['input_ids', 'attention_mask', 'pixel_values', 'image_position_ids']
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
    - get_image_features
    - get_video_features
    - get_audio_features
