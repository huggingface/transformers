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

> [!WARNING]
> This page is a work in progress. The examples are being revised while the model addition is under review, and the processor and multimodal sections will change once the processing feedback is addressed.

## Overview

EmbeddingGemma 2 is a multimodal embedding model built on the [Gemma 4](./gemma4) backbone. It turns text, images, video and audio into a single dense vector space, and is meant to be used for retrieval, clustering, classification and semantic similarity rather than for generation.

The key differences from Gemma 4 are:

- **No language modeling head.** There is no `ForCausalLM` and no `ForConditionalGeneration` class. [`EmbeddingGemma2Model`] returns a `last_hidden_state` of shape `(batch_size, sequence_length, embedding_dim)`, already projected by the embedding head.
- **An embedding head on the text backbone.** [`EmbeddingGemma2TextModel`] owns `embedding_projection`, a bias-free `nn.Linear(hidden_size, embedding_dim)` applied after the final norm. Because a linear map commutes with averaging, projecting per token is equivalent to projecting the mean-pooled sentence embedding.
- **Bidirectional attention.** The stack is an encoder: every layer attends bidirectionally, over the full sequence on `full_attention` layers and over a symmetric window on `sliding_attention` layers. There is no causal mask and no key-value cache.
- **Projection-only Per-Layer Embeddings (PLE).** Gemma 4 sums a token-identity term (an `embed_tokens_per_layer` lookup table) with a context-aware projection of `inputs_embeds`. EmbeddingGemma 2 keeps only the context-aware half: `EmbeddingGemma2TextPLE` takes `inputs_embeds` alone, and neither `vocab_size_per_layer_input` nor the lookup table exists. The text model computes the per-layer embeddings once with `EmbeddingGemma2TextPLE`; each decoder layer then mixes its own slice into the residual stream with `EmbeddingGemma2TextPLEBlock`, after attention and the MLP.
- **Reused Gemma 4 towers and processors.** `config.vision_config` is a [`Gemma4VisionConfig`] and `config.audio_config` is a [`Gemma4AudioConfig`]; the towers themselves are resolved through `AutoModel`, so they are a `Gemma4VisionModel` and a `Gemma4AudioModel`. The image processor ([`Gemma4ImageProcessor`]) and the audio feature extractor ([`Gemma4AudioFeatureExtractor`]) are reused as-is through the auto mappings. Only the video processor is specialized: [`EmbeddingGemma2VideoProcessor`] samples frames at 1 FPS (`fps=1`), caps a clip at 32 frames by uniformly subsampling anything longer (`max_frames=32`, `overflow_strategy="uniform"`), and leaves frame timestamps out of the prompt (`add_timestamps=False`).

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

The model is trained with a task prompt in front of every input, and the prompt is included in the pooled tokens. Sentence Transformers ships the catalog below in `config_sentence_transformers.json`, so pass `prompt_name` (or use `encode_query` / `encode_document`, which map to `query` and `document`). With [`AutoModel`], prepend the prompt string to the text yourself — that is the only difference.

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

For text, the prompt is prepended to the string. Inputs that carry media go through the chat template instead, where Sentence Transformers passes the prompt as a system message. The template ignores roles, emits the media placeholders first and then the text of each message in order, so the prompt lands between the media tokens and your own text. Pooling covers the prompt tokens in both cases.

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

Text, images, video and audio are mapped into the same vector space, so embeddings from different modalities are directly comparable. Each modality can also be embedded on its own — [`EmbeddingGemma2Processor`] synthesizes the placeholder tokens when no text is given.

<hfoptions id="usage">
<hfoption id="Sentence Transformers">

Inputs are dictionaries keyed by modality. A key may hold a PIL image, a local path, a URL or an array, and several keys can be combined in one dictionary to produce a single embedding. A list may mix plain strings and dictionaries.

```python
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-2")
print(model.modalities)
# ['text', 'image', 'audio', 'video', 'message']

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# one key per modality
text_embedding = model.encode({"text": "A photo of a cat"})
image_embedding = model.encode({"image": IMAGE})
audio_embedding = model.encode({"audio": "path/to/audio.wav"})
video_embedding = model.encode({"video": "path/to/video.mp4"})

# several modalities in one dictionary give one embedding
caption_embedding = model.encode({"text": "A photo of a cat", "image": IMAGE})
narration_embedding = model.encode({"text": "A 440 Hz tone", "audio": "path/to/audio.wav"})
scene_embedding = model.encode({"image": IMAGE, "audio": "path/to/audio.wav"})

# a batch may mix modalities, and task prompts apply to dictionaries too
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

Video preprocessing flags are forwarded through `processing_kwargs`.

```python
video_embedding = model.encode(
    {"video": "path/to/video.mp4"},
    processing_kwargs={"video": {"add_timestamps": True, "fps": 2, "max_frames": 16}},
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

inputs = processor(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    token_embeddings = model(**inputs).last_hidden_state

mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
image_embedding = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
image_embedding = F.normalize(image_embedding, p=2, dim=-1)
print(image_embedding.shape)
```

`audio=` and `videos=` work the same way, and any combination of the four modalities can be passed in a single call.

</hfoption>
</hfoptions>

## Processor

> [!WARNING]
> This section is a work in progress and will be revised once the processing review feedback is addressed.

[`EmbeddingGemma2Processor`] bundles the tokenizer, the image processor, the audio feature extractor and the video processor. Each modality can be passed on its own, in which case the processor synthesizes the placeholder tokens, so no text is required.

```python
from transformers import AutoProcessor


processor = AutoProcessor.from_pretrained("google/embeddinggemma-2")

IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

# text only
inputs = processor(text=["task: search result | query: Which planet is the Red Planet?"], return_tensors="pt")

# media only, placeholders are synthesized
inputs = processor(images=[IMAGE], return_tensors="pt")
inputs = processor(videos=["path/to/video.mp4"], return_tensors="pt")

# audio is passed as a raw waveform, a 1-D float array sampled at 16 kHz
inputs = processor(audio=[audio_array], return_tensors="pt")

# text and media together, one placeholder token per media item
inputs = processor(text=["<|image|> a photo of a cat"], images=[IMAGE], return_tensors="pt")
print(inputs.keys())
# dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_position_ids'])
```

The soft-token budget per image is configurable, and lowering it shortens the sequence. Supported values are 70, 140, 280, 560 and 1120.

```python
inputs = processor(images=[IMAGE], max_soft_tokens=70, return_tensors="pt")
print(inputs["input_ids"].shape, inputs["pixel_values"].shape)
# torch.Size([1, 68]) torch.Size([1, 630, 768])
```

[`EmbeddingGemma2VideoProcessor`] samples frames at 1 FPS, caps a clip at 32 frames, and leaves frame timestamps out of the prompt. Every knob is overridable per call.

Rate-based sampling needs to know the source frame rate, which only comes from decoding a file. A pre-decoded array carries no `fps` or `duration`, so for those inputs the processor warns, skips FPS sampling, and applies the `max_frames` budget alone — pass a `VideoMetadata` with a valid `fps` and `duration` if you want the array sampled at a target rate. Timestamps have no such fallback: `add_timestamps=True` on an array with no `fps` raises, because a guessed rate would write wrong `mm:ss` labels into the prompt.

```python
inputs = processor(
    videos=["path/to/video.mp4"],
    add_timestamps=True,
    fps=2,
    max_frames=16,
    return_tensors="pt",
)
```

Chat-style messages are also accepted, which is what Sentence Transformers uses internally for media inputs. The template ignores roles: it emits the media placeholders first and then the text of every message in order.

```python
messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE},
                {"type": "text", "text": "title: none | text: a photo of a cat"},
            ],
        }
    ]
]
inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
```

## Notes

- The model is an encoder: attention is bidirectional, there is no language modeling head and no key-value cache. `last_hidden_state` is already projected to `config.text_config.embedding_dim`, not `hidden_size`.

- Pooling must be mask-aware. Averaging over padding tokens changes the embedding, which is why the examples above weight by `attention_mask`.

- Use right padding, the tokenizer's default for this checkpoint. Positions default to `torch.arange(seq_len)`, which counts pad tokens.

- Because `embedding_projection` is linear, projecting every token and then averaging is equivalent to averaging and then projecting. Pooling the model output is therefore the same as pooling the backbone's hidden states and projecting once.

- Media inputs are expensive in tokens: an image costs 280 soft tokens by default, and a video costs `max_soft_tokens` per sampled frame, with up to `max_frames` (32) frames per clip.

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
