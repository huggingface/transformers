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
*This model was contributed to Hugging Face Transformers on 2026-09-14.*


# EmbeddingGemma2

## Overview

EmbeddingGemma 2 is a multimodal embedding model built on the [Gemma 4](./gemma4) backbone. It turns text, images, video and audio into a single dense vector space, and is meant to be used for retrieval, clustering, classification and semantic similarity rather than for generation.

The key differences from Gemma 4 are:

- **No language modeling head.** There is no `ForCausalLM` and no `ForConditionalGeneration` class. [`EmbeddingGemma2Model`] returns a `last_hidden_state` of shape `(batch_size, sequence_length, embedding_dim)`, already projected by the embedding head.
- **An embedding head on the text backbone.** [`EmbeddingGemma2TextModel`] owns `embedding_projection`, a bias-free `nn.Linear(hidden_size, embedding_dim)` applied after the final norm. Because a linear map commutes with averaging, projecting per token is equivalent to projecting the mean-pooled sentence embedding.
- **Bidirectional attention.** `use_bidirectional_attention="all"` makes the stack behave as an encoder, even though it reuses the Gemma 4 decoder layer.
- **Projection-only Per-Layer Embeddings (PLE).** Gemma 4 sums a token-identity term (an `embed_tokens_per_layer` lookup table) with a context-aware projection of `inputs_embeds`. EmbeddingGemma 2 keeps only the context-aware half: `get_per_layer_inputs()` returns `None`, `project_per_layer_inputs(inputs_embeds)` takes a single argument, and `vocab_size_per_layer_input` does not exist on [`EmbeddingGemma2TextConfig`].
- **Reused Gemma 4 towers and processors.** `config.vision_config` is a [`Gemma4VisionConfig`] and `config.audio_config` is a [`Gemma4AudioConfig`]; the towers themselves are resolved through `AutoModel`, so they are a `Gemma4VisionModel` and a `Gemma4AudioModel`. The image processor ([`Gemma4ImageProcessor`]) and the audio feature extractor ([`Gemma4AudioFeatureExtractor`]) are reused as-is through the auto mappings. Only the video processor is specialized: [`EmbeddingGemma2VideoProcessor`] defaults to 1-FPS linspace frame sampling (`use_1fps_linear_sampling=True`) and drops frame timestamps from the prompt (`exclude_timestamps=True`), matching the visual-only training distribution.

You can find all the original EmbeddingGemma checkpoints under the [EmbeddingGemma](https://huggingface.co/collections/google/embeddinggemma) collection. The examples below use the `google/embeddinggemma-2` identifier.

## Usage examples

### Sentence embeddings

The model is trained to be wrapped by SentenceTransformers-style mean pooling followed by L2 normalization. The snippet below reproduces that pooling by hand with the [`AutoModel`] class.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained("google/embeddinggemma-2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-2")

sentences = [
    "Which planet is known as the Red Planet?",
    "Mars is often called the Red Planet because of its reddish appearance.",
]
inputs = tokenizer(sentences, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    # (batch_size, sequence_length, config.text_config.embedding_dim)
    token_embeddings = model(**inputs).last_hidden_state

# Mean pooling over the non-padded tokens, then normalize
mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)

similarity = sentence_embeddings[0] @ sentence_embeddings[1]
print(similarity)
```

### Multimodal embeddings

[`EmbeddingGemma2Processor`] accepts any single modality on its own — text, images, video or audio — and synthesizes the placeholder tokens when no text is given.

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

image_embedding = F.normalize(token_embeddings.mean(dim=1), p=2, dim=-1)
print(image_embedding.shape)
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
