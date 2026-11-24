<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-24.*

# DeepSeek-V4.1

[DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) is a mixture-of-experts language model from
DeepSeek that also reads images. It keeps the building blocks of [DeepSeek-V4](./deepseek_v4) - a single shared KV
head, a sliding window in every layer, hyper-connections instead of plain residuals - but changes how the layers
reach beyond that window, and adds a few things on top.

If you already know DeepSeek-V4, here is what is new:

- **KV sharing.** In DeepSeek-V4, every layer that looks beyond its window compresses its own KV. Here only a few
  layers do, and the layers after them attend to that same compressed KV. A layer compresses either 2 tokens into 1,
  or not at all (a ratio of 1), in which case later layers share its full-resolution KV.
- **Shared, two-level selection.** An **indexer** picks the `index_topk` compressed entries that each query attends
  to. Only some layers run one, and the layers in between reuse the last selection. One indexer also keeps the best
  *blocks* of entries per query, and the indexers after it only choose inside those blocks.
- **Engram.** A few layers add an n-gram lookup to the residual stream: the last few token ids are hashed into a
  large table, and the rows that come back are gated by how well they match the stream.
- **Images.** A ViT with 2D RoPE encodes each image at its own resolution, and a projector merges every 3x3 square of
  patches into one token.

Each layer's variant is listed in `config.text_config.attention_layer_types`:

| Type | Compresses its own KV | Runs an indexer |
|---|---|---|
| `"sliding_attention"` | - | - |
| `"compressed_attention"` | yes | yes |
| `"kv_shared_indexed_attention"` | no, reads the last `"compressed_attention"` layer's | yes |
| `"kv_shared_attention"` | no | no, reuses the last selection |

## Usage

Let's make this concrete with a quick example. There is one step that most models don't need: the Engram n-grams are
hashed over a "compressed" vocabulary, where tokens that normalize alike (like `" The"` and `"the"`) share an id. The
checkpoint doesn't store that map (yet!), so you have to build it from the tokenizer once after loading the model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash", dtype=torch.bfloat16, device_map="auto")
model.set_engram_token_map(tokenizer)

messages = [{"role": "user", "content": "What is hashing an n-gram good for?"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :]))
```

If you forget that step, the model raises an error that tells you what to call, instead of quietly hashing with the
wrong ids.

### Images

Images go in the chat like any other content. The **processor** resizes each one to a whole number of 14x14 patches at
its own aspect ratio, and makes room for it in the prompt: one position for every 3x3 square of patches, a newline
after each row, and a delimiter at either end. A 1024x701 photo takes 444 positions, and no image takes more than 1024.

The checkpoint doesn't ship a processor config (yet!), so you build the processor from its two parts for now.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, DeepseekV41ImageProcessor, DeepseekV41Processor

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
processor = DeepseekV41Processor(
    image_processor=DeepseekV41ImageProcessor(), tokenizer=tokenizer, chat_template=tokenizer.chat_template
)
model = AutoModelForImageTextToText.from_pretrained(
    "deepseek-ai/DeepSeek-V4.1-Flash", dtype=torch.bfloat16, device_map="auto"
)
model.set_engram_token_map(tokenizer)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is in this picture?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
print(processor.decode(outputs[0, inputs["input_ids"].shape[1] :]))
```

## Notes

- The model needs `input_ids` even when you pass `inputs_embeds`, because Engram hashes token ids.
- `resize_token_embeddings` is not supported, because the Engram token map and n-gram tables are fixed to the
  vocabulary of the checkpoint. If you need a new token, reuse one of the placeholder tokens that the tokenizer
  already has.
- Batched generation with left padding matches unpadded generation. The compressor pools consecutive tokens, so it
  counts its groups from the first real token of each row. If you call the model directly on a left-padded batch,
  pass `position_ids` that count from the first real token of each row too, like [`~GenerationMixin.generate`] does.
  The rounding below happens after RoPE, so the model is not indifferent to a shift of the positions.
- Like the original implementation, the model rounds its KV and attention output to FP8, and its index keys, index
  queries and compressed KV to FP4. That is how they were quantized in training, so it happens whatever dtype you load
  the model in. The rounded values keep that dtype, so this does not make the cache any smaller (yet!).
- The checkpoint stores its experts in FP4, and its Engram tables and most of its linear layers in FP8. The Engram
  tables stay in FP8 however you load the model, even with `FineGrainedFP8Config(dequantize=True)`, because they would
  double in size otherwise. Their rows are rescaled as they are looked up, so this changes nothing about the outputs.
- The DSpark draft layers of the checkpoint (`mtp.*`) are not loaded.
- The image processor resizes with torchvision, while the original preprocessing uses PIL. Sizes and padding are the
  same, but resampling leaves a fraction of a percent of the pixel values off by one or two levels out of 255.

## DeepseekV41Config

[[autodoc]] DeepseekV41Config

## DeepseekV41TextConfig

[[autodoc]] DeepseekV41TextConfig

## DeepseekV41VisionConfig

[[autodoc]] DeepseekV41VisionConfig

## DeepseekV41ImageProcessor

[[autodoc]] DeepseekV41ImageProcessor
    - preprocess

## DeepseekV41Processor

[[autodoc]] DeepseekV41Processor
    - __call__

## DeepseekV41VisionModel

[[autodoc]] DeepseekV41VisionModel
    - forward

## DeepseekV41TextModel

[[autodoc]] DeepseekV41TextModel
    - forward
    - set_engram_token_map

## DeepseekV41Model

[[autodoc]] DeepseekV41Model
    - forward

## DeepseekV41ForCausalLM

[[autodoc]] DeepseekV41ForCausalLM
    - forward
    - set_engram_token_map
