<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BertGeneration

[BertGeneration](https://huggingface.co/papers/1907.12461) leverages pre-trained BERT checkpoints for sequence-to-sequence tasks using EncoderDecoderModel architecture.

BertGeneration adapts the powerful BERT encoder for generative tasks by using it in encoder-decoder architectures for tasks like summarization, translation, and text fusion. Think of it as taking BERT's deep understanding of language and teaching it to generate new text based on input context.

You can find all the original BertGeneration checkpoints under the [BERT Generation](https://huggingface.co/models?search=bert-generation) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the BertGeneration models in the right sidebar for more examples of how to apply BertGeneration to different sequence generation tasks.

The example below demonstrates how to use BertGeneration with [`EncoderDecoderModel`] for sequence-to-sequence tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

# Use pipeline for text generation with BERT-based models
generator = pipeline("text2text-generation", model="google/roberta2roberta_L-24_discofuse")
result = generator("This is the first sentence. This is the second sentence.")
print(result[0]['generated_text'])
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, EncoderDecoderModel

# Create encoder-decoder model from BERT checkpoints
encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
decoder = BertGenerationDecoder.from_pretrained(
    "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

# Create tokenizer
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

# Prepare input
input_ids = tokenizer("This is a long article to summarize", add_special_tokens=False, return_tensors="pt").input_ids

# Generate summary
outputs = model.generate(input_ids, max_length=50)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

</hfoption>
<hfoption id="transformers-cli">

```bash
# Using transformers-cli for quick inference
python -m transformers.models.bert_generation --model google/roberta2roberta_L-24_discofuse --input "This is the first sentence. This is the second sentence."
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [BitsAndBytesConfig](../main_classes/quantization#transformers.BitsAndBytesConfig) to quantize the weights to 4-bit.

```python
from transformers import BertGenerationEncoder, BertTokenizer, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load quantized model
encoder = BertGenerationEncoder.from_pretrained(
    "google-bert/bert-large-uncased",
    quantization_config=quantization_config,
    bos_token_id=101,
    eos_token_id=102
)
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")
```

## Notes

- BertGenerationEncoder and BertGenerationDecoder should be used in combination with EncoderDecoderModel for sequence-to-sequence tasks.
- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.
- No EOS token should be added to the end of the input for most generation tasks.

   ```python
   # Example of creating a complete encoder-decoder setup
   from transformers import EncoderDecoderModel, AutoTokenizer
   
   # Load pre-trained encoder-decoder model
   model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse")
   tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")
   
   # Generate text
   input_text = "This is the first sentence. This is the second sentence."
   input_ids = tokenizer(input_text, add_special_tokens=False, return_tensors="pt").input_ids
   outputs = model.generate(input_ids)
   result = tokenizer.decode(outputs[0])
   ```

## Resources

- [Original Paper: Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
- [Google Research Blog Post](https://ai.googleblog.com/2020/01/leveraging-bert-for-sequence-generation.html)

## BertGenerationConfig

[[autodoc]] BertGenerationConfig

## BertGenerationTokenizer

[[autodoc]] BertGenerationTokenizer
    - save_vocabulary

## BertGenerationEncoder

[[autodoc]] BertGenerationEncoder
    - forward

## BertGenerationDecoder

[[autodoc]] BertGenerationDecoder
    - forward