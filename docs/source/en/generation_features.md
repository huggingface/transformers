<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Generation features

The [`~GenerationMixin.generate`] API supports a couple features for building applications on top of it.

This guide will show you how to use these features.

## Streaming

Streaming starts returning text as soon as it is generated so you don't have to wait to see the entire generated response all at once. It is important in user-facing applications because it reduces perceived latency and allows users to see the generation progression.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/streaming-generation-visual-dark_360.gif"/>
</div>

> [!TIP]
> Learn more about streaming in the [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming) docs.

Create an instance of [`TextStreamer`] with the tokenizer. Pass [`TextStreamer`] to the `streamer` parameter in [`~GenerationMixin.generate`] to stream the output one word at a time.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt")
streamer = TextStreamer(tokenizer)

_ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
```

The `streamer` parameter is compatible with any class with a [`~TextStreamer.put`] and [`~TextStreamer.end`] method. [`~TextStreamer.put`] pushes new tokens and [`~TextStreamer.end`] flags the end of generation. You can create your own streamer class as long as they include these two methods, or you can use Transformers' basic streamer classes.

## Watermarking

Watermarking is useful for detecting whether text is generated. The [watermarking strategy](https://hf.co/papers/2306.04634) in Transformers randomly "colors" a subset of the tokens green. When green tokens are generated, they have a small bias added to their logits, and a higher probability of being generated. You can detect generated text by comparing the proportion of green tokens to the amount of green tokens typically found in human-generated text.

Watermarking is supported for any generative model in Transformers and doesn't require an extra classification model to detect the watermarked text.

Create a [`WatermarkingConfig`] with the bias value to add to the logits and watermarking algorithm. The example below uses the `"selfhash"` algorithm, where the green token selection only depends on the current token. Pass the [`WatermarkingConfig`] to [`~GenerationMixin.generate`].

> [!TIP]
> The [`WatermarkDetector`] class detects the proportion of green tokens in generated text, which is why it is recommended to strip the prompt text, if it is much longer than the generated text. Padding can also have an effect on [`WatermarkDetector`].

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkDetector, WatermarkingConfig

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

inputs = tokenizer(["This is the beginning of a long story", "Alice and Bob are"], padding=True, return_tensors="pt")
input_len = inputs["input_ids"].shape[-1]

watermarking_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
out = model.generate(**inputs, watermarking_config=watermarking_config, do_sample=False, max_length=20)
```

Create an instance of [`WatermarkDetector`] and pass the model output to it to detect whether the text is machine-generated. The [`WatermarkDetector`] must have the same [`WatermarkingConfig`] used during generation.

```py
detector = WatermarkDetector(model_config=model.config, device="cpu", watermarking_config=watermarking_config)
detection_out = detector(out, return_dict=True)
detection_out.prediction
array([True, True])
```
