<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# টেক্সট জেনারেশন

[[open-in-colab]]

বড় ল্যাঙ্গুয়েজ মডেল (LLM)-এর সবচেয়ে জনপ্রিয় ব্যবহার হলো টেক্সট জেনারেশন। একটি LLM-কে এমনভাবে ট্রেইন করা হয় যেন এটি কোনো শুরুর টেক্সট (prompt) পেলে পরের শব্দ বা টোকেন predict করতে পারে, আর এভাবেই এটি ধীরে ধীরে পুরো আউটপুট তৈরি করে যতক্ষণ না নির্দিষ্ট length-এ পৌঁছে যায় বা `EOS` token পায়।

Transformers-এ [`~GenerationMixin.generate`] API পুরো text generation handle করে, আর এটি সব generative model-এর জন্য available। এই গাইডে আমরা [`~GenerationMixin.generate`] দিয়ে basic text generation আর কিছু common mistake এড়িয়ে চলার উপায় দেখবো।

> [!TIP]
> নিচের command গুলো চালানোর আগে নিশ্চিত হয়ে নিও যে [`transformers serve` চালু আছে](https://huggingface.co/docs/transformers/main/en/serving)।
>
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

## Default generate

শুরু করার আগে [bitsandbytes](https://hf.co/docs/bitsandbytes/index) install করে নেওয়া ভালো। এটি বড় model-গুলোকে quantize করে memory usage কমাতে সাহায্য করে।

```bash
!pip install -U transformers bitsandbytes
```

Bitsandbytes শুধু CUDA GPU নয়, আরও কয়েক ধরনের backend support করে। বিস্তারিত জানতে multi-backend installation [guide](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend) দেখে নিতে পারো।

[`~PreTrainedModel.from_pretrained`] দিয়ে একটি LLM load করো এবং memory usage কমানোর জন্য নিচের দুইটা parameter ব্যবহার করো।

- `device_map="auto"` Automatically সবচেয়ে fast device (যেমন GPU)-এ model load করে।
- `quantization_config` quantization settings define করে। এখানে bitsandbytes backend ব্যবহার করে model-কে [4-bit](./quantization/bitsandbytes) mode-এ load করা হয়েছে।

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", quantization_config=quantization_config)
```

Input tokenize করো এবং [`~PreTrainedTokenizer.padding_side`] parameter `"left"` সেট করো কারণ LLM padding token থেকে generation continue করার জন্য train করা হয় না। Tokenizer input ids এবং attention mask return করবে।

> [!TIP]
> একসাথে একাধিক prompt process করতে tokenizer-এ string-এর list pass করতে পারো। এতে throughput বাড়ে, যদিও latency আর memory usage একটু বাড়তে পারে।

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
```

এরপর [`~GenerationMixin.generate`] দিয়ে token generate করো এবং [`~PreTrainedTokenizer.batch_decode`] দিয়ে generated token-গুলোকে আবার text-এ convert করো।

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
"A list of colors: red, blue, green, yellow, orange, purple, pink,"
```

## Generation configuration

সব generation settings [`GenerationConfig`]-এর মধ্যে থাকে। উপরের example-এ settings এসেছে [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)-এর `generation_config.json` file থেকে। যদি model-এর সাথে কোনো config save না থাকে তাহলে default decoding strategy ব্যবহার করা হয়।

`generation_config` attribute দিয়ে configuration inspect করতে পারো। এখানে শুধু default configuration থেকে যেগুলো আলাদা সেগুলোই দেখায়।

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model.generation_config
GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

[`GenerationConfig`] override করে [`~GenerationMixin.generate`] customize করতে পারো।

```py
# enable beam search sampling strategy
model.generate(**inputs, num_beams=4, do_sample=True)
```

[`~GenerationMixin.generate`] external library বা custom code দিয়েও extend করা যায়।

1. `logits_processor` custom [`LogitsProcessor`] ব্যবহার করে next token probability modify করতে পারে।
2. `stopping_criteria` custom [`StoppingCriteria`] ব্যবহার করে generation stop করতে পারে।
3. `custom_generate` flag দিয়ে custom decoding method load করা যায়।

আরও decoding strategy জানতে [Generation strategies](./generation_strategies) guide দেখে নিতে পারো।

### Saving

নিজের decoding parameter দিয়ে [`GenerationConfig`] instance তৈরি করো।

```py
from transformers import AutoModelForCausalLM, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("my_account/my_model")
generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)
```

[`~GenerationConfig.save_pretrained`] দিয়ে generation configuration save করো এবং `push_to_hub=True` দিলে Hub-এ upload হয়ে যাবে।

```py
generation_config.save_pretrained("my_account/my_model", push_to_hub=True)
```

`config_file_name` empty রাখো। এটি mainly একই directory-তে multiple generation config save করার জন্য ব্যবহার করা হয়।

```py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json", push_to_hub=True)

generation_config = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## Common Options

[`~GenerationMixin.generate`] অনেক powerful এবং heavily customizable। নতুনদের জন্য এটা একটু overwhelming লাগতে পারে। নিচে সবচেয়ে common generation options দেওয়া হলো।

| Option name | Type | সহজ ব্যাখ্যা |
|---|---|---|
| `max_new_tokens` | `int` | Maximum generation length control করে |
| `do_sample` | `bool` | Next token sampling হবে নাকি greedy generation হবে তা ঠিক করে |
| `temperature` | `float` | Output কতটা creative বা unpredictable হবে তা control করে |
| `num_beams` | `int` | Beam search enable করে |
| `repetition_penalty` | `float` | Model যেন একই জিনিস বারবার repeat না করে সেটা control করে |
| `eos_token_id` | `list[int]` | কোন token generation stop করবে তা define করে |

## Pitfalls

নিচে text generation-এর সময় যেসব common সমস্যা দেখা যায় আর সেগুলোর সমাধান দেওয়া হলো।

### Output length

[`~GenerationMixin.generate`] default ভাবে প্রায় 20 token পর্যন্ত generate করে যদি model config-এ অন্য কিছু define না থাকে। Output length control করার জন্য `max_new_tokens` manually set করা ভালো।

```py
model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to(model.device)
```

<hfoptions id="output-length">
<hfoption id="default length">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'
```

</hfoption>
<hfoption id="max_new_tokens">

```py
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

</hfoption>
</hfoptions>

### Decoding strategy

[`~GenerationMixin.generate`] default ভাবে greedy search ব্যবহার করে। এটি translation বা transcription-এর মতো task-এর জন্য ভালো হলেও creative task-এর জন্য খুব ভালো না।

আরও diverse output পাওয়ার জন্য multinomial sampling ব্যবহার করতে পারো।

```py
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to(model.device)
```

<hfoptions id="decoding">
<hfoption id="greedy search">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
<hfoption id="multinomial sampling">

```py
generated_ids = model.generate(**model_inputs, do_sample=True)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

### Padding side

সব input যদি same length না হয় তাহলে padding দরকার হয়। কিন্তু LLM padding token থেকে generation continue করতে trained না, তাই [`~PreTrainedTokenizer.padding_side`] `"left"` হওয়া দরকার।

<hfoptions id="padding">
<hfoption id="right pad">

```py
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'
```

</hfoption>
<hfoption id="left pad">

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

</hfoption>
</hfoptions>

### Prompt format

কিছু model বা task নির্দিষ্ট prompt format expect করে। Format ভুল হলে model suboptimal output দিতে পারে।

উদাহরণ হিসেবে, chat model সাধারণত [chat template](./chat_templating) expect করে যেখানে `role` আর `content` define করা থাকে।

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
```

<hfoptions id="format">
<hfoption id="no format">

```py
prompt = """How many cats does it take to change a light bulb? Reply as a pirate."""
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Aye, matey! 'Tis a simple task for a cat with a keen eye and nimble paws. First, the cat will climb up the ladder, carefully avoiding the rickety rungs. Then, with"
```

</hfoption>
<hfoption id="chat template">

```py
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many cats does it take to change a light bulb?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
input_length = model_inputs.shape[1]
generated_ids = model.generate(model_inputs, do_sample=True, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Arr, matey! According to me beliefs, 'twas always one cat to hold the ladder and another to climb up it an’ change the light bulb, but if yer looking to save some catnip, maybe yer can
```

</hfoption>
</hfoptions>

## Resources

আরও কিছু useful text generation library নিচে দেওয়া হলো।

- [Optimum](https://github.com/huggingface/optimum): নির্দিষ্ট hardware-এর জন্য training আর inference optimize করে
- [Outlines](https://github.com/dottxt-ai/outlines): constrained text generation-এর জন্য useful
- [SynCode](https://github.com/uiuc-focal-lab/syncode): grammar-guided generation support করে
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference): production-ready LLM server
- [Text generation web UI](https://github.com/oobabooga/text-generation-webui): text generation-এর জন্য Gradio UI
- [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo): অতিরিক্ত logits processor collection
