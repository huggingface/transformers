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

# পাইপলাইন

[`Pipeline`] হলো খুবই সহজ কিন্তু powerful inference API যেটা Hugging Face [Hub](https://hf.co/models)-এর প্রায় সব ধরনের machine learning task-এর জন্য ব্যবহার করা যায়।

তোমার task অনুযায়ী [`Pipeline`] customize করা যায়। যেমন automatic speech recognition (ASR)-এ timestamp add করে meeting note transcribe করা যায়। [`Pipeline`] GPU, Apple Silicon আর half-precision weight support করে, ফলে inference আরও fast হয় আর memory কম লাগে।

<Youtube id=tiZFewofSLM/>

Transformers-এ দুই ধরনের pipeline class আছে:

- Generic [`Pipeline`]
- Task-specific pipeline যেমন [`TextGenerationPipeline`]

Task-specific pipeline load করতে [`Pipeline`]-এর `task` parameter-এ task identifier দিতে হয়।

প্রতিটি task-এর জন্য default pretrained model আর preprocessor থাকে। তবে চাইলে `model` parameter দিয়ে অন্য model ব্যবহার করা যায়।

উদাহরণ হিসেবে [`TextGenerationPipeline`] আর [Gemma 2](./model_doc/gemma2) ব্যবহার করা যাক।

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}]
```

একাধিক input থাকলে list আকারে pass করো।

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
[[{'generated_text': 'the secret to baking a really good cake is 1. the right ingredients 2. the'}],
 [{'generated_text': 'a baguette is 100% bread.\n\na baguette is 100%'}]]
```

এই guide-এ আমরা [`Pipeline`] কীভাবে কাজ করে, এর feature গুলো, আর বিভিন্ন parameter কীভাবে configure করতে হয় তা দেখবো।

## Tasks

[`Pipeline`] অনেক ধরনের machine learning task support করে, বিভিন্ন modality-সহ।

শুধু সঠিক input pass করলেই pipeline বাকি সব handle করে নেয়।

নিচে কয়েকটা উদাহরণ দেওয়া হলো।

<hfoptions id="tasks">
<hfoption id="automatic speech recognition">

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</hfoption>
<hfoption id="image classification">

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224")
pipeline(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
[{'label': 'lynx, catamount', 'score': 0.43350091576576233},
 {'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
  'score': 0.034796204417943954},
 {'label': 'snow leopard, ounce, Panthera uncia',
  'score': 0.03240183740854263},
 {'label': 'Egyptian cat', 'score': 0.02394474856555462},
 {'label': 'tiger cat', 'score': 0.02288915030658245}]
```

</hfoption>
<hfoption id="visual question answering">

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</hfoption>
</hfoptions>

## Parameters

Minimum requirement হিসেবে [`Pipeline`] শুধু task identifier, model আর input চায়।

তবে performance optimize করা থেকে শুরু করে task-specific customization পর্যন্ত অনেক parameter available আছে।

এই section-এ সবচেয়ে গুরুত্বপূর্ণ parameter গুলো দেখবো।

### Device

[`Pipeline`] GPU, CPU, Apple Silicon সহ বিভিন্ন hardware support করে।

`device` parameter দিয়ে hardware select করা হয়।

Default ভাবে [`Pipeline`] CPU-তে run করে, যেটা `device=-1` দিয়ে বোঝানো হয়।

<hfoptions id="device">
<hfoption id="GPU">

GPU-তে [`Pipeline`] run করতে CUDA device id ব্যবহার করো।

যেমন `device=0` মানে প্রথম GPU।

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=0)
pipeline("the secret to baking a really good cake is ")
```

তুমি চাইলে [Accelerate](https://hf.co/docs/accelerate/index) ব্যবহার করে automatically device select করাতে পারো।

এটি সবচেয়ে fast device-এ model load করে তারপর প্রয়োজন হলে CPU বা hard drive ব্যবহার করে।

`device_map="auto"` সেট করলেই হবে।

> [!TIP]
> আগে নিশ্চিত হয়ে নিও যে [Accelerate](https://hf.co/docs/accelerate/basic_tutorials/install) install করা আছে।
>
> ```py
> !pip install -U accelerate
> ```

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device_map="auto")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
<hfoption id="Apple silicon">

Apple Silicon-এ run করতে `device="mps"` ব্যবহার করো।

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device="mps")
pipeline("the secret to baking a really good cake is ")
```

</hfoption>
</hfoptions>

### Batch inference

[`Pipeline`] `batch_size` parameter ব্যবহার করে batch inference support করে।

GPU-তে এটি অনেক সময় speed বাড়ায়, যদিও সবসময় guaranteed না।

উদাহরণ:

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device, batch_size=2)
pipeline(["the secret to baking a really good cake is", "a baguette is", "paris is the", "hotdogs are"])
[[{'generated_text': 'the secret to baking a really good cake is to use a good cake mix.\n\ni’'}],
 [{'generated_text': 'a baguette is'}],
 [{'generated_text': 'paris is the most beautiful city in the world.\n\ni’ve been to paris 3'}],
 [{'generated_text': 'hotdogs are a staple of the american diet. they are a great source of protein and can'}]]
```

Streaming data-এর ক্ষেত্রেও batch inference useful।

```py
from transformers import pipeline
from accelerate import Accelerator
from transformers.pipelines.pt_utils import KeyDataset
import datasets

device = Accelerator().device

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Batch inference ব্যবহার করার সময় কয়েকটা rule মাথায় রাখো:

1. নিজের hardware আর data দিয়ে performance measure করো।
2. Low latency application হলে batch ব্যবহার না করাই ভালো।
3. CPU-তে batch inference খুব useful না।
4. Sequence length unpredictable হলে OOM issue হতে পারে।
5. GPU বড় হলে batch inference বেশি useful হয়।
6. OOM error handle করার ব্যবস্থা রাখো।

### Task-specific parameters

[`Pipeline`] task অনুযায়ী আলাদা parameter support করে।

নিচে কয়েকটা useful example দেওয়া হলো।

<hfoptions id="task-specific-parameters">
<hfoption id="automatic speech recognition">

`return_timestamps="word"` দিলে প্রতিটি শব্দ কখন বলা হয়েছে তা return করবে।

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamp="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)}]}
```

</hfoption>
<hfoption id="text generation">

`return_full_text=False` দিলে শুধু generated text return করবে।

`num_return_sequences` ব্যবহার করে multiple output generate করা যায়।

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
pipeline("the secret to baking a good cake is", num_return_sequences=4, return_full_text=False)
[{'generated_text': ' how easy it is for me to do it with my hands. You must not go nuts, or the cake is going to fall out.'},
 {'generated_text': ' to prepare the cake before baking. The key is to find the right type of icing to use and that icing makes an amazing frosting cake.\n\nFor a good icing cake, we give you the basics'},
 {'generated_text': " to remember to soak it in enough water and don't worry about it sticking to the wall. In the meantime, you could remove the top of the cake and let it dry out with a paper towel.\n"},
 {'generated_text': ' the best time to turn off the oven and let it stand 30 minutes. After 30 minutes, stir and bake a cake in a pan until fully moist.\n\nRemove the cake from the heat for about 12'}]
```

</hfoption>
</hfoptions>

## Chunk batching

কিছু ক্ষেত্রে input chunk করে process করতে হয়।

যেমন:

- খুব বড় audio file
- zero-shot classification
- question answering

এই কাজের জন্য [ChunkPipeline](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/pipelines/base.py#L1387) ব্যবহার করা হয়।

এটি automatically batching handle করতে পারে।

```py
# ChunkPipeline
all_model_outputs = []
for preprocessed in pipeline.preprocess(inputs):
    model_outputs = pipeline.model_forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs =pipeline.postprocess(all_model_outputs)

# Pipeline
preprocessed = pipeline.preprocess(inputs)
model_outputs = pipeline.forward(preprocessed)
outputs = pipeline.postprocess(model_outputs)
```

## Large datasets

বড় dataset-এর জন্য dataset iterate করে inference চালানো যায়।

এতে পুরো dataset memory-তে load করতে হয় না।

```py
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from accelerate import Accelerator
from datasets import load_dataset

device = Accelerator().device

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipeline = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=device)
for out in pipeline(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
```

Iterator বা generator দিয়েও inference করা যায়।

```py
def data():
    for i in range(1000):
        yield f"My example {i}"

pipeline = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipeline(data()):
    generated_characters += len(out[0]["generated_text"])
```

## Large models

[Accelerate](https://hf.co/docs/accelerate/index) large model-এর জন্য অনেক optimization দেয়।

আগে install করে নাও:

```py
!pip install -U accelerate
```

`device_map="auto"` automatically fastest device-এ model distribute করে।

[`Pipeline`] half-precision weight (`torch.float16`) support করে যা inference fast করে আর memory save করে।

Hardware support করলে `torch.bfloat16` ব্যবহার করা আরও ভালো।

> [!TIP]
> Input internally `torch.float16`-এ convert হয় এবং এটি শুধু PyTorch backend model-এর জন্য কাজ করে।

[`Pipeline`] quantized model-ও support করে।

আগে `bitsandbytes` install করো।

```py
import torch
from transformers import pipeline, BitsAndBytesConfig

pipeline = pipeline(
    model="google/gemma-7b",
    dtype=torch.bfloat16,
    device_map="auto",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True)
    }
)

pipeline("the secret to baking a good cake is ")
[{'generated_text': 'the secret to baking a good cake is 1. the right ingredients 2. the right'}]
```
