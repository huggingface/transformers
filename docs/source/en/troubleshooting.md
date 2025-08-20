<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Troubleshoot

Sometimes errors occur, but we are here to help! This guide covers some of the most common issues we've seen and how you can resolve them. However, this guide isn't meant to be a comprehensive collection of every 🤗 Transformers issue. For more help with troubleshooting your issue, try:

<Youtube id="S2EEG3JIt2A"/>

1. Asking for help on the [forums](https://discuss.huggingface.co/). There are specific categories you can post your question to, like [Beginners](https://discuss.huggingface.co/c/beginners/5) or [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9). Make sure you write a good descriptive forum post with some reproducible code to maximize the likelihood that your problem is solved!

<Youtube id="_PAli-V4wj0"/>

2. Create an [Issue](https://github.com/huggingface/transformers/issues/new/choose) on the 🤗 Transformers repository if it is a bug related to the library. Try to include as much information describing the bug as possible to help us better figure out what's wrong and how we can fix it.

3. Check the [Migration](migration) guide if you use an older version of 🤗 Transformers since some important changes have been introduced between versions.

For more details about troubleshooting and getting help, take a look at [Chapter 8](https://huggingface.co/course/chapter8/1?fw=pt) of the Hugging Face course.


## Firewalled environments

Some GPU instances on cloud and intranet setups are firewalled to external connections, resulting in a connection error. When your script attempts to download model weights or datasets, the download will hang and then timeout with the following message:

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

In this case, you should try to run 🤗 Transformers on [offline mode](installation#offline-mode) to avoid the connection error.

## CUDA out of memory

Training large models with millions of parameters can be challenging without the appropriate hardware. A common error you may encounter when the GPU runs out of memory is:

```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

Here are some potential solutions you can try to lessen memory use:

- Reduce the [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) value in [`TrainingArguments`].
- Try using [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps) in [`TrainingArguments`] to effectively increase overall batch size.

<Tip>

Refer to the Performance [guide](performance) for more details about memory-saving techniques.

</Tip>

## ImportError

Another common error you may encounter, especially if it is a newly released model, is `ImportError`:

```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

For these error types, check to make sure you have the latest version of 🤗 Transformers installed to access the most recent models:

```bash
pip install transformers --upgrade
```

## CUDA error: device-side assert triggered

Sometimes you may run into a generic CUDA error about an error in the device code.

```
RuntimeError: CUDA error: device-side assert triggered
```

You should try to run the code on a CPU first to get a more descriptive error message. Add the following environment variable to the beginning of your code to switch to a CPU:

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

Another option is to get a better traceback from the GPU. Add the following environment variable to the beginning of your code to get the traceback to point to the source of the error:

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## Incorrect output when padding tokens aren't masked

In some cases, the output `hidden_state` may be incorrect if the `input_ids` include padding tokens. To demonstrate, load a model and tokenizer. You can access a model's `pad_token_id` to see its value. The `pad_token_id` may be `None` for some models, but you can always manually set it.

```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
>>> model.config.pad_token_id
0
```

The following example shows the output without masking the padding tokens:

```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

Here is the actual output of the second sequence:

```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

Most of the time, you should provide an `attention_mask` to your model to ignore the padding tokens to avoid this silent error. Now the output of the second sequence matches its actual output:

<Tip>

By default, the tokenizer creates an `attention_mask` for you based on your specific tokenizer's defaults.

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗 Transformers doesn't automatically create an `attention_mask` to mask a padding token if it is provided because:

- Some models don't have a padding token.
- For some use-cases, users want a model to attend to a padding token.

## ValueError: Unrecognized configuration class XYZ for this kind of AutoModel

Generally, we recommend using the [`AutoModel`] class to load pretrained instances of models. This class
can automatically infer and load the correct architecture from a given checkpoint based on the configuration. If you see
this `ValueError` when loading a model from a checkpoint, this means the Auto class couldn't find a mapping from
the configuration in the given checkpoint to the kind of model you are trying to load. Most commonly, this happens when a
checkpoint doesn't support a given task.
For instance, you'll see this error in the following example because there is no GPT2 for question answering:

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("openai-community/gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("openai-community/gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
