<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TensorFlow로 TPU에서 훈련하기[[training-on-tpu-with-tensorflow]]

<Tip>

If you don't need long explanations and just want TPU code samples to get started with, check out [our TPU example notebook!](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)

</Tip>

### TPU가 무엇인가요?[[what-is-a-tpu]]

A TPU is a **Tensor Processing Unit.** They are hardware designed by Google, which are used to greatly speed up the tensor computations within neural networks, much like GPUs. They can be used for both network training and inference. They are generally accessed through Google’s cloud services, but small TPUs can also be accessed directly for free through Google Colab and Kaggle Kernels.

Because [all TensorFlow models in 🤗 Transformers are Keras models](https://huggingface.co/blog/tensorflow-philosophy), most of the methods in this document are generally applicable to TPU training for any Keras model! However, there are a few points that are specific to the HuggingFace ecosystem (hug-o-system?) of Transformers and Datasets, and we’ll make sure to flag them up when we get to them.

### 어떤 종류의 TPU가 있나요?[[what-kinds-of-tpu-are-available]]

New users are often very confused by the range of TPUs, and the different ways to access them. The first key distinction to understand is the difference between **TPU Nodes** and **TPU VMs.**

When you use a **TPU Node**, you are effectively indirectly accessing a remote TPU. You will need a separate VM, which will initialize your network and data pipeline and then forward them to the remote node. When you use a TPU on Google Colab, you are accessing it in the **TPU Node** style.

Using TPU Nodes can have some quite unexpected behaviour for people who aren’t used to them! In particular, because the TPU is located on a physically different system to the machine you’re running your Python code on, your data cannot be local to your machine - any data pipeline that loads from your machine’s internal storage will totally fail! Instead, data must be stored in Google Cloud Storage where your data pipeline can still access it, even when the pipeline is running on the remote TPU node.

<Tip>

If you can fit all your data in memory as `np.ndarray` or `tf.Tensor`, then you can `fit()` on that data even when using Colab or a TPU Node, without needing to upload it to Google Cloud Storage.

</Tip>

<Tip>

**🤗Specific Hugging Face Tip🤗:** The methods `Dataset.to_tf_dataset()` and its higher-level wrapper `model.prepare_tf_dataset()` , which you will see throughout our TF code examples, will both fail on a TPU Node. The reason for this is that even though they create a `tf.data.Dataset` it is not a “pure” `tf.data` pipeline and uses `tf.numpy_function` or `Dataset.from_generator()` to stream data from the underlying HuggingFace `Dataset`. This HuggingFace `Dataset` is backed by data that is on a local disc and which the remote TPU Node will not be able to read.

</Tip>

The second way to access a TPU is via a **TPU VM.** When using a TPU VM, you connect directly to the machine that the TPU is attached to, much like training on a GPU VM. TPU VMs are generally easier to work with, particularly when it comes to your data pipeline. All of the above warnings do not apply to TPU VMs!

This is an opinionated document, so here’s our opinion: **Avoid using TPU Node if possible.** It is more confusing and more difficult to debug than TPU VMs. It is also likely to be unsupported in future - Google’s latest TPU, TPUv4, can only be accessed as a TPU VM, which suggests that TPU Nodes are increasingly going to become a “legacy” access method. However, we understand that the only free TPU access is on Colab and Kaggle Kernels, which uses TPU Node - so we’ll try to explain how to handle it if you have to! Check the [TPU example notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) for code samples that explain this in more detail.

### 어떤 크기의 TPU를 사용할 수 있나요?[[what-sizes-of-tpu-are-available]]

A single TPU (a v2-8/v3-8/v4-8) runs 8 replicas. TPUs exist in **pods** that can run hundreds or thousands of replicas simultaneously. When you use more than a single TPU but less than a whole pod (for example, a v3-32), your TPU fleet is referred to as a **pod slice.**

When you access a free TPU via Colab, you generally get a single v2-8 TPU.

### XLA에 대해 들어본 적이 있습니다. XLA란 무엇이고 TPU와 어떤 관련이 있나요?[[i-keep-hearing-about-this-xla-thing-whats-xla-and-how-does-it-relate-to-tpus]]

XLA is an optimizing compiler, used by both TensorFlow and JAX. In JAX it is the only compiler, whereas in TensorFlow it is optional (but mandatory on TPU!). The easiest way to enable it when training a Keras model is to pass the argument `jit_compile=True` to `model.compile()`. If you don’t get any errors and performance is good, that’s a great sign that you’re ready to move to TPU!

Debugging on TPU is generally a bit harder than on CPU/GPU, so we recommend getting your code running on CPU/GPU with XLA first before trying it on TPU. You don’t have to train for long, of course - just for a few steps to make sure that your model and data pipeline are working like you expect them to.

<Tip>

XLA compiled code is usually faster - so even if you’re not planning to run on TPU, adding `jit_compile=True` can improve your performance. Be sure to note the caveats below about XLA compatibility, though!

</Tip>

<Tip warning={true}>

**Tip born of painful experience:** Although using `jit_compile=True` is a good way to get a speed boost and test if your CPU/GPU code is XLA-compatible, it can actually cause a lot of problems if you leave it in when actually training on TPU. XLA compilation will happen implicitly on TPU, so remember to remove that line before actually running your code on a TPU!

</Tip>

### 제 XLA 모델과 호환하려면 어떻게 해야 하나요?[[how-do-i-make-my-model-xla-compatible]]

In many cases, your code is probably XLA-compatible already! However, there are a few things that work in normal TensorFlow that don’t work in XLA. We’ve distilled them into three core rules below:

<Tip>

**🤗Specific HuggingFace Tip🤗:** We’ve put a lot of effort into rewriting our TensorFlow models and loss functions to be XLA-compatible. Our models and loss functions generally obey rule #1 and #2 by default, so you can skip over them if you’re using `transformers` models. Don’t forget about these rules when writing your own models and loss functions, though!

</Tip>

#### XLA 규칙 #1: 코드에서 “데이터 의존 조건문”을 사용할 수 없습니다[[xla-rule-1-your-code-cannot-have-datadependent-conditionals]]

What that means is that any `if` statement cannot depend on values inside a `tf.Tensor`. For example, this code block cannot be compiled with XLA!

```python
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

This might seem very restrictive at first, but most neural net code doesn’t need to do this. You can often get around this restriction by using `tf.cond` (see the documentation [here](https://www.tensorflow.org/api_docs/python/tf/cond)) or by removing the conditional and finding a clever math trick with indicator variables instead, like so:

```python
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

This code has exactly the same effect as the code above, but by avoiding a conditional, we ensure it will compile with XLA without problems!

#### XLA 규칙 #2: 코드에서 "데이터 의존 크기"를 가질 수 없습니다[[xla-rule-2-your-code-cannot-have-datadependent-shapes]]

What this means is that the shape of all of the `tf.Tensor` objects in your code cannot depend on their values. For example, the function `tf.unique` cannot be compiled with XLA, because it returns a `tensor` containing one instance of each unique value in the input. The shape of this output will obviously be different depending on how repetitive the input `Tensor` was, and so XLA refuses to handle it!

In general, most neural network code obeys rule #2 by default. However, there are a few common cases where it becomes a problem. One very common one is when you use **label masking**, setting your labels to a negative value to indicate that those positions should be ignored when computing the loss. If you look at NumPy or PyTorch loss functions that support label masking, you will often see code like this that uses [boolean indexing](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing):

```python
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

This code is totally fine in NumPy or PyTorch, but it breaks in XLA! Why? Because the shape of `masked_outputs` and `masked_labels` depends on how many positions are masked - that makes it a **data-dependent shape.** However, just like for rule #1, we can often rewrite this code to yield exactly the same output without any data-dependent shapes.

```python
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask  # Set negative label positions to 0
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```

Here, we avoid data-dependent shapes by computing the loss for every position, but zeroing out the masked positions in both the numerator and denominator when we calculate the mean, which yields exactly the same result as the first block while maintaining XLA compatibility. Note that we use the same trick as in rule #1 - converting a `tf.bool` to `tf.float32` and using it as an indicator variable. This is a really useful trick, so remember it if you need to convert your own code to XLA!

#### XLA 규칙 #3: XLA는 각기 다른 입력 크기가 나타날 때마다 모델을 다시 컴파일해야 합니다[[xla-rule-3-xla-will-need-to-recompile-your-model-for-every-different-input-shape-it-sees]]

This is the big one. What this means is that if your input shapes are very variable, XLA will have to recompile your model over and over, which will create huge performance problems. This commonly arises in NLP models, where input texts have variable lengths after tokenization. In other modalities, static shapes are more common and this rule is much less of a problem.

How can you get around rule #3? The key is **padding** - if you pad all your inputs to the same length, and then use an `attention_mask`, you can get the same results as you’d get from variable shapes, but without any XLA issues. However, excessive padding can cause severe slowdown too - if you pad all your samples to the maximum length in the whole dataset, you might end up with batches consisting endless padding tokens, which will waste a lot of compute and memory!

There isn’t a perfect solution to this problem. However, you can try some tricks. One very useful trick is to **pad batches of samples up to a multiple of a number like 32 or 64 tokens.** This often only increases the number of tokens by a small amount, but it hugely reduces the number of unique input shapes, because every input shape now has to be a multiple of 32 or 64. Fewer unique input shapes means fewer XLA compilations!

<Tip>

**🤗Specific HuggingFace Tip🤗:** Our tokenizers and data collators have methods that can help you here. You can use `padding="max_length"` or `padding="longest"` when calling tokenizers to get them to output padded data. Our tokenizers and data collators also have a `pad_to_multiple_of` argument that you can use to reduce the number of unique input shapes you see!

</Tip>

### 실제 TPU로 모델을 훈련하려면 어떻게 해야 하나요?[[how-do-i-actually-train-my-model-on-tpu]]

Once your training is XLA-compatible and (if you’re using TPU Node / Colab) your dataset has been prepared appropriately, running on TPU is surprisingly easy! All you really need to change in your code is to add a few lines to initialize your TPU, and to ensure that your model and dataset are created inside a `TPUStrategy` scope. Take a look at [our TPU example notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) to see this in action!

### 요약[[summary]]

There was a lot in here, so let’s summarize with a quick checklist you can follow when you want to get your model ready for TPU training:

- Make sure your code follows the three rules of XLA
- Compile your model with `jit_compile=True` on CPU/GPU and confirm that you can train it with XLA
- Either load your dataset into memory or use a TPU-compatible dataset loading approach (see [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb))
- Migrate your code either to Colab (with accelerator set to “TPU”) or a TPU VM on Google Cloud
- Add TPU initializer code (see [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb))
- Create your `TPUStrategy` and make sure dataset loading and model creation are inside the `strategy.scope()` (see [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb))
- Don’t forget to take `jit_compile=True` out again when you move to TPU!
- 🙏🙏🙏🥺🥺🥺
- Call model.fit()
- You did it!