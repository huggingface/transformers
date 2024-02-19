<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 用于 TensorFlow 模型的 XLA 集成

[[open-in-colab]]

加速线性代数，也称为XLA，是一个用于加速TensorFlow模型运行时间的编译器。从[官方文档](https://www.tensorflow.org/xla)中可以看到：

XLA（加速线性代数）是一种针对线性代数的特定领域编译器，可以在可能不需要更改源代码的情况下加速TensorFlow模型。

在TensorFlow中使用XLA非常简单——它包含在`tensorflow`库中，并且可以使用任何图创建函数中的`jit_compile`参数来触发，例如[`tf.function`](https://www.tensorflow.org/guide/intro_to_graphs)。在使用Keras方法如`fit()`和`predict()`时，只需将`jit_compile`参数传递给`model.compile()`即可启用XLA。然而，XLA不仅限于这些方法 - 它还可以用于加速任何任意的`tf.function`。

在🤗 Transformers中，几个TensorFlow方法已经被重写为与XLA兼容，包括[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)、[T5](https://huggingface.co/docs/transformers/model_doc/t5)和[OPT](https://huggingface.co/docs/transformers/model_doc/opt)等文本生成模型，以及[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)等语音处理模型。

虽然确切的加速倍数很大程度上取决于模型，但对于🤗 Transformers中的TensorFlow文本生成模型，我们注意到速度提高了约100倍。本文档将解释如何在这些模型上使用XLA获得最大的性能。如果您有兴趣了解更多关于基准测试和我们在XLA集成背后的设计哲学的信息，我们还将提供额外的资源链接。


## 使用 XLA 运行 TensorFlow 函数

让我们考虑以下TensorFlow 中的模型：

```py
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

上述模型接受维度为 `(10,)` 的输入。我们可以像下面这样使用模型进行前向传播：

```py
# Generate random inputs for the model.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass.
_ = model(random_inputs)
```

为了使用 XLA 编译的函数运行前向传播，我们需要执行以下操作：

```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

`model`的默认`call()`函数用于编译XLA图。但如果你想将其他模型函数编译成XLA，也是可以的，如下所示：

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## 在🤗 Transformers库中使用XLA运行TensorFlow文本生成模型

要在🤗 Transformers中启用XLA加速生成，您需要安装最新版本的`transformers`。您可以通过运行以下命令来安装它：

```bash
pip install transformers --upgrade
```

然后您可以运行以下代码：

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Will error if the minimal version of Transformers is not installed.
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

# One line to create an XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# Generated -- TensorFlow is an open-source, open-source, distributed-source application # framework for the
```

正如您所注意到的，在`generate()`上启用XLA只需要一行代码。其余部分代码保持不变。然而，上面的代码片段中有一些与XLA相关的注意事项。您需要了解这些注意事项，以充分利用XLA可能带来的性能提升。我们将在下面的部分讨论这些内容。

## 需要关注的注意事项

当您首次执行启用XLA的函数（如上面的`xla_generate()`）时，它将在内部尝试推断计算图，这是一个耗时的过程。这个过程被称为[“tracing”](https://www.tensorflow.org/guide/intro_to_graphs#when_is_a_function_tracing)。

您可能会注意到生成时间并不快。连续调用`xla_generate()`（或任何其他启用了XLA的函数）不需要再次推断计算图，只要函数的输入与最初构建计算图时的形状相匹配。对于具有固定输入形状的模态（例如图像），这不是问题，但如果您正在处理具有可变输入形状的模态（例如文本），则必须注意。

为了确保`xla_generate()`始终使用相同的输入形状，您可以在调用`tokenizer`时指定`padding`参数。

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# Here, we call the tokenizer with padding options.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

通过这种方式，您可以确保`xla_generate()`的输入始终具有它跟踪的形状，从而加速生成时间。您可以使用以下代码来验证这一点：

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
    tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
    start = time.time_ns()
    generated_tokens = xla_generate(**tokenized_input, num_beams=2)
    end = time.time_ns()
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

在Tesla T4 GPU上，您可以期望如下的输出：

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```

第一次调用`xla_generate()`会因为`tracing`而耗时，但后续的调用会快得多。请注意，任何时候对生成选项的更改都会触发重新`tracing`，从而导致生成时间减慢。

在本文档中，我们没有涵盖🤗 Transformers提供的所有文本生成选项。我们鼓励您阅读文档以了解高级用例。

## 附加资源

以下是一些附加资源，如果您想深入了解在🤗 Transformers和其他库下使用XLA：

* [这个Colab Notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb) 提供了一个互动演示，让您可以尝试使用XLA兼容的编码器-解码器（例如[T5](https://huggingface.co/docs/transformers/model_doc/t5)）和仅解码器（例如[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)）文本生成模型。

* [这篇博客文章](https://huggingface.co/blog/tf-xla-generate) 提供了XLA兼容模型的比较基准概述，以及关于在TensorFlow中使用XLA的友好介绍。

* [这篇博客文章](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html) 讨论了我们在🤗 Transformers中为TensorFlow模型添加XLA支持的设计理念。

* 推荐用于更多学习XLA和TensorFlow图的资源：
    * [XLA：面向机器学习的优化编译器](https://www.tensorflow.org/xla)
    * [图和tf.function简介](https://www.tensorflow.org/guide/intro_to_graphs)
    * [使用tf.function获得更好的性能](https://www.tensorflow.org/guide/function)