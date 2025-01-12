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

# Export to ONNX

Deploying 🤗 Transformers models in production environments often requires, or can benefit from exporting the models into 
a serialized format that can be loaded and executed on specialized runtimes and hardware.

🤗 Optimum is an extension of Transformers that enables exporting models from PyTorch or TensorFlow to serialized formats 
such as ONNX and TFLite through its `exporters` module. 🤗 Optimum also provides a set of performance optimization tools to train 
and run models on targeted hardware with maximum efficiency.

This guide demonstrates how you can export 🤗 Transformers models to ONNX with 🤗 Optimum, for the guide on exporting models to TFLite, 
please refer to the [Export to TFLite page](tflite).

## Export to ONNX 

[ONNX (Open Neural Network eXchange)](http://onnx.ai) is an open standard that defines a common set of operators and a 
common file format to represent deep learning models in a wide variety of frameworks, including PyTorch and
TensorFlow. When a model is exported to the ONNX format, these operators are used to
construct a computational graph (often called an _intermediate representation_) which
represents the flow of data through the neural network.

By exposing a graph with standardized operators and data types, ONNX makes it easy to
switch between frameworks. For example, a model trained in PyTorch can be exported to
ONNX format and then imported in TensorFlow (and vice versa).

Once exported to ONNX format, a model can be:
- optimized for inference via techniques such as [graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) and [quantization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization). 
- run with ONNX Runtime via [`ORTModelForXXX` classes](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort),
which follow the same `AutoModel` API as the one you are used to in 🤗 Transformers.
- run with [optimized inference pipelines](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines),
which has the same API as the [`pipeline`] function in 🤗 Transformers. 

🤗 Optimum provides support for the ONNX export by leveraging configuration objects. These configuration objects come 
ready-made for a number of model architectures, and are designed to be easily extendable to other architectures.

For the list of ready-made configurations, please refer to [🤗 Optimum documentation](https://huggingface.co/docs/optimum/exporters/onnx/overview).

There are two ways to export a 🤗 Transformers model to ONNX, here we show both:

- export with 🤗 Optimum via CLI.
- export with 🤗 Optimum with `optimum.onnxruntime`.

### Exporting a 🤗 Transformers model to ONNX with CLI

To export a 🤗 Transformers model to ONNX, first install an extra dependency:

```bash
pip install optimum[exporters]
```

To check out all available arguments, refer to the [🤗 Optimum docs](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli), 
or view help in command line:

```bash
optimum-cli export onnx --help
```

To export a model's checkpoint from the 🤗 Hub, for example, `distilbert/distilbert-base-uncased-distilled-squad`, run the following command: 

```bash
optimum-cli export onnx --model distilbert/distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

You should see the logs indicating progress and showing where the resulting `model.onnx` is saved, like this:

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

The example above illustrates exporting a checkpoint from 🤗 Hub. When exporting a local model, first make sure that you 
saved both the model's weights and tokenizer files in the same directory (`local_path`). When using CLI, pass the 
`local_path` to the `model` argument instead of the checkpoint name on 🤗 Hub and provide the `--task` argument. 
You can review the list of supported tasks in the [🤗 Optimum documentation](https://huggingface.co/docs/optimum/exporters/task_manager).
If `task` argument is not provided, it will default to the model architecture without any task specific head.

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

The resulting `model.onnx` file can then be run on one of the [many
accelerators](https://onnx.ai/supported-tools.html#deployModel) that support the ONNX
standard. For example, we can load and run the model with [ONNX
Runtime](https://onnxruntime.ai/) as follows:

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

The process is identical for TensorFlow checkpoints on the Hub. For instance, here's how you would
export a pure TensorFlow checkpoint from the [Keras organization](https://huggingface.co/keras-io):

```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### Exporting a 🤗 Transformers model to ONNX with `optimum.onnxruntime`

Alternative to CLI, you can export a 🤗 Transformers model to ONNX programmatically like so: 

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # Load a model from transformers and export it to ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # Save the onnx model and tokenizer
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### Exporting a model for an unsupported architecture

If you wish to contribute by adding support for a model that cannot be currently exported, you should first check if it is
supported in [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview),
and if it is not, [contribute to 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)
directly.

### Exporting a model with `transformers.onnx`

<Tip warning={true}>

`transformers.onnx` is no longer maintained, please export models with 🤗 Optimum as described above. This section will be removed in the future versions.

</Tip>

To export a 🤗 Transformers model to ONNX with `transformers.onnx`, install extra dependencies:

```bash
pip install transformers[onnx]
```

Use `transformers.onnx` package as a Python module to export a checkpoint using a ready-made configuration:

```bash
python -m transformers.onnx --model=distilbert/distilbert-base-uncased onnx/
```

This exports an ONNX graph of the checkpoint defined by the `--model` argument. Pass any checkpoint on the 🤗 Hub or one that's stored locally.
The resulting `model.onnx` file can then be run on one of the many accelerators that support the ONNX standard. For example, 
load and run the model with ONNX Runtime as follows:

```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime expects NumPy arrays as input
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

The required output names (like `["last_hidden_state"]`) can be obtained by taking a look at the ONNX configuration of 
each model. For example, for DistilBERT we have:

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

The process is identical for TensorFlow checkpoints on the Hub. For example, export a pure TensorFlow checkpoint like so:

```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

To export a model that's stored locally, save the model's weights and tokenizer files in the same directory (e.g. `local-pt-checkpoint`), 
then export it to ONNX by pointing the `--model` argument of the `transformers.onnx` package to the desired directory:

```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```