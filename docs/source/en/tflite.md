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

# Export to TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) is a lightweight framework for deploying machine learning models 
on resource-constrained devices, such as mobile phones, embedded systems, and Internet of Things (IoT) devices. 
TFLite is designed to optimize and run models efficiently on these devices with limited computational power, memory, and 
power consumption.
A TensorFlow Lite model is represented in a special efficient portable format identified by the `.tflite` file extension. 

🤗 Optimum offers functionality to export 🤗 Transformers models to TFLite through the `exporters.tflite` module. 
For the list of supported model architectures, please refer to [🤗 Optimum documentation](https://huggingface.co/docs/optimum/exporters/tflite/overview).

To export a model to TFLite, install the required dependencies:
 
```bash
pip install optimum[exporters-tf]
```

To check out all available arguments, refer to the [🤗 Optimum docs](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model), 
or view help in command line:

```bash
optimum-cli export tflite --help
```

To export a model's checkpoint from the 🤗 Hub, for example, `bert-base-uncased`, run the following command:

```bash
optimum-cli export tflite --model bert-base-uncased --sequence_length 128 bert_tflite/
```

You should see the logs indicating progress and showing where the resulting `model.tflite` is saved, like this:

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
 ```

The example above illustrates exporting a checkpoint from 🤗 Hub. When exporting a local model, first make sure that you 
saved both the model's weights and tokenizer files in the same directory (`local_path`). When using CLI, pass the 
`local_path` to the `model` argument instead of the checkpoint name on 🤗 Hub. 