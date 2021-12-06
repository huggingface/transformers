<!---
Copyright 2021 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Bart + Beam Search to ONNX



This folder contains an example of exporting Bart + Beam Search generation (`BartForConditionalGeneration`) to ONNX.

Beam Search contains a for-loop workflow, so we need to make them TorchScript-compatible for exporting to ONNX. This example shows how to make a Bart model be TorchScript-compatible by wrapping up it into a new model. In addition, some changes were made to the `beam_search()` function to make it TorchScript-compatible.


## How to run the example

To make sure you can successfully run the latest versions of the example scripts, you have to **install the library from source** and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```
Then cd in this example folder and run
```bash
pip install -r requirements.txt
```

Now you can run the example command below to get the example ONNX file:

```bash
python run_onnx_exporter.py --model_name_or_path facebook/bart-base
```
