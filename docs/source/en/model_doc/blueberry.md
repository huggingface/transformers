<!--Copyright 2026 Dustin Loring and the HuggingFace team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Blueberry

[Blueberry](https://github.com/blueberry-llm/blueberry) is a decoder-only language model by Dustin Loring
(dustinwloring1988@gmail.com). It runs the first and last groups of decoder blocks once and repeats the middle half
with shared weights. Each execution maintains its own attention cache. An optional learned gate mixes the state
entering the loop with the previous pass's output at each pass boundary.
The training code builds on [NanoChat](https://github.com/karpathy/nanochat) by Andrej Karpathy.

The default configuration uses 8 distinct blocks and runs 16 block executions across 3 passes through the middle
4 blocks. It has about 125.8 million parameters and a training context of 2,048 tokens.
The supervised fine-tuned checkpoint is [CrowdMind/Blueberry](https://huggingface.co/CrowdMind/Blueberry).
The pretrained checkpoint is [CrowdMind/Blueberry-base](https://huggingface.co/CrowdMind/Blueberry-base).

## Convert a trained checkpoint

The [Blueberry trainer](https://github.com/blueberry-llm/blueberry) uses
`--depth 8 --aspect-ratio 64 --head-dim 128 --loops 3 --loop-gate --model-tag blueberry-125m` for the default architecture.
After training, convert the matching `model_<step>.pt` and `meta_<step>.json` files and the tokenizer from the same run.
Install `tiktoken` in the conversion environment. Use a trusted local `tokenizer.pkl`.

```bash
python -m transformers.models.blueberry.convert_blueberry_checkpoint \
  --checkpoint-dir /path/to/base_checkpoints/blueberry-125m \
  --step 1000 \
  --tokenizer /path/to/tokenizer/tokenizer.pkl \
  --output-dir ./Blueberry-125M
```

Replace `1000` with the step number of the checkpoint you plan to release. The output includes SafeTensors weights,
model and generation configs, the fast tokenizer, and a model card. Upload that directory to the model repo once
the checkpoint is ready.

```bash
hf upload CrowdMind/Blueberry-base ./Blueberry-125M .
```

## Generate text

Install a version of Transformers with Blueberry support before loading the local export or the Hub checkpoint.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "CrowdMind/Blueberry-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
inputs = tokenizer("The first blueberries of the season", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

## BlueberryConfig

[[autodoc]] BlueberryConfig

## BlueberryModel

[[autodoc]] BlueberryModel
    - forward

## BlueberryForCausalLM

[[autodoc]] BlueberryForCausalLM
    - forward
