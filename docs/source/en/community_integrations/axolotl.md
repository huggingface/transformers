<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Axolotl

[Axolotl](https://docs.axolotl.ai/) is a fine-tuning and post-training framework for large language models. It supports many tuning methods like LoRA, preference tuning, and reward modeling. Axolotl builds directly on Transformers for model loading, tokenization, training, and inference.

Create a YAML config file to define your training run.

```yaml
base_model: NousResearch/Nous-Hermes-llama-1b-v1
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: path/to/data
    type: alpaca

output_dir: ./outputs
sequence_len: 512
micro_batch_size: 1
gradient_accumulation_steps: 1
num_epochs: 1
learning_rate: 2.0e-5
```

Launch training with the [train](https://docs.axolotl.ai/docs/cli.html#train) command.

```bash
axolotl train my_training.yml
```

## Transformers integration

Axolotl's [ModelLoader](https://docs.axolotl.ai/docs/api/loaders.model.html#axolotl.loaders.model.ModelLoader) wraps the Transformers load flow.

- The model config is built from [`AutoConfig.from_pretrained`]. Preload setup configures the [device map](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map), [quantization config](../main_classes/quantization), and [attention backend](../attention_interface).

- `ModelLoader` selects [`AutoModelForCausalLM`], [`AutoModelForImageTextToText`], or a model-specific class from the multimodal mapping. Weights load with the selected loader's `from_pretrained`. When `reinit_weights` is set, Axolotl builds the model with the selected loader's `from_config` for random initialization.

- Adapters load after the base model when LoRA or other PEFT methods are enabled. A patch manager applies optimizations before and after loading.

- [AxolotlTrainer](https://docs.axolotl.ai/docs/api/core.trainers.base.html#axolotl.core.trainers.base.AxolotlTrainer) extends [`Trainer`], adding Axolotl mixins while using the [`Trainer`] training loop and APIs.

## Resources

- [Axolotl](https://docs.axolotl.ai/) docs