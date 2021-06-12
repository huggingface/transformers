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

# Language model training examples

The following example showcases how to train a language model from scratch 
using the JAX/Flax backend.

JAX/Flax allows you to trace pure functions and compile them into efficient, fused accelerator code on both GPU and TPU.
Models written in JAX/Flax are **immutable** and updated in a purely functional
way which enables simple and efficient model parallelism.

## Masked language modeling

In the following, we demonstrate how to train a bi-directional transformer model 
using masked language modeling objective as introduced in [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
More specifically, we demonstrate how JAX/Flax can be leveraged 
to pre-train [**`roberta-base`**](https://huggingface.co/roberta-base)
in Norwegian on a single TPUv3-8 pod.

The example script uses the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

Let's start by creating a folder to save the trained model and a symbolic link to the `run_mlm_flax.py` script.

```bash
export MODEL_DIR="./norwegian-roberta-base"
mkdir -p ${MODEL_DIR}
ln -s ~/transformers/examples/flax/language-modeling/run_mlm_flax.py run_mlm_flax.py
```

### Train tokenizer

In the first step, we train a tokenizer to efficiently process the text input for the model. Similar to how it is shown in [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train), we use a **`ByteLevelBPETokenizer`**.
The tokenizer is trained on the complete Norwegian dataset of OSCAR
and consequently saved in `${MODEL_DIR}`
This can take up to 10 minutes depending on your hardware ☕.

```python
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer

model_dir = "./norwegian-roberta-base"  # ${MODEL_DIR}

# load dataset
dataset = load_dataset("oscar", "unshuffled_deduplicated_no", split="train")

# Instantiate tokenizer
tokenizer = ByteLevelBPETokenizer()

def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# Customized training
tokenizer.train_from_iterator(batch_iterator(), vocab_size=50265, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save(f"{model_dir}/tokenizer.json")
```

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**roberta-base**`](https://huggingface.co/roberta-base)
in the local model folder:

```python
from transformers import RobertaConfig

model_dir = "./norwegian-roberta-base"  # ${MODEL_DIR}

config = RobertaConfig.from_pretrained("roberta-base")
config.save_pretrained(model_dir)
```

### Train model

Next we can run the example script to pretrain the model:

```bash
./run_mlm_flax.py \
        --output_dir="./runs" \
        --model_type="roberta" \
        --config_name="${MODEL_DIR}" \
        --tokenizer_name="${MODEL_DIR}" \
        --dataset_name="oscar" \
        --dataset_config_name="unshuffled_deduplicated_no" \
        --max_seq_length="128" \
        --weight_decay="0.01" \
        --per_device_train_batch_size="128" \
        --per_device_eval_batch_size="128" \
        --learning_rate="3e-4" \
        --warmup_steps="1000" \
        --overwrite_output_dir \
        --pad_to_max_length \
        --num_train_epochs="18" \
        --adam_beta1="0.9" \
        --adam_beta2="0.98"
```

Training should converge at a loss and accuracy 
of 1.78 and 0.64 respectively after 18 epochs on a single TPUv3-8.
This should take less than 18 hours.
Training statistics can be accessed on [tfhub.de](https://tensorboard.dev/experiment/GdYmdak2TWeVz0DDRYOrrg).

For a step-by-step walkthrough of how to do masked language modeling in Flax, please have a 
look at [this TODO: (Patrick)]() google colab.

## Causal language modeling

In the following, we demonstrate how to train an auto-regressive causal transformer model 
in JAX/Flax.
More specifically, we pretrain a randomely initialized [**`gpt2`**](https://huggingface.co/gpt2) model in Norwegian on a single TPUv3-8.
to pre-train 124M [**`gpt2`**](https://huggingface.co/gpt2)
in Norwegian on a single TPUv3-8 pod.

The example script uses the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

Let's start by creating a folder to save the trained model and a symbolic link to the `run_clm_flax.py` script.

```bash
export MODEL_DIR="./norwegian-gpt2"
mkdir -p ${MODEL_DIR}
ln -s ~/transformers/examples/flax/language-modeling/run_clm_flax.py run_clm_flax.py
```

Next, we'll follow the same steps as above in [Train tokenizer](#train-tokenizer) to train the tokenizer.

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**gpt2**`](https://huggingface.co/gpt2)
in the local model folder:

```python
from transformers import GPT2Config

model_dir = "./norwegian-gpt2"  # ${MODEL_DIR}

config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
config.save_pretrained(model_dir)
```

### Train model

Next we can run the example script to pretrain the model:

```bash
./run_clm_flax.py \
    --output_dir="./runs" \
    --model_type="gpt2" \
    --config_name="${MODEL_DIR}" \
    --tokenizer_name="${MODEL_DIR}" \
    --dataset_name="oscar" \
    --dataset_config_name="unshuffled_deduplicated_no" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" \
    --warmup_steps="1000" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
```

Training should converge at a loss and perplexity 
of 3.28 and 26.63 respectively after 20 epochs on a single TPUv3-8.
This should take less than ~21 hours.
Training statistics can be accessed on [tfhub.de](hhttps://tensorboard.dev/experiment/D1hRUJL1S8Wy3Hrz8hY8zQ/).

TODO(Suraj): Update the hyper-parameters and metrics


## TODO(Patrick): Add comparison with PyTorch GPU/TPU
