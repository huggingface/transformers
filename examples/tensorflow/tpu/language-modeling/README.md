# Training a model end-to-end from scratch on TPU

In this example, we're going to demonstrate model training on TPU from scratch. If you're interested in some background theory on training Hugging Face models with TensorFlow on TPU, please check out our 
[tutorial doc](https://huggingface.co/docs/transformers/main/perf_train_tpu_tf) on this topic!
If you're interested in smaller-scale TPU training from a pre-trained checkpoint, you can also check out the  [TPU fine-tuning example](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb).

This example will demonstrate pre-training language models at the 100M-1B parameter scale, similar to BERT or GPT-2. More concretely, we train a [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) (base model) from scratch on the [WikiText dataset (v1)](https://huggingface.co/datasets/wikitext).

We've tried to ensure that all the practices we show you here are scalable, though - with relatively few changes, the code could be scaled up to much larger models. 

Google's gargantuan [PaLM model](https://arxiv.org/abs/2204.02311), with
over 500B parameters, is a good example of how far you can go with pure TPU training, though gathering the dataset and the budget to train at that scale is not an easy task!

### Table of contents 

- [Setting up a TPU-VM](#setting-up-a-tpu-vm)
- [Training a tokenizer](#training-a-tokenizer)
- [Preparing the dataset](#preparing-the-dataset)
- [Training the model](#training-the-model)
- [Inference](#inference)

## Setting up a TPU-VM

Since this example focuses on using TPUs, the first step is to set up access to TPU hardware. For this example, we chose to use a TPU v3-8 VM. Follow [this guide](https://cloud.google.com/tpu/docs/run-calculation-tensorflow) to quickly create a TPU VM with TensorFlow pre-installed. 

> 💡 **Note**: You don't need a TPU-enabled hardware for tokenizer training and TFRecord shard preparation.

## Training a tokenizer

To train a language model from scratch, the first step is to tokenize text. In most Hugging Face examples, we begin from a pre-trained model and use its tokenizer. However, in this example, we're going to train a tokenizer from scratch as well. The script for this is `train_unigram.py`. An example command is:

```bash 
python train_unigram.py --batch_size 1000 --vocab_size 25000 --export_to_hub
```

> 💡 **Note**: In order for `export_to_hub` to work, you must authenticate yourself with the `huggingface-cli`. Run `huggingface-cli login` and follow the on-screen instructions.

## Preparing the dataset

The next step is to prepare the dataset. This consists of loading a text dataset from the Hugging Face Hub, tokenizing it and grouping it into chunks of a fixed length ready for training. The script for this is `prepare_tfrecord_shards.py`.

The reason we create TFRecord output files from this step is that these files work well with `tf.data` pipelines. This makes them very suitable for scalable TPU training - the dataset can easily be sharded and read in parallel just by tweaking a few parameters in the pipeline. An example command is:

```bash
python prepare_tfrecord_shards.py \
  --tokenizer_name_or_path tf-tpu/unigram-tokenizer-wikitext \
  --shard_size 5000  \
  --split test 
  --max_length 128 \
  --output_dir gs://tf-tpu-training-resources
```

**Notes**:

* While running the above script, you need to specify the `split` accordingly. The example command above will only filter the `test` split of the dataset. 
* If you append `gs://` in your `output_dir` the TFRecord shards will be directly serialized to a Google Cloud Storage bucket. 
* Additional CLI arguments are also supported. We encourage you to run `python prepare_tfrecord_shards.py -h` to know more about them.

## Training the model

Once that's done, the model is ready for training. By default, training takes place on TPU, but you can use the `--no_tpu` flag to train on CPU for testing purposes. An example command is:

```bash
python3 train_model.py \
       --train_dataset gs://tf-tpu-training-resources/train/ \
       --eval_dataset gs://tf-tpu-training-resources/validation/ \
       --tokenizer tf-tpu/unigram-tokenizer-wikitext \
       --output_dir trained_model  
```

If you had specified a `hub_model_id` while launching training, then your model will be pushed to a model repository on the Hugging Face Hub. You can find such an example repository here:
[tf-tpu/roberta-base-epochs-100](https://huggingface.co/tf-tpu/roberta-base-epochs-100).

## Inference

Once the model is trained, you can use 🤗 Pipelines to perform inference:

```python
from transformers import pipeline

model_id = "tf-tpu/roberta-base-epochs-100"
unmasker = pipeline("fill-mask", model="tf-tpu/roberta-base-epochs-100", framework="tf")
unmasker("Goal of my life is to [MASK].")

[{'score': 0.3213661313056946,
  'token': 52,
  'token_str': 'be',
  'sequence': 'Goal of my life is to be.'},
 {'score': 0.09109099209308624,
  'token': 36,
  'token_str': 'o',
  'sequence': 'Goal of my life is too.'},
 {'score': 0.0677114799618721,
  'token': 63,
  'token_str': 'r',
  'sequence': 'Goal of my life is tor.'},
 {'score': 0.033341776579618454,
  'token': 5,
  'token_str': '',
  'sequence': 'Goal of my life is to .'},
 {'score': 0.022657133638858795,
  'token': 105,
  'token_str': 'him',
  'sequence': 'Goal of my life is to him.'}]
```