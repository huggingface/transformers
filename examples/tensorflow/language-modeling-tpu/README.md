# Training a masked language model end-to-end from scratch on TPUs

In this example, we're going to demonstrate how to train a TensorFlow model from 🤗 Transformers from scratch. If you're interested in some background theory on training Hugging Face models with TensorFlow on TPU, please check out our 
[tutorial doc](https://huggingface.co/docs/transformers/main/perf_train_tpu_tf) on this topic!
If you're interested in smaller-scale TPU training from a pre-trained checkpoint, you can also check out the  [TPU fine-tuning example](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb).

This example will demonstrate pre-training language models at the 100M-1B parameter scale, similar to BERT or GPT-2. More concretely, we will show how to train a [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta) (base model) from scratch on the [WikiText dataset (v1)](https://huggingface.co/datasets/wikitext).

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

The script will automatically load the `train` split of the WikiText dataset and train a [Unigram tokenizer](https://huggingface.co/course/chapter6/7?fw=pt) on it.

> 💡 **Note**: In order for `export_to_hub` to work, you must authenticate yourself with the `huggingface-cli`. Run `huggingface-cli login` and follow the on-screen instructions.

## Preparing the dataset

The next step is to prepare the dataset. This consists of loading a text dataset from the Hugging Face Hub, tokenizing it and grouping it into chunks of a fixed length ready for training. The script for this is `prepare_tfrecord_shards.py`.

The reason we create TFRecord output files from this step is that these files work well with [`tf.data` pipelines](https://www.tensorflow.org/guide/data_performance). This makes them very suitable for scalable TPU training - the dataset can easily be sharded and read in parallel just by tweaking a few parameters in the pipeline. An example command is:

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
* If you append `gs://` in your `output_dir` the TFRecord shards will be directly serialized to a Google Cloud Storage (GCS) bucket. Ensure that you have already [created the GCS bucket](https://cloud.google.com/storage/docs). 
* If you're using a TPU node, you must stream data from a GCS bucket. Otherwise, if you're using a TPU VM,you can store the data locally. You may need to [attach](https://cloud.google.com/tpu/docs/setup-persistent-disk) a persistent storage to the VM. 
* Additional CLI arguments are also supported. We encourage you to run `python prepare_tfrecord_shards.py -h` to know more about them.

## Training the model

Once that's done, the model is ready for training. By default, training takes place on TPU, but you can use the `--no_tpu` flag to train on CPU for testing purposes. An example command is:

```bash
python3 run_mlm.py \
  --train_dataset gs://tf-tpu-training-resources/train/ \
  --eval_dataset gs://tf-tpu-training-resources/validation/ \
  --tokenizer tf-tpu/unigram-tokenizer-wikitext \
  --output_dir trained_model  
```

If you had specified a `hub_model_id` while launching training, then your model will be pushed to a model repository on the Hugging Face Hub. You can find such an example repository here:
[tf-tpu/roberta-base-epochs-500-no-wd](https://huggingface.co/tf-tpu/roberta-base-epochs-500-no-wd).

## Inference

Once the model is trained, you can use 🤗 Pipelines to perform inference:

```python
from transformers import pipeline

model_id = "tf-tpu/roberta-base-epochs-500-no-wd"
unmasker = pipeline("fill-mask", model=model_id, framework="tf")
unmasker("Goal of my life is to [MASK].")

[{'score': 0.1003185287117958,
  'token': 52,
  'token_str': 'be',
  'sequence': 'Goal of my life is to be.'},
 {'score': 0.032648514956235886,
  'token': 5,
  'token_str': '',
  'sequence': 'Goal of my life is to .'},
 {'score': 0.02152673341333866,
  'token': 138,
  'token_str': 'work',
  'sequence': 'Goal of my life is to work.'},
 {'score': 0.019547373056411743,
  'token': 984,
  'token_str': 'act',
  'sequence': 'Goal of my life is to act.'},
 {'score': 0.01939118467271328,
  'token': 73,
  'token_str': 'have',
  'sequence': 'Goal of my life is to have.'}]
```

You can also try out inference using the [Inference Widget](https://huggingface.co/tf-tpu/roberta-base-epochs-500-no-wd?text=Goal+of+my+life+is+to+%5BMASK%5D.) from the model page.