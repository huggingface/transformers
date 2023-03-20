Detailed README TBA.

# Training a model end-to-end from scratch on TPU

In this example, we're going to demonstrate model training on TPU from scratch. If you're interested in some
background theory on training Hugging Face models with TensorFlow on TPU, please check out our 
[tutorial doc](https://huggingface.co/docs/transformers/main/perf_train_tpu_tf) on this topic!
If you're interested in smaller-scale TPU training from a pre-trained checkpoint, you can also check out the 
[TPU fine-tuning example](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb).

This example will demonstrate pre-training language models at the 100M-1B parameter scale, similar to BERT or GPT-2.
We've tried to ensure that all the practices we show you here are scalable, though - with relatively few changes, the
code could be scaled up to much larger models. 

Google's gargantuan [PaLM model](https://arxiv.org/abs/2204.02311), with
over 500B parameters, is a good example of how far you can go with pure TPU training, though gathering the dataset
and the budget to train at that scale is not an easy task!

## Training a tokenizer

To train a language model from scratch, the first step is to tokenize text. In most Hugging Face examples, we begin
from a pre-trained model and use its tokenizer. However, in this example, we're going to train a tokenizer from
scratch as well. The script for this is `train_unigram.py`. An example command is:

```python
# Command goes here
```

## Preparing the dataset

The next step is to prepare the dataset. This consists of loading a text dataset from the Hugging Face Hub, tokenizing it
and grouping it into chunks of a fixed length ready for training. The script for this is `prepare_tfrecord_shards.py`.

The reason we create TFRecord output files from this step is that these files work well with `tf.data` pipelines. This
makes them very suitable for scalable TPU training - the dataset can easily be sharded and read in parallel just
by tweaking a few parameters in the pipeline. An example command is:

```python
# Command goes here
```

## Training the model

Once that's done, the model is ready for training. By default, training takes place on TPU, but you can use the
`--no_tpu` flag to train on CPU for testing purposes. An example command is:

```bash
python train_model.py \
       --train_dataset gs://tf-tpu-training-resources/train/ \
       --eval_dataset gs://tf-tpu-training-resources/validation/ \
       --tokenizer tf-tpu/unigram-tokenizer-wikitext \
       --output_dir trained_model  
```