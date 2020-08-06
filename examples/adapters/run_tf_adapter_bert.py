import modeling
import utility
import os
import tensorflow as tf
import tensorflow_datasets
import numpy as np
from transformers import (
    BertConfig,
    BertTokenizer,
    TFBertModel,
    glue_processors,
    glue_convert_examples_to_features,
)
from transformers.optimization_tf import create_optimizer
import argparse


def main():

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(gpu)
    except RuntimeError as e:
      print(e)

  parser = argparse.ArgumentParser()

  parser.add_argument('--casing', type=str, default="bert-base-uncased", help='BERT model')
  parser.add_argument('--bottleneck_size', type=int, default=64, help='Bottleneck size of adapters')
  parser.add_argument('--non_linearity', type=str, default='gelu', help='non_linearity function in adapters')
  parser.add_argument('--task', type=str, default='mrpc', help='GLUE task')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs')
  parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length')
  parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
  parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
  parser.add_argument('--saved_models_dir', type=str, default='saved_models', help='save directory')

  args = parser.parse_args()
  if not os.path.isdir(args.saved_models_dir):
    os.mkdir(args.saved_models_dir)
  if not os.path.isdir(os.path.join(args.saved_models_dir, args.task)):
    os.mkdir(os.path.join(args.saved_models_dir, args.task))

  if args.task == "sst-2":
      TFDS_TASK = "sst2"
      postfix = ""
  elif args.task == "sts-b":
      TFDS_TASK = "stsb"
      postfix = ""
  elif args.task == "mnli":
      TFDS_TASK = "mnli"
      postfix = "_matched"
  elif args.task == "mnli-mm":
      TFDS_TASK = "mnli"
      postfix = "_mismatched"
  else:
      TFDS_TASK = args.task
      postfix = ""

  TFDS_TASK = "glue/" + TFDS_TASK
  num_labels = len(glue_processors[args.task]().get_labels())

  # Load Model, Tokenizer & Datasets
  config = BertConfig.from_pretrained(args.casing, num_labels=num_labels)
  config.bottleneck_size = args.bottleneck_size
  config.non_linearity = args.non_linearity

  tokenizer = BertTokenizer.from_pretrained(args.casing)
  bert_model = TFBertModel(config).from_pretrained(args.casing)
  model = modeling.AdapterBertModel(bert_model, num_labels)

  data, info = tensorflow_datasets.load(TFDS_TASK, with_info=True)
  train_examples = info.splits["train"].num_examples
  valid_examples = info.splits["validation"+postfix].num_examples

  train_dataset = glue_convert_examples_to_features(data["train"], tokenizer, max_length=args.max_seq_length, task=args.task)
  valid_dataset = glue_convert_examples_to_features(data["validation"+postfix], tokenizer, max_length=args.max_seq_length, task=args.task)
    
  train_dataset = train_dataset.repeat().shuffle(buffer_size=100).batch(args.batch_size)
  valid_dataset = valid_dataset.batch(args.batch_size)

  train_steps = int(np.ceil(train_examples / args.batch_size))
  valid_steps = int(np.ceil(valid_examples / args.batch_size))


  # Add Adapters
  for i in range(config.num_hidden_layers):
    # instantiate
    model.bert.bert.encoder.layer[i] = modeling.TFBertLayer(model.bert.bert.encoder.layer[i].attention.self_attention,
                                                  model.bert.bert.encoder.layer[i].attention.dense_output.dense,
                                                  model.bert.bert.encoder.layer[i].attention.dense_output.LayerNorm,
                                                  model.bert.bert.encoder.layer[i].intermediate,
                                                  model.bert.bert.encoder.layer[i].bert_output.dense,
                                                  model.bert.bert.encoder.layer[i].bert_output.LayerNorm,
                                                  config)
    
  # Freeze BERT
  model.bert.bert.embeddings.trainable = False
  model.bert.bert.pooler.trainable = False
  for i in range(config.num_hidden_layers):
    model.bert.bert.encoder.layer[i].attention.self_attention.trainable = False
    model.bert.bert.encoder.layer[i].attention.dense_output.dense.trainable = False
    model.bert.bert.encoder.layer[i].attention.dense_output.LayerNorm.trainable = False
    model.bert.bert.encoder.layer[i].intermediate.trainable = False
    model.bert.bert.encoder.layer[i].bert_output.dense.trainable = False
    model.bert.bert.encoder.layer[i].bert_output.LayerNorm.trainable = False


  # Metrics, Loss & Optimizer
  if num_labels == 1:
      loss = tf.keras.losses.MeanSquaredError()
      metric = utility.spearman
      monitor = "val_spearman"
  else:
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      if args.task == 'cola':
        metric = utility.matthews_cc
        monitor = "val_matthews_cc"
      elif args.task in ['mrpc', 'qqp']:
        metric = utility.f1
        monitor = "val_f1"
      else:
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
        monitor = "val_accuracy"
                

  opt, scheduler = create_optimizer(init_lr=args.learning_rate,
                        num_train_steps=train_steps * args.epochs,
                        num_warmup_steps=int(train_steps * args.epochs * args.warmup_ratio),
                        adam_epsilon=1e-6,
                        weight_decay_rate = 0)
  model.compile(optimizer=opt, loss=loss, metrics=metric)

  # Callback
  checkpoint = utility.ModelCheckpoint(monitor, os.path.join(args.saved_models_dir, args.task))

  # Fine-tuning
  history = model.fit(
      train_dataset,
      epochs=args.epochs,
      steps_per_epoch=train_steps,
      validation_data=valid_dataset,
      validation_steps=valid_steps,
      callbacks=[checkpoint]
  )



if __name__ == "__main__":
    main()
