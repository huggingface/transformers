import modeling_tf_adapter_bert
import os
import tensorflow as tf
import tensorflow_datasets
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
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
  parser.add_argument('--non_linearity', type=str, default='gelu_new', help='non_linearity function in adapters')
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
  model = modeling_tf_adapter_bert.AdapterBertModel(bert_model, num_labels)

  data, info = tensorflow_datasets.load(TFDS_TASK, with_info=True)
  train_examples = info.splits["train"].num_examples
  eval_examples = info.splits["validation"+postfix].num_examples

  train_dataset = glue_convert_examples_to_features(data["train"], tokenizer, max_length=args.max_seq_length, task=args.task)
  eval_dataset = glue_convert_examples_to_features(data["validation"+postfix], tokenizer, max_length=args.max_seq_length, task=args.task)
    
  train_dataset = train_dataset.repeat().shuffle(buffer_size=100).batch(args.batch_size)
  eval_dataset = eval_dataset.batch(args.batch_size)

  train_steps = int(np.ceil(train_examples / args.batch_size))
  eval_steps = int(np.ceil(eval_examples / args.batch_size))


  # Add Adapters
  for i in range(config.num_hidden_layers):
    # instantiate
    model.bert.bert.encoder.layer[i].attention.dense_output = modeling_tf_adapter_bert.TFBertSelfOutput(
                                                  model.bert.bert.encoder.layer[i].attention.dense_output.dense,
                                                  model.bert.bert.encoder.layer[i].attention.dense_output.LayerNorm,
                                                  config)
    model.bert.bert.encoder.layer[i].bert_output = modeling_tf_adapter_bert.TFBertOutput(
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


  # Loss & Optimizer
  if num_labels == 1:
      loss = tf.keras.losses.MeanSquaredError()
      monitor = 'val_spearmanr'
  else:
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      if args.task == 'cola':
        monitor = 'val_matthews_corrcoef'
      elif args.task in ['mrpc', 'qqp']:
        monitor = 'val_f1'
      else:
        monitor = 'val_accuracy'
                

  opt, scheduler = create_optimizer(init_lr=args.learning_rate,
                        num_train_steps=train_steps * args.epochs,
                        num_warmup_steps=int(train_steps * args.epochs * args.warmup_ratio),
                        adam_epsilon=1e-6,
                        weight_decay_rate = 0)
  model.compile(optimizer=opt, loss=loss)

  # Callback to save the best model
  checkpoint = ModelCheckpoint(eval_dataset, args.batch_size, eval_steps, monitor, os.path.join(args.saved_models_dir, args.task))

  # Fine-tuning
  history = model.fit(
      train_dataset,
      epochs=args.epochs,
      steps_per_epoch=train_steps,
      callbacks=[checkpoint]
  )


class ModelCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, eval_dataset, eval_batch_size, eval_steps, monitor, save_path):
    super(ModelCheckpoint, self).__init__()
    self.eval_dataset = eval_dataset
    self.eval_batch_size = eval_batch_size
    self.eval_steps = eval_steps
    self.monitor = monitor
    self.save_path = save_path
    self.best_val_score = -np.Inf

    self.val_labels: np.ndarray = None
    for batch in self.eval_dataset:
      if self.val_labels is None:
        self.val_labels = batch[1].numpy()
      else:
        self.val_labels = np.append(self.val_labels, batch[1].numpy(), axis=0)

  def evaluate(self):
    val_preds = self.model.predict(self.eval_dataset, batch_size=self.eval_batch_size, steps=self.eval_steps)

    if self.monitor == 'val_spearmanr':
      val_preds = tf.clip_by_value(val_preds, clip_value_min=0.0, clip_value_max=5.0)
      return spearmanr(self.val_labels, val_preds)[0]

    elif self.monitor == 'val_matthews_corrcoef':
      val_preds = tf.math.argmax(val_preds, axis=1)
      return matthews_corrcoef(self.val_labels, val_preds)

    elif self.monitor == 'val_f1':
      val_preds = tf.math.argmax(val_preds, axis=1)
      return f1_score(self.val_labels, val_preds)

    else:
      val_preds = tf.math.argmax(val_preds, axis=1)
      return accuracy_score(self.val_labels, val_preds)

  def on_epoch_end(self, epoch, logs):
    val_score = self.evaluate()
    print(" - {0}: {1:.4f}".format(self.monitor, val_score))

    if val_score >= self.best_val_score:
      path = os.path.join(self.save_path, str(epoch+1))
      os.makedirs(path)
      self.model.save_weights(path+'/best_weights.h5')
      self.best_val_score = val_score
      print("Model saved in epoch {} as the best model".format(epoch+1))

if __name__ == "__main__":
    main()
