import tensorflow as tf
import tensorflow_datasets
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification, glue_convert_examples_to_features

# Load tokenizer, model, dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tf_model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
dataset = tensorflow_datasets.load("glue/mrpc")

# Prepare dataset for GLUE
train_dataset = glue_convert_examples_to_features(dataset['train'], tokenizer, task='mrpc', max_length=128)
valid_dataset = glue_convert_examples_to_features(dataset['validation'], tokenizer, task='mrpc', max_length=128)
train_dataset = train_dataset.shuffle(100).batch(32).repeat(3)
valid_dataset = valid_dataset.batch(64)

# Compile tf.keras model for training
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(2e-5, 345, end_learning_rate=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
tf_model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

# Train and evaluate using tf.keras.Model.fit()
tf_model.fit(train_dataset, epochs=1, steps_per_epoch=115, validation_data=valid_dataset, validation_steps=7)

# Save the model and load it in PyTorch
tf_model.save_pretrained('./runs/')
pt_model = BertForSequenceClassification.from_pretrained('./runs/')

# Quickly inspect a few predictions
inputs = tokenizer.encode_plus("I said the company is doing great", "The company has good results", add_special_tokens=True)
pred = pt_model(torch.tensor([tokens]))

# Divers
import torch

import tensorflow as tf
import tensorflow_datasets
from pytorch_transformers import BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification, glue_convert_examples_to_features

# Load tokenizer, model, dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

pt_train_dataset = torch.load('../../data/glue_data//MRPC/cached_train_bert-base-cased_128_mrpc')

def gen():
    for el in pt_train_dataset:
        yield ((el.input_ids, el.attention_mask, el.token_type_ids), (el.label,))

dataset = tf.data.Dataset.from_generator(gen,
            ((tf.int32, tf.int32, tf.int32), (tf.int64,)),
            ((tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])),
             (tf.TensorShape([]),)))

dataset = dataset.shuffle(100).batch(32)
next(iter(dataset))

learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(2e-5, 345, 0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(
                  learning_rate=learning_rate,
                  epsilon=1e-08,
                  clipnorm=1.0),
              loss=loss,
              metrics=[['sparse_categorical_accuracy']])

tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir='./runs/', update_freq=10, histogram_freq=1)

# Train model
model.fit(dataset, epochs=3, callbacks=[tensorboard_cbk])
