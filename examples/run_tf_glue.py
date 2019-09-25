import tensorflow as tf
import tensorflow_datasets
from transformers import *

# Load dataset, tokenizer, model from pretrained model/vocabulary
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
dataset = tensorflow_datasets.load('glue/mrpc')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(dataset['train'], tokenizer, task='mrpc')
valid_dataset = glue_convert_examples_to_features(dataset['validation'], tokenizer, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(3)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule 
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(2e-5, 345, end_learning_rate=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])

# Train and evaluate using tf.keras.Model.fit()
model.fit(train_dataset, epochs=3, steps_per_epoch=115,
          validation_data=valid_dataset, validation_steps=7)

# Save the TensorFlow model and load it in PyTorch
model.save_pretrained('./save/')
pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

# Quickly inspect a few predictions - MRPC is a paraphrasing task
inputs = tokenizer.encode_plus("The company is doing great",
                               "The company has good results",
                               add_special_tokens=True,
                               return_tensors='pt')
pred = pytorch_model(**inputs)
print("Paraphrase" if pred.argmax().item() == 0 else "Not paraphrase")
