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
tf_model.fit(train_dataset, epochs=3, steps_per_epoch=115, validation_data=valid_dataset, validation_steps=7)

# Save the model and load it in PyTorch
tf_model.save_pretrained('./runs/')
pt_model = BertForSequenceClassification.from_pretrained('./runs/', from_tf=True)

# Quickly inspect a few predictions
inputs = tokenizer.encode_plus("I said the company is doing great", "The company has good results", add_special_tokens=True, return_tensors='pt')
pred = pt_model(torch.tensor(tokens))
