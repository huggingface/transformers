import tensorflow as tf
import tensorflow_datasets
from pytorch_transformers import BertTokenizer, TFBertForSequenceClassification, glue_convert_examples_to_features

# Load tokenizer, model, dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
dataset, dataset_info = tensorflow_datasets.load("glue/mrpc", split="train", with_info=True)
print(dataset_info)

# Prepare dataset for GLUE
dataset = glue_convert_examples_to_features(dataset, tokenizer, task='mrpc', max_length=64)
dataset = dataset.batch(32)

# Compile model for training
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(5e-5, 300, 0)
model.compile(optimizer=tf.keras.optimizers.Adam(
              learning_rate=learning_rate),
              loss=lambda x, y: tf.keras.backend.sparse_categorical_crossentropy(x, y, from_logits=True),
              metrics=['accuracy'])

# Train model
model.fit(dataset, epochs=3)