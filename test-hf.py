import transformers
import tensorflow as tf

model = transformers.TFT5ForConditionalGeneration.from_pretrained('t5-small')
input_ids = tf.random.uniform([1, 512], maxval=1000, dtype=tf.int32)
labels = tf.random.uniform([1, 512], maxval=1000, dtype=tf.int32)

loss, *_ = model(dict(input_ids=input_ids, labels=labels))
print(loss)
