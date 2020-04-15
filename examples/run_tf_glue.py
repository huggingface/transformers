import os

import tensorflow as tf
import tensorflow_datasets

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    TFBertForSequenceClassification,
    glue_convert_examples_to_features,
    glue_processors,
)


# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 3

TASK = "mrpc"

if TASK == "sst-2":
    TFDS_TASK = "sst2"
elif TASK == "sts-b":
    TFDS_TASK = "stsb"
else:
    TFDS_TASK = TASK

num_labels = len(glue_processors[TASK]().get_labels())
print(num_labels)

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

# Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", config=config)

# Load dataset via TensorFlow Datasets
data, info = tensorflow_datasets.load(f"glue/{TFDS_TASK}", with_info=True)
train_examples = info.splits["train"].num_examples

# MNLI expects either validation_matched or validation_mismatched
valid_examples = info.splits["validation"].num_examples

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data["train"], tokenizer, max_length=128, task=TASK)

# MNLI expects either validation_matched or validation_mismatched
valid_dataset = glue_convert_examples_to_features(data["validation"], tokenizer, max_length=128, task=TASK)
train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
if USE_AMP:
    # loss scaling is currently required when using mixed precision
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")


if num_labels == 1:
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
model.compile(optimizer=opt, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
train_steps = train_examples // BATCH_SIZE
valid_steps = valid_examples // EVAL_BATCH_SIZE

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=valid_dataset,
    validation_steps=valid_steps,
)

# Save TF2 model
os.makedirs("./save/", exist_ok=True)
model.save_pretrained("./save/")

if TASK == "mrpc":
    # Load the TensorFlow model in PyTorch for inspection
    # This is to demo the interoperability between the two frameworks, you don't have to
    # do this in real life (you can run the inference on the TF model).
    pytorch_model = BertForSequenceClassification.from_pretrained("./save/", from_tf=True)

    # Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
    sentence_0 = "This research was consistent with his findings."
    sentence_1 = "His findings were compatible with this research."
    sentence_2 = "His findings were not compatible with this research."
    inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors="pt")
    inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors="pt")

    pred_1 = pytorch_model(**inputs_1)[0].argmax().item()
    pred_2 = pytorch_model(**inputs_2)[0].argmax().item()
    print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
    print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")
