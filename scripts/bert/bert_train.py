import tensorflow as tf
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
with strategy.scope():
    raw_datasets = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset  = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
    tf_train_dataset = small_train_dataset.remove_columns(["text"]).with_format("tensorflow")
    tf_eval_dataset  = small_eval_dataset.remove_columns(["text"]).with_format("tensorflow")
    
    train_features = {x: tf_train_dataset[x] for x in tokenizer.model_input_names}
    train_tf_dataset = tf.data.Dataset.from_tensor_slices((train_features, tf_train_dataset["label"]))
    train_tf_dataset = train_tf_dataset.shuffle(len(tf_train_dataset)).batch(8)
    
    eval_features = {x: tf_eval_dataset[x] for x in tokenizer.model_input_names}
    eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["label"]))
    eval_tf_dataset = eval_tf_dataset.batch(8)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

print("==================================== Evaluating Model =================================")
model.fit(train_tf_dataset, validation_data=eval_tf_dataset, epochs=1)
info = model.evaluate(eval_tf_dataset, verbose=2)

