import logging

logging.basicConfig(level=logging.INFO)

from transformers import TFTrainerForSequenceClassification


config = {
    "model_config": {
        "pretrained_model_name_or_path": "bert-base-cased",
        "optimizer_name": "adam",
        "learning_rate": 3e-5,
        "loss_name": "SparseCategoricalCrossentropy",
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "distributed": False,
        "epochs": 4,
        "metric_name": "SparseCategoricalAccuracy",
        "max_len": 128,
        "task": "glue/mrpc"
    },
    "data_processor_config": {
        "dataset_name": "glue/mrpc",
        "guid": "idx",
        "text_a": "sentence1",
        "text_b": "sentence2",
        "label": "label"
    }
}

trainer = TFTrainerForSequenceClassification(**config)
trainer.setup_training(data_cache_dir="data_cache", model_cache_dir="model_cache")
trainer.train()
trainer.save_model("save")
trainer.evaluate()
