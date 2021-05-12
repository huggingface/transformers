from transformers import FlaxAutoModelForSequenceClassification, AutoConfig

model_checkpoint = "bert-base-cased"

config = AutoConfig.from_pretrained(model_checkpoint, num_labels=3, return_dict=False)
model = FlaxAutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
