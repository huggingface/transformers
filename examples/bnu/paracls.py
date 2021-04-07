from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, EvalPrediction
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from datasets import load_metric


def read_material_split(file_path):
    texts = []
    labels = []
    with open(file_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            item = line.replace('\n', '').split('\t')
            texts.append(item[0])
            labels.append(int(item[1]))
    return texts, labels


train_texts, train_labels = read_material_split('./train.txt')
#
test_texts, test_labels = read_material_split('./test.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


class materialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = materialDataset(train_encodings, train_labels)
test_dataset = materialDataset(test_encodings, test_labels)

metric = load_metric("glue", 'mrpc', cache_dir='./metric_dir')

is_regression = False
task_name = "material_paragraphy_classify"


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if task_name is not None:
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()} 


training_args = TrainingArguments(
    output_dir='results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='logs',  # directory for storing logs
    logging_steps=10,
)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    compute_metrics=compute_metrics
)

trainer.train()
eval_result = trainer.evaluate()
print(eval_result)
trainer.save_model(training_args.output_dir)
