from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
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

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
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
val_dataset = materialDataset(val_encodings, val_labels)
test_dataset = materialDataset(test_encodings, test_labels)


metric = load_metric('glue', 'mrpc')


training_args = TrainingArguments(
    output_dir='./results_2',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs_2',  # directory for storing logs
    logging_steps=10,
)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    # compute_metrics=metric
)

trainer.train()
trainer.evaluate()


# Example of typical usage
for batch in val_dataset:
    inputs, references = batch
    predictions = model(inputs)
    metric.add_batch(predictions=predictions, references=references)
score = metric.compute()
print(score)

trainer.save_model(training_args.output_dir)
