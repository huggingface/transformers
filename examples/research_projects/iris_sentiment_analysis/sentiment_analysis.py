'''This project focuses on applying sentiment classification to the Iris dataset, 
using the Hugging Face Transformers library and a pre-trained DistilBERT model. 
Here, the Iris dataset's numeric features are transformed into text format for
sentiment analysis, with species mapped to three sentiment labels: Positive, Neutral, and Negative. '''

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


'''Loading the Iris dataset sepal_length, sepal_width, petal_length, petal_width, and a categorical species column, 
which is mapped to sentiment labels (Positive, Neutral, Negative) for classification'''

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#The species of Iris (Iris-setosa, Iris-versicolor, Iris-virginica) are mapped to sentiment labels:0,1 and 2 respectively
sentiment_mapping = {
    'Iris-setosa': 0,    # Positive
    'Iris-versicolor': 1,  # Neutral
    'Iris-virginica': 2   # Negative
}
iris['sentiment'] = iris['species'].map(sentiment_mapping)

#TextDataset class is implemented to handle the conversion of numeric Iris features into text format
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"Sepal Length: {self.texts[idx][0]}, Sepal Width: {self.texts[idx][1]}, Petal Length: {self.texts[idx][2]}, Petal Width: {self.texts[idx][3]}"
        label = self.labels[idx]

        #Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        #Convert encoding to a format that can be fed into the model
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['label'] = torch.tensor(label, dtype=torch.long)

        return item

#The Iris dataset is split into training and testing sets using an 80/20 split.
train_data, test_data = train_test_split(iris, test_size=0.2, random_state=42)

#The Hugging Face distilbert-base-uncased model is loaded, which is a pre-trained transformer model, along with the corresponding tokenizer.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

#Extracting Text and Labels from the DataFrame
train_texts = train_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
train_labels = train_data['sentiment'].values
test_texts = test_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
test_labels = test_data['sentiment'].values

#Creating Dataset Instances
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_length=128)

#Training arguments for the model are defined, including learning rate, batch size, number of epochs, evaluation strategy, and weight decay to regularize the model.
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,  # Experiment with different learning rates
    per_device_train_batch_size=8,  # Smaller batch size can help with learning
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increase the number of epochs
    weight_decay=0.1,  # Experiment with higher weight decay
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)


#Training the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)
trainer.train()


#Reporting

#Generating a classification report
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Evaluating the model and making predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

print(classification_report(test_labels, predicted_labels, target_names=["Positive", "Neutral", "Negative"]))

#Generating and plotting the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Positive', 'Neutral', 'Negative'], 
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

#Inference on New Data
new_texts = ["Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2",
             "Sepal Length: 6.0, Sepal Width: 2.2, Petal Length: 5.0, Petal Width: 1.5"]
new_encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors='pt')

# Make predictions
with torch.no_grad():
    outputs = model(**new_encodings)
    new_predictions = np.argmax(outputs.logits.numpy(), axis=1)

print("Predictions for new texts:", new_predictions)

#Evaluating the model
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

#Calculating accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")