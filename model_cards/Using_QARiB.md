
# Using QARiB models

## Using ktrain
[ktrain](https://github.com/amaiya/ktrain) is a wrapper around huggingface that can be used easily to load and fine-tune BERT models. It can be used to load QARiB models. Example usage:

### Binary/multi-class classification
```bash
import ktrain
from ktrain import text
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
categories = ['class1', 'class2', 'class3'] #classes present in the data
MODEL = "drive/My Drive/qarib-models/ckpt-1950000" #either name of BERT model or path to BERT model

def print_evaluation(gold_labels, predicted_labels):
    ## computes overall scores (accuracy, f1, recall, precision)
    accuracy = accuracy_score(gold_labels, predicted_labels) * 100
    f1 = f1_score(gold_labels, predicted_labels, average = "macro") * 100
    recall = recall_score(gold_labels, predicted_labels, average = "macro") * 100
    precision = precision_score(gold_labels, predicted_labels, average = "macro") * 100
    a = [accuracy, precision, recall, f1]
    return a

# read train, dev and test file. 
# train_X, dev_X, test_X are arrays of strings. train_y, dev_y, test_y are arrays of labels 
# It's not necessary to transform labels into one hot representation or any other format
train_X, train_y = readfile("train.tsv")
dev_X, dev_y = readfile("dev.tsv")
test_X, test_y = readfile("test.tsv")

t = text.Transformer(MODEL_PATH, maxlen=55, class_names=categories)
trn = t.preprocess_train(train_X, train_y)
val = t.preprocess_test(dev_X, dev_y)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=64)
# train for 3 epochs
learner.fit_onecycle(8e-5, 3)
predictor = ktrain.get_predictor(learner.model, preproc=t)
predicted_labels = clf.predict(test_input)
print_evaluation(test_labels, predicted_labels)
```
### Multi-label classification
```bash

import ktrain
from ktrain import text
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
categories = ['class1', 'class2', 'class3'] #classes present in the data
MODEL = "drive/My Drive/qarib-models/ckpt-1950000" #either name of BERT model or path to BERT model

def evaluate(labels, predictions):
  scores = {}
  scores['jaccard'] = jaccard_score(labels, predictions, average="samples")
  scores['f1'] = f1_score(labels, predictions, average="macro")
  all_accs = []
  for i in range(labels.shape[-1]):
    scores[f'accuracy-{i}'] = accuracy_score(labels[:, i], predictions[:, i])
    all_accs.append(scores[f'accuracy-{i}'])
  scores['accuracy'] = np.mean(np.array(all_accs))
  return scores
  
# read train, dev and test file. 
# train_X, dev_X, test_X are arrays of strings. 
# train_y, dev_y, test_y are two dimensional arrays of shape (n_samples, n_classes). 
# each row contains n_classes indicator variables to represent whether a label exists or not.
train_X, train_y = readfile("train.tsv")
dev_X, dev_y = readfile("dev.tsv")
test_X, test_y = readfile("test.tsv")

t = text.Transformer(MODEL_PATH, maxlen=55, class_names=categories)
trn = t.preprocess_train(train_X, train_y)
val = t.preprocess_test(dev_X, dev_y)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=64)
# train for 3 epochs
learner.fit_onecycle(8e-5, 3)
predictor = ktrain.get_predictor(learner.model, preproc=t)

# the predictor provides probability of each class being present. which then should be mapped to labels.
predicted_probas = clf.predict(test_X)
predicted_labels = []
for i in range (len(predicted_probas)):
  row = []
  for j in range (len(categories)):
      if predicted_probas[i][j][1] >= 0.5:
          row.append(1)
      else:
          row.append(0)
  predicted_labels.append(row)
predicted_labels = np.array(predicted_labels)
print(evaluate(test_labels, predicted_labels))
```