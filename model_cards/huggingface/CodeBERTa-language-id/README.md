---
language: code
thumbnail: https://hf-dinosaur.huggingface.co/CodeBERTa/CodeBERTa.png
---

# CodeBERTa-language-id: The World’s fanciest programming language identification algo 🤯


To demonstrate the usefulness of our CodeBERTa pretrained model on downstream tasks beyond language modeling, we fine-tune the [`CodeBERTa-small-v1`](https://huggingface.co/huggingface/CodeBERTa-small-v1) checkpoint on the task of classifying a sample of code into the programming language it's written in (*programming language identification*).

We add a sequence classification head on top of the model.

On the evaluation dataset, we attain an eval accuracy and F1 > 0.999 which is not surprising given that the task of language identification is relatively easy (see an intuition why, below).

## Quick start: using the raw model

```python
CODEBERTA_LANGUAGE_ID = "huggingface/CodeBERTa-language-id"

tokenizer = RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID)

input_ids = tokenizer.encode(CODE_TO_IDENTIFY)
logits = model(input_ids)[0]

language_idx = logits.argmax() # index for the resulting label
```


## Quick start: using Pipelines 💪

```python
from transformers import TextClassificationPipeline

pipeline = TextClassificationPipeline(
    model=RobertaForSequenceClassification.from_pretrained(CODEBERTA_LANGUAGE_ID),
    tokenizer=RobertaTokenizer.from_pretrained(CODEBERTA_LANGUAGE_ID)
)

pipeline(CODE_TO_IDENTIFY)
```

Let's start with something very easy:

```python
pipeline("""
def f(x):
    return x**2
""")
# [{'label': 'python', 'score': 0.9999965}]
```

Now let's probe shorter code samples:

```python
pipeline("const foo = 'bar'")
# [{'label': 'javascript', 'score': 0.9977546}]
```

What if I remove the `const` token from the assignment?
```python
pipeline("foo = 'bar'")
# [{'label': 'javascript', 'score': 0.7176245}]
```

For some reason, this is still statistically detected as JS code, even though it's also valid Python code. However, if we slightly tweak it:

```python
pipeline("foo = u'bar'")
# [{'label': 'python', 'score': 0.7638422}]
```
This is now detected as Python (Notice the `u` string modifier).

Okay, enough with the JS and Python domination already! Let's try fancier languages:

```python
pipeline("echo $FOO")
# [{'label': 'php', 'score': 0.9995257}]
```

(Yes, I used the word "fancy" to describe PHP 😅)

```python
pipeline("outcome := rand.Intn(6) + 1")
# [{'label': 'go', 'score': 0.9936151}]
```

Why is the problem of language identification so easy (with the correct toolkit)? Because code's syntax is rigid, and simple tokens such as `:=` (the assignment operator in Go) are perfect predictors of the underlying language:

```python
pipeline(":=")
# [{'label': 'go', 'score': 0.9998052}]
```

By the way, because we trained our own custom tokenizer on the [CodeSearchNet](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/) dataset, and it handles streams of bytes in a very generic way, syntactic constructs such `:=` are represented by a single token:

```python
self.tokenizer.encode(" :=", add_special_tokens=False)
# [521]
```

<br>

## Fine-tuning code

<details>

```python
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from tokenizers.implementations.byte_level_bpe import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange

from transformers import RobertaForSequenceClassification
from transformers.data.metrics import acc_and_f1, simple_accuracy


logging.basicConfig(level=logging.INFO)


CODEBERTA_PRETRAINED = "huggingface/CodeBERTa-small-v1"

LANGUAGES = [
    "go",
    "java",
    "javascript",
    "php",
    "python",
    "ruby",
]
FILES_PER_LANGUAGE = 1
EVALUATE = True

# Set up tokenizer
tokenizer = ByteLevelBPETokenizer("./pretrained/vocab.json", "./pretrained/merges.txt",)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Set up Tensorboard
tb_writer = SummaryWriter()


class CodeSearchNetDataset(Dataset):
    examples: List[Tuple[List[int], int]]

    def __init__(self, split: str = "train"):
        """
        train | valid | test
        """

        self.examples = []

        src_files = []
        for language in LANGUAGES:
            src_files += list(
                Path("../CodeSearchNet/resources/data/").glob(f"{language}/final/jsonl/{split}/*.jsonl.gz")
            )[:FILES_PER_LANGUAGE]
        for src_file in src_files:
            label = src_file.parents[3].name
            label_idx = LANGUAGES.index(label)
            print("🔥", src_file, label)
            lines = []
            fh = gzip.open(src_file, mode="rt", encoding="utf-8")
            for line in fh:
                o = json.loads(line)
                lines.append(o["code"])
            examples = [(x.ids, label_idx) for x in tokenizer.encode_batch(lines)]
            self.examples += examples
        print("🔥🔥")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return self.examples[i]


model = RobertaForSequenceClassification.from_pretrained(CODEBERTA_PRETRAINED, num_labels=len(LANGUAGES))

train_dataset = CodeSearchNetDataset(split="train")
eval_dataset = CodeSearchNetDataset(split="test")


def collate(examples):
    input_ids = pad_sequence([torch.tensor(x[0]) for x in examples], batch_first=True, padding_value=1)
    labels = torch.tensor([x[1] for x in examples])
    # ^^  uncessary .unsqueeze(-1)
    return input_ids, labels


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate)

batch = next(iter(train_dataloader))


model.to("cuda")
model.train()
for param in model.roberta.parameters():
    param.requires_grad = False
## ^^ Only train final layer.

print(f"num params:", model.num_parameters())
print(f"num trainable params:", model.num_parameters(only_trainable=True))


def evaluate():
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = np.empty((0), dtype=np.int64)
    out_label_ids = np.empty((0), dtype=np.int64)

    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=512, collate_fn=collate)
    for step, (input_ids, labels) in enumerate(tqdm(eval_dataloader, desc="Eval")):
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
            loss = outputs[0]
            logits = outputs[1]
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
        preds = np.append(preds, logits.argmax(dim=1).detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    acc = simple_accuracy(preds, out_label_ids)
    f1 = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    print("=== Eval: loss ===", eval_loss)
    print("=== Eval: acc. ===", acc)
    print("=== Eval: f1 ===", f1)
    # print(acc_and_f1(preds, out_label_ids))
    tb_writer.add_scalars("eval", {"loss": eval_loss, "acc": acc, "f1": f1}, global_step)


### Training loop

global_step = 0
train_iterator = trange(0, 4, desc="Epoch")
optimizer = torch.optim.AdamW(model.parameters())
for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")
    for step, (input_ids, labels) in enumerate(epoch_iterator):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"))
        loss = outputs[0]
        loss.backward()
        tb_writer.add_scalar("training_loss", loss.item(), global_step)
        optimizer.step()
        global_step += 1
        if EVALUATE and global_step % 50 == 0:
            evaluate()
            model.train()


evaluate()

os.makedirs("./models/CodeBERT-language-id", exist_ok=True)
model.save_pretrained("./models/CodeBERT-language-id")
```

</details>

<br>

## CodeSearchNet citation

<details>

```bibtex
@article{husain_codesearchnet_2019,
	title = {{CodeSearchNet} {Challenge}: {Evaluating} the {State} of {Semantic} {Code} {Search}},
	shorttitle = {{CodeSearchNet} {Challenge}},
	url = {http://arxiv.org/abs/1909.09436},
	urldate = {2020-03-12},
	journal = {arXiv:1909.09436 [cs, stat]},
	author = {Husain, Hamel and Wu, Ho-Hsiang and Gazit, Tiferet and Allamanis, Miltiadis and Brockschmidt, Marc},
	month = sep,
	year = {2019},
	note = {arXiv: 1909.09436},
}
```

</details>
