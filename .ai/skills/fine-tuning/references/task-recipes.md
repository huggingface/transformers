# Task recipes

This doc is a quick reference for task-specific choices. Each entry covers: model class, tokenization/preprocessing, data collator, metrics, and gotchas. Everything else follows the standard workflow in SKILL.md.

---

## Text tasks

### Sequence classification

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
```

- Tokenize with truncation; standard `DataCollatorWithPadding`

```python
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True)

dataset = dataset.map(preprocess, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)
```

- Metrics: `evaluate.load("accuracy")`

---

### Token classification (NER, POS, chunking)

```python
model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
```

- Data collator: `DataCollatorForTokenClassification`
- Labels must be realigned to subword tokens — use `word_ids()` and assign `-100` to special tokens and non-first subword pieces so the loss ignores them:

```python
def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        aligned = []
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])
            else:
                aligned.append(-100)  # non-first subword
            prev_word_id = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized
```

- Metrics: `evaluate.load("seqeval")` — filter out `-100` labels before computing

---

### Causal language modeling

```python
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
```

- Data collator: `DataCollatorForLanguageModeling(tokenizer, mlm=False)` — handles `labels = input_ids` automatically

```python
def preprocess(examples):
    return tokenizer(examples["text"])  # no padding; collator handles it

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

- For large datasets, concatenate and chunk into fixed-length blocks for efficiency:

```python
def chunk(examples, block_size=512):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total = (len(concatenated["input_ids"]) // block_size) * block_size
    return {k: [v[i:i+block_size] for i in range(0, total, block_size)]
            for k, v in concatenated.items()}
```

- Metrics: perplexity — `math.exp(eval_results["eval_loss"])`

---

### Masked language modeling

```python
model = AutoModelForMaskedLM.from_pretrained(model_id)
```

- Data collator: `DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)` — applies masking automatically; no manual label creation needed

```python
def preprocess(examples):
    return tokenizer(examples["text"])

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
# apply same chunking pattern as causal LM, then:
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
```

- Metrics: perplexity from eval loss

---

### Question answering (extractive)

```python
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
```

- Data collator: `DefaultDataCollator`

```python
def preprocess(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",        # truncate context, never question
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,  # long contexts produce multiple spans
        return_offsets_mapping=True,
        padding="max_length",
    )
    # map character-level answer offsets → token positions using
    # offset_mapping and sequence_ids(); unanswerable spans get (0, 0)
    # this mapping is complex — see the QA task doc for the full implementation
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DefaultDataCollator()
```

- Metrics: SQuAD-style EM/F1; requires significant postprocessing of logits back to answer strings

---

### Multiple choice

```python
model = AutoModelForMultipleChoice.from_pretrained(model_id)
```

- Replicate the context once per candidate answer, tokenize all `(context, candidate)` pairs, then unflatten to `(batch, num_choices, seq_len)`

```python
def preprocess(examples):
    # assumes columns: "sent1", "sent2", "ending0".."ending3", "label"
    choices = ["ending0", "ending1", "ending2", "ending3"]
    first = [[s] * len(choices) for s in examples["sent1"]]
    second = [
        [examples[c][i] for c in choices] for i in range(len(examples["sent1"]))
    ]
    # flatten for the tokenizer, then unflatten
    flat_first = sum(first, [])
    flat_second = sum(second, [])
    tokenized = tokenizer(flat_first, flat_second, truncation=True, max_length=256)
    return {k: [v[i:i+len(choices)] for i in range(0, len(v), len(choices))]
            for k, v in tokenized.items()}

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        batch_size, num_choices = len(features), len(features[0]["input_ids"])
        flattened = [{k: v[i] for k, v in f.items() if k != "label"}
                     for f in features for i in range(num_choices)]
        batch = self.tokenizer.pad(flattened, return_tensors="pt")
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor([f["label"] for f in features])
        return batch
```

- Metrics: `evaluate.load("accuracy")` on `argmax(logits, dim=1)`

---

### Summarization / Translation (seq2seq)

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
```

- Use `Seq2SeqTrainingArguments` with `predict_with_generate=True` (required — otherwise eval runs a forward pass, not generation)
- Tokenize source and target **separately** — use `text_target` to pass labels in the same call:

```python
def preprocess(examples):
    # for summarization: source_col="article", target_col="highlights"
    # for translation:   source_col="en", target_col="fr"  (adjust to your language pair)
    inputs = tokenizer(examples[source_col], max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples[target_col], max_length=128, truncation=True)
    # replace padding token IDs with -100 so loss ignores them
    labels["input_ids"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in l]
        for l in labels["input_ids"]
    ]
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer, model)
```
- Metrics: ROUGE (`evaluate.load("rouge")`) for summarization; SacreBLEU (`evaluate.load("sacrebleu")`) for translation — wrap label strings in a list: `[[label]]`
- **Summarization prefix** (T5): prepend `"summarize: "` to inputs
- **Translation prefix** (T5): prepend `"translate English to French: "` (or equivalent)

---

## Vision tasks

### Image classification

```python
model = AutoModelForImageClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True,
)
image_processor = AutoImageProcessor.from_pretrained(model_id)
```

- Use `dataset.with_transform(preprocess)` (not `dataset.map`) — applies transforms on-the-fly, avoids storing all pixel tensors

```python
from torchvision.transforms import (
    CenterCrop, Compose, Normalize, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = image_processor.size.get("shortest_edge", image_processor.size["height"])

train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
val_transforms   = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

def preprocess_train(examples):
    examples["pixel_values"] = [train_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

def preprocess_val(examples):
    examples["pixel_values"] = [val_transforms(img.convert("RGB")) for img in examples["image"]]
    return examples

train_dataset = dataset["train"].with_transform(preprocess_train)
eval_dataset  = dataset["validation"].with_transform(preprocess_val)
data_collator = DefaultDataCollator()
```

- **Set `remove_unused_columns=False`** — Trainer drops unknown columns before `with_transform` runs, losing the `image` column
- Metrics: `evaluate.load("accuracy")`

---

### Object detection

```python
model = AutoModelForObjectDetection.from_pretrained(
    model_id, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)
image_processor = AutoImageProcessor.from_pretrained(model_id, do_resize=True, do_pad=True)
```

- Reformat annotations to COCO format, then pass images + annotations together to the image processor:

```python
def preprocess(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    # reformat to COCO: list of dicts with image_id + annotations list
    annotations = [
        {"image_id": id_, "annotations": anns}
        for id_, anns in zip(examples["image_id"], examples["objects"])
    ]
    return image_processor(images=images, annotations=annotations)

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],  # list, not stacked — variable box counts
    }
```

- **Set `remove_unused_columns=False` and `eval_do_concat_batches=False`** — concatenating eval batches breaks per-image box tracking
- Metrics: mAP/mAR via `torchmetrics.detection.mean_ap.MeanAveragePrecision`; convert predicted boxes from YOLO → Pascal VOC format first

---

### Semantic segmentation

```python
model = AutoModelForSemanticSegmentation.from_pretrained(model_id, id2label=id2label, label2id=label2id)
image_processor = AutoImageProcessor.from_pretrained(model_id, do_reduce_labels=True)
```

- `do_reduce_labels=True` remaps the background class (index 0) to 255 so it's ignored in loss
- Pass images + segmentation masks to the image processor together:

```python
def preprocess(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    masks  = examples["annotation"]
    batch  = image_processor(images=images, segmentation_maps=masks)
    return batch

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DefaultDataCollator()
```

- **Set `remove_unused_columns=False`**
- Metrics: `evaluate.load("mean_iou")` with `ignore_index=255`; upsample logits to label size before argmax:

```python
upsampled = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
predictions = upsampled.argmax(dim=1)
```

---

### Visual question answering

```python
model = ViltForQuestionAnswering.from_pretrained(model_id)
processor = ViltProcessor.from_pretrained(model_id)
```

- VQA is multi-label classification with **soft labels** (weights from multiple annotators) — don't one-hot encode; create a float tensor of size `num_labels` and fill with annotation weights

```python
def preprocess(examples):
    questions = examples["question"]
    images    = examples["image"]
    encoding  = processor(images, questions, padding="max_length", truncation=True, return_tensors="pt")
    # build soft-label targets from annotator answer counts
    labels = []
    for answers in examples["answers"]:
        target = torch.zeros(len(id2label))
        for ans in answers:
            if ans["answer"] in label2id:
                target[label2id[ans["answer"]]] += ans["answer_confidence_score"]
        labels.append(target)
    encoding["labels"] = labels
    return encoding

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DefaultDataCollator()
```

---

### Image captioning

```python
model = AutoModelForCausalLM.from_pretrained(model_id)   # e.g. GIT
processor = AutoProcessor.from_pretrained(model_id)
```

- Pass images + captions together to processor; copy `input_ids` → `labels`

```python
def preprocess(examples):
    images   = [img.convert("RGB") for img in examples["image"]]
    captions = examples["caption"]                        # one caption per image
    encoding = processor(images=images, text=captions, padding="max_length", return_tensors="pt")
    labels = encoding["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore padding in loss
    encoding["labels"] = labels
    return encoding

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
data_collator = DefaultDataCollator()
```

- **Set `remove_unused_columns=False`** and `label_names=["labels"]` in TrainingArguments
- Metrics: WER (`evaluate.load("wer")`) — decode predictions and labels with `processor.batch_decode(..., skip_special_tokens=True)`

---

## Audio tasks

### Automatic speech recognition (CTC)

```python
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(
    model_id,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
```

- Resample audio to 16kHz, extract features, tokenize transcriptions, and pad them separately in a custom collator:

```python
def preprocess(examples):
    audio = [a["array"] for a in examples["audio"]]
    inputs = processor(audio, sampling_rate=16_000)
    examples["input_values"] = inputs.input_values
    examples["input_length"] = [len(v) for v in inputs.input_values]
    with processor.as_target_processor():
        examples["labels"] = processor(examples["text"]).input_ids
    return examples

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

@dataclass
class DataCollatorCTC:
    processor: Any
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
```

- Metrics: `evaluate.load("wer")` — argmax logits → token IDs → decode, filter `-100` from labels before decoding

---

### Audio classification

```python
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
```

- Resample audio to 16kHz, then pass through the feature extractor:

```python
def preprocess(examples):
    audio = [a["array"] for a in examples["audio"]]
    inputs = feature_extractor(
        audio, sampling_rate=feature_extractor.sampling_rate,
        max_length=16_000, truncation=True,
    )
    inputs["label"] = examples["label"]   # Trainer expects column named "label"
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
```

- Data collator: default (standard padding works)
- Metrics: `evaluate.load("accuracy")`

---

## Video tasks

### Video classification

```python
model = VideoMAEForVideoClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True,
)
```

- Use `VideoMAEImageProcessor` + PyTorchVideo transforms applied in a custom `collate_fn` (not `dataset.map`):

```python
from pytorchvideo.transforms import (
    ApplyTransformToKey, Normalize, RandomShortSideScale, UniformTemporalSubsample,
)
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomCrop, RandomHorizontalFlip

mean = image_processor.image_mean
std  = image_processor.image_std
h, w = image_processor.size["height"], image_processor.size["width"]
num_frames = image_processor.num_frames   # e.g. 16

train_transform = Compose([
    ApplyTransformToKey("video", Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop((h, w)),
        RandomHorizontalFlip(),
    ])),
])
val_transform = Compose([
    ApplyTransformToKey("video", Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        CenterCrop((h, w)),
    ])),
])

def collate_fn(examples):
    pixel_values = torch.stack([
        ex["video"].permute(1, 0, 2, 3)   # (C, T, H, W) → (T, C, H, W)
        for ex in examples
    ])
    labels = torch.tensor([ex["label"] for ex in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

- Video tensor from PyTorchVideo is `(channels, frames, H, W)` — permute to `(frames, channels, H, W)` in the collate_fn
- PyTorchVideo datasets don't implement `__len__` — use `max_steps` in TrainingArguments instead of `num_train_epochs`
- **Set `remove_unused_columns=False`**
- Metrics: `evaluate.load("accuracy")`
