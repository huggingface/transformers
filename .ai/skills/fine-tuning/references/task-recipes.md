# Task Recipes

Quick reference for task-specific choices. Each entry covers: model class, tokenization/preprocessing, data collator, metrics, and gotchas. Everything else follows the standard workflow in SKILL.md.

---

## Text tasks

### Sequence classification

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
```

- Tokenize with truncation; standard `DataCollatorWithPadding`
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
- Same chunking pattern as causal LM
- Metrics: perplexity from eval loss

---

### Question answering (extractive)

```python
model = AutoModelForQuestionAnswering.from_pretrained(model_id)
```

- Data collator: `DefaultDataCollator`
- Tokenize question + context with `truncation="only_second"` (truncate context, never question), `return_offsets_mapping=True`, `padding="max_length"`
- Map answer character offsets → token positions using `offset_mapping` and `sequence_ids()`; unanswerable spans get `(0, 0)`; this preprocessing is complex — refer to the [QA task doc](../tasks/question_answering.md) for the full label-mapping implementation
- Metrics: SQuAD-style EM/F1; requires significant postprocessing of logits back to answer strings

---

### Multiple choice

```python
model = AutoModelForMultipleChoice.from_pretrained(model_id)
```

- Replicate the context once per candidate answer, tokenize all `(context, candidate)` pairs, then unflatten to `(batch, num_choices, seq_len)`
- Data collator: custom — flattens, pads, then unflattens back to choice dimension
- Metrics: `evaluate.load("accuracy")` on `argmax(logits, dim=1)`

---

### Summarization / Translation (seq2seq)

```python
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
```

- Use `Seq2SeqTrainingArguments` with `predict_with_generate=True` (required — otherwise eval runs a forward pass, not generation)
- Tokenize source and target **separately** — targets go in `text_target`:

```python
with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=128, truncation=True)
```

Or with newer tokenizers: `tokenizer(source, text_target=targets, ...)`

- Replace padding token IDs in labels with `-100` so loss ignores them
- Data collator: `DataCollatorForSeq2Seq(tokenizer, model)`
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
- Apply train augmentations (RandomResizedCrop, RandomHorizontalFlip) and val transforms (Resize, CenterCrop); normalize with `image_processor.image_mean` / `image_processor.image_std`
- Data collator: `DefaultDataCollator`
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

- Reformat annotations to COCO format: `{'image_id': int, 'annotations': [{'image_id', 'category_id', 'bbox', 'area', 'iscrowd'}]}`
- Pass both images and annotations to the image processor: `image_processor(images=images, annotations=annotations)`
- Data collator: custom `collate_fn` — stacks `pixel_values`, keeps `labels` as a list (variable box counts per image)
- **Set `remove_unused_columns=False` and `eval_do_concat_batches=False`** — concatenating eval batches breaks per-image box tracking
- Metrics: mAP/mAR via `torchmetrics.detection.mean_ap.MeanAveragePrecision`; convert predicted boxes from YOLO → Pascal VOC format first

---

### Semantic segmentation

```python
model = AutoModelForSemanticSegmentation.from_pretrained(model_id, id2label=id2label, label2id=label2id)
image_processor = AutoImageProcessor.from_pretrained(model_id, do_reduce_labels=True)
```

- `do_reduce_labels=True` remaps the background class (index 0) to 255 so it's ignored in loss
- Pass images + segmentation masks to the image processor together
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
- `processor(image, question, padding="max_length", truncation=True)` returns `input_ids`, `token_type_ids`, `attention_mask`, `pixel_values`, `pixel_mask`
- Data collator: `DefaultDataCollator`

---

### Image captioning

```python
model = AutoModelForCausalLM.from_pretrained(model_id)   # e.g. GIT
processor = AutoProcessor.from_pretrained(model_id)
```

- Pass images + captions together to processor; copy `input_ids` → `labels`
- **Set `remove_unused_columns=False`** and `label_names=["labels"]` in TrainingArguments
- Metrics: WER (`evaluate.load("wer")`) — decode predictions and labels with `processor.batch_decode(..., skip_special_tokens=True)`

---

## Audio tasks

### Automatic speech recognition (CTC)

```python
model = AutoModelForCTC.from_pretrained(
    model_id,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
processor = AutoProcessor.from_pretrained(model_id)
```

- Resample audio to 16kHz; extract `input_values` via `processor(audio, sampling_rate=16000)`
- Tokenize transcription text; no built-in data collator — implement a custom one that pads `input_values` and `labels` separately and replaces label padding with `-100`
- Metrics: `evaluate.load("wer")` — argmax logits → token IDs → decode, filter `-100` from labels before decoding

---

### Audio classification

```python
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
```

- Resample audio to 16kHz; pass through `feature_extractor` to get `input_values`; rename label column to `"label"`
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

- Use `VideoMAEImageProcessor` + PyTorchVideo transforms: `UniformTemporalSubsample`, normalize, resize, random crop
- Video tensor from dataset is `(channels, frames, H, W)` — collate_fn must permute to `(frames, channels, H, W)`
- PyTorchVideo datasets don't implement `__len__` — use `max_steps` in TrainingArguments instead of `num_train_epochs`
- **Set `remove_unused_columns=False`**
- Metrics: `evaluate.load("accuracy")`
