# Migrating from pytorch-pretrained-bert


Here is a quick summary of what you should take care of when migrating from `pytorch-pretrained-bert` to `pytorch-transformers`

### Models always output `tuples`

The main breaking change when migrating from `pytorch-pretrained-bert` to `pytorch-transformers` is that the models forward method always outputs a `tuple` with various elements depending on the model and the configuration parameters.

The exact content of the tuples for each model are detailled in the models' docstrings and the [documentation](https://huggingface.co/pytorch-transformers/).

In pretty much every case, you will be fine by taking the first element of the output as the output you previously used in `pytorch-pretrained-bert`.

Here is a `pytorch-pretrained-bert` to `pytorch-transformers` conversion example for a `BertForSequenceClassification` classification model:

```python
# Let's load our model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# If you used to have this line in pytorch-pretrained-bert:
loss = model(input_ids, labels=labels)

# Now just use this line in pytorch-transformers to extract the loss from the output tuple:
outputs = model(input_ids, labels=labels)
loss = outputs[0]

# In pytorch-transformers you can also have access to the logits:
loss, logits = outputs[:2]

# And even the attention weigths if you configure the model to output them (and other outputs too, see the docstrings and documentation)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=True)
outputs = model(input_ids, labels=labels)
loss, logits, attentions = outputs
```

### Serialization

Breaking change in the `from_pretrained()`method:

1. Models are now set in evaluation mode by default when instantiated with the `from_pretrained()` method. To train them don't forget to set them back in training mode (`model.train()`) to activate the dropout modules.

2. The additional `*inputs` and `**kwargs` arguments supplied to the `from_pretrained()` method used to be directly passed to the underlying model's class `__init__()` method. They are now used to update the model configuration attribute first which can break derived model classes build based on the previous `BertForSequenceClassification` examples. More precisely, the positional arguments `*inputs` provided to `from_pretrained()` are directly forwarded the model `__init__()` method while the keyword arguments `**kwargs` (i) which match configuration class attributes are used to update said attributes (ii) which don't match any configuration class attributes are forwarded to the model `__init__()` method.

Also, while not a breaking change, the serialization methods have been standardized and you probably should switch to the new method `save_pretrained(save_directory)` if you were using any other serialization method before.

Here is an example:

```python
### Let's load a model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

### Do some stuff to our model and tokenizer
# Ex: add new tokens to the vocabulary and embeddings of our model
tokenizer.add_tokens(['[SPECIAL_TOKEN_1]', '[SPECIAL_TOKEN_2]'])
model.resize_token_embeddings(len(tokenizer))
# Train our model
train(model)

### Now let's save our model and tokenizer to a directory
model.save_pretrained('./my_saved_model_directory/')
tokenizer.save_pretrained('./my_saved_model_directory/')

### Reload the model and the tokenizer
model = BertForSequenceClassification.from_pretrained('./my_saved_model_directory/')
tokenizer = BertTokenizer.from_pretrained('./my_saved_model_directory/')
```

### Optimizers: BertAdam & OpenAIAdam are now AdamW, schedules are standard PyTorch schedules

The two optimizers previously included, `BertAdam` and `OpenAIAdam`, have been replaced by a single `AdamW` optimizer which has a few differences:

- it only implements weights decay correction,
- schedules are now externals (see below),
- gradient clipping is now also external (see below).

The new optimizer `AdamW` matches PyTorch `Adam` optimizer API and let you use standard PyTorch or apex methods for the schedule and clipping.

The schedules are now standard [PyTorch learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) and not part of the optimizer anymore.

Here is a conversion examples from `BertAdam` with a linear warmup and decay schedule to `AdamW` and the same schedule:

```python
# Parameters:
lr = 1e-3
max_grad_norm = 1.0
num_total_steps = 1000
num_warmup_steps = 100
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1

### Previously BertAdam optimizer was instantiated like this:
optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, t_total=num_total_steps)
### and used like this:
for batch in train_data:
    loss = model(batch)
    loss.backward()
    optimizer.step()

### In PyTorch-Transformers, optimizer and schedules are splitted and instantiated like this:
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
### and used like this:
for batch in train_data:
    loss = model(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
    scheduler.step()
    optimizer.step()
```
