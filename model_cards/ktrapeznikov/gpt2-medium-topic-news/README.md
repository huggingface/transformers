---
language: 
- en
thumbnail:
widget:
 - text: "topic: climate article:"
---

# GPT2-medium-topic-news

## Model description

GPT2-medium fine tuned on a large news corpus conditioned on a topic

## Intended uses & limitations

#### How to use

To generate a news article text conditioned on a topic, prompt model with: 
`topic: climate article:`

The following tags were used during training:
`arts law international science business politics disaster world conflict football sport sports artanddesign environment music film lifeandstyle business health commentisfree books technology media education politics travel stage uk society us money culture religion science news tv fashion uk australia cities global childrens sustainable global voluntary housing law local healthcare theguardian`

Zero shot generation works pretty well as long as `topic` is a single word and not too specific.

```python
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/gpt2-medium-topic-news")
model = AutoModelWithLMHead.from_pretrained("ktrapeznikov/gpt2-medium-topic-news")
model.to(device)
topic = "climate"
prompt = tokenizer(f"topic: {topic} article:", return_tensors="pt")
out = model.generate(prompt["input_ids"].to(device), do_sample=True,max_length=500, early_stopping=True, top_p=.9)
print(tokenizer.decode(list(out.cpu()[0])))
```

## Training data


## Training procedure
