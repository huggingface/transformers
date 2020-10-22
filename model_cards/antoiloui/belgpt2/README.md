---
language: "fr"
---

# BelGPT-2

**BelGPT-2** (*Belgian GPT-2* 🇧🇪) is a "small" GPT-2 model pre-trained on a very large and heterogeneous French corpus (around 60Gb). Please check [antoiloui/gpt2-french](https://github.com/antoiloui/gpt2-french) for more information about the pre-trained model, the data, the code to use the model and the code to pre-train it.


## Using BelGPT-2 for Text Generation in French

You can use BelGPT-2 with [🤗 transformers](https://github.com/huggingface/transformers) library as follows:

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pretrained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("antoiloui/belgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("antoiloui/belgpt2")

# Generate a sample of text
model.eval()
output = model.generate(
            bos_token_id=random.randint(1,50000),
            do_sample=True,   
            top_k=50, 
            max_length=100,
            top_p=0.95, 
            num_return_sequences=1
)

# Decode it
decoded_output = []
for sample in output:
    decoded_output.append(tokenizer.decode(sample, skip_special_tokens=True))
print(decoded_output)
```

## Data

Below is the list of all French copora used to pre-trained the model:

| Dataset | `$corpus_name` | Raw size | Cleaned size |
| :------|   :--- | :---: | :---: | 
| CommonCrawl |  `common_crawl`   |  200.2 GB   |  40.4 GB   |
| NewsCrawl |   `news_crawl`  |   10.4 GB  |  9.8 GB   |
| Wikipedia |   `wiki`  |   19.4 GB  |  4.1 GB   |
| Wikisource |   `wikisource`  |  4.6  GB  |  2.3 GB   |
| Project Gutenberg |  `gutenberg`   |  1.3 GB   |  1.1 GB   |
| EuroParl |  `europarl`   |  289.9 MB   |   278.7 MB  |
| NewsCommentary |  `news_commentary`   |   61.4 MB  |  58.1 MB   |
| **Total** |     |   **236.3 GB**  |  **57.9 GB**   |
