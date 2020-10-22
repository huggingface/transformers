## About the model

The model has been trained on a collection of 500k articles with headings. Its purpose is to create a one-line heading suitable for the given article.

Sample code with a WikiNews article:

```python
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = model.to(device)

article = '''
Very early yesterday morning, the United States President Donald Trump reported he and his wife First Lady Melania Trump tested positive for COVID-19. Officials said the Trumps' 14-year-old son Barron tested negative as did First Family and Senior Advisors Jared Kushner and Ivanka Trump.
Trump took to social media, posting at 12:54 am local time (0454 UTC) on Twitter, "Tonight, [Melania] and I tested positive for COVID-19. We will begin our quarantine and recovery process immediately. We will get through this TOGETHER!" Yesterday afternoon Marine One landed on the White House's South Lawn flying Trump to Walter Reed National Military Medical Center (WRNMMC) in Bethesda, Maryland.
Reports said both were showing "mild symptoms". Senior administration officials were tested as people were informed of the positive test. Senior advisor Hope Hicks had tested positive on Thursday.
Presidential physician Sean Conley issued a statement saying Trump has been given zinc, vitamin D, Pepcid and a daily Aspirin. Conley also gave a single dose of the experimental polyclonal antibodies drug from Regeneron Pharmaceuticals.
According to official statements, Trump, now operating from the WRNMMC, is to continue performing his duties as president during a 14-day quarantine. In the event of Trump becoming incapacitated, Vice President Mike Pence could take over the duties of president via the 25th Amendment of the US Constitution. The Pence family all tested negative as of yesterday and there were no changes regarding Pence's campaign events.
'''

text =  "headline: " + article

max_len = 256

encoding = tokenizer.encode_plus(text, return_tensors = "pt")
input_ids = encoding["input_ids"].to(device)
attention_masks = encoding["attention_mask"].to(device)

beam_outputs = model.generate(
    input_ids = input_ids,
    attention_mask = attention_masks,
    max_length = 64,
    num_beams = 3,
    early_stopping = True,
)

result = tokenizer.decode(beam_outputs[0])
print(result)
```

Result:

```Trump and First Lady Melania Test Positive for COVID-19```
