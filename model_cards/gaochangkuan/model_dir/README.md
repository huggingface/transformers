## Generating Chinese poetry by topic.

```python
from transformers import *

tokenizer = BertTokenizer.from_pretrained("gaochangkuan/model_dir")

model = AutoModelWithLMHead.from_pretrained("gaochangkuan/model_dir")


prompt= '''<s>田园躬耕'''

length= 84    
stop_token='</s>'        

temperature = 1.2 
  
repetition_penalty=1.3 
 
k= 30
p= 0.95
 
device ='cuda'
seed=2020          
no_cuda=False      
 
prompt_text = prompt if prompt else input("Model prompt >>> ")

encoded_prompt = tokenizer.encode(
                                  '<s>'+prompt_text+'<sep>',
                                  add_special_tokens=False, 
                                  return_tensors="pt"
                                 )

encoded_prompt = encoded_prompt.to(device)

output_sequences = model.generate(
    input_ids=encoded_prompt,
    max_length=length,
    min_length=10,
    do_sample=True,
    early_stopping=True,
    num_beams=10,
    temperature=temperature,
    top_k=k,
    top_p=p,
    repetition_penalty=repetition_penalty,
    bad_words_ids=None,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    length_penalty=1.2,
    no_repeat_ngram_size=2,
    num_return_sequences=1,
    attention_mask=None,
    decoder_start_token_id=tokenizer.bos_token_id,)
    
    
    generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence)


text = text[: text.find(stop_token) if stop_token else None]

print(''.join(text).replace(' ','').replace('<pad>','').replace('<s>',''))
```
