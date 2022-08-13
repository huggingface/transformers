from transformers import FlaxT5ForConditionalGeneration, T5Tokenizer
from jax import numpy as jnp, random
from flax import linen as nn

tokenizer = T5Tokenizer.from_pretrained("t5-small",model_max_length=512)
model = FlaxT5ForConditionalGeneration.from_pretrained("t5-small")

class TokenEmbed(nn.Module):
    vocab_size: int = None
    d_model: int = None
    def setup(self):
        self.shared = nn.Embed(
            self.vocab_size,
            self.d_model,
        )

    def __call__(self,x):
        return self.shared(x)

ARTICLE_TO_SUMMARIZE = "summarize: The US has \"passed the peak\" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world. At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors."

print('-'*80)
print('Model Input ->', ARTICLE_TO_SUMMARIZE)

inputs = tokenizer([ARTICLE_TO_SUMMARIZE], return_tensors="np")

# Generate summary from input_ids
summary_ids = model.generate(inputs["input_ids"],max_new_tokens=50,do_sample=True).sequences
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print('-'*80)
print('Summary from input_ids ->', pred_summary)

# Create input_embeds from input_ids
vocab_size, d_model = model.params['shared']['embedding'].shape
shared = TokenEmbed(vocab_size, d_model)
# input_embeds: batch x length x d_model
input_embeds = shared.apply({'params':{'shared':model.params['shared']}},inputs['input_ids'])

# Generate summary from input_embeds
summary_ids = model.generate(None,max_new_tokens=50,input_embeds=input_embeds,do_sample=True).sequences
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print('-'*80)
print('Summary from input_embeds ->',pred_summary)

# Concatenate with vision embeddings (eg. patch embeddings from a ViT)
batch = input_embeds.shape[0]
vision_embeddings = random.uniform(random.PRNGKey(0),[batch,100,d_model])
input_embeds = jnp.concatenate([input_embeds,vision_embeddings],axis=1)
summary_ids = model.generate(None,max_new_tokens=50,input_embeds=input_embeds,do_sample=True).sequences
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print('-'*80)
print('Summary after concatenating random visual embeddings -> ',pred_summary)