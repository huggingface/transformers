---
language: italian
---

# GePpeTto GPT2 Model 🇮🇹

Pretrained GPT2 117M model for Italian.

You can find further details in the paper:

Lorenzo De Mattei, Michele Cafagna, Felice Dell’Orletta, Malvina Nissim, Marco Guerini "GePpeTto Carves Italian into a Language Model", arXiv preprint. Pdf available at: https://arxiv.org/abs/2004.14253

## Pretraining Corpus

The pretraining set comprises two main sources. The first one is a dump of Italian Wikipedia (November 2019), 
consisting of 2.8GB of text. The second one is the ItWac corpus (Baroni et al., 2009), which amounts to 11GB of web
texts. This collection provides a mix of standard and less standard Italian, on a rather wide chronological span, 
with older texts than the Wikipedia dump (the latter stretches only to the late 2000s).

## Pretraining details

This model was trained using GPT2's Hugging Face implemenation on 4 NVIDIA Tesla T4 GPU for 620k steps.

Training parameters:

- GPT-2 small configuration
- vocabulary size: 30k
- Batch size: 32
- Block size: 100
- Adam Optimizer
- Initial learning rate: 5e-5
- Warm up steps: 10k

## Perplexity scores

| Domain | Perplexity |
|---|---|
| Wikipedia | 26.1052 |
| ItWac | 30.3965 |
| Legal | 37.2197 |
| News | 45.3859 |
| Social Media | 84.6408 |

For further details, qualitative analysis and human evaluation check out: https://arxiv.org/abs/2004.14253

## Load Pretrained Model

You can use this model by installing Huggingface library `transformers`. And you can use it directly by initializing it like this:  

```python
from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('LorenzoDeMattei/GePpeTto')
tokenizer = GPT2Tokenizer.from_pretrained(
    'LorenzoDeMattei/GePpeTto',
)
```

## Example using GPT2LMHeadModel

```python
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")

text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "Wikipedia Geppetto",
    "Maestro Ciliegia regala il pezzo di legno al suo amico Geppetto, il quale lo prende per fabbricarsi un burattino maraviglioso"]


samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)


for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:", sample_output['generated_text'])
        print()

```

Output is,

```
----------------------------------------------------------------------------------------------------
Prompt: Wikipedia Geppetto
Sample: Wikipedia Geppetto rosso (film 1920)

Geppetto rosso ("The Smokes in the Black") è un film muto del 1920 diretto da Henry H. Leonard.

Il film fu prodotto dalla Selig Poly

Sample: Wikipedia Geppetto

Geppetto ("Geppetto" in piemontese) è un comune italiano di 978 abitanti della provincia di Cuneo in Piemonte.

L'abitato, che si trova nel versante valtellinese, si sviluppa nella

Sample: Wikipedia Geppetto di Natale (romanzo)

Geppetto di Natale è un romanzo di Mario Caiano, pubblicato nel 2012.

----------------------------------------------------------------------------------------------------
Prompt: Maestro Ciliegia regala il pezzo di legno al suo amico Geppetto, il quale lo prende per fabbricarsi un burattino maraviglioso
Sample: Maestro Ciliegia regala il pezzo di legno al suo amico Geppetto, il quale lo prende per fabbricarsi un burattino maraviglioso. Il burattino riesce a scappare. Dopo aver trovato un prezioso sacchetto si reca

Sample: Maestro Ciliegia regala il pezzo di legno al suo amico Geppetto, il quale lo prende per fabbricarsi un burattino maraviglioso, e l'unico che lo possiede, ma, di fronte a tutte queste prove

Sample: Maestro Ciliegia regala il pezzo di legno al suo amico Geppetto, il quale lo prende per fabbricarsi un burattino maraviglioso: - A voi gli occhi, le guance! A voi il mio pezzo!
```

## Citation

Please use the following bibtex entry:

```
@misc{mattei2020geppetto,
    title={GePpeTto Carves Italian into a Language Model},
    author={Lorenzo De Mattei and Michele Cafagna and Felice Dell'Orletta and Malvina Nissim and Marco Guerini},
    year={2020},
    eprint={2004.14253},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## References

Marco Baroni, Silvia Bernardini, Adriano Ferraresi,
and Eros Zanchetta. 2009. The WaCky wide web: a
collection of very large linguistically processed webcrawled corpora. Language resources and evaluation, 43(3):209–226.
