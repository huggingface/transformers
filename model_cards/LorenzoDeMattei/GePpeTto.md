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
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('LorenzoDeMattei/GePpeTto')
model = GPT2LMHeadModel.from_pretrained(
    'LorenzoDeMattei/GePpeTto', pad_token_id = tokenizer.eos_token_id
)

input_ids = tokenizer.encode(
    'Wikipedia Geppetto', return_tensors = 'pt'
)
sample_outputs = model.generate(
    input_ids,
    do_sample = True,
    max_length = 50,
    top_k = 50,
    top_p = 0.95,
    num_return_sequences = 3,
)

print('Output:\n' + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print(
        '{}: {}'.format(
            i, tokenizer.decode(sample_output, skip_special_tokens = True)
        )
    )
```

Output is,

```text
Output:
----------------------------------------------------------------------------------------------------
0: Wikipedia Geppetto

Geppetto è una città degli Stati Uniti d'America, situata nello Stato dell'Iowa, nella Contea di Greene.

Wikipedia The Sax

The Sax è il primo album discografico
2: Wikipedia Geppetto/Passione

Geppetto è il primo album in studio dei Saturday Night Live, pubblicato dalla Iron Maiden nel 1974.

L'album è un lavoro di debutto che lo porta a definire
3: Wikipedia Geppetto

Geppetto ("Fenëvëv" in calabrese) è un comune italiano di abitanti della regione Calabria.

Zona di particolare pregio storico-artistico, paesaggistico, storico-artistico,
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
