---
language: en
license: apache-2.0
datasets:
- cnn_dailymail
---

# Roberta2Roberta_L-24_cnn_daily_mail EncoderDecoder model

The model was introduced in 
[this paper](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn and first released in [this repository](https://tfhub.dev/google/bertseq2seq/roberta24_cnndm/1). 

The model is an encoder-decoder model that was initialized on the `roberta-large` checkpoints for both the encoder 
and decoder and fine-tuned on summarization on the CNN / Dailymail dataset, which is linked above.

Disclaimer: The model card has been written by the Hugging Face team.

## How to use

You can use this model for summarization, *e.g.*

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_cnn_daily_mail")

article = """	(The Hollywood Reporter)"The Rocky Horror Picture
Show" is the latest musical getting the small-
screen treatment. Fox is developing a two-hour
remake of the 1975 cult classic to be directed,
executive-produced and choreographed by Kenneth
Ortega ("High School Musical"). The project,
tentatively titled "The Rocky Horror Picture Show
Event," is casting-contingent. The special will be
filmed in advance and not air live, but few
details beyond that are known. In addition to
Ortega, Gail Berman and Lou Adler, who produced
the original film, are also attached as executive
producers. The special will be produced by Fox 21
Television Studios, and Berman's The Jackal Group.
The special is timed to celebrate the 40th
anniversary of the film, which has grossed more
than $112 million and still plays in theaters
across the country. TV premiere dates: The
complete guide . This isn't the first stab at
adapting "The Rocky Horror Picture Show." In 2002,
Fox unveiled plans for an adaptation timed to the
30th anniversary that never came to fruition. The
faces of pilot season 2015 . Fox's "Glee" covered
several of the show's most popular songs for a
Season 2 episode and even released a special "The
Rocky Horror Glee Show" EP. There is no plan yet
for when the adaptation will air. Fox also has a
live musical production of "Grease", starring
Julianne Hough and Vanessa Hudgens, scheduled to
air on Jan. 31, 2016. Broadcast TV scorecard .
Following in the footsteps of "The Sound of Music"
and "Peter Pan," NBC recently announced plans to
air a live version of The Wiz later this year.
Ortega's credits include "Gilmore Girls," "This Is
It" and "Hocus Pocus." He is repped by Paradigm
and Hanson, Jacobson. ©2015 The Hollywood
Reporter. All rights reserved."""

input_ids = tokenizer(article, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# Fox is developing a two-hour remake of the 1975 cult classic. The special will be directed, executive-produced and choreographed by Kenneth Ortega. 
# The special is timed to celebrate the 40th anniversary of the film, which has grossed more than $112 million.

```
